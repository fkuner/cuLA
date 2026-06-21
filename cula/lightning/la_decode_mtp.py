# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightning Attention MTP (Multi-Token Processing) Decode Kernel.

Processes T > 1 tokens in one launch with h held in registers across the
whole T-loop. Targeted at speculative-decoding verify scenarios.

Per timestep:
    h_t = exp(-decay_scales[h]) * h_{t-1} + k_t ⊗ v_t
    o_t = (h_t @ q_t) * softmax_scale

`decay_scales` is per-head and time-invariant, so `r_decay` is computed ONCE
outside the T-loop.

Grid: (B * HV * num_v_tiles, 1, 1). Each block handles one [tile_v] slice
across all T timesteps; h for that slice stays in registers.

Reference: flashinfer/flashinfer/gdn_kernels/gdn_decode_mtp.py (inline variant).
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.utils import USE_FAST_MATH, get_device_sm_version

# ============================================================================
# Global configuration
# ============================================================================
TILE_K_MTP = 128
NUM_THREADS_MTP = 128  # 4 warps


# ============================================================================
# FMA pair helpers (packed F32x2 on SM100; scalar fallback on SM90)
# ============================================================================
@cute.jit
def la_update_pair(h_lo, h_hi, k_lo, k_hi, v_j, decay, use_packed_fma: cutlass.Constexpr[bool]):
    """Inner LA recurrence on a (lo, hi) pair: h = h*decay + k*v_j."""
    if cutlass.const_expr(use_packed_fma):
        # h *= decay   (packed mul implemented as FMA with src_c=0)
        h_lo, h_hi = cute.arch.fma_packed_f32x2(
            src_a=(h_lo, h_hi),
            src_b=(decay, decay),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
        # h += k * v_j
        h_lo, h_hi = cute.arch.fma_packed_f32x2(
            src_a=(k_lo, k_hi),
            src_b=(v_j, v_j),
            src_c=(h_lo, h_hi),
        )
        return h_lo, h_hi
    else:
        return h_lo * decay + k_lo * v_j, h_hi * decay + k_hi * v_j


@cute.jit
def hq_dot_pair(h_lo, h_hi, q_lo, q_hi, sum_lo, sum_hi, use_packed_fma: cutlass.Constexpr[bool]):
    """Accumulate dot product over a (lo, hi) pair: sum += h * q."""
    if cutlass.const_expr(use_packed_fma):
        return cute.arch.fma_packed_f32x2(
            src_a=(h_lo, h_hi),
            src_b=(q_lo, q_hi),
            src_c=(sum_lo, sum_hi),
        )
    else:
        return h_lo * q_lo + sum_lo, h_hi * q_hi + sum_hi


# TODO: re-tune for LA after first benchmark.
# TODO (perf): for configs with row_iters > 1 (e.g. tile_v=64, ilp=4), q/k are
# reloaded from global on every row-loop iteration because the row-outer / T-inner
# structure is required to keep h register-resident across T (r_h budget is 8 rows).
# Stage q/k in SMEM per i_t (cooperative load + barrier) to avoid the (row_iters - 1)
# redundant reads; worst case (tile_v=64, ilp=4) wastes 3x the q/k bandwidth.
def get_mtp_config(B: int, T: int, HV: int, V: int, disable_state_update: bool) -> tuple:
    """Pick (tile_v, vec_size, ilp_rows, use_smem_v) based on work units.

    Thresholds ported from GDN MTP (B200 grid search on Qwen3.5, HV=64).
    LA's per-step compute is ~30% lighter (no delta rule), so we may need
    to retune; the structure is preserved for now.
    """
    work_units = B * HV
    vec_size = 4

    if work_units <= 64:
        tile_v, ilp_rows, use_smem_v = 8, 2, False
    elif work_units <= 128:
        tile_v, ilp_rows, use_smem_v = 16, 4, False
    elif work_units <= 448:
        if T <= 2:
            tile_v, ilp_rows, use_smem_v = 16, 2, False
        else:
            tile_v, ilp_rows, use_smem_v = 32, 4, False
    elif work_units <= 1024:
        tile_v, ilp_rows, use_smem_v = 32, 4, False
    else:
        tile_v = 64
        use_smem_v = True
        ilp_rows = 4
        if not disable_state_update and T <= 2:
            ilp_rows = 8
            use_smem_v = False

    tile_v = min(tile_v, V)
    rows_per_group = tile_v // 4
    assert rows_per_group % ilp_rows == 0, (
        f"tile_v={tile_v} / num_groups=4 / ilp_rows={ilp_rows} doesn't divide cleanly "
        f"(rows_per_group={rows_per_group}); the ILP loop would run zero iterations."
    )
    return tile_v, vec_size, ilp_rows, use_smem_v


# ============================================================================
# Kernel
# ============================================================================
@cute.kernel
def la_verify_kernel_mtp(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] fp32 (or dummy)
    decay_scales: cute.Tensor,  # [H] fp32
    q: cute.Tensor,  # [B, T, H, K] bf16
    k: cute.Tensor,  # [B, T, H, K] bf16
    v: cute.Tensor,  # [B, T, HV, V] bf16
    o: cute.Tensor,  # [B, T, HV, V] bf16
    h0_indices: cute.Tensor,  # [B] int32
    cu_seqlens: cute.Tensor,  # [B+1] int32 (dummy when is_varlen=False)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    ilp_rows: cutlass.Constexpr[int],
    use_smem_v: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]

    # ------------------------------------------------------------------
    # SMEM allocation (sVdata + sOutput only — LA has no Phase 1 work)
    # ------------------------------------------------------------------
    smem = cutlass.utils.SmemAllocator()
    sVdata = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16)
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16)

    # ------------------------------------------------------------------
    # Register tensors
    # ------------------------------------------------------------------
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    # r_h always declared with 8 rows; ilp_rows constexpr picks which are used.
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)

    if cache_idx >= 0:
        # r_decay is a T-loop invariant — computed ONCE.
        r_decay = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

        # Optional v preload to SMEM (cooperative load across the whole block).
        if cutlass.const_expr(use_smem_v):
            for i_t in cutlass.range_constexpr(T):
                v_tile_start = i_v * tile_v
                if tidx < tile_v:
                    v_global_idx = v_tile_start + tidx
                    if v_global_idx < V:
                        sVdata[(i_t, tidx)] = cutlass.Float32(v[i_n, i_t, i_hv, v_global_idx])
            cute.arch.barrier()

        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # Process `ilp_rows` V-rows per iteration. ilp_rows is a compile-time
        # constant, so range_constexpr fully unrolls the slot loops below — the
        # generated SASS is identical to hand-unrolling each ilp_rows value, but
        # one loop covers ilp_rows ∈ {2, 4, 8}.
        num_chunks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for chunk in cutlass.range_constexpr(num_chunks):
            v_idx_0 = i_v * tile_v + group_idx * rows_per_group + chunk * ilp_rows
            if v_idx_0 + (ilp_rows - 1) < V:
                # Load ilp_rows h-state rows ONCE; they stay register-resident across T.
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_0 + slot, lane_in_group),
                    )
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                for i_t in cutlass.range_constexpr(T):
                    # ---- inline q/k load for this t ----
                    q_tile = cute.local_tile(
                        q,
                        (1, 1, 1, vec_size),
                        (i_n, i_t, i_h, lane_in_group),
                    )
                    k_tile = cute.local_tile(
                        k,
                        (1, 1, 1, vec_size),
                        (i_n, i_t, i_h, lane_in_group),
                    )
                    cute.autovec_copy(q_tile, r_q_bf16)
                    cute.autovec_copy(k_tile, r_k_bf16)
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i]) * scale
                        r_k[i] = cutlass.Float32(r_k_bf16[i])

                    # Per-row dot-product accumulators (lo, hi) — zeroed each t step.
                    r_dot_lo = cute.make_rmem_tensor(cute.make_layout((ilp_rows,), stride=(1,)), cutlass.Float32)
                    r_dot_hi = cute.make_rmem_tensor(cute.make_layout((ilp_rows,), stride=(1,)), cutlass.Float32)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        r_dot_lo[slot] = cutlass.Float32(0.0)
                        r_dot_hi[slot] = cutlass.Float32(0.0)

                    # ---- fused decay + rank-1 update (per V-row) ----
                    for slot in cutlass.range_constexpr(ilp_rows):
                        if cutlass.const_expr(use_smem_v):
                            r_v_s = sVdata[(i_t, v_idx_0 - i_v * tile_v + slot)]
                        else:
                            r_v_s = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_0 + slot])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[slot, j], r_h[slot, j + 1] = la_update_pair(
                                r_h[slot, j],
                                r_h[slot, j + 1],
                                r_k[j],
                                r_k[j + 1],
                                r_v_s,
                                r_decay,
                                use_packed_fma,
                            )

                    # ---- optional intermediate-state cache ----
                    if cutlass.const_expr(cache_intermediate_states):
                        flat_idx = i_n * T * HV + i_t * HV + i_hv
                        for slot in cutlass.range_constexpr(ilp_rows):
                            inter_tile = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_0 + slot, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (slot, None)), inter_tile)

                    # ---- o_t = h_t @ q_t (per-row warp reduce) ----
                    for slot in cutlass.range_constexpr(ilp_rows):
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_dot_lo[slot], r_dot_hi[slot] = hq_dot_pair(
                                r_h[slot, j],
                                r_h[slot, j + 1],
                                r_q[j],
                                r_q[j + 1],
                                r_dot_lo[slot],
                                r_dot_hi[slot],
                                use_packed_fma,
                            )
                        r_acc = r_dot_lo[slot] + r_dot_hi[slot]
                        for offset in [16, 8, 4, 2, 1]:
                            r_acc += cute.arch.shuffle_sync_bfly(r_acc, offset=offset, mask=-1, mask_and_clamp=31)
                        r_dot_lo[slot] = r_acc  # reuse slot for final result

                    # ---- writeback ----
                    if lane_in_group == 0:
                        if cutlass.const_expr(use_smem_v):
                            vla = v_idx_0 - i_v * tile_v
                            for slot in cutlass.range_constexpr(ilp_rows):
                                sOutput[(i_t, vla + slot)] = cutlass.BFloat16(r_dot_lo[slot])
                        else:
                            for slot in cutlass.range_constexpr(ilp_rows):
                                o[(i_n, i_t, i_hv, v_idx_0 + slot)] = cutlass.BFloat16(r_dot_lo[slot])

                # Final state writeback
                if cutlass.const_expr(not disable_state_update):
                    for slot in cutlass.range_constexpr(ilp_rows):
                        h_tile_out = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_0 + slot, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_tile_out)

        # Cooperative output writeback (only when use_smem_v staged outputs to SMEM)
        if cutlass.const_expr(use_smem_v):
            cute.arch.barrier()
            v_tile_base = i_v * tile_v
            for t_idx in cutlass.range_constexpr(T):
                if tidx < tile_v:
                    v_global = v_tile_base + tidx
                    if v_global < V:
                        o[(i_n, t_idx, i_hv, v_global)] = sOutput[(t_idx, tidx)]


# ============================================================================
# Launcher
# ============================================================================
@cute.jit
def run_la_verify_kernel_mtp(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_smem_v: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _, v_dim, _ = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = B * HV * num_v_tiles

    smem_bytes = (
        4 * T * tile_v  # sVdata
        + 2 * T * tile_v  # sOutput
        + 128  # alignment
    )

    la_verify_kernel_mtp(
        h0_source,
        intermediate_states,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        cu_seqlens,
        vec_size,
        num_v_tiles,
        tile_v,
        scale,
        B,
        T,
        H,
        HV,
        K,
        V,
        disable_state_update,
        cache_intermediate_states,
        is_varlen,
        ilp_rows,
        use_smem_v,
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# Compile cache
# ============================================================================
@functools.cache
def _get_compiled_la_mtp_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    softmax_scale: float,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    is_varlen: bool,
    tile_v: int,
    vec_size: int,
    ilp_rows: int,
    use_smem_v: bool,
    use_packed_fma: bool,
):
    return {}


# ============================================================================
# Public Python entry point
# ============================================================================
def linear_attention_decode_mtp(
    q: torch.Tensor,  # [B, T, H, K] bf16
    k: torch.Tensor,  # [B, T, H, K] bf16
    v: torch.Tensor,  # [B, T, HV, V] bf16
    s: torch.Tensor,  # [pool_size, HV, V, K] fp32
    intermediate_states: torch.Tensor,  # [pool_size*T*HV, V, K] fp32 (or dummy)
    out: torch.Tensor,  # [B, T, HV, V] bf16
    decay_scales: torch.Tensor,  # [H] fp32
    s_offsets: torch.Tensor,  # [B] int32 (-1 to skip)
    cu_seqlens: torch.Tensor,  # [B+1] int32 (reserved; see note below)
    softmax_scale: float,
    T: int,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    is_varlen: bool,
) -> None:
    """
    Lightning Attention multi-token decode (T > 1).

    Writes to ``out``; updates ``s`` in place unless ``disable_state_update`` is True;
    writes ``intermediate_states`` when ``cache_intermediate_states`` is True.

    NOTE: For any batch ``i`` where ``s_offsets[i] < 0`` the kernel skips that batch
    entirely — ``out[i]`` is LEFT UNCHANGED, and neither ``s`` nor
    ``intermediate_states`` is written for that slot. Callers must initialize ``out``
    to a known value (e.g. ``torch.zeros``) before the call if any downstream code
    may read those slots.

    NOTE: ``is_varlen`` and ``cu_seqlens`` are reserved in the signature to keep the
    public API stable, but the early-stop branch is NOT implemented yet — same as
    upstream flashinfer GDN MTP, which also exposes the flag without consuming it.
    Callers should pass ``is_varlen=False`` and any int32 tensor for ``cu_seqlens``.
    The kernel descriptor is built with ``assumed_align=16``, so even the dummy
    ``cu_seqlens`` must be 16-byte aligned; pass a fresh ``torch.empty(N, dtype=int32)``
    (CUDA allocator guarantees alignment) — do NOT pass a slice that may misalign.
    """
    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    tile_v, vec_size, ilp_rows, use_smem_v = get_mtp_config(B, T, HV, V, disable_state_update)
    assert V % ilp_rows == 0, f"V={V} % ilp_rows={ilp_rows} ≠ 0: partial row-blocks would be silently skipped"
    major, _ = get_device_sm_version(q.device)
    use_packed_fma = major >= 10

    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        softmax_scale,
        disable_state_update,
        cache_intermediate_states,
        is_varlen,
        tile_v,
        vec_size,
        ilp_rows,
        use_smem_v,
        use_packed_fma,
    )
    cache = _get_compiled_la_mtp_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        compiled = cute.compile(
            run_la_verify_kernel_mtp,
            from_dlpack(h0_view, assumed_align=16),
            from_dlpack(intermediate_states, assumed_align=16),
            from_dlpack(decay_scales, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(s_offsets, assumed_align=16),
            from_dlpack(cu_seqlens, assumed_align=16),
            scale=softmax_scale,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_smem_v=use_smem_v,
            use_packed_fma=use_packed_fma,
            disable_state_update=disable_state_update,
            cache_intermediate_states=cache_intermediate_states,
            is_varlen=is_varlen,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        intermediate_states,
        decay_scales,
        q,
        k,
        v,
        out,
        s_offsets,
        cu_seqlens,
        stream,
    )
