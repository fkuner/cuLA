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
Lightning Attention KVBuffer verify kernel (paper Eq. 7 for LA).

Closed-form parallel verification — derives each step's state from (h0, k, v)
instead of materializing it:

    o_t = alpha^{t+1} * (h0 @ q_t * scale)
        + sum_{i=0..t} alpha^{t-i} * (q_t . k_i * scale) * v_i

Reads s, q, k, v; writes o. Never touches s (no state side-effect), never writes
intermediate states. The post-acceptance state write is the separate
linear_attention_state_update_kvbuffer kernel.

Grid: (B * HV * num_v_tiles, 1, 1), 128 threads/block — identical to baseline.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.utils import USE_FAST_MATH, get_device_sm_version
from cula.lightning.la_decode_mtp import (
    NUM_THREADS_MTP,
    get_mtp_config,
    hq_dot_pair,
)


@cute.kernel
def la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,     # [pool_size * HV, V, K] fp32 (READ ONLY)
    decay_scales: cute.Tensor,  # [H] fp32
    q: cute.Tensor,             # [B, T, H,  K] bf16
    k: cute.Tensor,             # [B, T, H,  K] bf16
    v: cute.Tensor,             # [B, T, HV, V] bf16
    o: cute.Tensor,             # [B, T, HV, V] bf16 (WRITTEN)
    h0_indices: cute.Tensor,    # [B] int32
    k_buf: cute.Tensor,         # [pool_size, T, H, K] bf16 (WRITTEN when write_kv)
    v_buf: cute.Tensor,         # [pool_size, T, HV, V] bf16 (WRITTEN when write_kv)
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
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    write_kv: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size           # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group   # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp            # 4

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

    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_q_seq = cute.make_rmem_tensor(cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_k_seq = cute.make_rmem_tensor(cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_decay_pow = cute.make_rmem_tensor(cute.make_layout((T + 1,), stride=(1,)), cutlass.Float32)
    o_partial = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)

    smem = cutlass.utils.SmemAllocator()
    s_qk_scaled = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16
    )
    # v staged to SMEM (block-shared over the whole v-tile). v has no K dim, so
    # keeping it in per-lane registers wasted 8*T regs/thread and capped occupancy;
    # SMEM costs only T*tile_v*4 bytes and is read warp-uniformly (broadcast).
    sVdata = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
    )

    if cache_idx >= 0:
        alpha = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

        # alpha^0 .. alpha^T  (T+1 powers; term1 uses alpha^{t+1})
        r_decay_pow[0] = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(1, T + 1):
            r_decay_pow[t] = r_decay_pow[t - 1] * alpha

        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # Stage all T q (scaled) and k (fp32) for this lane's K-slice.
        for t in cutlass.range_constexpr(T):
            q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)
            for j in cutlass.range_constexpr(vec_size):
                r_q_seq[t, j] = cutlass.Float32(r_q_bf16[j]) * scale
                r_k_seq[t, j] = cutlass.Float32(r_k_bf16[j])

            # Write k to buffer — gated: only one block per (b, h, t) writes
            if cutlass.const_expr(write_kv):
                if i_v == 0 and i_hv % (HV // H) == 0:
                    kb_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                              (cache_idx, t, i_h, lane_in_group))
                    cute.autovec_copy(r_k_bf16, kb_tile)

        # Phase 1: cooperative QK matrix — 4 warps split T*(T+1)/2 qk dot products.
        # Warp w handles rows where min(t, T-1-t) % 4 == w (head-tail pairing) so that
        # each warp's total row-length is balanced: heavy tail rows are paired with light
        # head rows, making per-warp work ≈ T*(T+1)/8 regardless of T.
        for t_assign in cutlass.range_constexpr(T):
            if min(t_assign, T - 1 - t_assign) % 4 == warp_idx:
                for i in cutlass.range_constexpr(t_assign + 1):
                    qk_lo = cutlass.Float32(0.0)
                    qk_hi = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(0, vec_size, 2):
                        qk_lo, qk_hi = hq_dot_pair(
                            r_q_seq[t_assign, j], r_q_seq[t_assign, j + 1],
                            r_k_seq[i, j], r_k_seq[i, j + 1],
                            qk_lo, qk_hi, use_packed_fma)
                    qk = qk_lo + qk_hi
                    for offset in [16, 8, 4, 2, 1]:
                        qk += cute.arch.shuffle_sync_bfly(qk, offset=offset, mask=-1, mask_and_clamp=31)
                    if lane_in_group == 0:
                        s_qk_scaled[(t_assign, i)] = r_decay_pow[t_assign - i] * qk

        # Cooperative v load: first tile_v threads each stage one v-row for all T
        # steps into SMEM. v_buf write (when enabled) is fused here — every
        # (cache_idx, t, hv, v_row) is written exactly once by its owning thread,
        # replacing the old single-lane scalar store.
        v_tile_start = i_v * tile_v
        for t in cutlass.range_constexpr(T):
            if tidx < tile_v:
                v_global_idx = v_tile_start + tidx
                if v_global_idx < V:
                    vv = v[i_n, t, i_hv, v_global_idx]
                    sVdata[(t, tidx)] = cutlass.Float32(vv)
                    if cutlass.const_expr(write_kv):
                        v_buf[(cache_idx, t, i_hv, v_global_idx)] = vv

        cute.arch.barrier()

        num_row_blocks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for row_block in cutlass.range_constexpr(num_row_blocks):
            v_base = i_v * tile_v + group_idx * rows_per_group + row_block * ilp_rows
            v_local = group_idx * rows_per_group + row_block * ilp_rows  # offset within sVdata's v-tile
            if v_base + (ilp_rows - 1) < V:
                # Load h_init rows (persistent across the T loop).
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_base + slot, lane_in_group))
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                for t in cutlass.range_constexpr(T):
                    # term1: alpha^{t+1} * (h_init @ q_t)  (per-slot warp reduce)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        hq_lo = cutlass.Float32(0.0)
                        hq_hi = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            hq_lo, hq_hi = hq_dot_pair(
                                r_h[slot, j], r_h[slot, j + 1],
                                r_q_seq[t, j], r_q_seq[t, j + 1],
                                hq_lo, hq_hi, use_packed_fma)
                        hq = hq_lo + hq_hi
                        for offset in [16, 8, 4, 2, 1]:
                            hq += cute.arch.shuffle_sync_bfly(hq, offset=offset, mask=-1, mask_and_clamp=31)
                        o_partial[slot] = r_decay_pow[t + 1] * hq

                    # term2: read pre-computed decay-scaled qk + staged v from SMEM
                    for i in cutlass.range_constexpr(t + 1):
                        coeff = s_qk_scaled[(t, i)]
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o_partial[slot] = o_partial[slot] + coeff * sVdata[(i, v_local + slot)]

                    # writeback (all lanes hold the reduced value; lane 0 writes)
                    if lane_in_group == 0:
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o[(i_n, t, i_hv, v_base + slot)] = cutlass.BFloat16(o_partial[slot])


@cute.jit
def run_la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
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
    use_packed_fma: cutlass.Constexpr[bool],
    write_kv: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    smem_bytes = T * T * 4 + T * tile_v * 4  # s_qk_scaled[T][T] + sVdata[T][tile_v]

    la_verify_kvbuffer_kernel(
        h0_source,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        k_buf,
        v_buf,
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
        ilp_rows,
        use_packed_fma,
        write_kv,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_verify_kvbuffer_kernel(
    B: int, T: int, H: int, HV: int, K: int, V: int,
    pool_size: int, softmax_scale: float,
    tile_v: int, vec_size: int, ilp_rows: int, use_packed_fma: bool,
    write_kv: bool,
):
    return {}


def linear_attention_verify_kvbuffer(
    q: torch.Tensor,            # [B, T, H,  K] bf16
    k: torch.Tensor,            # [B, T, H,  K] bf16
    v: torch.Tensor,            # [B, T, HV, V] bf16
    s: torch.Tensor,            # [pool_size, HV, V, K] fp32, READ ONLY
    out: torch.Tensor,          # [B, T, HV, V] bf16, WRITTEN
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,   # [B] int32, -1 to skip
    softmax_scale: float,
    T: int,
    k_buf: torch.Tensor | None = None,  # [pool_size, T, H, K] bf16, WRITTEN
    v_buf: torch.Tensor | None = None,  # [pool_size, T, HV, V] bf16, WRITTEN
) -> None:
    """
    Closed-form parallel verify (KVBuffer Eq. 7). Writes out; does not touch s.

    When k_buf and v_buf are provided, also writes k,v to pool-indexed buffers
    so the caller can free the original k,v tensors after this call returns.

    For batch b with h0_indices[b] < 0, out[b] is LEFT UNCHANGED — callers must
    pre-initialize out if downstream code reads those slots.
    """
    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    write_kv = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")

    tile_v, vec_size, ilp_rows, _ = get_mtp_config(B, T, HV, V, True)
    major, _ = get_device_sm_version(q.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, softmax_scale,
        tile_v, vec_size, ilp_rows, use_packed_fma,
        write_kv,
    )
    cache = _get_compiled_verify_kvbuffer_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

    # Dummy tensors when write_kv=False (never accessed by kernel)
    if not write_kv:
        k_buf_t = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.bfloat16)
        v_buf_t = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.bfloat16)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_la_verify_kvbuffer_kernel,
            from_dlpack(h0_view, assumed_align=16),
            from_dlpack(decay_scales, assumed_align=16),
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(h0_indices, assumed_align=16),
            from_dlpack(k_buf_t, assumed_align=16),
            from_dlpack(v_buf_t, assumed_align=16),
            scale=softmax_scale,
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            write_kv=write_kv,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        q, k, v, out,
        h0_indices,
        k_buf_t, v_buf_t,
        stream,
    )
