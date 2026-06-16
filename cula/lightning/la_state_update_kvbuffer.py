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
Lightning Attention KVBuffer state-update kernel (paper Eq. 8 for LA).

After a parallel-verify cycle, advances the pooled state from h_init to
h_state_L for a per-batch accepted prefix length L = accepted_len[b]:

    h_running = h_init
    for i in 0..L-1:
        h_running = exp(-decay_scales[h]) * h_running + k_i ⊗ v_i
    s[cache_idx] = h_running

The loop body is bit-identical to the baseline T-loop body, so at L == T the
result is bit-equivalent to running the baseline with disable_state_update=False.

Reads s, k, v; writes s. Never touches q or o.

Grid: (B * HV * num_v_tiles, 1, 1), 128 threads/block — identical layout to the
baseline verify kernel, so the state write aligns with the verify kernel's h0 read.
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
    la_update_pair,
)


@cute.kernel
def la_state_update_kernel(
    h0_source: cute.Tensor,     # [pool_size * HV, V, K] fp32 (read + written in place)
    decay_scales: cute.Tensor,  # [H] fp32
    k: cute.Tensor,             # [B, T, H,  K] bf16
    v: cute.Tensor,             # [B, T, HV, V] bf16
    h0_indices: cute.Tensor,    # [B] int32
    accepted_len: cute.Tensor,  # [B] int32
    k_buf: cute.Tensor,         # [pool_size, T, H, K] bf16 (READ when read_from_buf)
    v_buf: cute.Tensor,         # [pool_size, T, HV, V] bf16 (READ when read_from_buf)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    read_from_buf: cutlass.Constexpr[bool],
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
    L = accepted_len[i_n]

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
    )

    if cache_idx >= 0 and L > 0:
        r_decay = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        if cutlass.const_expr(ilp_rows == 2):
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2
            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1
                if v_idx_b < V:
                    h_tile_a = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group))
                    h_tile_b = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group))
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    for i in cutlass.range(0, L, unroll=0):
                        if cutlass.const_expr(read_from_buf):
                            k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                                      (cache_idx, i, i_h, lane_in_group))
                        else:
                            k_tile = cute.local_tile(k, (1, 1, 1, vec_size),
                                                      (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])

                        if cutlass.const_expr(read_from_buf):
                            r_v_a = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_b])
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i, i_hv, v_idx_b])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[0, j], r_h[0, j + 1] = la_update_pair(
                                r_h[0, j], r_h[0, j + 1], r_k[j], r_k[j + 1], r_v_a, r_decay, use_packed_fma)
                            r_h[1, j], r_h[1, j + 1] = la_update_pair(
                                r_h[1, j], r_h[1, j + 1], r_k[j], r_k[j + 1], r_v_b, r_decay, use_packed_fma)

                    h_out_a = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group))
                    cute.autovec_copy(cute.slice_(r_h, (0, None)), h_out_a)
                    h_out_b = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group))
                    cute.autovec_copy(cute.slice_(r_h, (1, None)), h_out_b)

        elif cutlass.const_expr(ilp_rows == 4):
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4
            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3
                if v_idx_d < V:
                    for off, slot in ((0, 0), (1, 1), (2, 2), (3, 3)):
                        h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a + off, lane_in_group))
                        cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                    for i in cutlass.range(0, L, unroll=0):
                        if cutlass.const_expr(read_from_buf):
                            k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                                      (cache_idx, i, i_h, lane_in_group))
                        else:
                            k_tile = cute.local_tile(k, (1, 1, 1, vec_size),
                                                      (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])
                        if cutlass.const_expr(read_from_buf):
                            r_v_a = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_b])
                            r_v_c = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_c])
                            r_v_d = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_d])
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i, i_hv, v_idx_b])
                            r_v_c = cutlass.Float32(v[i_n, i, i_hv, v_idx_c])
                            r_v_d = cutlass.Float32(v[i_n, i, i_hv, v_idx_d])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[0, j], r_h[0, j + 1] = la_update_pair(
                                r_h[0, j], r_h[0, j + 1], r_k[j], r_k[j + 1], r_v_a, r_decay, use_packed_fma)
                            r_h[1, j], r_h[1, j + 1] = la_update_pair(
                                r_h[1, j], r_h[1, j + 1], r_k[j], r_k[j + 1], r_v_b, r_decay, use_packed_fma)
                            r_h[2, j], r_h[2, j + 1] = la_update_pair(
                                r_h[2, j], r_h[2, j + 1], r_k[j], r_k[j + 1], r_v_c, r_decay, use_packed_fma)
                            r_h[3, j], r_h[3, j + 1] = la_update_pair(
                                r_h[3, j], r_h[3, j + 1], r_k[j], r_k[j + 1], r_v_d, r_decay, use_packed_fma)

                    for off, slot in ((0, 0), (1, 1), (2, 2), (3, 3)):
                        h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a + off, lane_in_group))
                        cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)

        elif cutlass.const_expr(ilp_rows == 8):
            eighth_rows: cutlass.Constexpr[int] = rows_per_group // 8
            for row_oct in cutlass.range_constexpr(eighth_rows):
                v_idx_0 = i_v * tile_v + group_idx * rows_per_group + row_oct * 8
                if v_idx_0 + 7 < V:
                    for slot in cutlass.range_constexpr(8):
                        h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                        cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                    for i in cutlass.range(0, L, unroll=0):
                        if cutlass.const_expr(read_from_buf):
                            k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                                      (cache_idx, i, i_h, lane_in_group))
                        else:
                            k_tile = cute.local_tile(k, (1, 1, 1, vec_size),
                                                      (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])
                        for slot in cutlass.range_constexpr(8):
                            if cutlass.const_expr(read_from_buf):
                                r_v_s = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_0 + slot])
                            else:
                                r_v_s = cutlass.Float32(v[i_n, i, i_hv, v_idx_0 + slot])
                            for j in cutlass.range_constexpr(0, vec_size, 2):
                                r_h[slot, j], r_h[slot, j + 1] = la_update_pair(
                                    r_h[slot, j], r_h[slot, j + 1], r_k[j], r_k[j + 1], r_v_s, r_decay, use_packed_fma)

                    for slot in cutlass.range_constexpr(8):
                        h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                        cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)


@cute.jit
def run_la_state_update_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
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
    read_from_buf: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    la_state_update_kernel(
        h0_source, decay_scales, k, v, h0_indices, accepted_len,
        k_buf, v_buf,
        vec_size, num_v_tiles, tile_v,
        B, T, H, HV, K, V, ilp_rows, use_packed_fma, read_from_buf,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        stream=stream,
    )


@functools.cache
def _get_compiled_state_update_kernel(
    B: int, T: int, H: int, HV: int, K: int, V: int,
    pool_size: int, tile_v: int, vec_size: int, ilp_rows: int, use_packed_fma: bool,
    read_from_buf: bool,
):
    return {}


def linear_attention_state_update_kvbuffer(
    k: torch.Tensor,            # [B, T, H,  K] bf16 — read when k_buf is None
    v: torch.Tensor,            # [B, T, HV, V] bf16 — read when v_buf is None
    s: torch.Tensor,            # [pool_size, HV, V, K] fp32, WRITTEN IN PLACE
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,   # [B] int32, -1 to skip
    accepted_len: torch.Tensor, # [B] int32, in [0, T]
    T: int,
    k_buf: torch.Tensor | None = None,  # [pool_size, T, H, K] bf16
    v_buf: torch.Tensor | None = None,  # [pool_size, T, HV, V] bf16
) -> None:
    """
    Advance pooled state from h_init to h_state_L per batch (KVBuffer Eq. 8).

    When k_buf and v_buf are provided, reads k,v from pool-indexed buffers
    instead of batch-indexed input tensors.
    """
    B, T_k, H, K = k.shape
    assert T_k == T, f"k.shape[1]={T_k} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    read_from_buf = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")

    tile_v, vec_size, ilp_rows, _use_smem_v = get_mtp_config(B, T, HV, V, False)
    major, _ = get_device_sm_version(k.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, tile_v, vec_size, ilp_rows, use_packed_fma,
        read_from_buf,
    )
    cache = _get_compiled_state_update_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

    if not read_from_buf:
        k_buf_t = torch.empty(1, 1, 1, 1, device=k.device, dtype=torch.bfloat16)
        v_buf_t = torch.empty(1, 1, 1, 1, device=k.device, dtype=torch.bfloat16)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_la_state_update_kernel,
            from_dlpack(h0_view, assumed_align=16),
            from_dlpack(decay_scales, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(h0_indices, assumed_align=16),
            from_dlpack(accepted_len, assumed_align=16),
            from_dlpack(k_buf_t, assumed_align=16),
            from_dlpack(v_buf_t, assumed_align=16),
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            read_from_buf=read_from_buf,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        k, v,
        h0_indices,
        accepted_len,
        k_buf_t, v_buf_t,
        stream,
    )
