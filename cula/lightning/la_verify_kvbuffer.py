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
    use_smem_v: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
):
    # KERNEL BODY FILLED IN TASK 5.
    return


@cute.jit
def run_la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
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
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    # SMEM: staged v per (t, tile) + staged output per (t, tile), same as baseline.
    smem_bytes = 0
    if cutlass.const_expr(use_smem_v):
        smem_bytes = T * tile_v * 4 + T * tile_v * 2  # fp32 sVdata + bf16 sOutput

    la_verify_kvbuffer_kernel(
        h0_source,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
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
        use_smem_v,
        use_packed_fma,
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
    tile_v: int, vec_size: int, ilp_rows: int, use_smem_v: bool, use_packed_fma: bool,
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
) -> None:
    """
    Closed-form parallel verify (KVBuffer Eq. 7). Writes out; does not touch s.

    For batch b with h0_indices[b] < 0, out[b] is LEFT UNCHANGED — callers must
    pre-initialize out if downstream code reads those slots.
    """
    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    tile_v, vec_size, ilp_rows, use_smem_v = get_mtp_config(B, T, HV, V, True)
    major, _ = get_device_sm_version(q.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, softmax_scale,
        tile_v, vec_size, ilp_rows, use_smem_v, use_packed_fma,
    )
    cache = _get_compiled_verify_kvbuffer_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

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
            scale=softmax_scale,
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_smem_v=use_smem_v,
            use_packed_fma=use_packed_fma,
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
        stream,
    )
