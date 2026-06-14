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
):
    # KERNEL BODY FILLED IN TASK 3.
    return


@cute.jit
def run_la_state_update_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
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
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    la_state_update_kernel(
        h0_source,
        decay_scales,
        k,
        v,
        h0_indices,
        accepted_len,
        vec_size,
        num_v_tiles,
        tile_v,
        B,
        T,
        H,
        HV,
        K,
        V,
        ilp_rows,
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        stream=stream,
    )


@functools.cache
def _get_compiled_state_update_kernel(
    B: int, T: int, H: int, HV: int, K: int, V: int,
    pool_size: int, tile_v: int, vec_size: int, ilp_rows: int, use_packed_fma: bool,
):
    return {}


def linear_attention_state_update_kvbuffer(
    k: torch.Tensor,            # [B, T, H,  K] bf16
    v: torch.Tensor,            # [B, T, HV, V] bf16
    s: torch.Tensor,            # [pool_size, HV, V, K] fp32, WRITTEN IN PLACE
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,   # [B] int32, -1 to skip
    accepted_len: torch.Tensor, # [B] int32, in [0, T]
    T: int,                     # = k.shape[1]
) -> None:
    """
    Advance pooled state from h_init to h_state_L per batch (KVBuffer Eq. 8).

    For batch b: if h0_indices[b] < 0 OR accepted_len[b] == 0, the pool slot is
    left unchanged. Otherwise s[h0_indices[b]] is overwritten with the state after
    accepted_len[b] recurrent steps over (k, v).
    """
    B, T_k, H, K = k.shape
    assert T_k == T, f"k.shape[1]={T_k} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    # disable_state_update is irrelevant here; pass False to get the same tiling
    # the verify kernel uses for the h0 read alignment.
    tile_v, vec_size, ilp_rows, _use_smem_v = get_mtp_config(B, T, HV, V, False)
    major, _ = get_device_sm_version(k.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, tile_v, vec_size, ilp_rows, use_packed_fma,
    )
    cache = _get_compiled_state_update_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

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
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
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
        k, v,
        h0_indices,
        accepted_len,
        stream,
    )
