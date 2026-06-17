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
Minimal Nsight Compute (ncu) profiling harness for la_state_update_kvbuffer.

Isolates the state-update (commit) kernel so ncu profiles only that launch.
Wraps measured launches in an NVTX range named "commit".

Usage (on the GPU box):
    python benchmarks/profile_state_update_kvbuffer.py --B 128 --T 8 --read-buf

    ncu --nvtx --nvtx-include "commit/" --launch-count 1 --metrics ... \
        python benchmarks/profile_state_update_kvbuffer.py --B 128 --T 8 --read-buf --iters 1
"""

import argparse

import torch

from cula.lightning.la_state_update_kvbuffer import linear_attention_state_update_kvbuffer


def build_inputs(B, T, H, K, V, layer_idx, num_layers, device, read_buf):
    dtype = torch.bfloat16
    HV = H

    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(
        H, device=device, dtype=torch.float32
    )
    decay_scales = -g_gamma

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    state_init_kmaj = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01
    s = state_init_kmaj.permute(0, 1, 3, 2).contiguous()  # [B, HV, V, K]

    h0_indices = torch.arange(B, device=device, dtype=torch.int32)
    accepted_len = torch.full((B,), T, device=device, dtype=torch.int32)  # worst case L=T

    k_buf = v_buf = None
    if read_buf:
        k_buf = torch.zeros(B, T, H, K, device=device, dtype=dtype)
        v_buf = torch.zeros(B, T, HV, V, device=device, dtype=dtype)

    return dict(
        k=k, v=v, s=s, decay_scales=decay_scales, h0_indices=h0_indices,
        accepted_len=accepted_len, T=T, k_buf=k_buf, v_buf=v_buf,
        state_init=state_init_kmaj,
    )


def call_commit(d):
    # state-update mutates s in place; restore each call so timing is steady.
    d["s"].copy_(d["state_init"].permute(0, 1, 3, 2))
    linear_attention_state_update_kvbuffer(
        d["k"], d["v"], d["s"], d["decay_scales"], d["h0_indices"],
        d["accepted_len"], d["T"], k_buf=d["k_buf"], v_buf=d["v_buf"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layer-idx", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--read-buf", action="store_true",
                   help="match benchmark's cu_cmt (read k/v from pool buffers)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=1)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    device = "cuda"
    K = V = args.head_dim

    d = build_inputs(args.B, args.T, args.heads, K, V,
                     args.layer_idx, args.num_layers, device, args.read_buf)

    for _ in range(args.warmup):
        call_commit(d)
    torch.cuda.synchronize()

    # NVTX region must contain ONLY the state-update launch (no copy_ restore),
    # so ncu's --launch-count 1 picks the right kernel.
    torch.cuda.nvtx.range_push("commit")
    for _ in range(args.iters):
        linear_attention_state_update_kvbuffer(
            d["k"], d["v"], d["s"], d["decay_scales"], d["h0_indices"],
            d["accepted_len"], d["T"], k_buf=d["k_buf"], v_buf=d["v_buf"],
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    n = 50
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        call_commit(d)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med = times[len(times) // 2]
    print(f"B={args.B} T={args.T} read_buf={args.read_buf} median_commit={med * 1e3:.2f} us")


if __name__ == "__main__":
    main()
