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
Minimal Nsight Compute (ncu) profiling harness for la_verify_kvbuffer.

Isolates the verify kernel so ncu profiles ONLY that launch (no SGLang /
decode_mtp / state_update noise). Wraps the measured launches in an NVTX
range named "verify" so you can filter with `--nvtx --nvtx-include "verify/"`.

Usage (on the GPU box):
    # quick wall-clock sanity check, no ncu
    python benchmarks/profile_verify_kvbuffer.py --B 128 --T 8 --write-kv

    # under ncu (see commands in the chat / docstring below)
    ncu --set full --nvtx --nvtx-include "verify/" -o verify_B128_T8 \
        python benchmarks/profile_verify_kvbuffer.py --B 128 --T 8 --write-kv --iters 1
"""

import argparse

import torch

from cula.lightning.la_verify_kvbuffer import linear_attention_verify_kvbuffer


def build_inputs(B, T, H, K, V, layer_idx, num_layers, device, write_kv):
    dtype = torch.bfloat16
    scale = K ** -0.5
    HV = H  # SGLang seg_la does not support GQA; keep parity with the benchmark

    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(
        H, device=device, dtype=torch.float32
    )
    decay_scales = -g_gamma  # cuLA convention: exp(-decay_scales)

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    state_init_kmaj = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

    s_kvbuf = state_init_kmaj.permute(0, 1, 3, 2).contiguous()  # [B, HV, V, K]
    out = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    h0_indices = torch.arange(B, device=device, dtype=torch.int32)

    k_buf = v_buf = None
    if write_kv:
        k_buf = torch.zeros(B, T, H, K, device=device, dtype=dtype)
        v_buf = torch.zeros(B, T, HV, V, device=device, dtype=dtype)

    return dict(
        q=q, k=k, v=v, s=s_kvbuf, out=out, decay_scales=decay_scales,
        h0_indices=h0_indices, scale=scale, T=T, k_buf=k_buf, v_buf=v_buf,
    )


def call_verify(args_d):
    linear_attention_verify_kvbuffer(
        args_d["q"], args_d["k"], args_d["v"], args_d["s"], args_d["out"],
        args_d["decay_scales"], args_d["h0_indices"], args_d["scale"], args_d["T"],
        k_buf=args_d["k_buf"], v_buf=args_d["v_buf"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layer-idx", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--write-kv", action="store_true",
                   help="match benchmark's cu_vfy (verify + KV buffer write)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=1,
                   help="measured launches inside the NVTX range (use 1 under ncu)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    device = "cuda"
    K = V = args.head_dim

    d = build_inputs(args.B, args.T, args.heads, K, V,
                     args.layer_idx, args.num_layers, device, args.write_kv)

    # Warmup: triggers cute.compile (first call) + steady state.
    for _ in range(args.warmup):
        call_verify(d)
    torch.cuda.synchronize()

    # Measured region — wrapped in an NVTX range for ncu filtering.
    torch.cuda.nvtx.range_push("verify")
    for _ in range(args.iters):
        call_verify(d)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Wall-clock fallback when not running under ncu.
    n = 50
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        call_verify(d)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med = times[len(times) // 2]
    print(f"B={args.B} T={args.T} write_kv={args.write_kv} "
          f"median_verify={med * 1e3:.2f} us")


if __name__ == "__main__":
    main()
