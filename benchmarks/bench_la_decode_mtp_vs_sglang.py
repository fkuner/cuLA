#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Benchmark: cuLA LA decode MTP vs SGLang seg_la_mtp_kernel.

Compares three implementations of T > 1 Lightning Attention decode:
  1. sglang  `seg_la_mtp_kernel`                 (Triton, from SGLang upstream)
  2. cula    `linear_attention_decode_mtp`         (CuTe DSL, fused single-launch)
  3. cula    `linear_attention_verify_kvbuffer`     (CuTe DSL, verify + state-update)
     + `linear_attention_state_update_kvbuffer`

Correctness is validated against a shared PyTorch reference.

Usage:
    python benchmarks/bench_la_decode_mtp_vs_sglang.py
    python benchmarks/bench_la_decode_mtp_vs_sglang.py --batch-sizes 1 4 16 64 --T 2 4 8
    python benchmarks/bench_la_decode_mtp_vs_sglang.py --heads 32 --head-dim 128 --T 4
"""

import argparse
import os
import sys

import torch
import triton

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import cuda.bindings.driver as cuda_drv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sglang", "python"))

from sglang.srt.layers.attention.linear.seg_la import (
    SegLaMeta,
    seg_la_mtp_kernel,
    seg_la_sum_kernel,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)

from cula.lightning.la_decode_mtp import (
    get_mtp_config,
    linear_attention_decode_mtp,
)
from cula.lightning.la_verify_kvbuffer import (
    _get_compiled_verify_kvbuffer_kernel,
    linear_attention_verify_kvbuffer,
)
from cula.lightning.la_state_update_kvbuffer import (
    _get_compiled_state_update_kernel,
    linear_attention_state_update_kvbuffer,
)
from cula.utils import USE_FAST_MATH, get_device_sm_version


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch reference
# ─────────────────────────────────────────────────────────────────────────────
def torch_la_mtp_ref(q, k, v, state, decay_scales, softmax_scale):
    """
    Pure PyTorch reference for MTP decode.

    Args:
        q, k:  [B, T, H, K] bf16
        v:     [B, T, H, V] bf16  (H == HV for SGLang compat)
        state: [B, H, K, V] fp32  (K-major, SGLang convention)
        decay_scales: [H] fp32
        softmax_scale: float

    Returns:
        out:    [B, T, H, V] fp32
        state:  [B, H, K, V] fp32  (updated)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    state = state.clone().float()
    out = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)

    decay = torch.exp(-decay_scales).float()  # [H]

    for t in range(T):
        qt = q[:, t].float() * softmax_scale  # [B, H, K]
        kt = k[:, t].float()                  # [B, H, K]
        vt = v[:, t].float()                  # [B, H, V]
        state = state * decay[None, :, None, None] + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", qt, state)

    return out, state


# ─────────────────────────────────────────────────────────────────────────────
# SGLang seg_la MTP wrapper (matches seg_la_fwd MTP path)
# ─────────────────────────────────────────────────────────────────────────────
def run_sglang_mtp(q_3d, k_3d, v_3d, s_sglang, caches_sglang,
                   s_offsets, cache_indices, decay_scales, meta, softmax_scale,
                   HEAD_DIM, step, K_SPLIT_DIM=32, V_SPLIT_DIM=64):
    """
    Invoke seg_la_mtp_kernel the same way seg_la_fwd does for the MTP path.

    q_3d, k_3d, v_3d: [length, qo_heads, HEAD_DIM]  (contiguous, length = B*step)
    s_sglang:   [pool_size, qo_heads, HEAD_DIM, HEAD_DIM]  fp32
    caches_sglang: [pool_size * step, qo_heads, HEAD_DIM, HEAD_DIM] fp32
    """
    length = q_3d.shape[0]
    qo_heads = q_3d.shape[1]
    bs = meta.batch_size

    k_dim_block = HEAD_DIM // K_SPLIT_DIM
    v_dim_block = HEAD_DIM // V_SPLIT_DIM
    tmp = torch.empty(
        (k_dim_block, length, qo_heads, HEAD_DIM), device=q_3d.device, dtype=q_3d.dtype
    )
    grid = (bs, qo_heads, k_dim_block * v_dim_block)
    num_warps = 2
    num_stages = 3

    seg_la_mtp_kernel[grid](
        q_3d,
        k_3d,
        v_3d,
        s_sglang,
        caches_sglang,
        tmp,
        softmax_scale,
        q_3d.stride(0),
        k_3d.stride(0),
        v_3d.stride(0),
        s_sglang.stride(0),
        caches_sglang.stride(0),
        tmp.stride(0),
        s_offsets,
        cache_indices,
        decay_scales,
        step,
        HEAD_DIM=HEAD_DIM,
        K_SPLIT_DIM=K_SPLIT_DIM,
        V_SPLIT_DIM=V_SPLIT_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if k_dim_block > 1:
        if length < 2048:
            o = tmp.sum(0)
        else:
            o = torch.empty(
                (length, qo_heads, HEAD_DIM), device=q_3d.device, dtype=q_3d.dtype
            )
            seg_la_sum_kernel[(length,)](
                tmp, o,
                DIM=qo_heads * HEAD_DIM,
                NUM_BLOCK=k_dim_block,
                num_warps=2,
                num_stages=3,
            )
    else:
        o = tmp[0]
    return o


# ─────────────────────────────────────────────────────────────────────────────
# SGLang commit wrapper (fused_mamba_state_scatter_with_mask)
# ─────────────────────────────────────────────────────────────────────────────
def run_sglang_commit(s_sglang, caches_sglang, s_offsets, step_indices, B, H, K, V, T):
    """
    Invoke fused_mamba_state_scatter_with_mask the way hybrid_linear_attn_backend does.

    dst: [1, pool_size, H*K*V]  — state pool (1 layer)
    src: [1, B, T, H*K*V]       — intermediate caches (1 layer)
    """
    elem_per_entry = H * K * V
    dst = s_sglang.reshape(1, -1, elem_per_entry)
    src = caches_sglang.reshape(1, B, T, elem_per_entry)
    fused_mamba_state_scatter_with_mask(dst, src, s_offsets, step_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_fn(fn, warmup=30, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    iqr = times[n // 4 : 3 * n // 4]
    return sum(iqr) / len(iqr)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark for one (B, T) configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(B, T, H, K, V, layer_idx, num_layers):
    device = "cuda"
    dtype = torch.bfloat16
    scale = K ** -0.5
    HV = H  # SGLang seg_la does not support GQA

    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, device=device, dtype=torch.float32)
    decay_scales = -g_gamma  # cuLA convention: exp(-decay_scales)

    torch.manual_seed(42)
    q_4d = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k_4d = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v_4d = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    state_init_kmaj = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

    # ── PyTorch reference ──────────────────────────────────────────────────
    with torch.no_grad():
        o_ref, _ = torch_la_mtp_ref(q_4d, k_4d, v_4d, state_init_kmaj, decay_scales, scale)

    # ── SGLang setup ───────────────────────────────────────────────────────
    length = B * T
    q_3d = q_4d.reshape(length, H, K).contiguous()
    k_3d = k_4d.reshape(length, H, K).contiguous()
    v_3d = v_4d.reshape(length, HV, V).contiguous()

    pool_size = B
    s_sglang = state_init_kmaj.reshape(pool_size, H, K, V).contiguous()
    caches_sglang = torch.zeros(pool_size * T, H, K, V, device=device, dtype=torch.float32)

    s_offsets_sg = torch.arange(B, device=device, dtype=torch.int64)
    cache_indices_sg = torch.arange(B, device=device, dtype=torch.int64) * T

    q_offsets = torch.arange(B + 1, device=device, dtype=torch.int64) * T
    q_lengths = torch.full((B,), T, device=device, dtype=torch.int64)
    s_scales = torch.ones(B, device=device, dtype=torch.int64)

    meta = SegLaMeta(
        batch_size=B,
        max_q_length=T,
        q_offsets=q_offsets,
        s_offsets=s_offsets_sg,
        q_lengths=q_lengths,
        s_scales=s_scales,
    )

    K_SPLIT_DIM = 32
    V_SPLIT_DIM = 32 if B <= 2 else 64

    # warmup sglang (Triton JIT compile)
    with torch.no_grad():
        s_sg_run = s_sglang.clone()
        c_sg_run = caches_sglang.clone()
        o_sg = run_sglang_mtp(
            q_3d, k_3d, v_3d, s_sg_run, c_sg_run,
            s_offsets_sg, cache_indices_sg, decay_scales, meta, scale,
            K, T, K_SPLIT_DIM, V_SPLIT_DIM,
        )

    # ── SGLang correctness ─────────────────────────────────────────────────
    o_sg_4d = o_sg.reshape(B, T, HV, V).float()
    rmse_sg = torch.sqrt(torch.mean((o_sg_4d - o_ref) ** 2)).item()
    max_ref = torch.abs(o_ref).max().item()
    reldiff_sg = torch.abs(o_sg_4d - o_ref).max().item() / (max_ref + 1e-8)

    # ── cuLA MTP setup ─────────────────────────────────────────────────────
    # SGLang seg_la_mtp writes intermediate caches but does NOT write back S,
    # so the fair comparison is cache_intermediate_states=True, disable_state_update=True.
    cache_inter = True
    disable_su = True

    s_cute = state_init_kmaj.permute(0, 1, 3, 2).contiguous()  # [B, HV, V, K]
    out_cute = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    s_offsets_cu = torch.arange(B, device=device, dtype=torch.int32)
    inter = torch.zeros(B * T * HV, V, K, device=device, dtype=torch.float32)
    cu_seqlens_dummy = torch.empty(1, device=device, dtype=torch.int32)

    with torch.no_grad():
        linear_attention_decode_mtp(
            q_4d, k_4d, v_4d, s_cute, inter, out_cute,
            decay_scales=decay_scales,
            s_offsets=s_offsets_cu,
            cu_seqlens=cu_seqlens_dummy,
            softmax_scale=scale,
            T=T,
            cache_intermediate_states=cache_inter,
            disable_state_update=disable_su,
            is_varlen=False,
        )

    out_cute_cmp = out_cute.float()
    rmse_cu = torch.sqrt(torch.mean((out_cute_cmp - o_ref) ** 2)).item()
    reldiff_cu = torch.abs(out_cute_cmp - o_ref).max().item() / (max_ref + 1e-8)

    # ── KVBuffer verify + state-update setup ───────────────────────────────
    s_kvbuf = state_init_kmaj.permute(0, 1, 3, 2).contiguous()  # [B, HV, V, K]
    out_kvbuf = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    h0_indices_kv = torch.arange(B, device=device, dtype=torch.int32)
    accepted_len_kv = torch.full((B,), T, device=device, dtype=torch.int32)

    with torch.no_grad():
        linear_attention_verify_kvbuffer(
            q_4d, k_4d, v_4d, s_kvbuf, out_kvbuf,
            decay_scales, h0_indices_kv, scale, T,
        )
        s_kvbuf_warmup = state_init_kmaj.permute(0, 1, 3, 2).contiguous()
        linear_attention_state_update_kvbuffer(
            k_4d, v_4d, s_kvbuf_warmup, decay_scales,
            h0_indices_kv, accepted_len_kv, T,
        )

    out_kvbuf_cmp = out_kvbuf.float()
    rmse_kv = torch.sqrt(torch.mean((out_kvbuf_cmp - o_ref) ** 2)).item()
    reldiff_kv = torch.abs(out_kvbuf_cmp - o_ref).max().item() / (max_ref + 1e-8)

    # ==================================================================
    # Kernel-only timing: pre-compiled handles, no Python overhead
    # ==================================================================

    # ---- cuLA kernel-only setup ----
    pool_size = B
    tile_v, vec_size, ilp_rows, use_smem_v = get_mtp_config(B, T, HV, V, disable_su)
    major, _ = get_device_sm_version(q_4d.device)
    use_packed_fma = major >= 10
    stream_handle = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    # ---- SGLang: Triton kernel is already "kernel-only" (no Python wrapper overhead).
    #      We just avoid the redundant .clone() on state S, since seg_la_mtp_kernel
    #      does NOT write back to S (it writes to CACHES only). ----
    s_sg_bench = s_sglang  # no clone needed, kernel only reads S
    c_sg_bench = caches_sglang

    def kernel_sglang():
        run_sglang_mtp(
            q_3d, k_3d, v_3d, s_sg_bench, c_sg_bench,
            s_offsets_sg, cache_indices_sg, decay_scales, meta, scale,
            K, T, K_SPLIT_DIM, V_SPLIT_DIM,
        )

    # ---- SGLang commit setup ----
    step_indices_sg = torch.full((B,), T - 1, device=device, dtype=torch.int32)

    def kernel_sglang_commit():
        run_sglang_commit(
            s_sg_bench, c_sg_bench, s_offsets_sg.int(),
            step_indices_sg, B, H, K, V, T,
        )

    # ---- cuLA KVBuffer with actual buffer write/read ----
    k_buf_bench = torch.zeros(pool_size, T, H, K, device=device, dtype=dtype)
    v_buf_bench = torch.zeros(pool_size, T, HV, V, device=device, dtype=dtype)

    # Trigger compilation for write_kv=True variant
    s_kvbuf_compile = state_init_kmaj.permute(0, 1, 3, 2).contiguous()
    out_compile = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    linear_attention_verify_kvbuffer(
        q_4d, k_4d, v_4d, s_kvbuf_compile, out_compile,
        decay_scales, h0_indices_kv, scale, T,
        k_buf=k_buf_bench, v_buf=v_buf_bench,
    )

    tile_v_kv, vec_size_kv, ilp_rows_kv, use_smem_v_kv = get_mtp_config(B, T, HV, V, True)
    verify_buf_cache_key = (
        B, T, H, HV, K, V, pool_size, scale,
        tile_v_kv, vec_size_kv, ilp_rows_kv, use_smem_v_kv, use_packed_fma,
        True,  # write_kv
    )
    verify_buf_cache = _get_compiled_verify_kvbuffer_kernel(*verify_buf_cache_key)
    compiled_verify_buf = verify_buf_cache["compiled"]

    s_kvbuf_kk_vb = state_init_kmaj.permute(0, 1, 3, 2).contiguous().view(pool_size * HV, V, K)
    out_kvbuf_kk = torch.empty(B, T, HV, V, device=device, dtype=dtype)

    def kernel_kvbuf_verify_with_write():
        compiled_verify_buf(
            s_kvbuf_kk_vb,
            decay_scales, q_4d, k_4d, v_4d, out_kvbuf_kk,
            h0_indices_kv,
            k_buf_bench, v_buf_bench,
            stream_handle,
        )

    # Trigger compilation for read_from_buf=True variant
    s_kvbuf_warmup2 = state_init_kmaj.permute(0, 1, 3, 2).contiguous()
    linear_attention_state_update_kvbuffer(
        k_4d, v_4d, s_kvbuf_warmup2, decay_scales,
        h0_indices_kv, accepted_len_kv, T,
        k_buf=k_buf_bench, v_buf=v_buf_bench,
    )

    tile_v_su, vec_size_su, ilp_rows_su, _smem_su = get_mtp_config(B, T, HV, V, False)
    update_buf_cache_key = (
        B, T, H, HV, K, V, pool_size, tile_v_su, vec_size_su, ilp_rows_su, use_packed_fma,
        True,  # read_from_buf
    )
    update_buf_cache = _get_compiled_state_update_kernel(*update_buf_cache_key)
    compiled_update_buf = update_buf_cache["compiled"]

    s_kvbuf_kk_ub = state_init_kmaj.permute(0, 1, 3, 2).contiguous().view(pool_size * HV, V, K)

    def kernel_kvbuf_update_from_buf():
        compiled_update_buf(
            s_kvbuf_kk_ub,
            decay_scales, k_4d, v_4d,
            h0_indices_kv, accepted_len_kv,
            k_buf_bench, v_buf_bench,
            stream_handle,
        )

    with torch.no_grad():
        sg_vfy_ms = benchmark_fn(kernel_sglang)
        sg_cmt_ms = benchmark_fn(kernel_sglang_commit)
        cu_vfy_ms = benchmark_fn(kernel_kvbuf_verify_with_write)
        cu_cmt_ms = benchmark_fn(kernel_kvbuf_update_from_buf)

    sg_total_ms = sg_vfy_ms + sg_cmt_ms
    cu_total_ms = cu_vfy_ms + cu_cmt_ms

    return {
        "B": B,
        "T": T,
        "sg_vfy_ms": sg_vfy_ms,
        "sg_cmt_ms": sg_cmt_ms,
        "sg_total_ms": sg_total_ms,
        "cu_vfy_ms": cu_vfy_ms,
        "cu_cmt_ms": cu_cmt_ms,
        "cu_total_ms": cu_total_ms,
        "speedup": sg_total_ms / cu_total_ms,
        "rmse_sg": rmse_sg,
        "rmse_cu": rmse_cu,
        "rmse_kv": rmse_kv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark la_decode_mtp vs SGLang seg_la")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--T", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    H = args.heads
    K = V = args.head_dim

    print("LA Decode MTP: cuLA vs SGLang seg_la Benchmark")
    print(f"  H={H}, K={K}, V={V}, layer={args.layer_idx}/{args.num_layers}")
    print(f"  dtype=bf16, state=fp32")
    print(f"  USE_FAST_MATH={USE_FAST_MATH}")
    print(f"  cuLA MTP: cache_intermediate_states=True, disable_state_update=True")
    print(f"  Timing: kernel-only (cuLA pre-compiled handle; SGLang no extra .clone())")

    hdr = (
        f"{'B':>4} | {'T':>3} | "
        f"{'sg_vfy(ms)':>10} | {'sg_cmt(ms)':>10} | {'sg_total':>9} | "
        f"{'cu_vfy(ms)':>10} | {'cu_cmt(ms)':>10} | {'cu_total':>9} | "
        f"{'speedup':>7} | "
        f"{'rmse_sg':>9} | {'rmse_cu':>9} | {'rmse_kv':>9}"
    )
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for T_val in args.T:
        for B in args.batch_sizes:
            r = run_config(B, T_val, H, K, V, args.layer_idx, args.num_layers)
            print(
                f"{r['B']:>4} | {r['T']:>3} | "
                f"{r['sg_vfy_ms']:>10.4f} | {r['sg_cmt_ms']:>10.4f} | {r['sg_total_ms']:>9.4f} | "
                f"{r['cu_vfy_ms']:>10.4f} | {r['cu_cmt_ms']:>10.4f} | {r['cu_total_ms']:>9.4f} | "
                f"{r['speedup']:>6.2f}x | "
                f"{r['rmse_sg']:>9.6f} | {r['rmse_cu']:>9.6f} | {r['rmse_kv']:>9.6f}"
            )
        print()

    # Memory comparison
    sg_mem = B * T_val * H * K * V * 4
    cu_mem = B * T_val * (H * K + H * V) * 2
    print(f"Memory per-pool (B={args.batch_sizes[-1]}, T={args.T[-1]}):")
    print(f"  SGLang intermediate caches: {sg_mem / 1e6:.1f} MB")
    print(f"  cuLA KV buffer:             {cu_mem / 1e6:.1f} MB")
    print(f"  Ratio:                      {sg_mem / cu_mem:.0f}×")

    print("\nColumns:")
    print("  sg_vfy     : seg_la_mtp_kernel (Triton, SGLang upstream)")
    print("  sg_cmt     : fused_mamba_state_scatter_with_mask (Triton, SGLang)")
    print("  cu_vfy     : verify_kvbuffer with KV buffer write (CuTe DSL)")
    print("  cu_cmt     : state_update_kvbuffer reading from buffer (CuTe DSL)")
    print("  speedup    : sg_total / cu_total")
    print("  rmse_*     : RMSE vs PyTorch reference")


if __name__ == "__main__":
    main()
