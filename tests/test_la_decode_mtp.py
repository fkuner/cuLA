#!/usr/bin/env python3
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
Unit tests for la_decode_mtp (CuTe DSL Lightning Attention MTP decode kernel).

Compares against a PyTorch reference implementation of multi-token
Lightning Attention decode (T > 1).

Layouts:
  q, k:                [B, T, H,  K]   bf16
  v:                   [B, T, HV, V]   bf16
  s:                   [pool_size, HV, V, K]  fp32  (V-major, K-last)
  intermediate_states: [pool_size * T * HV, V, K] fp32, or 1-elem dummy
  out:                 [B, T, HV, V]   bf16
  decay_scales:        [H]             fp32  (positive; kernel does exp(-x))
  s_offsets:           [B]             int32 (pool index per batch; -1 to skip)
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.lightning.la_decode_mtp import linear_attention_decode_mtp


def _skip_if_no_sm90_or_later():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = torch.cuda.get_device_capability("cuda")
    if cc[0] < 9:
        pytest.skip(f"requires SM90+, got SM{cc[0]}{cc[1]}")


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def torch_la_mtp_ref(
    q, k, v, state, decay_scales, scale, T,
    cache_intermediate_states=False, disable_state_update=False,
):
    """
    Pure PyTorch reference for multi-token Lightning Attention decode.

    Args:
        q, k:        [B, T, H,  D] bf16
        v:           [B, T, HV, D] bf16
        state:       [B, HV, D, D] fp32 (K-major, V-minor at this layout)
                     i.e. state[b, h, k, v] is element (k, v).
        decay_scales: [H] fp32 (positive; kernel does exp(-x))
        scale: float
        T: int
        cache_intermediate_states: cache per-step state to inter
        disable_state_update: do not update state_new at end (return state.clone())

    Returns:
        out:        [B, T, HV, D] bf16
        state_new:  [B, HV, D, D] fp32
        inter:      [B*T*HV, D, D] fp32 or None
    """
    B, _, H, D = q.shape
    HV = v.shape[2]
    q_f = q.float() * scale
    k_f, v_f = k.float(), v.float()
    # decay_scales is per-q-head [H]; broadcast over HV via i_h = i_hv // (HV//H).
    decay_per_q_head = torch.exp(-decay_scales)  # [H]
    decay_per_hv = decay_per_q_head.repeat_interleave(HV // H).view(1, HV, 1, 1)

    state_running = state.clone()
    out = torch.zeros(B, T, HV, D, dtype=torch.bfloat16, device=q.device)
    inter = (
        torch.zeros(B * T * HV, D, D, dtype=torch.float32, device=q.device)
        if cache_intermediate_states
        else None
    )

    for t in range(T):
        # broadcast q[B,H,K] over HV via repeat_interleave on the i_h mapping
        q_hv = q_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        k_hv = k_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        v_t = v_f[:, t]  # [B, HV, D]

        # h_t = decay * h_{t-1} + k ⊗ v
        state_running = state_running * decay_per_hv + k_hv.unsqueeze(-1) * v_t.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_hv, state_running).bfloat16()

        if cache_intermediate_states:
            # inter[b*T*HV + t*HV + hv] = state_running[b, hv]
            for b in range(B):
                inter[b * T * HV + t * HV : b * T * HV + (t + 1) * HV] = state_running[b]

    state_final = state.clone() if disable_state_update else state_running
    return out, state_final, inter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(B, T, H, HV, D, device="cuda", seed=42):
    """Returns q[B,T,H,D] bf16, k[B,T,H,D] bf16, v[B,T,HV,D] bf16, state[B,HV,D,D] fp32."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, D, device=device, dtype=torch.bfloat16)
    state = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    return q, k, v, state


def run_la_mtp(
    q, k, v, state_4d, decay_scales, scale, T,
    cache_intermediate_states=False, disable_state_update=False,
):
    """
    Wraps linear_attention_decode_mtp with proper state-layout conversion.

    state_4d: [B, HV, K, V] fp32 (K-major)
    Kernel expects s: [pool_size=B, HV, V, K]; we transpose K and V.
    """
    B, HV, K, V = state_4d.shape
    H = q.shape[2]
    assert HV % H == 0, "HV must be a multiple of H"

    # pretranspose: [B, HV, V, K]
    s_cute = state_4d.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.bfloat16)
    s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)

    if cache_intermediate_states:
        inter = torch.zeros(B * T * HV, V, K, device=q.device, dtype=torch.float32)
    else:
        inter = torch.empty(1, 1, 1, device=q.device, dtype=torch.float32)  # dummy

    cu_seqlens = torch.empty(1, device=q.device, dtype=torch.int32)  # dummy when is_varlen=False

    linear_attention_decode_mtp(
        q,
        k,
        v,
        s_cute,
        inter,
        out,
        decay_scales=decay_scales,
        s_offsets=s_offsets,
        cu_seqlens=cu_seqlens,
        softmax_scale=scale,
        T=T,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        is_varlen=False,
    )

    # convert state back: [B, HV, V, K] -> [B, HV, K, V]
    state_out = s_cute.permute(0, 1, 3, 2).contiguous()

    if cache_intermediate_states:
        # inter (kernel): [B*T*HV, V, K] -> ref layout [B*T*HV, K, V]
        inter_out = inter.permute(0, 2, 1).contiguous()
    else:
        inter_out = None

    return out, state_out, inter_out


# ---------------------------------------------------------------------------
# Tests vs PyTorch reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("B", [4])
def test_output_vs_torch_ref(B, T):
    """Step 2 baseline: B=4, T∈{2,4}, H=HV=16, D=128 — ILP=4 hardcoded path."""
    _skip_if_no_sm90_or_later()
    H, HV, D = 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    o_ref, state_ref, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    o_cute, state_cute, _ = run_la_mtp(q, k, v, state, decay_scales, scale, T)

    # Output check
    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel = rmse / (max_ref + 1e-8)
    assert rel < 0.01, f"B={B} T={T}: output rel RMSE {rel:.6f} too large"

    # State check
    state_rmse = torch.sqrt(torch.mean((state_cute - state_ref) ** 2)).item()
    state_max = torch.abs(state_ref).max().item()
    state_rel = state_rmse / (state_max + 1e-8)
    assert state_rel < 0.001, f"B={B} T={T}: state rel RMSE {state_rel:.6f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
