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

"""Unit tests for the KVBuffer verify + state-update kernels."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cula.lightning.la_state_update_kvbuffer import linear_attention_state_update_kvbuffer
from _la_mtp_ref import torch_la_mtp_ref


def _skip_if_no_sm90_or_later():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = torch.cuda.get_device_capability("cuda")
    if cc[0] < 9:
        pytest.skip(f"requires SM90+, got SM{cc[0]}{cc[1]}")


def _make_inputs(B, T, H, HV, D, device="cuda", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, D, device=device, dtype=torch.bfloat16)
    state = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    return q, k, v, state


def test_state_update_L0_no_op():
    """accepted_len=0 everywhere: s must be byte-for-byte unchanged."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()  # [B, HV, V, K]
    s_snapshot = s_cute.clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    accepted_len = torch.zeros(B, device="cuda", dtype=torch.int32)

    linear_attention_state_update_kvbuffer(
        k, v, s_cute, decay_scales, h0_indices, accepted_len, T,
    )
    assert torch.equal(s_cute, s_snapshot), "L=0 must leave state unchanged"


def _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T):
    """state[B,HV,K,V] fp32; returns the per-batch state after L recurrent steps."""
    B, HV, K, V = state.shape
    H = k.shape[2]
    k_f, v_f = k.float(), v.float()
    decay_per_q_head = torch.exp(-decay_scales)
    decay_per_hv = decay_per_q_head.repeat_interleave(HV // H).view(HV, 1, 1)
    out = state.clone()
    for b in range(B):
        L = int(L_per_batch[b].item())
        running = state[b].clone()
        for i in range(L):
            k_hv = k_f[b, i].repeat_interleave(HV // H, dim=0)  # [HV, K]
            v_i = v_f[b, i]  # [HV, V]
            running = running * decay_per_hv + k_hv.unsqueeze(-1) * v_i.unsqueeze(-2)
        out[b] = running
    return out


@pytest.mark.parametrize("B,T,H,HV,D", [(4, 4, 16, 16, 128), (8, 4, 64, 64, 128)])
def test_state_update_full_accept(B, T, H, HV, D):
    """accepted_len=T everywhere: bit-exact vs baseline recurrence reference."""
    _skip_if_no_sm90_or_later()
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)  # [B,HV,K,V]

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_state_update_kvbuffer(
        k, v, s_cute, decay_scales, h0_indices, L_per_batch, T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()  # back to [B,HV,K,V]
    rmse = torch.sqrt(torch.mean((got - ref) ** 2)).item()
    rel = rmse / (torch.abs(ref).max().item() + 1e-8)
    assert rel < 1e-3, f"full-accept state rel RMSE {rel:.6f} too large"


@pytest.mark.parametrize("L", [0, 1, 3])
def test_state_update_partial(L):
    """Uniform accepted_len=L across all batches."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.full((B,), L, device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_state_update_kvbuffer(
        k, v, s_cute, decay_scales, h0_indices, L_per_batch, T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()
    rel = torch.sqrt(torch.mean((got - ref) ** 2)).item() / (torch.abs(ref).max().item() + 1e-8)
    assert rel < 1e-3, f"L={L} state rel RMSE {rel:.6f}"


def test_state_update_per_batch_L():
    """accepted_len varies per batch: [0, 1, T-1, T]."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.tensor([0, 1, T - 1, T], device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_state_update_kvbuffer(
        k, v, s_cute, decay_scales, h0_indices, L_per_batch, T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()
    for b in range(B):
        rel = torch.sqrt(torch.mean((got[b] - ref[b]) ** 2)).item() / (torch.abs(ref[b]).max().item() + 1e-8)
        assert rel < 1e-3, f"batch {b} (L={int(L_per_batch[b])}) rel RMSE {rel:.6f}"


def test_state_update_skip_negative_h0_indices():
    """h0_indices[b]=-1: that pool slot is untouched even with accepted_len>0."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    snapshot_b2 = s_cute[2].clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1
    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)

    linear_attention_state_update_kvbuffer(
        k, v, s_cute, decay_scales, h0_indices, L_per_batch, T,
    )
    assert torch.equal(s_cute[2], snapshot_b2), "skipped batch slot was modified"
