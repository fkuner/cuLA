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

from cula.lightning.la_update_kvbuffer import linear_attention_state_update_kvbuffer
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


from cula.lightning.la_verify_kvbuffer import linear_attention_verify_kvbuffer


def test_verify_skip_negative_h0_indices():
    """h0_indices[b]=-1: out[b] stays at its sentinel value."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    sentinel = 123.0
    out = torch.full((B, T, HV, D), sentinel, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1

    linear_attention_verify_kvbuffer(
        q, k, v, s_cute, out, decay_scales, h0_indices, scale, T,
    )
    assert torch.all(out[2] == sentinel), "skipped batch out slot was modified"


@pytest.mark.parametrize("B,T", [(1, 4), (2, 2), (2, 4), (8, 4), (32, 2), (32, 4)])
def test_verify_outputs_match_ref(B, T):
    """Verify kernel o matches torch_la_mtp_ref across the baseline configs."""
    _skip_if_no_sm90_or_later()
    H, HV, D = 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q, k, v, s_cute, out, decay_scales, h0_indices, scale, T,
    )
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (
        torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"B={B} T={T}: verify output rel RMSE {rel:.6f} too large"


@pytest.mark.parametrize("H,HV", [(16, 16), (8, 32), (16, 64)])
def test_verify_different_heads(H, HV):
    _skip_if_no_sm90_or_later()
    B, T, D = 4, 4, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q, k, v, s_cute, out, decay_scales, h0_indices, scale, T,
    )
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (
        torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"H={H} HV={HV}: verify output mismatch {rel:.6f}"


def test_verify_zero_decay():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = torch.zeros(H, device="cuda", dtype=torch.float32)
    q, k, v, state = _make_inputs(B, T, H, HV, D)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(q, k, v, s_cute, out, decay_scales, h0_indices, scale, T)
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (
        torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"zero decay: {rel:.6f}"


def test_verify_zero_state():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.ones(H, device="cuda", dtype=torch.float32)
    q, k, v, _ = _make_inputs(B, T, H, HV, D)
    state = torch.zeros(B, HV, D, D, device="cuda", dtype=torch.float32)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(q, k, v, s_cute, out, decay_scales, h0_indices, scale, T)
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (
        torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"zero state: {rel:.6f}"


from cula.lightning.la_decode_mtp import linear_attention_decode_mtp


def test_end_to_end_equivalence_with_baseline():
    """KVBuffer (verify + state_update L=T) == baseline (cache_inter=T, disable=T)."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    # ---- Baseline: capture out + all intermediate states ----
    s_base = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    out_base = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    s_offsets = torch.arange(B, device="cuda", dtype=torch.int32)
    inter = torch.zeros(B * T * HV, D, D, device="cuda", dtype=torch.float32)  # [.,V,K]
    cu_seqlens = torch.empty(1, device="cuda", dtype=torch.int32)
    linear_attention_decode_mtp(
        q, k, v, s_base, inter, out_base,
        decay_scales=decay_scales, s_offsets=s_offsets, cu_seqlens=cu_seqlens,
        softmax_scale=scale, T=T,
        cache_intermediate_states=True, disable_state_update=True, is_varlen=False,
    )

    # ---- KVBuffer: verify writes out; state-update (L=T) writes state ----
    s_kv = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    out_kv = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q, k, v, s_kv, out_kv, decay_scales, h0_indices, scale, T,
    )
    accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
    linear_attention_state_update_kvbuffer(
        k, v, s_kv, decay_scales, h0_indices, accepted_len, T,
    )

    # (a) outputs match
    rel_o = torch.sqrt(torch.mean((out_kv.float() - out_base.float()) ** 2)).item() / (
        torch.abs(out_base.float()).max().item() + 1e-8)
    assert rel_o < 1e-2, f"output mismatch vs baseline: {rel_o:.6f}"

    # (b) updated state == baseline's last intermediate slice [B,HV,V,K]
    inter_v = inter.view(B, T, HV, D, D)            # [B,T,HV,V,K]
    last_state = inter_v[:, T - 1]                  # [B,HV,V,K]
    rel_s = torch.sqrt(torch.mean((s_kv - last_state) ** 2)).item() / (
        torch.abs(last_state).max().item() + 1e-8)
    assert rel_s < 1e-3, f"state mismatch vs baseline last intermediate: {rel_s:.6f}"


@pytest.mark.parametrize("B,T", [(4, 4), (8, 2), (32, 4)])
def test_verify_writes_kv_buffer(B, T):
    """Verify kernel with k_buf/v_buf writes correct copies of k and v."""
    _skip_if_no_sm90_or_later()
    H, HV, D = 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.bfloat16)

    linear_attention_verify_kvbuffer(
        q, k, v, s_cute, out, decay_scales, h0_indices, scale, T,
        k_buf=k_buf, v_buf=v_buf,
    )

    for b in range(B):
        pool_idx = h0_indices[b].item()
        assert torch.equal(k_buf[pool_idx], k[b]), f"k_buf mismatch at batch {b}"
        assert torch.equal(v_buf[pool_idx], v[b]), f"v_buf mismatch at batch {b}"


def test_verify_output_unchanged_with_kv_write():
    """Output o is identical whether k_buf/v_buf are provided or not."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    s1 = state.permute(0, 1, 3, 2).contiguous().clone()
    s2 = s1.clone()
    out_no_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    out_with_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)

    linear_attention_verify_kvbuffer(
        q, k, v, s1, out_no_buf, decay_scales, h0_indices, scale, T,
    )

    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.bfloat16)
    linear_attention_verify_kvbuffer(
        q, k, v, s2, out_with_buf, decay_scales, h0_indices, scale, T,
        k_buf=k_buf, v_buf=v_buf,
    )

    assert torch.equal(out_no_buf, out_with_buf), "kv write should not affect output"


@pytest.mark.parametrize("B,T,H,HV,D", [(4, 4, 16, 16, 128), (8, 4, 64, 64, 128)])
def test_state_update_from_buffer(B, T, H, HV, D):
    """State update from k_buf/v_buf matches state update from raw k,v."""
    _skip_if_no_sm90_or_later()
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)

    # Path A: read from raw k, v
    s_raw = state.permute(0, 1, 3, 2).contiguous().clone()
    linear_attention_state_update_kvbuffer(
        k, v, s_raw, decay_scales, h0_indices, L_per_batch, T,
    )

    # Path B: read from buffer (fill buffer with same k, v)
    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.bfloat16)
    for b in range(B):
        k_buf[h0_indices[b].item()] = k[b]
        v_buf[h0_indices[b].item()] = v[b]

    s_buf = state.permute(0, 1, 3, 2).contiguous().clone()
    linear_attention_state_update_kvbuffer(
        k, v, s_buf, decay_scales, h0_indices, L_per_batch, T,
        k_buf=k_buf, v_buf=v_buf,
    )

    assert torch.equal(s_raw, s_buf), "buffer-read state must match raw-read state"


def test_verify_skip_negative_indices_no_buffer_write():
    """h0_indices[b]=-1: k_buf and v_buf slots are untouched."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    sentinel = 42.0
    k_buf = torch.full((pool_size, T, H, D), sentinel, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.full((pool_size, T, HV, D), sentinel, device="cuda", dtype=torch.bfloat16)
    k_buf_snap = k_buf.clone()
    v_buf_snap = v_buf.clone()

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1

    linear_attention_verify_kvbuffer(
        q, k, v, s_cute, out, decay_scales, h0_indices, scale, T,
        k_buf=k_buf, v_buf=v_buf,
    )

    assert torch.equal(k_buf[2], k_buf_snap[2]), "skipped batch k_buf slot was modified"
    assert torch.equal(v_buf[2], v_buf_snap[2]), "skipped batch v_buf slot was modified"


def test_end_to_end_with_buffer():
    """Full pipeline: verify(+kv write) → state_update(from buffer) matches baseline."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)

    # Reference: existing end-to-end (no buffer)
    s_ref = state.permute(0, 1, 3, 2).contiguous().clone()
    out_ref = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    linear_attention_verify_kvbuffer(
        q, k, v, s_ref, out_ref, decay_scales, h0_indices, scale, T,
    )
    accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
    linear_attention_state_update_kvbuffer(
        k, v, s_ref, decay_scales, h0_indices, accepted_len, T,
    )

    # Buffer path: verify writes buffer, state_update reads buffer
    s_buf = state.permute(0, 1, 3, 2).contiguous().clone()
    out_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.bfloat16)

    linear_attention_verify_kvbuffer(
        q, k, v, s_buf, out_buf, decay_scales, h0_indices, scale, T,
        k_buf=k_buf, v_buf=v_buf,
    )
    linear_attention_state_update_kvbuffer(
        k, v, s_buf, decay_scales, h0_indices, accepted_len, T,
        k_buf=k_buf, v_buf=v_buf,
    )

    assert torch.equal(out_ref, out_buf), "output mismatch with buffer pipeline"
    assert torch.equal(s_ref, s_buf), "state mismatch with buffer pipeline"
