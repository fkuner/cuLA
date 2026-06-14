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
