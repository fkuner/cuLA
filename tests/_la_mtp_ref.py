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

"""Shared PyTorch reference for multi-token Lightning Attention decode."""

import torch


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
        q_hv = q_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        k_hv = k_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        v_t = v_f[:, t]  # [B, HV, D]

        state_running = state_running * decay_per_hv + k_hv.unsqueeze(-1) * v_t.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_hv, state_running).bfloat16()

        if cache_intermediate_states:
            for b in range(B):
                inter[b * T * HV + t * HV : b * T * HV + (t + 1) * HV] = state_running[b]

    state_final = state.clone() if disable_state_update else state_running
    return out, state_final, inter
