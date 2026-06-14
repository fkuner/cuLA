# LA Decode MTP — KVBuffer Parallel-Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new CuTe DSL kernels (`linear_attention_verify_kvbuffer`, `linear_attention_state_update_kvbuffer`) that implement the KVBuffer §3.3 parallel-verify reformulation of Lightning Attention MTP decode, eliminating the dominant intermediate-state DRAM write.

**Architecture:** The baseline `linear_attention_decode_mtp` uses the recurrent form and (under bench config) materializes T full fp32 states per (b,hv). KVBuffer derives those states in closed form from `(h0, k, v)`, so they need never be written. We split the work into (a) a **verify** kernel that reads `s,q,k,v` and writes only `o` via `o_t = α^{t+1}·(h0@q_t·scale) + Σ_{i≤t} α^{t-i}·(q_t·k_i·scale)·v_i`, and (b) a **state-update** kernel that reads `s,k,v` plus per-batch `accepted_len` and writes the post-acceptance state back to `s` via the recurrence (bit-equivalent to baseline at L=T). Both reuse baseline helpers (`la_update_pair`, `hq_dot_pair`, `get_mtp_config`, `TILE_K_MTP`, `NUM_THREADS_MTP`) and the baseline grid/lane partitioning verbatim. The baseline kernel is untouched for head-to-head comparison.

**Tech Stack:** Python, CuTe DSL (`cutlass.cute`), PyTorch (bf16 in / fp32 state), pytest, CUDA events for benchmarking. Target SM90+ (packed-FMA path on SM100+).

**Spec:** `docs/superpowers/specs/2026-06-15-la-mtp-kvbuffer-design.md` — read it before starting; this plan implements it verbatim.

**Reference code to mirror (do NOT modify):** `cula/lightning/la_decode_mtp.py`
- Helpers + config: `la_update_pair` (lines 54-72), `hq_dot_pair` (75-85), `get_mtp_config` (94-129), `TILE_K_MTP`/`NUM_THREADS_MTP` (47-48).
- Kernel signature + grid/lane setup: `la_verify_kernel_mtp` (135-223).
- ilp=2 row loop: lines 225-339. ilp=4 row loop: 341-536. ilp=8 row loop: 538-744 (read the file for the ilp=8 body).
- Launcher + compile-cache + Python entry: `run_la_verify_kernel_mtp` (~750-833), `_get_compiled_la_mtp_kernel` (839-858), `linear_attention_decode_mtp` (864-958).

---

## File Structure

**Create:**
- `cula/lightning/la_state_update_kvbuffer.py` — state-update kernel + launcher + Python entry + compile cache.
- `cula/lightning/la_verify_kvbuffer.py` — verify kernel + launcher + Python entry + compile cache.
- `tests/_la_mtp_ref.py` — shared PyTorch reference (extracted from the existing test file).
- `tests/test_la_verify_kvbuffer.py` — tests for both new kernels + end-to-end equivalence.

**Modify:**
- `cula/lightning/__init__.py` — export the two new entry points.
- `tests/test_la_decode_mtp.py` — import `torch_la_mtp_ref` from the new shared module instead of defining it inline.
- `benchmarks/bench_la_decode_mtp.py` — add KVBuffer timing columns + bytes model.

**Implementation order rationale:** state-update kernel first (simpler — pure sequential recurrence, no cross-terms, bit-exact to baseline so trivially testable), then verify kernel (adds the q·k cross-terms), then end-to-end equivalence, then benchmark wiring.

---

## Task 1: Extract shared PyTorch reference

**Files:**
- Create: `tests/_la_mtp_ref.py`
- Modify: `tests/test_la_decode_mtp.py:54-109` (remove inline def), `tests/test_la_decode_mtp.py:40` (add import)

- [ ] **Step 1: Create the shared reference module**

Copy `torch_la_mtp_ref` verbatim from `tests/test_la_decode_mtp.py` lines 54-109 into a new file `tests/_la_mtp_ref.py` with this exact content:

```python
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
```

- [ ] **Step 2: Update the existing test file to import from the shared module**

In `tests/test_la_decode_mtp.py`, delete the inline `torch_la_mtp_ref` function (lines 54-109, including the `# PyTorch reference` banner comment block above it at lines 51-53). Then ensure the `tests/` dir is on `sys.path` and add the import. The existing line 38 is `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))` (repo root); add a second insert for the `tests/` dir directly below it, and replace the line-40 import block:

```python
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cula.lightning.la_decode_mtp import linear_attention_decode_mtp
from _la_mtp_ref import torch_la_mtp_ref
```

- [ ] **Step 3: Run the existing suite to confirm the refactor is behavior-preserving**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_decode_mtp.py -q`
Expected: same pass/fail set as before the refactor (on a GPU box, all pass; on this no-GPU box, all SKIP — that is fine, the import must still resolve without error).

- [ ] **Step 4: Commit**

```bash
git add tests/_la_mtp_ref.py tests/test_la_decode_mtp.py
git commit -m "refactor: extract torch_la_mtp_ref into shared tests/_la_mtp_ref.py"
```

---

## Task 2: State-update kernel — module scaffold + Python entry + export

**Files:**
- Create: `cula/lightning/la_state_update_kvbuffer.py`
- Modify: `cula/lightning/__init__.py`
- Test: `tests/test_la_verify_kvbuffer.py` (smoke test only in this task)

This task lands the file with the full Python entry point and an EMPTY kernel body (the launcher compiles and launches, but the kernel does no work yet). The smoke test only checks the call signature wires up and a skipped batch (`accepted_len=0`) leaves `s` unchanged — which an empty kernel trivially satisfies. Task 3 fills in the kernel body.

- [ ] **Step 1: Write the module with imports, kernel stub, launcher, compile cache, and Python entry**

Create `cula/lightning/la_state_update_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Export from the lightning package**

In `cula/lightning/__init__.py`, add the import after line 15 and the name to `__all__`:

```python
from cula.lightning.la_decode_mtp import linear_attention_decode_mtp
from cula.lightning.la_state_update_kvbuffer import linear_attention_state_update_kvbuffer
from cula.ops.la_decode import linear_attention_decode
from cula.ops.lightning_attn import (
    LinearAttentionChunkwiseDecay,
    lightning_attn_fwd,
    lightning_attn_fwd_varlen,
)

__all__ = [
    "LinearAttentionChunkwiseDecay",
    "lightning_attn_fwd",
    "lightning_attn_fwd_varlen",
    "linear_attention_decode",
    "linear_attention_decode_mtp",
    "linear_attention_state_update_kvbuffer",
]
```

- [ ] **Step 3: Write the smoke test (failing until module imports cleanly)**

Create `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 4: Run the smoke test**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py::test_state_update_L0_no_op -q`
Expected (no-GPU box): SKIP (CUDA required). The import of `linear_attention_state_update_kvbuffer` and `torch_la_mtp_ref` must resolve with no ImportError — that is what this step verifies. On a GPU box: PASS (empty kernel leaves state unchanged; L=0 path is a no-op regardless).

- [ ] **Step 5: Commit**

```bash
git add cula/lightning/la_state_update_kvbuffer.py cula/lightning/__init__.py tests/test_la_verify_kvbuffer.py
git commit -m "feat: scaffold linear_attention_state_update_kvbuffer (empty kernel + Python entry)"
```

---

## Task 3: State-update kernel — full body

**Files:**
- Modify: `cula/lightning/la_state_update_kvbuffer.py` (replace the `la_state_update_kernel` body)
- Test: `tests/test_la_verify_kvbuffer.py` (add full-accept + partial tests)

**Approach:** mirror the baseline kernel's grid/lane setup (`la_decode_mtp.py:163-223`) and its per-ilp h-row load + writeback patterns (ilp=2 at 225-339, ilp=4 at 341-536, ilp=8 at 538-744), but replace the `for i_t in range(T)` body with a **dynamic** loop `for i in range_dynamic(L)` whose body is exactly the `la_update_pair` block (no q, no o, no intermediate, no `disable_state_update` branch — the writeback always happens here). `L = accepted_len[i_n]`.

- [ ] **Step 1: Write the failing tests (full accept + uniform partial + per-batch L)**

Append to `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail (empty kernel produces unchanged state)**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -q -k "state_update"`
Expected (GPU box): `test_state_update_full_accept`, `test_state_update_partial[1]`, `test_state_update_partial[3]`, `test_state_update_per_batch_L` FAIL (empty kernel leaves state at h_init); `test_state_update_partial[0]`, `test_state_update_L0_no_op`, `test_state_update_skip_negative_h0_indices` PASS. No-GPU box: all SKIP.

- [ ] **Step 3: Fill in the kernel body**

Replace the `la_state_update_kernel` body (everything between the signature's closing `):` and the stub `return`) with the following. The grid/lane setup is copied verbatim from `la_decode_mtp.py:163-181`; the h-row load and writeback are copied from the corresponding ilp branch; the inner recurrence loop uses `cutlass.range_dynamic(L)`.

```python
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size           # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group   # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp            # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    L = accepted_len[i_n]

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
    )

    if cache_idx >= 0 and L > 0:
        r_decay = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        if cutlass.const_expr(ilp_rows == 2):
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2
            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1
                if v_idx_b < V:
                    h_tile_a = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group))
                    h_tile_b = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group))
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    for i in cutlass.range_dynamic(L):
                        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])
                        r_v_a = cutlass.Float32(v[i_n, i, i_hv, v_idx_a])
                        r_v_b = cutlass.Float32(v[i_n, i, i_hv, v_idx_b])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[0, j], r_h[0, j + 1] = la_update_pair(
                                r_h[0, j], r_h[0, j + 1], r_k[j], r_k[j + 1], r_v_a, r_decay, use_packed_fma)
                            r_h[1, j], r_h[1, j + 1] = la_update_pair(
                                r_h[1, j], r_h[1, j + 1], r_k[j], r_k[j + 1], r_v_b, r_decay, use_packed_fma)

                    h_out_a = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group))
                    cute.autovec_copy(cute.slice_(r_h, (0, None)), h_out_a)
                    h_out_b = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group))
                    cute.autovec_copy(cute.slice_(r_h, (1, None)), h_out_b)

        elif cutlass.const_expr(ilp_rows == 4):
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4
            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3
                if v_idx_d < V:
                    for off, slot in ((0, 0), (1, 1), (2, 2), (3, 3)):
                        h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a + off, lane_in_group))
                        cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                    for i in cutlass.range_dynamic(L):
                        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])
                        r_v_a = cutlass.Float32(v[i_n, i, i_hv, v_idx_a])
                        r_v_b = cutlass.Float32(v[i_n, i, i_hv, v_idx_b])
                        r_v_c = cutlass.Float32(v[i_n, i, i_hv, v_idx_c])
                        r_v_d = cutlass.Float32(v[i_n, i, i_hv, v_idx_d])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[0, j], r_h[0, j + 1] = la_update_pair(
                                r_h[0, j], r_h[0, j + 1], r_k[j], r_k[j + 1], r_v_a, r_decay, use_packed_fma)
                            r_h[1, j], r_h[1, j + 1] = la_update_pair(
                                r_h[1, j], r_h[1, j + 1], r_k[j], r_k[j + 1], r_v_b, r_decay, use_packed_fma)
                            r_h[2, j], r_h[2, j + 1] = la_update_pair(
                                r_h[2, j], r_h[2, j + 1], r_k[j], r_k[j + 1], r_v_c, r_decay, use_packed_fma)
                            r_h[3, j], r_h[3, j + 1] = la_update_pair(
                                r_h[3, j], r_h[3, j + 1], r_k[j], r_k[j + 1], r_v_d, r_decay, use_packed_fma)

                    for off, slot in ((0, 0), (1, 1), (2, 2), (3, 3)):
                        h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a + off, lane_in_group))
                        cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)

        elif cutlass.const_expr(ilp_rows == 8):
            eighth_rows: cutlass.Constexpr[int] = rows_per_group // 8
            for row_oct in cutlass.range_constexpr(eighth_rows):
                v_idx_0 = i_v * tile_v + group_idx * rows_per_group + row_oct * 8
                if v_idx_0 + 7 < V:
                    for slot in cutlass.range_constexpr(8):
                        h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                        cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                    for i in cutlass.range_dynamic(L):
                        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])
                        for slot in cutlass.range_constexpr(8):
                            r_v_s = cutlass.Float32(v[i_n, i, i_hv, v_idx_0 + slot])
                            for j in cutlass.range_constexpr(0, vec_size, 2):
                                r_h[slot, j], r_h[slot, j + 1] = la_update_pair(
                                    r_h[slot, j], r_h[slot, j + 1], r_k[j], r_k[j + 1], r_v_s, r_decay, use_packed_fma)

                    for slot in cutlass.range_constexpr(8):
                        h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                        cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)
```

NOTE on `range_dynamic`: confirm the exact spelling against the installed CuTe DSL (`python -c "import cutlass; print([n for n in dir(cutlass) if 'range' in n.lower()])"`). The baseline uses `cutlass.range_constexpr` for static loops; the runtime-bound variant is `cutlass.range_dynamic`. If the installed version names it differently (e.g. `cutlass.range`), use that — the loop bound `L` is a runtime `int32`, so it must be the dynamic form, not `range_constexpr`.

- [ ] **Step 4: Run the state-update tests**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -q -k "state_update"`
Expected (GPU box): all PASS. No-GPU box: all SKIP.

- [ ] **Step 5: Commit**

```bash
git add cula/lightning/la_state_update_kvbuffer.py tests/test_la_verify_kvbuffer.py
git commit -m "feat: implement linear_attention_state_update_kvbuffer kernel body (Eq. 8)"
```

---

## Task 4: Verify kernel — module scaffold + Python entry + export

**Files:**
- Create: `cula/lightning/la_verify_kvbuffer.py`
- Modify: `cula/lightning/__init__.py`
- Test: `tests/test_la_verify_kvbuffer.py` (smoke test only)

Lands the file with the full Python entry point and an EMPTY kernel body. Task 5 fills the body.

- [ ] **Step 1: Write the module (imports, kernel stub, launcher, compile cache, Python entry)**

Create `cula/lightning/la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Export from the lightning package**

In `cula/lightning/__init__.py`, add the import and the `__all__` entry (place the verify import directly above the state-update import added in Task 2):

```python
from cula.lightning.la_decode_mtp import linear_attention_decode_mtp
from cula.lightning.la_verify_kvbuffer import linear_attention_verify_kvbuffer
from cula.lightning.la_state_update_kvbuffer import linear_attention_state_update_kvbuffer
```

and add `"linear_attention_verify_kvbuffer",` to `__all__`.

- [ ] **Step 3: Add a verify smoke test**

Append to `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 4: Run the verify smoke test**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py::test_verify_skip_negative_h0_indices -q`
Expected (no-GPU box): SKIP (import must resolve). GPU box: PASS (empty kernel never writes, so the skipped slot — and every slot — stays at sentinel; this test only asserts the skipped slot, so it passes even with the empty kernel).

- [ ] **Step 5: Commit**

```bash
git add cula/lightning/la_verify_kvbuffer.py cula/lightning/__init__.py tests/test_la_verify_kvbuffer.py
git commit -m "feat: scaffold linear_attention_verify_kvbuffer (empty kernel + Python entry)"
```

---

## Task 5: Verify kernel — full body

**Files:**
- Modify: `cula/lightning/la_verify_kvbuffer.py` (replace `la_verify_kvbuffer_kernel` body, add `hq_dot_pair` to imports)
- Test: `tests/test_la_verify_kvbuffer.py` (add output-match test)

**Approach:** unlike the baseline (which hand-expands a/b/c/d per ilp), the verify body uses a single `for slot in range_constexpr(ilp_rows)` form parameterized by `ilp_rows` — correctness-first, no per-ilp duplication. The q·k cross-term is a K-dimension dot reduced across the 32-lane group (same `shuffle_sync_bfly` ladder as baseline); it is independent of the v-row, so it is computed once per (t, i) and broadcast to all slots. `r_q_seq` is pre-scaled, so both term1 and term2 inherit `scale` automatically.

- [ ] **Step 1: Write the failing output-match test**

Append to `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure (empty kernel leaves out at zeros)**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -q -k "verify_outputs or different_heads or zero_decay or zero_state"`
Expected (GPU box): all FAIL (empty kernel → out stays zero, ref is nonzero). No-GPU box: SKIP.

- [ ] **Step 3: Add `hq_dot_pair` to the import**

In `cula/lightning/la_verify_kvbuffer.py`, extend the helper import:

```python
from cula.lightning.la_decode_mtp import (
    NUM_THREADS_MTP,
    get_mtp_config,
    hq_dot_pair,
)
```

- [ ] **Step 4: Fill in the kernel body**

Replace the `la_verify_kvbuffer_kernel` body (between `):` and the stub `return`) with:

```python
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size           # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group   # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp            # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]

    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_q_seq = cute.make_rmem_tensor(cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_k_seq = cute.make_rmem_tensor(cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_v_seq = cute.make_rmem_tensor(cute.make_layout((T, 8), stride=(8, 1)), cutlass.Float32)
    r_decay_pow = cute.make_rmem_tensor(cute.make_layout((T + 1,), stride=(1,)), cutlass.Float32)
    o_partial = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        alpha = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

        # alpha^0 .. alpha^T  (T+1 powers; term1 uses alpha^{t+1})
        r_decay_pow[0] = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(1, T + 1):
            r_decay_pow[t] = r_decay_pow[t - 1] * alpha

        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # Stage all T q (scaled) and k (fp32) for this lane's K-slice.
        for t in cutlass.range_constexpr(T):
            q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)
            for j in cutlass.range_constexpr(vec_size):
                r_q_seq[t, j] = cutlass.Float32(r_q_bf16[j]) * scale
                r_k_seq[t, j] = cutlass.Float32(r_k_bf16[j])

        num_row_blocks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for row_block in cutlass.range_constexpr(num_row_blocks):
            v_base = i_v * tile_v + group_idx * rows_per_group + row_block * ilp_rows
            if v_base + (ilp_rows - 1) < V:
                # Load h_init rows (persistent across the T loop).
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_base + slot, lane_in_group))
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                # Load all T v-values for these rows.
                for t in cutlass.range_constexpr(T):
                    for slot in cutlass.range_constexpr(ilp_rows):
                        r_v_seq[t, slot] = cutlass.Float32(v[i_n, t, i_hv, v_base + slot])

                for t in cutlass.range_constexpr(T):
                    # term1: alpha^{t+1} * (h_init @ q_t)  (per-slot warp reduce)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        hq_lo = cutlass.Float32(0.0)
                        hq_hi = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            hq_lo, hq_hi = hq_dot_pair(
                                r_h[slot, j], r_h[slot, j + 1],
                                r_q_seq[t, j], r_q_seq[t, j + 1],
                                hq_lo, hq_hi, use_packed_fma)
                        hq = hq_lo + hq_hi
                        for offset in [16, 8, 4, 2, 1]:
                            hq += cute.arch.shuffle_sync_bfly(hq, offset=offset, mask=-1, mask_and_clamp=31)
                        o_partial[slot] = r_decay_pow[t + 1] * hq

                    # term2: sum_{i=0..t} alpha^{t-i} * (q_t . k_i) * v_i
                    for i in cutlass.range_constexpr(t + 1):
                        qk_lo = cutlass.Float32(0.0)
                        qk_hi = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            qk_lo, qk_hi = hq_dot_pair(
                                r_q_seq[t, j], r_q_seq[t, j + 1],
                                r_k_seq[i, j], r_k_seq[i, j + 1],
                                qk_lo, qk_hi, use_packed_fma)
                        qk = qk_lo + qk_hi
                        for offset in [16, 8, 4, 2, 1]:
                            qk += cute.arch.shuffle_sync_bfly(qk, offset=offset, mask=-1, mask_and_clamp=31)
                        coeff = r_decay_pow[t - i] * qk
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o_partial[slot] = o_partial[slot] + coeff * r_v_seq[i, slot]

                    # writeback (all lanes hold the reduced value; lane 0 writes)
                    if lane_in_group == 0:
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o[(i_n, t, i_hv, v_base + slot)] = cutlass.BFloat16(o_partial[slot])
```

NOTE on `r_q_seq` pre-scaling: term2 computes `q_t · k_i` where `q_t` is already `* scale`, so the product carries `scale` exactly as the spec's `(q_t · k_i · scale)`. Do NOT multiply by `scale` again. term1's `h_init @ q_t` likewise inherits `scale` from the pre-scaled q. This mirrors the baseline, which scales q once at load (`la_decode_mtp.py:259`, `:389`).

NOTE on `use_smem_v`: the verify body loads v directly from global into `r_v_seq` (hoisted out of the T loop, so each v is read once). The `use_smem_v` flag is kept in the signature/cache-key for parity with `get_mtp_config`, but SMEM staging is not used here — it is a deferred perf optimization (spec §4.5, §11). The launcher still reserves the SMEM, which is harmless when unused.

- [ ] **Step 5: Run the verify tests**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -q -k "verify"`
Expected (GPU box): all PASS. No-GPU box: SKIP.

- [ ] **Step 6: Commit**

```bash
git add cula/lightning/la_verify_kvbuffer.py tests/test_la_verify_kvbuffer.py
git commit -m "feat: implement linear_attention_verify_kvbuffer kernel body (Eq. 7)"
```

---

## Task 6: End-to-end equivalence with baseline

**Files:**
- Test: `tests/test_la_verify_kvbuffer.py` (add end-to-end test)

Asserts the KVBuffer path (verify + state-update at L=T) reproduces the baseline (`cache_intermediate_states=True, disable_state_update=True`): (a) outputs match in bf16 tolerance, (b) the post-update state equals the baseline's last intermediate slice `intermediate_states[:, T-1]` to fp32 tolerance.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py::test_end_to_end_equivalence_with_baseline -q`
Expected (GPU box): PASS. No-GPU box: SKIP.

- [ ] **Step 3: Run the whole new test file**

Run: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -q`
Expected (GPU box): all PASS. No-GPU box: all SKIP, zero import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_la_verify_kvbuffer.py
git commit -m "test: end-to-end equivalence of KVBuffer path vs baseline"
```

---

## Task 7: Benchmark integration

**Files:**
- Modify: `benchmarks/bench_la_decode_mtp.py`

Add a `kvbuf_bytes()` model, time the two new kernels (kernel-only, pre-compiled), and print `kvbuf_total_ms`, `spd_kvbuf`, and a KVBuffer SOL% column.

- [ ] **Step 1: Add the bytes model**

In `benchmarks/bench_la_decode_mtp.py`, add after `la_mtp_bytes` (after line 96):

```python
def kvbuf_bytes(B, T, H, HV, K, V):
    """DRAM traffic for verify + state-update (worst case L=T). Spec §8."""
    bf16, fp32 = 2, 4
    qkv    = B * T * H * K * bf16 * 2 + B * T * HV * V * bf16   # q,k,v reads (verify)
    out_w  = B * T * HV * V * bf16                              # o writes (verify)
    h0_r   = B * HV * V * K * fp32                              # h0 reads (verify)
    update = B * HV * V * K * fp32 * 2 + B * T * H * K * bf16 + B * T * HV * V * bf16
    return qkv + out_w + h0_r + update
```

- [ ] **Step 2: Add imports for the new entry points + their compile caches**

Extend the cula imports near line 55:

```python
from cula.lightning.la_decode_mtp import (
    _get_compiled_la_mtp_kernel,
    get_mtp_config,
    linear_attention_decode_mtp,
)
from cula.lightning.la_verify_kvbuffer import linear_attention_verify_kvbuffer
from cula.lightning.la_state_update_kvbuffer import linear_attention_state_update_kvbuffer
```

- [ ] **Step 3: Time the two KVBuffer kernels in `run_config`**

In `run_config`, after the `cute_seq_ms = benchmark_fn(kernel_cute_seq)` block (around line 216), add:

```python
    # ── KVBuffer verify + state-update (kernel-only via Python wrapper) ──
    s_kvbuf_verify = state_init.clone().permute(0, 1, 3, 2).contiguous()   # [B,HV,V,K]
    out_kvbuf = torch.empty(B, T, HV, V, device=device, dtype=dtype)
    h0_indices_kv = torch.arange(B, device=device, dtype=torch.int32)
    accepted_len_kv = torch.full((B,), T, device=device, dtype=torch.int32)
    s_kvbuf_update = state_init.clone().permute(0, 1, 3, 2).contiguous()   # [B,HV,V,K]

    def kernel_kvbuf_verify():
        linear_attention_verify_kvbuffer(
            q_4d, k_4d, v_4d, s_kvbuf_verify, out_kvbuf,
            decay_scales, h0_indices_kv, scale, T,
        )

    def kernel_kvbuf_update():
        linear_attention_state_update_kvbuffer(
            k_4d, v_4d, s_kvbuf_update, decay_scales,
            h0_indices_kv, accepted_len_kv, T,
        )

    with torch.no_grad():
        kvbuf_verify_ms = benchmark_fn(kernel_kvbuf_verify)
        kvbuf_update_ms = benchmark_fn(kernel_kvbuf_update)
    kvbuf_total_ms = kvbuf_verify_ms + kvbuf_update_ms
```

- [ ] **Step 4: Compute SOL% + speedup and add to the return dict**

After the existing `sol = sol_pct(...)` line (around line 264), add:

```python
    kvbuf_bytes_moved = kvbuf_bytes(B, T, H, HV, K, V)
    kvbuf_sol = sol_pct(kvbuf_bytes_moved, kvbuf_total_ms, peak_bps)
    spd_kvbuf = cute_mtp_ms / kvbuf_total_ms
```

and add these keys to the returned dict (inside the `return {...}`):

```python
        "kvbuf_verify_ms": kvbuf_verify_ms,
        "kvbuf_update_ms": kvbuf_update_ms,
        "kvbuf_total_ms": kvbuf_total_ms,
        "spd_kvbuf": spd_kvbuf,
        "kvbuf_sol_pct": kvbuf_sol,
```

- [ ] **Step 5: Add columns to the table header + row printf**

Extend the `cols` header string (line 319-323) to append:

```python
        f" | {'kvbuf(ms)':>9} | {'spd_kv':>6} | {'kvSOL%':>6}"
```

and extend the per-row print (the `print(...)` at lines 335-343) to append:

```python
                f" | {r['kvbuf_total_ms']:>9.4f} | {r['spd_kvbuf']:>5.2f}x | {r['kvbuf_sol_pct']:>6.1f}"
```

- [ ] **Step 6: Add a Notes line**

Append to the Notes block (after line 353):

```python
    print("  kvbuf     : linear_attention_verify_kvbuffer + _state_update_kvbuffer (L=T worst case)")
    print("  spd_kv    : cute_mtp / kvbuf_total  (KVBuffer speedup vs baseline)")
```

- [ ] **Step 7: Smoke-run the benchmark help (no GPU needed for arg parse)**

Run: `cd /Users/fankun/kernel/cuLA && python benchmarks/bench_la_decode_mtp.py --help`
Expected: argparse help prints with no ImportError (verifies the new imports resolve). On a GPU box, run the full benchmark to confirm the KVBuffer columns populate and `spd_kv > 1` at the default config.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/bench_la_decode_mtp.py
git commit -m "bench: add KVBuffer verify+state-update timing columns and bytes model"
```

---

## Final verification

- [ ] Run the full new suite: `cd /Users/fankun/kernel/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py tests/test_la_decode_mtp.py -q`
- [ ] Confirm `linear_attention_verify_kvbuffer` and `linear_attention_state_update_kvbuffer` import from `cula.lightning`:
  `python -c "from cula.lightning import linear_attention_verify_kvbuffer, linear_attention_state_update_kvbuffer; print('ok')"`
- [ ] On a GPU box: run `python benchmarks/bench_la_decode_mtp.py --batch-sizes 64 --T 4` and confirm `spd_kv ≥ 1.3` at the default config (spec §10).

## Notes for the executing engineer

- **No GPU here.** The author's machine has no GPU; tests will SKIP locally. They must be run on an SM90+ box (the user verifies). Treat a clean import + clean SKIP as the local green bar; correctness is confirmed on GPU.
- **`range_dynamic` spelling** (Task 3): verify the exact attribute name in the installed CuTe DSL before relying on it — see the inline NOTE in Task 3 Step 3.
- **Baseline is sacred:** do not edit `cula/lightning/la_decode_mtp.py`. Both new kernels import from it.
- **`s` layout** is `[pool_size, HV, V, K]` (V-major, K-last) everywhere, viewed internally as `[pool_size·HV, V, K]`. The tests convert to/from the reference's K-major `[B,HV,K,V]` via `.permute(0,1,3,2)`.
