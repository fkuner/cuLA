# LA KVBuffer — SGLang Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `linear_backend="cula"` Lightning-Attention backend to a SGLang fork that routes prefill/decode/verify/commit to cuLA's V-major kernels and replaces the per-draft-token `intermediate_ssm` cache with a slim fixed-T KV buffer, turning per-request memory savings into ~5× concurrent-request capacity for speculative decoding.

**Architecture:** cuLA owns the whole LA layer (wholesale replacement of the uniform Triton `seg_la` path). The persistent state pool (`MambaPool.temporal`) is allocated V-major to match cuLA; the draft `k,v` are buffered in `MambaPool.SpeculativeState` (replacing `intermediate_ssm`) so the post-acceptance commit can recompute `h_L` via cuLA's `state_update`. cuLA kernels are unchanged.

**Tech Stack:** SGLang (Python serving framework, fork at `~/kernel/sglang`), cuLA (CuTe DSL kernels, dependency), PyTorch, Triton (existing seg_la for the equivalence oracle), pytest, CUDA. Target SM90+. **cuLA dependency is one-directional: SGLang imports cuLA; cuLA is never modified by this plan.**

**Spec:** `docs/superpowers/specs/2026-06-15-la-kvbuffer-sglang-integration-design.md` — read it first; this plan implements it.

**Reference (do NOT modify) — cuLA entry points the backend will call:**
- prefill: `cula.ops.lightning_attn.lightning_attn_fwd_varlen(Q,K,V, decay, cu_seqlens, scale, state_pool, initial_state_indices, chunk_size, persistent)` — Q/K/V `[1,T,H,D]`, `state_pool [pool,H,D,D]` fp32 V-major (K-contiguous), INPLACE_UPDATE.
- decode: `cula.ops.la_decode.linear_attention_decode(q,k,v,s,out, softmax_scale, stride_*, s_offsets, decay_scales, HEAD_DIM,K_SPLIT_DIM,V_SPLIT_DIM)` — q/k/v `[B,1,H,K]`, `s [pool,HV,V,K]`.
- verify: `cula.lightning.linear_attention_verify_kvbuffer(q,k,v,s,out, decay_scales, h0_indices, softmax_scale, T)` — read-only on `s`.
- commit: `cula.lightning.linear_attention_state_update_kvbuffer(k,v,s, decay_scales, h0_indices, accepted_len, T)` — writes `s` in place.

**Reference (mirror, do NOT break) — SGLang files this plan edits:**
- `python/sglang/srt/layers/attention/linear/lightning_backend.py` — `LightningAttentionBackend(MambaAttnBackendBase)`: `__init__` (`:39`, `linear_backend` selector `:77`), `forward_extend` (`:275`, seg_la branch `:306`), `forward_decode` (`:349`, seg_la branch `:373`), `_linear_attention_entry` (`:238`).
- `python/sglang/srt/mem_cache/memory_pool.py` — `MambaPool.SpeculativeState` (`:305`), allocation (`:369-417`), `intermediate_ssm` shape `[layers, spec_size+1, T, HV, K, V]` (`:384`).
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` — `update_mamba_state_after_mtp_verify` (`:873`).
- `python/sglang/srt/speculative/spec_utils.py` — `commit_mamba_states_after_verify` (`:555`, calls the hook at `:625`).

---

## File Structure

**Create (SGLang fork):**
- `python/sglang/srt/layers/attention/linear/cula_entry.py` — thin adapters wrapping the 4 cuLA calls (layout reshape, decay/scale conversion, index mapping). One responsibility: translate SGLang batch tensors ↔ cuLA signatures. Keeps `lightning_backend.py` readable and isolates the cuLA dependency.
- `test/srt/models/test_la_cula_equivalence.py` — L1 operator-equivalence tests (cuLA vs seg_la), no model load.

**Modify (SGLang fork):**
- `lightning_backend.py` — add `linear_backend=="cula"` branches in `__init__`, `forward_extend`, `forward_decode`; write draft `k,v` into the KV buffer during verify.
- `memory_pool.py` — `MambaPool`: for the cula backend, allocate `temporal` V-major and replace `intermediate_ssm` with a `draft_kv` buffer.
- `hybrid_linear_attn_backend.py` — `update_mamba_state_after_mtp_verify`: cula variant calling `state_update_kvbuffer` from the KV buffer.
- `spec_utils.py` — `commit_mamba_states_after_verify`: pass `accept_lens` through to the cula commit hook (already in scope at `:558`).

**Do NOT modify:** anything under `cula/` (dependency only); the seg_la / minimax paths (kept as the equivalence oracle and for other models).

---

## Phase 0 — Verification & grounding (resolve the 4 gating unknowns first)

These are investigation tasks. Each produces a recorded answer that the later phases depend on. Do them before writing integration code.

### Task 0.1: Confirm the deployed Bailing `linear_backend` value

**Why:** if the target Bailing config uses `linear_backend="minimax"` (not `"seg_la"`), prefill/decode route to the Bailing-specific kernels and the "wholesale seg_la replacement" framing changes.

- [ ] **Step 1: Inspect the model's HF config for `linear_backend`**

Run: `python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<bailing-model-path>'); print(getattr(c,'linear_backend','<unset→defaults to seg_la>'))"`
Expected: prints `seg_la` (or unset → default `seg_la` per `lightning_backend.py:78`). Record the value.

- [ ] **Step 2: If it is `minimax`, STOP and revise the spec** — the prefill/decode replacement targets `jit_linear_forward_prefix`/`linear_decode_forward_triton` instead of seg_la. Do not proceed with the seg_la-replacement assumption.

### Task 0.2: Pin the decay/scale convention (`tp_slope` ↔ cuLA `decay_scales`)

**Why:** cuLA expects positive `decay_scales[h]` with `λ = exp(-decay_scales[h])`, and pre-scales q by `softmax_scale` internally. The backend builds `tp_slope` via `_build_slope_tensor` (`lightning_backend.py:74`). These must be reconciled or the outputs diverge.

- [ ] **Step 1: Read `_build_slope_tensor` and record what `tp_slope[layer_id]` contains**

Run: `grep -n "_build_slope_tensor" -A 30 python/sglang/srt/layers/attention/linear/lightning_backend.py`
Record: the shape (`[H]`? `[H,1,1]`?), the sign, and whether it already equals `decay_scales` cuLA expects (positive `s_h`) or its negative/log form.

- [ ] **Step 2: Read how seg_la consumes it** (it does `decay_scale = -tl.load(decay_scales+hid)` then `exp(decay_scale)`, `seg_la.py:514/545`) and confirm cuLA's convention (`exp(-decay_scales)`, `la_verify_kvbuffer.py` / `la_decode_mtp.py:210`) matches the SAME `tp_slope` input. Record the exact conversion (likely identity, possibly a `.view(H)`).

- [ ] **Step 3: Confirm `softmax_scale` is applied exactly once** — read the Bailing mixer to check whether q is pre-scaled there (the minimax path sets `linear_scale`, `bailing_moe_linear.py:458`). For the seg_la/cula path, q must be scaled once. Record where.

### Task 0.3: Confirm `temporal` dtype is fp32 and head config is MHA

- [ ] **Step 1: Confirm state dtype**

Run: `grep -n "temporal\|ssm_dtype\|dtype" python/sglang/srt/layers/attention/linear/linear_metadata.py python/sglang/srt/models/bailing_moe_linear.py | grep -i dtype`
Expected: temporal/ssm dtype is fp32 (cuLA state is fp32). Record; if not fp32, cuLA needs a dtype variant (out of current scope — flag).

- [ ] **Step 2: Confirm MHA (`HV == H`)**

Run: `grep -n "num_attention_heads\|total_kv_heads\|num_key_value_heads\|head_dim" python/sglang/srt/models/bailing_moe_linear.py | head`
Expected: `total_kv_heads == num_attention_heads` (MHA), and `head_dim == 128` so `K == V == 128`. Record; if GQA, cuLA prefill needs HV support (out of scope — flag).

### Task 0.4: Smoke-confirm the Bailing nextn + linear spec-decode path runs (baseline, seg_la)

**Why:** establishes the working seg_la baseline that the cula backend must match token-for-token, and proves the LA+MTP path is live in this build.

- [ ] **Step 1: Run a tiny greedy generation with the existing seg_la backend + nextn draft**

Run (adjust model path/args): `python -m sglang.bench_one_batch --model <bailing-model> --speculative-algorithm EAGLE --speculative-draft-model-path <bailing-nextn> --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 --batch-size 1 --input-len 16 --output-len 16`
Expected: runs without error, emits tokens. Record the exact command + the output tokens for a fixed seed — this is the **golden reference** for Task 5.2 (end-to-end equivalence).

- [ ] **Step 2: Record the seg_la `intermediate_ssm` GB from the startup log** (`memory_pool.py:423` logs it) — this is the memory the KV buffer will reclaim; the Task 5.4 capacity test compares against it.

---

## Phase 1 — MambaPool: V-major `temporal` + slim KV buffer

The pool is constructed before the backend, so it needs to know the cula mode. We thread a `linear_backend: str` argument into `MambaPool.__init__` and branch the spec allocation: V-major `temporal`, and a `draft_k`/`draft_v` buffer instead of `intermediate_ssm`/`intermediate_conv_window`.

### Task 1.1: Thread `linear_backend` into MambaPool construction

**Files:**
- Modify: `python/sglang/srt/mem_cache/memory_pool.py:309` (`MambaPool.__init__` signature)
- Modify: the MambaPool construction site (locate in Step 1)

- [ ] **Step 1: Locate where MambaPool is constructed and how hf_config reaches it**

Run: `grep -rn "MambaPool(" python/sglang/srt/ | grep -v "class MambaPool"`
Record the call site(s) and confirm `hf_config.linear_backend` (or the already-parsed value) is reachable there. (`HybridReqToTokenPool` wraps it — check `memory_pool.py` around `:547`.)

- [ ] **Step 2: Add the `linear_backend` parameter to `MambaPool.__init__`**

In `memory_pool.py`, add to the `__init__` keyword args (after `speculative_num_draft_tokens`, `:318`):

```python
        linear_backend: str = "seg_la",
```

Store it: `self.linear_backend = linear_backend` near `self.size = size` (`:329`).

- [ ] **Step 3: Pass it at the construction site**

At the call site found in Step 1, pass `linear_backend=getattr(hf_config, "linear_backend", "seg_la")`.

- [ ] **Step 4: Verify import resolves**

Run: `python -c "import sglang.srt.mem_cache.memory_pool"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/mem_cache/memory_pool.py <call-site-file>
git commit -m "feat(cula): thread linear_backend into MambaPool"
```

### Task 1.2: Add `CulaSpeculativeState` + V-major / KV-buffer allocation branch

**Files:**
- Modify: `python/sglang/srt/mem_cache/memory_pool.py:304` (new dataclass), `:374-417` (allocation branch)
- Test: `test/srt/test_mamba_pool_cula.py`

- [ ] **Step 1: Write the failing test (shapes + dtypes of the cula allocation)**

Create `test/srt/test_mamba_pool_cula.py`:

```python
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.mem_cache.memory_pool import MambaPool


def _make_pool(linear_backend):
    # Minimal cache_params stub mirroring a Lightning/Mamba2 state: HV=4, K=V=128.
    from sglang.srt.layers.attention.mamba.mamba_utils import Mamba2StateShape  # adjust if path differs
    # Build via the model's real cache_params in practice; here assert post-construction shapes.
    ...


def test_cula_temporal_is_v_major_and_has_kv_buffer():
    # Construct a MambaPool in cula mode with T=4 draft tokens and inspect mamba_cache.
    pool = _make_pool("cula")
    st = pool.mamba_cache
    # temporal V-major: last two dims are (V, K) == (128,128) and K is contiguous (stride 1)
    assert st.temporal.shape[-2:] == (128, 128)
    assert st.temporal.stride()[-1] == 1
    # KV buffer present, intermediate_ssm absent
    assert hasattr(st, "draft_k") and hasattr(st, "draft_v")
    assert not hasattr(st, "intermediate_ssm")
    # draft buffer is bf16 and sized for T draft tokens
    assert st.draft_k.dtype == torch.bfloat16
    assert st.draft_k.shape[2] == 4  # speculative_num_draft_tokens
```

NOTE: the `_make_pool` helper must build the real `cache_params`/`mamba_layer_ids` the model uses (read them from `bailing_moe_linear` cache params during Phase 0). Fill it in from the confirmed config — do not stub shapes that diverge from the model.

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/kernel/sglang && python -m pytest test/srt/test_mamba_pool_cula.py -q`
Expected (GPU): FAIL — cula mode not implemented, `draft_k` missing. (No-GPU: module skip.)

- [ ] **Step 3: Add the `CulaSpeculativeState` dataclass**

In `memory_pool.py` after `SpeculativeState` (`:307`):

```python
    @dataclass(frozen=True, kw_only=True)
    class CulaSpeculativeState(State):
        # Slim KV buffer (paper §3.3): draft k,v persisted verify→commit,
        # replacing the per-draft-token intermediate_ssm cache.
        draft_k: torch.Tensor  # [num_layers, spec_size+1, T, H,  K] bf16
        draft_v: torch.Tensor  # [num_layers, spec_size+1, T, HV, V] bf16
```

- [ ] **Step 4: Add the allocation branch**

In `__init__`, inside `if speculative_num_draft_tokens is not None:` (`:374`), branch on `self.linear_backend == "cula"`. The cula branch allocates V-major `temporal` (transpose last two dims of `temporal_state_shape`, mirroring the NPU transpose at `:376-381`) and the `draft_k`/`draft_v` buffers instead of the intermediate caches:

```python
            if speculative_num_draft_tokens is not None:
                if self.linear_backend == "cula":
                    # V-major temporal: (HV, K, V) -> (HV, V, K), K contiguous.
                    temporal_state = temporal_state.transpose(-1, -2).contiguous()
                    HV, Kd, Vd = temporal_state_shape  # original (HV, K, V)
                    T = speculative_num_draft_tokens
                    H = HV  # MHA target (confirmed in Task 0.3); GQA out of scope
                    draft_k = torch.zeros(
                        (num_mamba_layers, spec_state_size + 1, T, H, Kd),
                        dtype=torch.bfloat16, device="cuda",
                    )
                    draft_v = torch.zeros(
                        (num_mamba_layers, spec_state_size + 1, T, HV, Vd),
                        dtype=torch.bfloat16, device="cuda",
                    )
                    self.mamba_cache = self.CulaSpeculativeState(
                        conv=conv_state, temporal=temporal_state,
                        draft_k=draft_k, draft_v=draft_v,
                    )
                    logger.info(
                        f"Mamba Cache (cula) allocated. temporal V-major {tuple(temporal_state.shape)}, "
                        f"draft_kv size: {(get_tensor_size_bytes(draft_k)+get_tensor_size_bytes(draft_v))/GB:.3f}GB"
                    )
                elif _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    # ... existing NPU + default seg_la path unchanged below ...
```

Keep the existing `intermediate_ssm` allocation as the `else` (seg_la/minimax) path verbatim.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd ~/kernel/sglang && python -m pytest test/srt/test_mamba_pool_cula.py -q`
Expected (GPU): PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/mem_cache/memory_pool.py test/srt/test_mamba_pool_cula.py
git commit -m "feat(cula): V-major temporal + slim draft-KV buffer in MambaPool"
```

---

## Phase 2 — cula backend: ops routing + KV-buffer write

Isolate the cuLA calls in `cula_entry.py` (layout/scale/decay/index translation), then branch the backend on `linear_backend == "cula"`.

### Task 2.1: `cula_entry.py` — the 4 adapters (validated by Phase 5 L1, not unit-tested in isolation)

**Files:**
- Create: `python/sglang/srt/layers/attention/linear/cula_entry.py`

- [ ] **Step 1: Write the adapter module**

Create `cula_entry.py`. Each function takes SGLang's packed tensors + metadata and calls the corresponding cuLA entry. Layout/convention values come from Phase 0 (`DECAY = tp_slope.view(H)` per Task 0.2; `SCALE` applied once per Task 0.2 Step 3). Use the exact cuLA signatures from the plan header.

```python
# Copyright 2025-2026 ...
"""Adapters: SGLang LA batch tensors <-> cuLA Lightning-Attention kernels.
Isolates the one-directional cuLA dependency. Correctness validated by
test/srt/models/test_la_cula_equivalence.py (Phase 5)."""
import torch
from cula.ops.lightning_attn import lightning_attn_fwd_varlen
from cula.ops.la_decode import linear_attention_decode
from cula.lightning import (
    linear_attention_verify_kvbuffer,
    linear_attention_state_update_kvbuffer,
)


def cula_prefill(q, k, v, temporal, cache_indices, cu_seqlens, decay, scale):
    # q,k,v packed [tokens, H, D] -> cuLA expects [1, T, H, D]
    return lightning_attn_fwd_varlen(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        decay, cu_seqlens, scale=scale,
        state_pool=temporal, initial_state_indices=cache_indices,
    )


def cula_decode(q, k, v, temporal, cache_indices, decay, scale, out):
    # packed [B, H, K] -> cuLA decode expects [B, 1, H, K]
    HEAD_DIM = q.shape[-1]
    linear_attention_decode(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), temporal, out,
        softmax_scale=scale, stride_q=0, stride_k=0, stride_v=0, stride_s=0, stride_o=0,
        s_offsets=cache_indices, decay_scales=decay,
        HEAD_DIM=HEAD_DIM, K_SPLIT_DIM=HEAD_DIM, V_SPLIT_DIM=v.shape[-1],
    )
    return out


def cula_verify(q, k, v, temporal, cache_indices, decay, scale, T, out):
    # packed [B*T, H, K] -> [B, T, H, K]; uniform T = draft_token_num
    B = cache_indices.shape[0]
    q4 = q.view(B, T, *q.shape[1:]); k4 = k.view(B, T, *k.shape[1:]); v4 = v.view(B, T, *v.shape[1:])
    out4 = out.view(B, T, *out.shape[1:])
    linear_attention_verify_kvbuffer(
        q4, k4, v4, temporal, out4, decay, cache_indices, scale, T)
    return out


def cula_commit(draft_k, draft_v, temporal, cache_indices, accepted_len, decay, T):
    # draft_k/draft_v: [B, T, H/HV, *] already gathered for this layer
    linear_attention_state_update_kvbuffer(
        draft_k, draft_v, temporal, decay, cache_indices, accepted_len, T)
```

NOTE: the exact `stride_*` args to `linear_attention_decode` and whether `out` is preallocated by the caller must match cuLA's `la_decode.py:581` signature — confirm against that file when writing (the plan header lists the params). If cuLA's decode computes strides internally from the tensors, pass `0` placeholders as shown; otherwise pass `q.stride(0)` etc.

- [ ] **Step 2: Verify import resolves (needs cuLA installed in the SGLang env)**

Run: `cd ~/kernel/sglang && python -c "import sglang.srt.layers.attention.linear.cula_entry"`
Expected: no ImportError (requires `pip install -e ~/kernel/cuLA` in the env). If cuLA import fails, fix the env before proceeding.

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/layers/attention/linear/cula_entry.py
git commit -m "feat(cula): SGLang<->cuLA adapter entry points"
```

### Task 2.2: Branch the backend on `linear_backend == "cula"`

**Files:**
- Modify: `lightning_backend.py` `__init__` (`:77`), `forward_extend` (`:306`), `forward_decode` (`:373`)

- [ ] **Step 1: `__init__` — accept "cula" as a valid backend**

The selector at `:77` already reads `linear_backend`. No code change needed there, but confirm the `topk > 1` guard (`:44`) still applies (cula is chain-only too — keep it).

- [ ] **Step 2: `forward_extend` — add the cula branch**

In `forward_extend`, alongside the `elif self.linear_backend == "seg_la":` block (`:306`), add `elif self.linear_backend == "cula":`. For non-verify (prefill) call `cula_prefill`; for `is_target_verify()` call `cula_verify` AND write the draft `k,v` into the KV buffer for commit:

```python
        elif self.linear_backend == "cula":
            decay = self.tp_slope[layer_id].view(-1)  # [H] per Task 0.2
            scale = self.scale  # applied once; per Task 0.2 Step 3
            if forward_batch.forward_mode.is_target_verify():
                T = self.forward_metadata.draft_token_num  # = speculative_num_draft_tokens
                out = torch.empty_like(v)
                o = cula_verify(q, k, v, ssm_states, cache_indices, decay, scale, T, out)
                # buffer draft k,v for commit (slim KV buffer)
                st = mamba_cache_params  # CulaSpeculativeState slice for this layer
                B = cache_indices.shape[0]
                st.draft_k[cache_indices] = k.view(B, T, *k.shape[1:]).to(torch.bfloat16)
                st.draft_v[cache_indices] = v.view(B, T, *v.shape[1:]).to(torch.bfloat16)
            else:
                o = cula_prefill(q, k, v, ssm_states, cache_indices,
                                 self.forward_metadata.query_start_loc, decay, scale)
```

NOTE: confirm how a per-layer `CulaSpeculativeState` slice exposes `draft_k`/`draft_v` (mirror how `mamba_cache_params.intermediate_ssm` is sliced at `:325` and `at_layer_idx`, `memory_pool.py:285`). Index `[cache_indices]` writes the per-request rows.

- [ ] **Step 3: `forward_decode` — add the cula branch**

Alongside `elif self.linear_backend == "seg_la":` (`:373`):

```python
        elif self.linear_backend == "cula":
            decay = self.tp_slope[layer_id].view(-1)
            out = torch.empty_like(v)
            o = cula_decode(q, k, v, ssm_states, cache_indices, decay, self.scale, out)
```

Add `from sglang.srt.layers.attention.linear.cula_entry import cula_prefill, cula_decode, cula_verify` at the top.

- [ ] **Step 4: Verify import + a no-GPU構造 smoke (backend instantiation needs a model runner — defer real run to Phase 5)**

Run: `cd ~/kernel/sglang && python -c "import sglang.srt.layers.attention.linear.lightning_backend"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/attention/linear/lightning_backend.py
git commit -m "feat(cula): route prefill/decode/verify to cuLA + buffer draft kv"
```

---

## Phase 3 — Commit via cuLA `state_update`

### Task 3.1: cula commit hook on the backend

**Files:**
- Modify: `hybrid_linear_attn_backend.py` (`update_mamba_state_after_mtp_verify`, `:873`) or add a cula override
- Modify: `spec_utils.py` (`commit_mamba_states_after_verify`, `:625`) to pass `accept_lens`

- [ ] **Step 1: Pass `accept_lens` to the commit hook**

`commit_mamba_states_after_verify` already has `accept_lens` (`:558`). At the hook call (`:625`), add `accept_lens=accept_lens` to the kwargs (the seg_la hook can ignore it; the cula hook uses it).

- [ ] **Step 2: Implement the cula commit**

In the backend's `update_mamba_state_after_mtp_verify`, branch for cula: derive `accepted_len` and call `cula_commit`. For topk==1, `accepted_len = accept_lens` (drafts accepted; `accept_lens` already includes the bonus token per `spec_utils.py:581` — confirm whether to subtract 1 for the state recurrence: the state must advance over exactly the accepted DRAFT tokens, so `L = accept_lens - 1` if the bonus is the target's sampled token, else `accept_lens`; pin this against the seg_la scatter's `last_correct_step_indices = accept_lens - 1`, `:593`). Use `L = (accept_lens - 1).clamp(min=0).to(torch.int32)` to match `last_correct_step_indices`.

```python
    def update_mamba_state_after_mtp_verify(self, *, last_correct_step_indices,
                                            accept_lens=None, model=None, **kw):
        if self.linear_backend != "cula":
            return self._scatter_commit(last_correct_step_indices=last_correct_step_indices, model=model, **kw)
        from sglang.srt.layers.attention.linear.cula_entry import cula_commit
        L = last_correct_step_indices.to(torch.int32)  # == accept_lens-1 for topk==1; the state advances L draft steps
        T = self.forward_metadata.draft_token_num
        for layer_id in self.mamba_layer_ids:
            st = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            cache_indices = self.forward_metadata.mamba_cache_indices
            cula_commit(
                st.draft_k[cache_indices].view(-1, T, *st.draft_k.shape[-2:]),
                st.draft_v[cache_indices].view(-1, T, *st.draft_v.shape[-2:]),
                st.temporal, cache_indices, L,
                self.tp_slope[layer_id].view(-1), T,
            )
```

NOTE: rename the existing scatter body to `_scatter_commit` (keep verbatim) so the seg_la path is untouched. Confirm `mamba_layer_ids` / `mamba2_layer_cache` accessors against `memory_pool.py` (`HybridReqToTokenPool`). The exact `L` semantics (accept_lens vs accept_lens-1) MUST be pinned by the Phase 5 L2 token-equivalence test — if tokens diverge by one position, flip the off-by-one here.

- [ ] **Step 3: Verify import resolves**

Run: `cd ~/kernel/sglang && python -c "import sglang.srt.layers.attention.hybrid_linear_attn_backend, sglang.srt.speculative.spec_utils"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py python/sglang/srt/speculative/spec_utils.py
git commit -m "feat(cula): commit accepted state via cuLA state_update from KV buffer"
```

---

## Phase 4 — CUDA-graph warmup

cuLA compiles lazily (`cute.compile` on first call), which cannot happen inside graph capture. Warm up every captured shape first. SGLang captures Decode and Target-Verify (`lightning_backend.py:32-36`).

### Task 4.1: Warm up cula kernels before capture

**Files:**
- Modify: `lightning_backend.py` `init_cuda_graph_capture` (or `init_cuda_graph_state`) — locate exact method in Step 1.

- [ ] **Step 1: Locate the capture entry**

Run: `grep -n "def init_cuda_graph_state\|def init_cuda_graph_capture\|def init_forward_metadata_capture_cuda_graph\|capture_bs\|cuda_graph_bs" python/sglang/srt/layers/attention/linear/lightning_backend.py python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
Record the method that runs once per captured batch size and the list of captured batch sizes.

- [ ] **Step 2: Add a warmup that calls each cula op once per captured `(B, T)`**

In the cula branch of that method, before the graph is captured, run a throwaway call of `cula_decode` and `cula_verify` at each captured batch size `B` (T = `draft_token_num`), on the real pool tensors with dummy inputs, so `cute.compile` populates its cache. Note `linear_attention_decode` branches on `B<=32` vs `B>32` (`la_decode.py:644`) — warm a `B<=32` and a `B>32` size if both are captured.

```python
        if self.linear_backend == "cula":
            from sglang.srt.layers.attention.linear.cula_entry import cula_decode, cula_verify
            for bs in self.cuda_graph_bs:  # the captured batch sizes
                decay = self.tp_slope[0].view(-1)
                # dummy decode warmup
                qd = torch.zeros((bs, self.num_heads, self.head_dim), device=self.device, dtype=torch.bfloat16)
                cula_decode(qd, qd, qd, self.req_to_token_pool.mamba2_layer_cache(0).temporal,
                            torch.zeros(bs, dtype=torch.int32, device=self.device), decay, self.scale,
                            torch.empty_like(qd))
                # dummy verify warmup (B*T tokens)
                T = self.forward_metadata.draft_token_num
                qv = torch.zeros((bs * T, self.num_heads, self.head_dim), device=self.device, dtype=torch.bfloat16)
                cula_verify(qv, qv, qv, self.req_to_token_pool.mamba2_layer_cache(0).temporal,
                            torch.zeros(bs, dtype=torch.int32, device=self.device), decay, self.scale, T,
                            torch.empty_like(qv))
            torch.cuda.synchronize()
```

NOTE: adjust attribute names (`self.cuda_graph_bs`, `self.num_heads`, `self.head_dim`, `self.scale`) to the backend's actual fields (found in Step 1 / `__init__`). Use `h0_indices` of all-zeros (dummy slot 0) so warmup doesn't corrupt real state — slot 0 is the padding row.

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/layers/attention/linear/lightning_backend.py
git commit -m "feat(cula): warm up cuLA kernels before CUDA graph capture"
```

---

## Phase 5 — Tests

### Task 5.1: L1 — operator equivalence (cuLA vs seg_la), no model

**Files:**
- Create: `test/srt/models/test_la_cula_equivalence.py`

- [ ] **Step 1: Write the equivalence test**

Same random `(q,k,v, initial state, accept_len)` through both paths; assert verify outputs and committed state match. This validates the §4.5 adapters (layout/scale/decay/index) and the commit off-by-one.

```python
import pytest, torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from cula.lightning import linear_attention_verify_kvbuffer, linear_attention_state_update_kvbuffer

def test_cula_verify_matches_seg_la_mtp():
    B, T, H, D = 2, 4, 4, 128
    torch.manual_seed(0)
    scale = D ** -0.5
    decay = (0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H)
    q = torch.randn(B*T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B*T, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B*T, H, D, device="cuda", dtype=torch.bfloat16)
    # seg_la state pool [pool, H, K, V] K-major; cuLA pool [pool, H, V, K] V-major (transpose)
    state_kmajor = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01
    s_seg = state_kmajor.clone()
    s_cula = state_kmajor.transpose(-1, -2).contiguous()  # V-major

    # seg_la MTP verify (writes per-step intermediate cache)
    caches = torch.zeros(B*T, H, D, D, device="cuda", dtype=torch.float32)
    cache_idx = torch.arange(B, device="cuda", dtype=torch.int32)
    meta = SegLaMeta(batch_size=B, max_q_length=T,
                     q_offsets=torch.arange(0, B*T+1, T, device="cuda", dtype=torch.int32),
                     s_offsets=cache_idx, q_lengths=torch.full((B,), T, device="cuda", dtype=torch.int32),
                     s_scales=torch.ones(B, device="cuda", dtype=torch.int32))
    o_seg = seg_la_fwd(q, k, v, s_seg, decay, meta, caches=caches,
                       cache_indices=cache_idx, softmax_scale=scale)

    # cuLA verify
    out_cula = torch.zeros(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    linear_attention_verify_kvbuffer(
        q.view(B,T,H,D), k.view(B,T,H,D), v.view(B,T,H,D),
        s_cula, out_cula, decay, cache_idx, scale, T)

    rel = (out_cula.view(B*T,H,D).float() - o_seg.float()).pow(2).mean().sqrt() / (o_seg.float().abs().max()+1e-8)
    assert rel < 1e-2, f"verify output mismatch vs seg_la: {rel:.5f}"
```

NOTE: the exact seg_la output layout (`o_seg` is `[tokens, H, D]`) and `SegLaMeta` field names must match `seg_la.py:17`/`:657`; adjust if Phase 0 reading shows differences. Add a companion `test_cula_commit_matches_seg_la_scatter` that runs `state_update_kvbuffer(L)` and compares the committed `temporal` against seg_la's scattered `intermediate_ssm[L-1]`.

- [ ] **Step 2: Run**

Run: `cd ~/kernel/sglang && python -m pytest test/srt/models/test_la_cula_equivalence.py -q`
Expected (GPU): PASS once adapters/conventions are right. **If it fails, the convention (decay sign / scale double-apply / V-major transpose) is wrong — fix in `cula_entry.py`, not by changing cuLA.**

- [ ] **Step 3: Commit**

```bash
git add test/srt/models/test_la_cula_equivalence.py
git commit -m "test(cula): operator equivalence vs seg_la (verify + commit)"
```

### Task 5.2: L2 — single-model end-to-end token equivalence

- [ ] **Step 1: Generate with cula backend, compare to the seg_la golden from Task 0.4**

Run the exact command from Task 0.4 Step 1 but with the model configured `linear_backend="cula"` (set in hf_config or via an override flag), same seed/prompt/greedy.
Expected: **identical output tokens** to the seg_la golden. Speculative decoding is output-preserving, so any divergence = a bug in verify/commit (most likely the commit off-by-one in Task 3.1 — flip `accept_lens` vs `accept_lens-1`).

- [ ] **Step 2: Record PASS/FAIL + tokens. Commit any fix to the off-by-one if needed (in cula_entry/commit, not cuLA).**

### Task 5.3: L3 — partial-accept correctness

- [ ] **Step 1: Force partial acceptance (draft mostly wrong) and assert tokens still equal non-spec-decode greedy output**

Run a prompt where the draft is frequently rejected (e.g. low draft quality / temperature 0 mismatch), with `linear_backend="cula"`, and compare tokens to the SAME model run WITHOUT speculative decoding.
Expected: identical tokens (exercises `state_update`'s `L<T` path). Divergence → `L<T` commit bug.

### Task 5.4: L4 — capacity / throughput (the payoff)

- [ ] **Step 1: Measure max concurrent requests at fixed HBM, cula vs seg_la**

Run `python -m sglang.bench_serving` (or `bench_one_batch` with rising `--max-running-requests`) under memory pressure for both backends; record max concurrency and tokens/s.
Expected: cula admits more concurrent requests (the reclaimed `intermediate_ssm` GB from Task 0.4 Step 2 → more MambaPool slots), targeting the paper's ~5× / ~1.46×. Record the numbers; this is the result that justifies the work.

---

## Self-Review

**1. Spec coverage:** §2 unified replacement → Phase 2; §4.1 backend routing → Task 2.2; §4.2 V-major pool → Task 1.2; §4.3 commit → Phase 3; §4.4 KV buffer → Task 1.2 + 2.2 (write) + 3.1 (read); §4.5 adapters → Task 2.1; §4.6 warmup → Phase 4; §5 must-verify → Phase 0; §6 tests L0-L4 → Phase 5 (L0 is the existing cuLA suite, run as a precondition). Covered.

**2. Placeholder scan:** No "TODO/TBD". The `NOTE:` blocks are *grounding caveats* (confirm an exact SGLang field/signature against a cited file before writing the line) — required because this targets a large GPU-only repo that cannot be executed from the authoring environment; they cite the exact file:line to check, not "figure it out later." Phase 0 outputs (decay convention, config, dtype, MHA) feed concrete values into Phases 1-4.

**3. Type consistency:** `CulaSpeculativeState.draft_k/draft_v` (Task 1.2) are written in Task 2.2 and read in Task 3.1 under the same names; `cula_entry` functions `cula_prefill/cula_decode/cula_verify/cula_commit` (Task 2.1) are called with matching signatures in Tasks 2.2/3.1/4.1; `linear_backend=="cula"` is the consistent selector throughout.

## Known plan risk

This plan edits a large GPU-only framework (SGLang) that cannot be run from the authoring box, so several steps carry a `NOTE:` to confirm an exact field/signature against a cited SGLang file before finalizing that line. Phase 0 must complete first — its findings (esp. Task 0.1 backend value, Task 0.2 decay/scale convention, Task 0.3 dtype/MHA) can require adjusting later-phase specifics. Treat the L1 (Task 5.1) and L2 (Task 5.2) tests as the correctness arbiters: convention bugs surface in L1, the commit off-by-one in L2.

