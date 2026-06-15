# LA KVBuffer — SGLang Integration Design

**Status:** Design (awaiting plan & implementation)
**Depends on:** `2026-06-15-la-mtp-kvbuffer-design.md` (the cuLA verify/state-update kernels, branch `la-decode-mtp`)
**Reference paper:** KVBuffer: IO-aware Serving for Linear Attention (Zou & Zhong, 2026) — note the paper implements KVBuffer for **Qwen3-Next / Gated DeltaNet**; this work targets the **Lightning Attention** variant instead.
**Repos:** cuLA `/Users/fankun/kernel/cuLA`, SGLang `/Users/fankun/kernel/sglang`

---

## 1. Goal

Make cuLA's Lightning-Attention (LA) KVBuffer kernels usable inside SGLang so that the per-request **memory saving** turns into **serving-level concurrency capacity** (paper Fig. 5: ~5× requests, ~1.46× throughput). The kernel-only speedup is small and not the point; the win is **replacing the large per-draft-token intermediate state cache (`intermediate_ssm`) with a small draft-KV buffer** — the paper's KVBuffer mechanism for the §3.3 parallel-verify scenario.

## 2. Core architectural decision: cuLA owns the whole LA layer

cuLA already has a complete, internally-consistent LA operator set, **all in V-major (`[…, V, K]`, K-contiguous) layout**:

| LA op | cuLA entry | file |
|---|---|---|
| prefill (varlen, pool-indexed) | `lightning_attn_fwd_varlen` | `cula/ops/lightning_attn.py` |
| single-token decode | `linear_attention_decode` | `cula/ops/la_decode.py` |
| MTP verify (KVBuffer) | `linear_attention_verify_kvbuffer` | `cula/lightning/la_verify_kvbuffer.py` |
| MTP state-commit (KVBuffer) | `linear_attention_state_update_kvbuffer` | `cula/lightning/la_state_update_kvbuffer.py` |

Therefore we introduce a **new `linear_backend="cula"`** that routes **all four** LA layer operations to cuLA, **replacing the Triton `seg_la` path entirely** for models that opt in. The persistent state pool stays in cuLA's native V-major layout end-to-end — **no per-call transpose, no K-major rework of cuLA, no layout copy.**

Rejected alternatives:
- *Splice cuLA only into the verify step, keep seg_la for prefill/decode* — forces cuLA to read/write the shared pool in seg_la's K-major layout (the pool is shared across prefill/decode/verify/commit), which means either reworking cuLA's coalescing-sensitive lane layout or a 134 MB transpose-copy per call that erases the memory win.
- *Boundary transpose-copy* — same 134 MB copy, rejected.

**Consequence:** cuLA's kernel interfaces do **not** change for layout. The adaptation work lives in SGLang (a backend class + a pool-allocation branch) plus thin reshape/convention adapters.

## 3. cuLA-side changes (minimal)

cuLA stays framework-agnostic; dependency direction is **SGLang → cuLA** only. Expected cuLA changes are limited to:

1. **None required for layout** — all four ops are already V-major and pool-indexed.
2. **Possible convention shim** (TBD after verification): if SGLang's `tp_slope` is not numerically identical to cuLA's expected positive `decay_scales` (`λ = exp(-decay_scales[h])`), the conversion is done on the SGLang side, not in cuLA. cuLA's contract is unchanged.
3. **Optional GQA note:** cuLA decode/verify/commit carry an `HV` axis; cuLA **prefill** is single-`H` only. The first target model is MHA (`HV==H`), so this is a non-issue now. A future GQA LA model would need HV support added to cuLA prefill — out of scope here.

If a later perf pass shows the V-major pool hurts any SGLang-side access, that is a cuLA-internal tuning question, not an interface change.

## 4. SGLang-side adaptations

All in a SGLang fork / feature branch. Ordered by dependency.

### 4.1 New backend `linear_backend="cula"`
A backend class (new, or a branch inside `layers/attention/linear/lightning_backend.py`) selected when `hf_config.linear_backend == "cula"` (the selector already exists, `lightning_backend.py:78`, default `"seg_la"`). It implements:
- `forward_extend` (prefill) → `lightning_attn_fwd_varlen(state_pool=temporal, initial_state_indices=mamba_cache_indices, cu_seqlens=query_start_loc, …)`.
- `forward_decode` (single token) → `linear_attention_decode(s=temporal, s_offsets=mamba_cache_indices, …)`.
- `forward_extend` under `is_target_verify()` → `linear_attention_verify_kvbuffer(s=temporal, h0_indices=mamba_cache_indices, …)`, **NOT** writing any intermediate state cache. Additionally, the backend **copies the draft `k,v` into the slim KV buffer (§4.4)** so they survive to the commit step (a cheap bf16 copy; `k,v` are already the verify inputs).

**Routing fact (verified):** for the Bailing target the model's `linear_backend` defaults to `"seg_la"` (`bailing_moe_linear.py:456`), and with that value **prefill, decode, AND verify all go through `seg_la_fwd`** (`lightning_backend.py:306`/`:373` → `_linear_attention_entry`). The Bailing-specific kernels (`jit_linear_forward_prefix`, `linear_decode_forward_triton`) are only used when `linear_backend == "minimax"` (`:295`/`:371`). So replacing the whole seg_la path with cuLA is a **clean wholesale swap of one uniform K-major Triton path** — there is no separate Bailing prefill to also patch. (Must-verify: confirm the deployed Bailing config is `seg_la`, not `minimax`.)

### 4.2 MambaPool V-major allocation branch
`mem_cache/memory_pool.py` allocates `temporal` as `[HV, K, V]` (V-contiguous = K-major). Add a `linear_backend=="cula"` branch (alongside the existing NPU-only `transpose(-1,-2)` at ~`:376`) that allocates/transposes `temporal` to cuLA's **V-major `[HV, V, K]` (K-contiguous)**. Confirm `K == V == 128` for the target model so the slot is square. **This is the only structural SGLang memory change.**

### 4.3 Commit via cuLA state-update (replace the scatter)
SGLang's current commit (`update_mamba_state_after_mtp_verify`, `hybrid_linear_attn_backend.py:873`) scatters `intermediate_ssm[last_correct_step] → temporal`. Replace, for the cula backend, with:
1. Derive `accepted_len[b] ∈ [0, T]` from `last_correct_step_indices` (commit length = correct steps; the existing `spec_utils.py:593` already computes `last_correct_step_indices = accept_index[…, accept_lens-1] - offset`; `accept_lens` is already in scope at the commit site, `spec_utils.py:558`).
2. Read the draft `k,v` back **from the slim KV buffer (§4.4)** and call `linear_attention_state_update_kvbuffer(k, v, temporal, decay_scales, h0_indices=mamba_cache_indices, accepted_len, T)`.
This recomputes `h_L` from `(h_0, accepted k, v)` instead of reading a per-step state cache.

**Invariant:** verify is read-only on `temporal`, so between verify and commit the pool slot still holds `h_0`; the commit reads `h_0` + buffered `k,v`. SGLang must not advance `temporal` for these requests between the two steps.

### 4.4 The slim KV buffer (the paper's KVBuffer, §3.3 special case) — replaces `intermediate_ssm`
This is the **core mechanism**, not a side effect. Linear attention keeps no per-token KV cache (only the state), so the draft `k,v` are transient activations freed when the verify forward returns — but the commit (post-sampling, outside the forward) needs them. So we **persist the draft `k,v` in a buffer**: exactly the paper's KVBuffer for the verify scenario (paper Fig. 1 / §3.3).

- **Shape:** fixed per-request `[slots, T, H, K]` (k) + `[slots, T, HV, V]` (v), bf16. **No paging** — `T = draft_token_num` is fixed and known, the buffer is written in verify and consumed in commit each cycle, so there is no dynamic growth or fragmentation to manage.
- **Where:** `MambaPool.SpeculativeState`, **replacing** the `intermediate_ssm` allocation (`memory_pool.py:384`, GB logged at `:423`).
- **Memory win (the entire point):** out goes `intermediate_ssm` = `slots·T·HV·V·K·4` (per-step fp32 states); in comes the KV buffer = `slots·T·H·K·2 + slots·T·HV·V·2` (bf16, no K·V product). Net saving is large → more MambaPool slots → ~5× concurrent requests (paper Fig. 5).
- **Not built here:** the general **paged** KV buffer (§3.1) and the §3.2 chunkwise-decode / §3.4 KV-only scenarios that need it. Those unlock the separate ~45% *decode-latency* win but require new chunkwise-decode machinery; out of scope (see §7).

### 4.5 Tensor / convention adapters
- **q/k/v layout:** mixer emits packed `[total_tokens, H, head_dim]`. Reshape to cuLA's expected `[1,T,H,D]` (prefill), `[B,1,H,K]` (decode), `[B,T,H,K]`/`[B,T,HV,V]` (verify). Valid because target-verify uses a uniform `draft_token_num` per request. Mixed prefill+decode batches must be split into per-mode sub-calls (cuLA has no single fused mixed-mode entry).
- **decay/scale:** pass `tp_slope.view(H)` as `decay_scales` after confirming sign/magnitude equals cuLA's positive `s_h`; ensure `softmax_scale` is applied exactly once (cuLA pre-scales q internally — do not also scale in the mixer).
- **h0_indices / s_offsets / cu_seqlens:** map to `mamba_cache_indices` and `query_start_loc` (`forward_metadata`).

### 4.6 CUDA-graph warmup
cuLA compiles kernels lazily (`@functools.cache` + `cute.compile`). SGLang captures CUDA graphs for **Decode** and **Target-Verify**, and compilation cannot run inside capture. **Before `init_cuda_graph_capture`, force a warmup call of every cula op at every captured shape** `(B, T, pool_size)` — and note `linear_attention_decode` branches on `B<=32` vs `B>32` (two kernels → two warmups). Varlen prefill is not graph-captured, so its lazy compile is fine.

## 5. Validation target + must-verify before implementation

**Validation target (RESOLVED): `bailing_moe_linear` + `bailing_moe_nextn`.**
`BailingMoELinearAttention` (`models/bailing_moe_linear.py:417`) routes its linear layers through the lightning/seg_la backend (`linear_backend` default `"seg_la"`, `:456`); the model is **hybrid** (`is_linear_layer(layer_idx, layer_group_size)`, `:129/:940` — some layers linear, some full attention). `bailing_moe_nextn.py` is its **NextN/MTP draft head** (DeepSeek-V3-style `eh_proj`/`enorm`/`hnorm`, `:86`). So the LA layers participate in MTP target-verify — exactly the seg_la `intermediate_ssm` caching path KVBuffer replaces. This is a complete in-tree LA+MTP speculative-decoding path; the KVBuffer verify/commit can be exercised end-to-end. (`minimax_m2` also has `num_mtp_modules` but does not route through seg_la as cleanly — secondary candidate.)

**Must-verify before/early in implementation (not design blockers):**
1. **Smoke-confirm the nextn+linear combo actually runs spec-decode** in this SGLang build (trace nextn worker → bailing linear layer → `lightning_backend` TARGET_VERIFY) — the pieces are all present; confirm end-to-end before relying on it.
2. **`tp_slope` ↔ cuLA `decay_scales` sign/magnitude** — numeric equality check.
3. **Bailing `temporal` dtype is fp32** (cuLA state is fp32).
4. **Bailing linear-layer head config is MHA** (`HV==H`) — cuLA prefill is MHA-only; confirmed `total_kv_heads == num_attention_heads` is expected for this model.

## 6. Test strategy (layered; all need an SM90+ GPU)

Speculative decoding is **output-preserving**: the oracle is "cula backend output == seg_la backend output == non-spec-decode output" for the same model/prompt/sampling.

- **Layer 0 — cuLA unit tests** (already exist): `pytest tests/test_la_verify_kvbuffer.py` — verify/state-update vs PyTorch reference.
- **Layer 1 — operator equivalence in SGLang fork** (highest value, no model needed): same `(q,k,v,h0,accept_len)` → SGLang `seg_la_fwd(caches=…)` + scatter-commit vs cuLA `verify_kvbuffer` + `state_update_kvbuffer`; assert outputs and committed state match. This is also where the §4.5 convention adapters (layout/scale/decay/index) get validated.
- **Layer 2 — single-model end-to-end**: `bailing_moe_linear` (+ `bailing_moe_nextn` draft) with `linear_backend="cula"` vs `"seg_la"`, same prompt, greedy, assert identical tokens. Validates prefill+decode+verify+commit wiring + `mamba_cache_indices` + CUDA-graph warmup.
- **Layer 3 — partial-accept correctness**: force `L<T`; assert tokens still equal non-spec-decode output (exercises `state_update`'s `L<T` path).
- **Layer 4 — capacity/throughput** (the actual payoff): drop `intermediate_ssm`; at fixed HBM, measure max concurrent requests / tokens-per-sec, cula vs seg_la (`bench_serving` under memory pressure). The `intermediate_ssm` GB log at `memory_pool.py:423` quantifies the saving.

## 7. Scope

**In scope (paper §3.3 only):** the `linear_backend="cula"` LA backend (prefill+decode+verify+commit routing as a wholesale replacement of the uniform seg_la path), the MambaPool V-major branch, the **slim fixed-T KV buffer** replacing `intermediate_ssm`, the cuLA-commit path, adapters, CUDA-graph warmup, and the test layers above. Target win: ~5× concurrent requests / ~1.46× throughput.

**Out of scope:**
- **The general paged KV buffer (§3.1) and §3.2 chunkwise decode / §3.4 KV-only.** These need dynamic/paged buffering and new chunkwise-decode machinery, and deliver the *separate* ~45% decode-latency win — a follow-up, not this work. The slim buffer here is a clean special case that can later be generalized to paged when §3.2 has a real consumer.
- GDN/delta-rule (the paper's Qwen3-Next variant).
- Tree-structured (`topk>1`) speculative verify — cuLA assumes chain/MTP, consistent with seg_la's MTP path which is also chain-only.
- cuLA prefill GQA support (target Bailing layers are MHA).

## 7a. Three friction points (analyzed; all resolved/managed)
1. **State layout / which kernels (resolved):** Bailing's whole LA path is uniform seg_la (K-major). cuLA replaces it wholesale with its V-major ops + a V-major pool → cuLA kernels untouched. No partial-splice K-major rework, no separate Bailing-prefill to patch.
2. **`k,v` lifetime to commit (resolved = the KV buffer, §4.4):** draft `k,v` are freed after the verify forward; the slim KV buffer persists them to commit. This *is* the paper's KVBuffer mechanism, not a hazard.
3. **CUDA-graph vs lazy compile (managed, §4.6):** warm up every captured `(B,T,pool)` shape before capture; add a capture→replay numerical-equivalence test to confirm cuLA's steady-state launch is graph-safe.

## 7b. Paper-fidelity of cuLA's interface (no change needed)
cuLA's `linear_attention_verify_kvbuffer` ↔ paper Eq. 7 (`O = Q·S_{t-j} + ((QK^T)⊙M)·V`) and `linear_attention_state_update_kvbuffer` ↔ paper Eq. 8 (`S_t = S_{t-j} + K_acc^T V_acc`): every equation variable maps to a parameter (`s`+`h0_indices`=`S_{t-j}`, `q,k,v`=draft KVs, `accepted_len`=`|accepted|`, `T`=`m`). Extra params (`softmax_scale`, `T`, `h0_indices`) are serving necessities the abstract matrix notation omits. `accepted_len` (vs paper's pre-sliced `K_acc`) is the correct *batched* form for per-request variable accept length. **No signature change for paper fidelity.** Two intentional impl divergences are §11-style perf follow-ups, not interface issues: cuLA's commit uses a recurrent loop vs Eq. 8's batched outer product; cuLA's verify uses per-pair `q·k` vs the chunkwise `QK^T` matmul.

## 8. Open risks / follow-ups (non-blocking)
- cuLA prefill emits final-state only (no per-chunk intermediate `h`), so sub-sequence chunk-boundary prefix-cache tracking has no cuLA equivalent — fine for final-state-only models, revisit if a target needs unaligned-chunk tracking.
- V-major pool may shift cuLA's own access patterns vs its standalone benchmark; re-tune if measured.
- Fusing verify + L=T commit into one launch (cuLA spec §11) would further cut the per-cycle cost; defer until end-to-end works.
