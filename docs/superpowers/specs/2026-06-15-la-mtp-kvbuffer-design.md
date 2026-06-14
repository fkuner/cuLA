# LA Decode MTP — KVBuffer Parallel-Verify Redesign

**Status:** Design (awaiting plan & implementation)
**Branch context:** `la-decode-mtp` (baseline `linear_attention_decode_mtp` already lands)
**Reference paper:** KVBuffer: IO-aware Serving for Linear Attention (Zou & Zhong, 2026), §3.3

---

## 1. Motivation

The current `cula.lightning.la_decode_mtp.linear_attention_decode_mtp` kernel processes T draft tokens in a single launch using the **recurrent form**:

```
h_t = α · h_{t-1} + k_t ⊗ v_t
o_t = h_t @ q_t · scale
```

Under the bench-default config (`cache_intermediate_states=True, disable_state_update=True`, the canonical spec-verify workload), the kernel writes T full fp32 state slices per (b, hv) to a `[pool_size·T·HV, V, K]` buffer.

For B=64, T=4, HV=64, K=V=128, that intermediate write alone is

```
intermediate = B · T · HV · V · K · 4B = 64·4·64·128·128·4 ≈ 537 MB
```

which is roughly **3.5×** the rest of the kernel's DRAM traffic combined and is the dominant bandwidth cost.

KVBuffer (§3.3) observes that for speculative-decoding verify, all T draft KVs are already present in caller-owned tensors. The intermediate states `h_1, …, h_T` are derivable from `(h_0, k, v)` in closed form, so storing them is pure waste. The paper's parallel-verify reformulation:

```
o_t      = γ_t · (h_0 @ q_t · scale) + Σ_{i<t} α^{t-1-i} · (q_t · k_i · scale) · v_i
h_L      = α^L · h_0 + Σ_{i<L} α^{L-1-i} · k_i ⊗ v_i        # only computed once, post-acceptance
```

eliminates the intermediate writes entirely. Per-request memory savings translate to ~5× concurrent capacity (paper Fig. 5) and 2.78× verify-kernel speedup at T=8 (paper Fig. 4).

This document specifies the redesign for the Lightning Attention (LA) variant — no delta rule, scalar per-head decay `α = exp(-decay_scale[h])`.

## 2. Scope

**In scope:**
- New `linear_attention_verify_kvbuffer` kernel (paper Eq. 7 for LA)
- New `linear_attention_state_update_kvbuffer` kernel (paper Eq. 8 for LA, per-batch `accepted_len`)
- Tests and benchmarks covering both new kernels and side-by-side comparison with baseline
- Reuse of existing helpers from `la_decode_mtp.py` (`la_update_pair`, `hq_dot_pair`, `get_mtp_config`, FMA pair helpers)

**Out of scope:**
- Paper §3.2 chunkwise single-token decoding (requires multi-step KV-buffer page management at the serving-system layer)
- Paper §3.4 KV-only short-context decoding (same reason)
- Tree-structured speculative verification (current MTP path is purely sequential; matches paper's assumption)
- GDN/delta-rule variants (out of LA's recurrence shape)
- Removing or modifying `linear_attention_decode_mtp` — the baseline kernel is retained verbatim for head-to-head comparison

## 3. Math

### 3.1 Verify kernel (Eq. 7 for LA)

The baseline kernel updates `h` **before** computing `o_t` (see `la_decode_mtp.py:247-327` per-timestep block: update → output). Naming the loaded state from the pool `h_init` and the state after t updates `h_state_t`:

```
h_state_t = α · h_state_{t-1} + k_{t-1} ⊗ v_{t-1}       # t ≥ 1
h_state_0 = h_init                                       # loaded from pool
o_t       = h_state_{t+1} @ q_t · scale                  # output at token index t ∈ [0, T)
```

Unrolling the recurrence:

```
h_state_{t+1} = α^{t+1} · h_init + Σ_{i=0..t} α^{t-i} · k_i ⊗ v_i
```

Substituting into the output gives the parallel-verify form:

```
o_t = α^{t+1} · (h_init @ q_t · scale)
      + Σ_{i=0..t} α^{t-i} · (q_t · k_i · scale) · v_i
```

The `i ≤ t` causal mask is enforced by loop bounds (the sum includes the self-pair `i=t`, which corresponds to the just-added `k_t⊗v_t` term in `h_state_{t+1}`). There is no normalization / softmax in LA — the form is exactly bilinear.

This matches the `torch_la_mtp_ref` semantics in `tests/test_la_decode_mtp.py:93-101` (which is the contract the baseline kernel obeys).

### 3.2 State update (Eq. 8 for LA)

For accepted prefix length `L ∈ [0, T]`, the target state in the pool slot after this verify cycle is `h_state_L`:

```
h_state_L = α^L · h_init + Σ_{i=0..L-1} α^{L-1-i} · k_i ⊗ v_i
```

Equivalently (numerically identical, simpler loop body):

```
h_running = h_init
for i in 0..L-1:
    h_running = α · h_running + k_i ⊗ v_i
return h_running
```

We implement the latter form because it reuses `la_update_pair` from the baseline and exhibits the same numerical behavior as the original recurrence — making bit-equivalence to the baseline (when `L == T`) easy to assert.

`L == 0`: skip; `s[…]` is left unchanged (state stays at `h_init`).

## 4. Kernel design — verify

### 4.1 Grid and thread mapping

Identical to baseline `la_verify_kernel_mtp`:

```
grid   = (B · HV · num_v_tiles, 1, 1)
block  = (128, 1, 1)            # 4 warps
mapping:
  block_idx → (i_v, i_hv, i_n)
  i_h = i_hv // (HV // H)
  cache_idx = h0_indices[i_n]   # ← we use h0_indices, not s_offsets, matching baseline rename in commit 90e7dfa
```

Lane partitioning (sticking with `vec_size=4`, K=128):

```
threads_per_group = K // vec_size = 32
groups_per_warp   = 32 // threads_per_group = 1
num_groups        = 4 * groups_per_warp = 4
lane_in_group     = lane % 32
group_idx         = warp_idx · 1 + group_in_warp
rows_per_group    = tile_v // num_groups
```

### 4.2 Register tensors (per lane)

| Tensor | Shape | Bytes (T=4, ilp=4, vec=4) |
|---|---|---|
| `r_h[ilp_rows][vec_size]` | h_init tile, persistent across T loop | 64 |
| `r_q_seq[T][vec_size]`    | All T q vectors, fp32 (scaled)        | 64 |
| `r_k_seq[T][vec_size]`    | All T k vectors, fp32                  | 64 |
| `r_v_seq[T][ilp_rows]`    | All T v values for this v-tile         | 64 |
| `r_decay_pow[T+1]`        | α⁰..α^T (extra slot for term1)         | 20 |

Worst case (T=8, ilp_rows=8, vec=4): ~256 B/lane. Well within budget.

### 4.3 Control flow

```python
# (1) Skip path
cache_idx = h0_indices[i_n]
if cache_idx < 0: return

α = exp(-decay_scales[i_h])

# (2) Precompute α^0..α^T  (T+1 powers)
r_decay_pow[0] = 1.0
for t in 1..T: r_decay_pow[t] = r_decay_pow[t-1] * α

# (3) Load h_init tile (reuses baseline's per-ilp-rows load pattern)
load h_init[cache_idx, v_idx_*, K_lane*] → r_h

# (4) Load all T (q, k) pairs and v_tile slices
for t in 0..T-1:
    load q[i_n, t, i_h, K_lane*] → r_q_seq[t]   (bf16 → fp32, ×scale)
    load k[i_n, t, i_h, K_lane*] → r_k_seq[t]   (bf16 → fp32)
    for row in 0..ilp_rows-1:
        load v[i_n, t, i_hv, v_idx_row] → r_v_seq[t][row]
        (use_smem_v path: pre-staged in sVdata, same as baseline)

# (5) Per-t output computation
for t in 0..T-1:
    # term1: α^{t+1} · (h_init @ q_t)
    for row in 0..ilp_rows-1:
        hq = Σ_i r_h[row][i] · r_q_seq[t][i]
        hq = warp_reduce(hq)                      # 5-stage shuffle, same as baseline
        o_partial[row] = r_decay_pow[t+1] · hq

    # term2: Σ_{i=0..t} α^{t-i} · (q_t · k_i) · v_i   (includes self-pair i=t)
    for i in 0..t:
        qk = Σ_j r_q_seq[t][j] · r_k_seq[i][j]
        qk = warp_reduce(qk)                       # 5-stage shuffle
        coeff = r_decay_pow[t-i] · qk
        for row in 0..ilp_rows-1:
            o_partial[row] += coeff · r_v_seq[i][row]

    # writeback (lane_in_group == 0 holds the reduced value, same as baseline)
    if lane_in_group == 0:
        for row in 0..ilp_rows-1:
            o[i_n, t, i_hv, v_idx_row] = bf16(o_partial[row])
        (use_smem_v path: stage into sOutput, cooperative writeback at end)
```

### 4.4 Differences from baseline

- **Removed:** `la_update_pair` call inside T-loop; no h mutation.
- **Removed:** `cache_intermediate_states` branch; no `intermediate_states` parameter.
- **Removed:** `disable_state_update` branch and final h writeback; verify kernel never touches `s`.
- **Added:** `r_q_seq[T]`, `r_k_seq[T]` register storage so all T q/k are simultaneously available for cross-terms.
- **Added:** `r_v_seq[T][ilp_rows]` to hold the T v-values for this tile.
- **Added:** T·(T+1)/2 extra q·k dot products (causal incl. self-pair). For T=4: 10 extras; for T=8: 36 extras. Each is one 5-stage shuffle reduce — comparable to the per-t hq reductions already present (T·ilp_rows of those), still a small fraction of the per-block work.

### 4.5 Tuning surface (sticking with baseline heuristics)

- `tile_v, vec_size, ilp_rows, use_smem_v`: identical to `get_mtp_config()` — imported and reused.
- `use_packed_fma`: `major >= 10` (SM100+), identical predicate.
- `USE_FAST_MATH`: identical to baseline.

Re-tuning is left for a follow-up pass after first benchmark.

## 5. Kernel design — state update

### 5.1 Grid and thread mapping

Identical layout to verify kernel — same `(B · HV · num_v_tiles, 1, 1)` grid, same lane partitioning. This keeps the state writeback aligned with the verify kernel's h₀ read, and means we can reuse `la_update_pair` directly.

### 5.2 Register tensors (per lane)

| Tensor | Shape | Notes |
|---|---|---|
| `r_h[ilp_rows][vec_size]` | accumulator | loaded once from `h_init` |
| `r_k[vec_size]`           | current k_i | reloaded per i |
| `r_v[ilp_rows]`           | current v_i | reloaded per i |

No need to stage all T (k, v) at once — the loop is purely sequential over i with no cross-terms.

### 5.3 Control flow

```python
cache_idx = h0_indices[i_n]
if cache_idx < 0: return
L = accepted_len[i_n]
if L == 0: return

α = exp(-decay_scales[i_h])

# Load h_init → r_h (same load pattern as verify kernel)
load h_init[cache_idx, v_idx_*, K_lane*] → r_h

# Dynamic-bound recurrence (cutlass.range_dynamic over L)
for i in 0..L-1:
    load k[i_n, i, i_h, K_lane*] → r_k    (bf16 → fp32)
    for row in 0..ilp_rows-1:
        load v[i_n, i, i_hv, v_idx_row] → r_v[row]

    # r_h[row][j] = α · r_h[row][j] + r_k[j] · r_v[row]   (la_update_pair)
    for row in 0..ilp_rows-1:
        for j in 0..vec_size-1 step 2:
            r_h[row][j], r_h[row][j+1] = la_update_pair(
                r_h[row][j], r_h[row][j+1],
                r_k[j], r_k[j+1],
                r_v[row], α,
                use_packed_fma)

# Writeback (same lane-collective pattern as baseline's final state write)
store r_h → s[cache_idx, v_idx_*, K_lane*]
```

### 5.4 Why no `q` involvement

State update never reads `q` and never writes `o`. The kernel signature drops both. This is the cleanest split possible — verify reads `s, q, k, v` and writes `o`; state update reads `s, k, v` and writes `s`.

### 5.5 Numerical equivalence to baseline

Because the loop body is bit-identical to the baseline's T-loop body (same `la_update_pair`, same operand order), the result for `L = T` is bit-equivalent to running the baseline kernel with `disable_state_update=False`. This makes the equivalence test trivially strict.

## 6. Python API

### 6.1 New entry points (in `cula/lightning/__init__.py`)

```python
def linear_attention_verify_kvbuffer(
    q: torch.Tensor,           # [B, T, H,  K] bf16
    k: torch.Tensor,           # [B, T, H,  K] bf16
    v: torch.Tensor,           # [B, T, HV, V] bf16
    s: torch.Tensor,           # [pool_size, HV, V, K] fp32, READ-ONLY in this call
    out: torch.Tensor,         # [B, T, HV, V] bf16, WRITTEN
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,  # [B] int32, -1 to skip a batch
    softmax_scale: float,
    T: int,
) -> None: ...

def linear_attention_state_update_kvbuffer(
    k: torch.Tensor,            # [B, T, H,  K] bf16
    v: torch.Tensor,            # [B, T, HV, V] bf16
    s: torch.Tensor,            # [pool_size, HV, V, K] fp32, WRITTEN IN PLACE
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,   # [B] int32, -1 to skip
    accepted_len: torch.Tensor, # [B] int32, ∈ [0, T]
    T: int,                     # = k.shape[1]
) -> None: ...
```

### 6.2 Caller usage pattern (matching paper §3.3)

```python
# Per decode iteration:
linear_attention_verify_kvbuffer(q, k, v, s, out, decay_scales, h0_indices, scale, T)
# host samples on `out`, determines accepted_len per batch
linear_attention_state_update_kvbuffer(k, v, s, decay_scales, h0_indices, accepted_len, T)
```

### 6.3 Invariants

- `s` layout `[pool_size, HV, V, K]` (V-major, K-last) — identical to baseline. Internally viewed as `[pool_size · HV, V, K]` for indexing.
- `q, k, v` must be the **same tensors** passed to both kernels (verify reads, state-update reads). Caller cannot free/mutate between calls.
- `decay_scales` is positive; kernel computes `exp(-decay_scales)`. Matches baseline convention.
- `h0_indices[b] < 0` skips that batch entirely — `out[b]` left untouched by verify; `s[…]` left untouched by state-update. Symmetric to baseline `s_offsets` semantics; renamed for clarity and to match commit 90e7dfa.
- `accepted_len[b] == 0` skips state-update for that batch (state stays at h₀).
- Both kernels use the per-call compile cache pattern from baseline (`@functools.cache` keyed by shape + config flags).
- Both kernels expose the same `assumed_align=16` `from_dlpack` requirement as baseline.

### 6.4 Old kernel retention

`linear_attention_decode_mtp` is **not modified or deprecated**. It remains the bench baseline and the recurrent-form reference. Removing it is out of scope.

## 7. Tests

New file: `tests/test_la_verify_kvbuffer.py`.

| Test | Purpose |
|---|---|
| `test_verify_outputs_match_ref[(B,T)…]` | Verify kernel `o` matches `torch_la_mtp_ref` on all 6 baseline parameterized configs (`B,T ∈ {(1,4),(2,2),(2,4),(8,4),(32,2),(32,4)}`); rel-RMSE < 1e-2. |
| `test_verify_different_heads[(H,HV)…]` | GQA cases `(16,16),(8,32),(16,64)`; same tolerance. |
| `test_verify_skip_negative_h0_indices` | `h0_indices[b]=-1` leaves `out[b]` at sentinel value. |
| `test_verify_zero_decay` | `decay_scales = 0` (α = 1) edge case. |
| `test_verify_zero_state` | h₀ = 0 edge case (output = causal-attention over draft KVs only). |
| `test_state_update_full_accept` | `accepted_len = [T,T,…]`; new `s` matches baseline `state_running` after T recurrent steps; **bit-exact** (same loop body). |
| `test_state_update_partial[L=0,1,T-1]` | Uniform `accepted_len = L`; matches manual `α^L·h₀ + Σ α^{L-1-i} k_i⊗v_i`; rel-RMSE < 1e-3. |
| `test_state_update_per_batch_L` | `accepted_len = [0, 1, T-1, T]`; each batch independently correct. |
| `test_state_update_skip_negative_h0_indices` | `h0_indices[b]=-1`; that pool slot is untouched. |
| `test_state_update_L0_no_op` | `accepted_len=0`; `s` unchanged for that batch. |
| `test_end_to_end_equivalence_with_baseline` | Run baseline (cache_intermediate=True, disable_state_update=True), then run (verify + state_update with L=T): (a) outputs match in bf16 tolerance; (b) `s_after_kvbuffer == intermediate_states[T-1]` (bit-exact). |

`torch_la_mtp_ref` is reused from `tests/test_la_decode_mtp.py` — refactor into a shared `tests/_la_mtp_ref.py` (or import directly; whichever the executing plan picks).

## 8. Benchmarks

Modifying `benchmarks/bench_la_decode_mtp.py`:

Add three timing columns next to the existing `cute_mtp_ms`:

| Column | Computation |
|---|---|
| `kvbuf_verify_ms` | Kernel-only time for `linear_attention_verify_kvbuffer` (pre-compiled, pre-stream) |
| `kvbuf_update_ms` | Kernel-only time for `linear_attention_state_update_kvbuffer` with `accepted_len = T` (worst case) |
| `kvbuf_total_ms`  | `kvbuf_verify_ms + kvbuf_update_ms` — paper Fig. 4 metric |
| `spd_kvbuf`       | `cute_mtp_ms / kvbuf_total_ms` |

Bandwidth model addition:

```python
def kvbuf_bytes(B, T, H, HV, K, V):
    bf16, fp32 = 2, 4
    qkv   = B*T*H*K*bf16*2 + B*T*HV*V*bf16    # q, k, v reads (verify)
    out_w = B*T*HV*V*bf16                      # o writes (verify)
    h0_r  = B*HV*V*K*fp32                      # h0 reads (verify)
    # state-update: reads h0 again, reads k+v again (worst case L=T), writes h_new
    update = B*HV*V*K*fp32 * 2 + B*T*H*K*bf16 + B*T*HV*V*bf16
    return qkv + out_w + h0_r + update
```

The default config remains `cache_intermediate=True, disable_state_update=True` for the legacy `cute_mtp_*` columns — this is the spec-verify workload the paper targets and where the speedup is measurable.

A separate SOL% column for the KVBuffer total is reported alongside the baseline's SOL%.

## 9. Module layout

```
cula/lightning/
├── __init__.py                       # exports the two new entry points
├── la_decode_mtp.py                  # baseline, untouched
├── la_verify_kvbuffer.py             # new: verify kernel + launcher + Python entry
└── la_state_update_kvbuffer.py       # new: state-update kernel + launcher + Python entry
```

Both new files **import** helpers from `la_decode_mtp.py` (`la_update_pair`, `hq_dot_pair`, `get_mtp_config`, `TILE_K_MTP`, `NUM_THREADS_MTP`). No duplication.

## 10. Expected outcomes

Predicted DRAM-traffic reduction at the bench-default config (B=64, T=4, HV=64, K=V=128):

| Path | Bytes (MB) | Notes |
|---|---|---|
| Baseline `cute_mtp` (cache_inter=T, disable=T) | ~545 | 537 of which is `intermediate_states` write |
| KVBuffer verify | ~151 | qkv + out + h₀ read; no state side-effects |
| KVBuffer state-update (L=T) | ~270 | h₀ read + k+v read + h_new write |
| KVBuffer total | ~421 | non-overlapping launches |

Both kernels are memory-bound. Predicted total-time speedup vs baseline: ≥ 1.3× at this config; larger at higher T (paper measures up to 2.78× at T=8). End-to-end serving throughput improvements (paper Fig. 5, ~1.46×) are out of scope — that depends on serving-system integration.

## 11. Open risks / follow-ups (non-blocking)

- `get_mtp_config` was tuned for the recurrent-form work mix; re-tuning may shift the optimum (e.g., higher ilp_rows now that v-load is hoisted out of the per-t inner loop).
- For T ≥ 8, the q/k register footprint pushes lane register count up; may need to spill to SMEM. Defer until measured.
- State-update kernel's `accepted_len`-dependent dynamic loop can be hot-restricted to a constexpr T path when most batches accept all drafts; defer until measured.
- The post-verify state-update doesn't fuse easily with the next layer's prefill; if this becomes a bottleneck on small T, a fused "verify + L=T update" launch could be added as a fast-path. Defer until measured.
