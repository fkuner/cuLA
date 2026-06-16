# LA KVBuffer: Fused KV Write + Fair Benchmark Redesign

**Status:** Design (awaiting plan & implementation)
**Branch context:** `la-decode-mtp`
**Reference paper:** KVBuffer: IO-aware Serving for Linear Attention (Zou & Zhong, 2026), §3.3
**Predecessor specs:** `2026-06-15-la-mtp-kvbuffer-design.md`, `2026-06-15-la-kvbuffer-sglang-integration-design.md`

---

## 1. Motivation

The current `linear_attention_verify_kvbuffer` kernel computes correct output but has two gaps vs. the paper's KVBuffer design (§3.3, Fig. 2):

### Gap 1: Verify does not write KV to buffer

The paper's design buffers draft KVs during verification so that:
- The state-update (commit) reads from the buffer, not from transient activation tensors
- The caller can safely free the original `k, v` tensors after verify returns

Current implementation requires the caller to keep `k, v` alive between `verify` and `state_update` calls — a fragile implicit contract that breaks in serving systems where intermediate activations are freed after the forward pass.

### Gap 2: Benchmark does not compare full verify+commit pipelines

The current `bench_la_decode_mtp_vs_sglang.py` times:
- SGLang: `seg_la_mtp_kernel` only (verify, writes intermediate state caches)
- cuLA: `verify_kvbuffer` + `state_update_kvbuffer` (verify + commit)

This is unfair — SGLang's commit cost (`fused_mamba_state_scatter_with_mask`) is excluded. The paper's claimed speedup (Fig. 4) includes both verify and commit.

## 2. Scope

**In scope:**
- Fuse KV buffer writes into `linear_attention_verify_kvbuffer` kernel
- Modify `linear_attention_state_update_kvbuffer` to optionally read from KV buffer instead of raw input tensors
- Add pool-indexed KV buffer tensors (`k_buf`, `v_buf`) as new parameters
- Redesign benchmark to compare full verify+commit pipelines on both sides
- Backward compatibility: `k_buf=None, v_buf=None` preserves current behavior

**Out of scope:**
- Restructuring verify computation from serial warp-reduce to parallel QK matrix form (kernel is memory-bound; no performance gain — deferred)
- KV buffer paged memory management (serving-system concern, not kernel-level)
- Changes to `linear_attention_decode_mtp` baseline kernel

## 3. KV Buffer Design

### 3.1 Tensor shapes and layout

```
k_buf: [pool_size, T, H,  K]  bf16   — keyed by pool slot, not batch index
v_buf: [pool_size, T, HV, V]  bf16   — same indexing
```

Pool-indexed layout matches the state pool `s: [pool_size, HV, V, K]` and uses `h0_indices[b]` for lookup — same index used by both verify and state-update kernels.

### 3.2 Memory cost

Per-pool-slot: `T × (H×K + HV×V) × 2B`

At B=64, T=4, H=8, HV=64, K=V=128:
```
k_buf: 64 × 4 × 8  × 128 × 2 =   0.5 MB
v_buf: 64 × 4 × 64 × 128 × 2 =   4.0 MB
Total:                             4.5 MB
```

vs. SGLang intermediate caches: `B × T × H × K × V × 4B = 537 MB`

**118× smaller.** This is the core memory advantage from the paper.

### 3.3 Indexing in kernel

Verify kernel writes:
```
k_buf[h0_indices[b], t, i_h, K_lane*vec_size : K_lane*vec_size+vec_size] = k[b, t, i_h, ...]
v_buf[h0_indices[b], t, i_hv, v_row] = v[b, t, i_hv, v_row]
```

State-update kernel reads:
```
k_i = k_buf[h0_indices[b], i, i_h, K_lane*vec_size : ...]   (instead of k[b, i, ...])
v_i = v_buf[h0_indices[b], i, i_hv, v_row]                   (instead of v[b, i, ...])
```

## 4. Verify Kernel Changes

### 4.1 New kernel parameters

```python
# Added to la_verify_kvbuffer_kernel signature:
k_buf: cute.Tensor,       # [pool_size, T, H,  K] bf16 (WRITTEN) — or dummy if disabled
v_buf: cute.Tensor,       # [pool_size, T, HV, V] bf16 (WRITTEN) — or dummy if disabled
write_kv: cutlass.Constexpr[bool],  # compile-time flag
```

### 4.2 Write location

KV writes are fused into the existing q/k/v load loop (`la_verify_kvbuffer.py:113-120`). After loading k_t and v_t into registers:

```python
# Existing: load k, v into registers
for t in cutlass.range_constexpr(T):
    load q[b, t, h, lane*] → r_q_seq[t]  (bf16→fp32, ×scale)
    load k[b, t, h, lane*] → r_k_seq[t]  (bf16→fp32)

    # NEW: write k to buffer (all lanes write their vec_size slice)
    if cutlass.const_expr(write_kv):
        k_buf_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                      (cache_idx, t, i_h, lane_in_group))
        cute.autovec_copy(r_k_bf16, k_buf_tile)

# Existing: load v values per row
for t in cutlass.range_constexpr(T):
    for slot in cutlass.range_constexpr(ilp_rows):
        r_v_seq[t, slot] = Float32(v[b, t, hv, v_row])

        # NEW: write v to buffer (only one lane per v_row writes the scalar)
        if cutlass.const_expr(write_kv):
            if lane_in_group == 0:
                v_buf[(cache_idx, t, i_hv, v_row)] = BFloat16(r_v_seq[t, slot])
```

### 4.3 Write redundancy analysis

Each `k[b, t, h, :]` is written by every V-tile block that shares the same `(b, h)`. With num_v_tiles = V/tile_v = 4 blocks per (b, hv), and HV/H = 8 hv per h:

- k_buf write: redundant 4×8 = 32× per element (all V-tile blocks and all hv sharing the same h write the same k)
- v_buf write: redundant only across row_blocks within a tile (1-2×)

The k redundancy is harmless: writes are to the same address with the same value (no race), and at ~0.5 MB total the bandwidth is negligible vs. the 18 KB/block state reads.

To avoid k redundancy entirely, an alternative is to restrict the k write to `i_v == 0 and i_hv % (HV//H) == 0`. This adds two branch checks but eliminates redundant writes. We implement the gated version.

### 4.4 Performance impact

Additional DRAM writes per block with gated k writes:
```
k_buf: 0 (gated off for most blocks, only i_v==0 && first-hv-of-head writes)
v_buf: T × ilp_rows × 2B = 4 × 4 × 2 = 32 B per row_block iteration
Total additional per block: ~256 B on V-write blocks only
```

On a base of ~18.5 KB/block, this is < 1.5%. No measurable latency impact.

### 4.5 Backward compatibility

When `k_buf` and `v_buf` are None at the Python level:
- `write_kv = False` is passed as a constexpr
- Kernel compiles without write instructions (dead code eliminated)
- Behavior is identical to current implementation
- Separate compile cache entry (keyed by `write_kv`)

## 5. State-Update Kernel Changes

### 5.1 New kernel parameters

```python
# Added to la_state_update_kernel signature:
k_buf: cute.Tensor,       # [pool_size, T, H,  K] bf16 (READ) — or dummy
v_buf: cute.Tensor,       # [pool_size, T, HV, V] bf16 (READ) — or dummy
read_from_buf: cutlass.Constexpr[bool],  # compile-time flag
```

### 5.2 Read path change

When `read_from_buf` is True, the recurrence loop reads from buffer instead of input tensors:

```python
for i in cutlass.range(0, L, unroll=0):
    if cutlass.const_expr(read_from_buf):
        # Read from pool-indexed buffer
        k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                  (cache_idx, i, i_h, lane_in_group))
        r_v = Float32(v_buf[cache_idx, i, i_hv, v_row])
    else:
        # Read from batch-indexed input (current behavior)
        k_tile = cute.local_tile(k, (1, 1, 1, vec_size),
                                  (i_n, i, i_h, lane_in_group))
        r_v = Float32(v[i_n, i, i_hv, v_row])
```

### 5.3 Python API

```python
def linear_attention_state_update_kvbuffer(
    k: torch.Tensor,            # [B, T, H,  K] bf16 — read when k_buf is None
    v: torch.Tensor,            # [B, T, HV, V] bf16 — read when v_buf is None
    s: torch.Tensor,            # [pool_size, HV, V, K] fp32, WRITTEN IN PLACE
    decay_scales: torch.Tensor, # [H] fp32
    h0_indices: torch.Tensor,   # [B] int32, -1 to skip
    accepted_len: torch.Tensor, # [B] int32, in [0, T]
    T: int,
    k_buf: torch.Tensor | None = None,  # [pool_size, T, H, K] bf16
    v_buf: torch.Tensor | None = None,  # [pool_size, T, HV, V] bf16
) -> None:
```

When `k_buf` and `v_buf` are provided, `k` and `v` are still passed for shape inference (B, H, HV, K, V) but their data is not read by the kernel. Shapes are cross-checked: `k_buf.shape[1] == T`, `k_buf.shape[2] == H`, etc.

## 6. Python API (updated)

### 6.1 Verify entry point

```python
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
    k_buf: torch.Tensor | None = None,  # [pool_size, T, H, K] bf16, WRITTEN
    v_buf: torch.Tensor | None = None,  # [pool_size, T, HV, V] bf16, WRITTEN
) -> None:
```

### 6.2 Caller usage pattern

```python
# Allocate persistent buffers (once per pool)
k_buf = torch.zeros(pool_size, T, H, K, device="cuda", dtype=torch.bfloat16)
v_buf = torch.zeros(pool_size, T, HV, V, device="cuda", dtype=torch.bfloat16)

# Per decode iteration:
linear_attention_verify_kvbuffer(
    q, k, v, s, out, decay_scales, h0_indices, scale, T,
    k_buf=k_buf, v_buf=v_buf,
)
# k, v can now be freed — data persists in k_buf, v_buf

# Host samples on `out`, determines accepted_len per batch
linear_attention_state_update_kvbuffer(
    k, v, s, decay_scales, h0_indices, accepted_len, T,  # k,v ignored
    k_buf=k_buf, v_buf=v_buf,
)
```

### 6.3 Invariants

- `k_buf` and `v_buf` must both be None or both be provided.
- Both buffers are pool-indexed: `k_buf[h0_indices[b]]` and `v_buf[h0_indices[b]]` are the slots for batch element b.
- `k_buf` layout is `[pool_size, T, H, K]` with K contiguous — matches q/k input convention.
- `v_buf` layout is `[pool_size, T, HV, V]` with V contiguous — matches v input convention.
- When `h0_indices[b] < 0`, no write occurs to buffer (same skip logic as output and state).

## 7. Benchmark Redesign

### 7.1 Fair comparison structure

```
SGLang full pipeline:
  verify:  seg_la_mtp_kernel          → writes output + intermediate caches
  commit:  fused_mamba_state_scatter   → copies accepted state from caches to pool

cuLA KVBuffer full pipeline:
  verify:  verify_kvbuffer (+kv write) → writes output + k,v to buffer
  commit:  state_update_kvbuffer       → reads buffer, updates state pool
```

### 7.2 SGLang commit setup

The `fused_mamba_state_scatter_with_mask` kernel expects:
```python
# dst: [num_layers, cache_size, *state_shape] — state pool (written)
# src: [num_layers, spec_size, draft_tokens, *state_shape] — intermediate caches (read)
# dst_indices_raw: [total_requests] int32 — pool slot per batch
# step_indices_raw: [total_requests] int32 — accepted step index (0-based), -1 = rejected

# For benchmark (num_layers=1, all accepted, worst case):
dst = s_sglang.unsqueeze(0)             # [1, pool_size, H*K*V]  (flattened state)
src = caches_sglang.unsqueeze(0)        # [1, B, T, H*K*V]      (intermediate caches)
dst_indices_raw = s_offsets.int()        # [B] — pool slot per batch
step_indices_raw = (T - 1) * ones       # [B] — last step accepted (0-based)
```

For the benchmark we use `num_layers=1` (single-layer timing) and `accepted_len = T` (all accepted, worst-case commit). The tensor reshaping matches how `hybrid_linear_attn_backend.py` calls this kernel in production.

### 7.3 Output columns

```
B | T | sg_vfy(ms) | sg_cmt(ms) | sg_total(ms) | cu_vfy(ms) | cu_cmt(ms) | cu_total(ms) | speedup
```

Where:
- `sg_vfy`: `seg_la_mtp_kernel` (+ optional `seg_la_sum_kernel` for k_dim_block > 1)
- `sg_cmt`: `fused_mamba_state_scatter_with_mask`
- `sg_total`: `sg_vfy + sg_cmt`
- `cu_vfy`: `linear_attention_verify_kvbuffer` with `write_kv=True`
- `cu_cmt`: `linear_attention_state_update_kvbuffer` with `read_from_buf=True`
- `cu_total`: `cu_vfy + cu_cmt`
- `speedup`: `sg_total / cu_total`

### 7.4 Memory comparison row

Print a summary line showing per-request memory:
```
SGLang intermediate caches: B × T × H × K² × 4B
cuLA KV buffer:             B × T × (H×K + HV×V) × 2B
Ratio:                      ~118× smaller
```

## 8. Tests

### 8.1 New tests (added to `tests/test_la_verify_kvbuffer.py`)

| Test | Purpose |
|---|---|
| `test_verify_writes_kv_buffer` | After verify with `k_buf, v_buf`, check `k_buf[pool_idx, t, h, :] == k[b, t, h, :]` and `v_buf[pool_idx, t, hv, :] == v[b, t, hv, :]` for all t. |
| `test_verify_output_unchanged_with_kv_write` | Output `o` is identical whether `k_buf/v_buf` are provided or not. |
| `test_state_update_from_buffer` | State update with `k_buf, v_buf` produces same result as with raw `k, v`. |
| `test_verify_skip_negative_indices_no_buffer_write` | When `h0_indices[b] < 0`, `k_buf` and `v_buf` slots for that pool index are untouched. |
| `test_end_to_end_with_buffer` | Full pipeline: verify(+kv write) → state_update(from buffer) matches baseline with L=T. |

### 8.2 Existing tests

All existing tests pass unchanged (backward compatible via `k_buf=None, v_buf=None`).

## 9. Module layout (unchanged)

```
cula/lightning/
├── __init__.py                       # updated exports
├── la_decode_mtp.py                  # baseline, untouched
├── la_verify_kvbuffer.py             # modified: +kv buffer write
└── la_state_update_kvbuffer.py       # modified: +read from buffer option
```

## 10. Open items (non-blocking)

- Verify kernel computation structure (serial warp-reduce vs parallel QK matrix) deferred — kernel is memory-bound, no perf gain.
- KV buffer paged memory management for serving (SGLang-side concern, per integration spec).
- `k_buf` write gating (`i_v==0 && first-hv-of-head`) may be overly conservative; profiling will determine if ungated writes are cheaper (avoid branch divergence).
