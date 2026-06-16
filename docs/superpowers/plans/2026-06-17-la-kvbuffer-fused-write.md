# LA KVBuffer: Fused KV Write + Fair Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse KV buffer writes into the verify kernel, add buffer-read mode to state-update kernel, and redesign the benchmark for fair SGLang vs cuLA full-pipeline comparison.

**Architecture:** The verify kernel gains an optional `write_kv` constexpr flag and two new tensor parameters (`k_buf`, `v_buf`). When enabled, it writes k,v to pool-indexed buffers alongside computing output. The state-update kernel gains a symmetric `read_from_buf` flag to read k,v from the buffer instead of batch-indexed input tensors. The benchmark adds SGLang's commit operator (`fused_mamba_state_scatter_with_mask`) to compare full verify+commit pipelines.

**Tech Stack:** CuTe DSL (CUTLASS Python), PyTorch, Triton (SGLang kernels), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `cula/lightning/la_verify_kvbuffer.py` | Modify | Add `k_buf`, `v_buf`, `write_kv` params; fuse KV writes into load loop |
| `cula/lightning/la_state_update_kvbuffer.py` | Modify | Add `k_buf`, `v_buf`, `read_from_buf` params; conditional read source |
| `tests/test_la_verify_kvbuffer.py` | Modify | Add 5 new test functions for buffer read/write paths |
| `benchmarks/bench_la_decode_mtp_vs_sglang.py` | Modify | Add SGLang commit timing, cuLA buffer-mode timing, new output columns |

---

### Task 1: Verify kernel — add `k_buf/v_buf` write path

**Files:**
- Modify: `cula/lightning/la_verify_kvbuffer.py`

The kernel function, JIT launcher, compile-cache key, and Python entry point all need the new parameters. The kernel writes k,v to pool-indexed buffers when `write_kv=True`, gated to avoid redundant writes.

- [ ] **Step 1: Add parameters to `la_verify_kvbuffer_kernel`**

Add three new parameters after `h0_indices`:

```python
@cute.kernel
def la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,     # [pool_size * HV, V, K] fp32 (READ ONLY)
    decay_scales: cute.Tensor,  # [H] fp32
    q: cute.Tensor,             # [B, T, H,  K] bf16
    k: cute.Tensor,             # [B, T, H,  K] bf16
    v: cute.Tensor,             # [B, T, HV, V] bf16
    o: cute.Tensor,             # [B, T, HV, V] bf16 (WRITTEN)
    h0_indices: cute.Tensor,    # [B] int32
    k_buf: cute.Tensor,         # [pool_size, T, H, K] bf16 (WRITTEN when write_kv)
    v_buf: cute.Tensor,         # [pool_size, T, HV, V] bf16 (WRITTEN when write_kv)
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
    write_kv: cutlass.Constexpr[bool],
):
```

- [ ] **Step 2: Add KV write logic after the k/v load loop**

Inside the kernel body, after the existing k load at line 116-120 (`cute.autovec_copy(k_tile, r_k_bf16)` and bf16→fp32 conversion), add the k_buf write. Gate with `i_v == 0` and first-hv-of-head to avoid redundancy:

```python
        # Stage all T q (scaled) and k (fp32) for this lane's K-slice.
        for t in cutlass.range_constexpr(T):
            q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t, i_h, lane_in_group))
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)
            for j in cutlass.range_constexpr(vec_size):
                r_q_seq[t, j] = cutlass.Float32(r_q_bf16[j]) * scale
                r_k_seq[t, j] = cutlass.Float32(r_k_bf16[j])

            # Write k to buffer — gated: only one block per (b, h, t) writes
            if cutlass.const_expr(write_kv):
                if i_v == 0 and i_hv % (HV // H) == 0:
                    kb_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                              (cache_idx, t, i_h, lane_in_group))
                    cute.autovec_copy(r_k_bf16, kb_tile)
```

Then, inside the per-row-block loop, after loading v values at line 133-135 (`r_v_seq[t, slot] = ...`), add the v_buf write:

```python
                # Load all T v-values for these rows.
                for t in cutlass.range_constexpr(T):
                    for slot in cutlass.range_constexpr(ilp_rows):
                        r_v_seq[t, slot] = cutlass.Float32(v[i_n, t, i_hv, v_base + slot])

                        # Write v to buffer — each (cache_idx, t, hv, v_row) written once
                        if cutlass.const_expr(write_kv):
                            if lane_in_group == 0:
                                v_buf[(cache_idx, t, i_hv, v_base + slot)] = v[i_n, t, i_hv, v_base + slot]
```

- [ ] **Step 3: Update `run_la_verify_kvbuffer_kernel` JIT launcher**

Add the new parameters to the launcher function signature and pass them to the kernel:

```python
@cute.jit
def run_la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
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
    write_kv: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    smem_bytes = 0
    if cutlass.const_expr(use_smem_v):
        smem_bytes = T * tile_v * 4 + T * tile_v * 2

    la_verify_kvbuffer_kernel(
        h0_source, decay_scales, q, k, v, o, h0_indices,
        k_buf, v_buf,
        vec_size, num_v_tiles, tile_v, scale,
        B, T, H, HV, K, V, ilp_rows,
        use_smem_v, use_packed_fma, write_kv,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )
```

- [ ] **Step 4: Update `_get_compiled_verify_kvbuffer_kernel` cache key**

Add `write_kv` to the cache key:

```python
@functools.cache
def _get_compiled_verify_kvbuffer_kernel(
    B: int, T: int, H: int, HV: int, K: int, V: int,
    pool_size: int, softmax_scale: float,
    tile_v: int, vec_size: int, ilp_rows: int, use_smem_v: bool, use_packed_fma: bool,
    write_kv: bool,
):
    return {}
```

- [ ] **Step 5: Update `linear_attention_verify_kvbuffer` Python entry point**

Add `k_buf` and `v_buf` optional parameters. When provided, set `write_kv=True` and pass actual tensors; when None, pass dummy tensors and `write_kv=False`:

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
    """
    Closed-form parallel verify (KVBuffer Eq. 7). Writes out; does not touch s.

    When k_buf and v_buf are provided, also writes k,v to pool-indexed buffers
    so the caller can free the original k,v tensors after this call returns.
    """
    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    write_kv = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")

    tile_v, vec_size, ilp_rows, use_smem_v = get_mtp_config(B, T, HV, V, True)
    major, _ = get_device_sm_version(q.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, softmax_scale,
        tile_v, vec_size, ilp_rows, use_smem_v, use_packed_fma,
        write_kv,
    )
    cache = _get_compiled_verify_kvbuffer_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

    # Dummy tensors when write_kv=False (never accessed by kernel)
    if not write_kv:
        k_buf_t = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.bfloat16)
        v_buf_t = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.bfloat16)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

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
            from_dlpack(k_buf_t, assumed_align=16),
            from_dlpack(v_buf_t, assumed_align=16),
            scale=softmax_scale,
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_smem_v=use_smem_v,
            use_packed_fma=use_packed_fma,
            write_kv=write_kv,
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
        k_buf_t, v_buf_t,
        stream,
    )
```

- [ ] **Step 6: Run existing tests to verify backward compatibility**

Run: `cd /root/work/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -v`

Expected: All existing tests PASS (no `k_buf`/`v_buf` passed → `write_kv=False` → identical to old behavior).

- [ ] **Step 7: Commit**

```bash
git add cula/lightning/la_verify_kvbuffer.py
git commit -m "feat: fuse KV buffer writes into verify kernel (write_kv flag)"
```

---

### Task 2: State-update kernel — add buffer-read path

**Files:**
- Modify: `cula/lightning/la_state_update_kvbuffer.py`

- [ ] **Step 1: Add parameters to `la_state_update_kernel`**

Add three new parameters after `accepted_len`:

```python
@cute.kernel
def la_state_update_kernel(
    h0_source: cute.Tensor,     # [pool_size * HV, V, K] fp32 (read + written in place)
    decay_scales: cute.Tensor,  # [H] fp32
    k: cute.Tensor,             # [B, T, H,  K] bf16
    v: cute.Tensor,             # [B, T, HV, V] bf16
    h0_indices: cute.Tensor,    # [B] int32
    accepted_len: cute.Tensor,  # [B] int32
    k_buf: cute.Tensor,         # [pool_size, T, H, K] bf16 (READ when read_from_buf)
    v_buf: cute.Tensor,         # [pool_size, T, HV, V] bf16 (READ when read_from_buf)
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
    read_from_buf: cutlass.Constexpr[bool],
):
```

- [ ] **Step 2: Add conditional read source in recurrence loop**

For each of the three ILP paths (`ilp_rows == 2`, `4`, `8`), change the k and v loads to conditionally read from buffer. Example for `ilp_rows == 2` path (apply the same pattern to the `4` and `8` paths):

```python
                    for i in cutlass.range(0, L, unroll=0):
                        if cutlass.const_expr(read_from_buf):
                            k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size),
                                                      (cache_idx, i, i_h, lane_in_group))
                        else:
                            k_tile = cute.local_tile(k, (1, 1, 1, vec_size),
                                                      (i_n, i, i_h, lane_in_group))
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for j in cutlass.range_constexpr(vec_size):
                            r_k[j] = cutlass.Float32(r_k_bf16[j])

                        if cutlass.const_expr(read_from_buf):
                            r_v_a = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_b])
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i, i_hv, v_idx_b])
                        # ... la_update_pair calls unchanged ...
```

Repeat the same `if cutlass.const_expr(read_from_buf):` pattern for the `ilp_rows == 4` and `ilp_rows == 8` paths, substituting `k_buf/v_buf[cache_idx, i, ...]` for `k/v[i_n, i, ...]` in each v load.

- [ ] **Step 3: Update `run_la_state_update_kernel` launcher**

Add `k_buf`, `v_buf`, `read_from_buf` to the launcher signature and pass them to the kernel:

```python
@cute.jit
def run_la_state_update_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
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
    read_from_buf: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    la_state_update_kernel(
        h0_source, decay_scales, k, v, h0_indices, accepted_len,
        k_buf, v_buf,
        vec_size, num_v_tiles, tile_v,
        B, T, H, HV, K, V, ilp_rows, use_packed_fma, read_from_buf,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        stream=stream,
    )
```

- [ ] **Step 4: Update compile cache key**

Add `read_from_buf`:

```python
@functools.cache
def _get_compiled_state_update_kernel(
    B: int, T: int, H: int, HV: int, K: int, V: int,
    pool_size: int, tile_v: int, vec_size: int, ilp_rows: int, use_packed_fma: bool,
    read_from_buf: bool,
):
    return {}
```

- [ ] **Step 5: Update `linear_attention_state_update_kvbuffer` Python entry point**

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
    """
    Advance pooled state from h_init to h_state_L per batch (KVBuffer Eq. 8).

    When k_buf and v_buf are provided, reads k,v from pool-indexed buffers
    instead of batch-indexed input tensors.
    """
    B, T_k, H, K = k.shape
    assert T_k == T, f"k.shape[1]={T_k} doesn't match T={T}"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]

    read_from_buf = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")

    tile_v, vec_size, ilp_rows, _use_smem_v = get_mtp_config(B, T, HV, V, False)
    major, _ = get_device_sm_version(k.device)
    use_packed_fma = major >= 10

    cache_key = (
        B, T, H, HV, K, V, pool_size, tile_v, vec_size, ilp_rows, use_packed_fma,
        read_from_buf,
    )
    cache = _get_compiled_state_update_kernel(*cache_key)

    h0_view = s.view(pool_size * HV, V, K)

    if not read_from_buf:
        k_buf_t = torch.empty(1, 1, 1, 1, device=k.device, dtype=torch.bfloat16)
        v_buf_t = torch.empty(1, 1, 1, 1, device=k.device, dtype=torch.bfloat16)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

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
            from_dlpack(k_buf_t, assumed_align=16),
            from_dlpack(v_buf_t, assumed_align=16),
            B=B, T=T, H=H, HV=HV, K=K, V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            read_from_buf=read_from_buf,
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
        k_buf_t, v_buf_t,
        stream,
    )
```

- [ ] **Step 6: Run existing tests to verify backward compatibility**

Run: `cd /root/work/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -v`

Expected: All existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add cula/lightning/la_state_update_kvbuffer.py
git commit -m "feat: add buffer-read mode to state-update kernel (read_from_buf flag)"
```

---

### Task 3: Tests — KV buffer write and read paths

**Files:**
- Modify: `tests/test_la_verify_kvbuffer.py`

- [ ] **Step 1: Write `test_verify_writes_kv_buffer`**

Append to `tests/test_la_verify_kvbuffer.py`:

```python
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
```

- [ ] **Step 2: Write `test_verify_output_unchanged_with_kv_write`**

```python
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
```

- [ ] **Step 3: Write `test_state_update_from_buffer`**

```python
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
```

- [ ] **Step 4: Write `test_verify_skip_negative_indices_no_buffer_write`**

```python
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
```

- [ ] **Step 5: Write `test_end_to_end_with_buffer`**

```python
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
```

- [ ] **Step 6: Run all tests**

Run: `cd /root/work/cuLA && python -m pytest tests/test_la_verify_kvbuffer.py -v`

Expected: All tests PASS (existing + 5 new).

- [ ] **Step 7: Commit**

```bash
git add tests/test_la_verify_kvbuffer.py
git commit -m "test: add KV buffer write/read path tests for verify + state-update"
```

---

### Task 4: Benchmark — fair SGLang vs cuLA full-pipeline comparison

**Files:**
- Modify: `benchmarks/bench_la_decode_mtp_vs_sglang.py`

- [ ] **Step 1: Add SGLang scatter import**

Add the import at the top of the file, after the existing SGLang imports:

```python
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
```

- [ ] **Step 2: Add SGLang commit wrapper function**

Add after `run_sglang_mtp`:

```python
def run_sglang_commit(s_sglang, caches_sglang, s_offsets, step_indices, B, H, K, V, T):
    """
    Invoke fused_mamba_state_scatter_with_mask the way hybrid_linear_attn_backend does.

    dst: [1, pool_size, H*K*V]  — state pool (1 layer)
    src: [1, B, T, H*K*V]       — intermediate caches (1 layer)
    """
    elem_per_entry = H * K * V
    dst = s_sglang.reshape(1, -1, elem_per_entry)
    src = caches_sglang.reshape(1, B, T, elem_per_entry)
    fused_mamba_state_scatter_with_mask(dst, src, s_offsets, step_indices)
```

- [ ] **Step 3: Update `run_config` — add SGLang commit timing**

After the existing SGLang correctness check block (~line 256), add SGLang commit setup and timing. In the timing section, add:

```python
    # ---- SGLang commit setup ----
    step_indices_sg = torch.full((B,), T - 1, device=device, dtype=torch.int32)

    def kernel_sglang_commit():
        run_sglang_commit(
            s_sg_bench, c_sg_bench, s_offsets_sg.int(),
            step_indices_sg, B, H, K, V, T,
        )
```

- [ ] **Step 4: Update `run_config` — add cuLA buffer-mode timing**

Add KV buffer allocation and buffer-mode kernel timing:

```python
    # ---- cuLA KVBuffer with actual buffer write/read ----
    k_buf_bench = torch.zeros(pool_size, T, H, K, device=device, dtype=dtype)
    v_buf_bench = torch.zeros(pool_size, T, HV, V, device=device, dtype=dtype)

    # Need new compiled kernels with write_kv=True / read_from_buf=True
    verify_buf_cache_key = (
        B, T, H, HV, K, V, pool_size, scale,
        tile_v_kv, vec_size_kv, ilp_rows_kv, use_smem_v_kv, use_packed_fma,
        True,  # write_kv
    )
    verify_buf_cache = _get_compiled_verify_kvbuffer_kernel(*verify_buf_cache_key)

    # Trigger compilation for write_kv=True variant
    s_kvbuf_compile = state_init_kmaj.permute(0, 1, 3, 2).contiguous()
    out_compile = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    linear_attention_verify_kvbuffer(
        q_4d, k_4d, v_4d, s_kvbuf_compile, out_compile,
        decay_scales, h0_indices_kv, scale, T,
        k_buf=k_buf_bench, v_buf=v_buf_bench,
    )
    compiled_verify_buf = verify_buf_cache["compiled"]

    s_kvbuf_kk_vb = state_init_kmaj.permute(0, 1, 3, 2).contiguous().view(pool_size * HV, V, K)

    def kernel_kvbuf_verify_with_write():
        compiled_verify_buf(
            s_kvbuf_kk_vb,
            decay_scales, q_4d, k_4d, v_4d, out_kvbuf_kk,
            h0_indices_kv,
            k_buf_bench, v_buf_bench,
            stream_handle,
        )

    # State-update reading from buffer
    from cula.lightning.la_state_update_kvbuffer import _get_compiled_state_update_kernel
    update_buf_cache_key = (
        B, T, H, HV, K, V, pool_size, tile_v_su, vec_size_su, ilp_rows_su, use_packed_fma,
        True,  # read_from_buf
    )
    update_buf_cache = _get_compiled_state_update_kernel(*update_buf_cache_key)

    # Trigger compilation for read_from_buf=True variant
    s_kvbuf_warmup2 = state_init_kmaj.permute(0, 1, 3, 2).contiguous()
    linear_attention_state_update_kvbuffer(
        k_4d, v_4d, s_kvbuf_warmup2, decay_scales,
        h0_indices_kv, accepted_len_kv, T,
        k_buf=k_buf_bench, v_buf=v_buf_bench,
    )
    compiled_update_buf = update_buf_cache["compiled"]

    s_kvbuf_kk_ub = state_init_kmaj.permute(0, 1, 3, 2).contiguous().view(pool_size * HV, V, K)

    def kernel_kvbuf_update_from_buf():
        compiled_update_buf(
            s_kvbuf_kk_ub,
            decay_scales, k_4d, v_4d,
            h0_indices_kv, accepted_len_kv,
            k_buf_bench, v_buf_bench,
            stream_handle,
        )
```

- [ ] **Step 5: Update timing section and return dict**

Replace the timing and return section:

```python
    with torch.no_grad():
        sg_vfy_ms = benchmark_fn(kernel_sglang)
        sg_cmt_ms = benchmark_fn(kernel_sglang_commit)
        cu_vfy_ms = benchmark_fn(kernel_kvbuf_verify_with_write)
        cu_cmt_ms = benchmark_fn(kernel_kvbuf_update_from_buf)

    sg_total_ms = sg_vfy_ms + sg_cmt_ms
    cu_total_ms = cu_vfy_ms + cu_cmt_ms

    return {
        "B": B,
        "T": T,
        "sg_vfy_ms": sg_vfy_ms,
        "sg_cmt_ms": sg_cmt_ms,
        "sg_total_ms": sg_total_ms,
        "cu_vfy_ms": cu_vfy_ms,
        "cu_cmt_ms": cu_cmt_ms,
        "cu_total_ms": cu_total_ms,
        "speedup": sg_total_ms / cu_total_ms,
        "rmse_sg": rmse_sg,
        "rmse_cu": rmse_cu,
        "rmse_kv": rmse_kv,
    }
```

- [ ] **Step 6: Update `main()` output formatting**

Replace the header and print statements:

```python
    hdr = (
        f"{'B':>4} | {'T':>3} | "
        f"{'sg_vfy(ms)':>10} | {'sg_cmt(ms)':>10} | {'sg_total':>9} | "
        f"{'cu_vfy(ms)':>10} | {'cu_cmt(ms)':>10} | {'cu_total':>9} | "
        f"{'speedup':>7} | "
        f"{'rmse_sg':>9} | {'rmse_cu':>9} | {'rmse_kv':>9}"
    )
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for T_val in args.T:
        for B in args.batch_sizes:
            r = run_config(B, T_val, H, K, V, args.layer_idx, args.num_layers)
            print(
                f"{r['B']:>4} | {r['T']:>3} | "
                f"{r['sg_vfy_ms']:>10.4f} | {r['sg_cmt_ms']:>10.4f} | {r['sg_total_ms']:>9.4f} | "
                f"{r['cu_vfy_ms']:>10.4f} | {r['cu_cmt_ms']:>10.4f} | {r['cu_total_ms']:>9.4f} | "
                f"{r['speedup']:>6.2f}x | "
                f"{r['rmse_sg']:>9.6f} | {r['rmse_cu']:>9.6f} | {r['rmse_kv']:>9.6f}"
            )
        print()

    # Memory comparison
    sg_mem = B * T_val * H * K * V * 4
    cu_mem = B * T_val * (H * K + H * V) * 2
    print(f"Memory per-pool (B={args.batch_sizes[-1]}, T={args.T[-1]}):")
    print(f"  SGLang intermediate caches: {sg_mem / 1e6:.1f} MB")
    print(f"  cuLA KV buffer:             {cu_mem / 1e6:.1f} MB")
    print(f"  Ratio:                      {sg_mem / cu_mem:.0f}×")

    print("\nColumns:")
    print("  sg_vfy     : seg_la_mtp_kernel (Triton, SGLang upstream)")
    print("  sg_cmt     : fused_mamba_state_scatter_with_mask (Triton, SGLang)")
    print("  cu_vfy     : verify_kvbuffer with KV buffer write (CuTe DSL)")
    print("  cu_cmt     : state_update_kvbuffer reading from buffer (CuTe DSL)")
    print("  speedup    : sg_total / cu_total")
```

- [ ] **Step 7: Test the benchmark runs**

Run: `cd /root/work/cuLA && python benchmarks/bench_la_decode_mtp_vs_sglang.py --batch-sizes 4 --T 4`

Expected: Runs without error, prints a table with the new columns.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/bench_la_decode_mtp_vs_sglang.py
git commit -m "bench: fair full-pipeline comparison (SGLang verify+commit vs cuLA KVBuffer)"
```
