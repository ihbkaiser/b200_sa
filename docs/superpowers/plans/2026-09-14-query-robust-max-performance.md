# Query-Robust Maximum-Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize Qwen3 Query-Robust 128K decode and summary updates without changing routing semantics, then produce verified native Modal B200 metrics.

**Architecture:** Keep `query_robust.py` as the numerical reference and add a CUDA-only Triton scorer for the cache's contiguous page range. Add a reusable summary workspace and an opt-in compiled summary backend, with eager PyTorch fallback. Instrument the cache and run the real local Qwen3 checkpoint through the same model path on a Modal B200.

**Tech Stack:** PyTorch 2.11, Triton 3.6, CUDA 13.0, FlashAttention, existing ShadowKV CUDA extension, pytest, Modal.

**Spec:** `docs/superpowers/specs/2026-09-14-query-robust-max-performance-design.md`

## Global Constraints

- Preserve QR's FP32 solver, BF16 runtime landmarks, page validity, shared page order, and `torch.topk` semantics.
- Use `solver_iters=24`, `solver_lr=0.25`, `alpha=1.0`, page size 8, budget 4096, and context length 131072 for the production benchmark.
- Do not put model weights, HF tokens, GitHub tokens, or Modal secrets in source, logs committed to Git, or benchmark metadata.
- Do not report RTX 3090/4090 measurements as B200 performance.
- A B200 performance claim requires native B200 GPU, Qwen3 model load, FlashAttention import, ShadowKV CUDA-extension import, and metric markers in the saved run log.

### Task 1: Lock the baseline and add failing fused-scorer tests

**Files:**
- Create: `source/ShadowKV/tests/test_query_robust_triton.py`
- Modify: `source/ShadowKV/tests/test_query_robust.py` only if a shared fixture is needed
- Test: `source/ShadowKV/tests/test_query_robust_triton.py`

**Interfaces:**
- The tests will require `models.query_robust_triton.query_robust_page_scores` with keyword arguments `first_page`, `last_page`, `scale`, and `alpha`.
- The function returns `[batch, pages]` FP32 scores for contiguous pages.

- [ ] **Step 1: Write the deterministic parity tests before adding the module.**

  Cover: valid pages, invalid metadata in one KV head, GQA max reduction, nonzero `alpha`, nonzero `first_page`, and a randomized top-k comparison against `score_query_robust_pages_reference`.

  The test must assert the invalid mask exactly, `torch.testing.assert_close` on valid values with `atol=2e-2, rtol=2e-2`, and equality of top-`min(64, pages)` indices on fixtures with separated scores. Mark CUDA tests skipped when CUDA or Triton is unavailable.

- [ ] **Step 2: Run the new test and verify the expected RED failure.**

  Run:

  ```bash
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust_triton.py
  ```

  Expected result: collection fails with `ModuleNotFoundError` for the not-yet-created `models.query_robust_triton`, not with a fixture or assertion error.

- [ ] **Step 3: Record the existing eager baseline without changing production code.**

  Run the current scorer on a 16,384-page, 32-query-head, 8-KV-head, 128-dimensional BF16 fixture and save the per-call CUDA-event time in the local run log. This baseline is for comparison only and is not a B200 claim.

- [ ] **Step 4: Commit the red tests.**

  ```bash
  git add source/ShadowKV/tests/test_query_robust_triton.py
  git commit -m "test: define QR fused scorer parity contract"
  ```

### Task 2: Implement the fused Triton QR scorer

**Files:**
- Create: `source/ShadowKV/models/query_robust_triton.py`
- Test: `source/ShadowKV/tests/test_query_robust_triton.py`

**Interfaces:**
- Produce `query_robust_page_scores(query, landmark, bias, epsilon, metadata_valid, *, first_page, last_page, scale, alpha) -> torch.Tensor`.
- Accept query `[B,QH,D]`, landmark `[P,KVH,D]`, scalar metadata `[P,KVH]`, validity `[P,KVH]`, and contiguous page bounds.
- Reject CPU tensors, non-contiguous page ranges, invalid shapes, non-divisible GQA heads, and empty page ranges with explicit `ValueError`/`RuntimeError` messages.

- [ ] **Step 1: Add the minimal public wrapper and a kernel stub that raises a clear unavailable error.**

- [ ] **Step 2: Run the parity test and confirm it fails because the implementation is intentionally unavailable.**

- [ ] **Step 3: Implement `_query_robust_page_score_kernel` with one page tile per batch item.**

  Load BF16 landmarks and query values as FP32, loop over compile-time KV heads and GQA groups, reduce dot products over `D`, add `scale`, `bias`, and `alpha * epsilon`, reduce groups and heads by max, reduce validity with logical AND, and store `+inf` for an invalid page.

  Use `BLOCK_P=64` for fewer than 8192 candidates and `BLOCK_P=128` otherwise, and `BLOCK_D=triton.next_power_of_2(head_dim)`. Keep the wrapper's output FP32 and do not implement a fused top-k in this task.

- [ ] **Step 4: Run the parity test until it passes, including the invalid-page and top-k assertions.**

  ```bash
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust_triton.py
  ```

- [ ] **Step 5: Commit the kernel.**

  ```bash
  git add source/ShadowKV/models/query_robust_triton.py source/ShadowKV/tests/test_query_robust_triton.py
  git commit -m "perf: add fused Triton Query-Robust scorer"
  ```

### Task 3: Integrate the fused scorer and runtime timing counters

**Files:**
- Modify: `source/ShadowKV/models/query_robust_cache.py:1-225`
- Modify: `source/ShadowKV/models/streaming_cache.py:500-535`
- Test: `source/ShadowKV/tests/test_query_robust_triton.py`

**Interfaces:**
- Add `QUERY_ROBUST_ROUTER_BACKEND=auto|triton|torch`, defaulting to `auto`.
- Add `qr_runtime_stats() -> dict[str, object]` with router milliseconds, top-k milliseconds, update milliseconds, and event counts.

- [ ] **Step 1: Add a dispatch test that sets `QUERY_ROBUST_ROUTER_BACKEND=triton`, calls `_score_blocks` on a CUDA cache fixture, and asserts the fused scorer is selected.**

- [ ] **Step 2: Run the dispatch test and verify RED because the cache currently calls the reference scorer directly.**

- [ ] **Step 3: Add the backend selector and route only contiguous CUDA decode ranges to the Triton function.**

  Keep `torch` as an explicit reference backend. In `auto`, use Triton only when available and fall back to the reference scorer on import/shape failure. Preserve the cache's existing one shared page order and return shape `[B,KVH,P]` by broadcasting the fused `[B,P]` result.

- [ ] **Step 4: Instrument router, `torch.topk`, and summary update durations with CUDA events when the tensors are CUDA-resident.**

  Accumulate raw samples in bounded Python lists (maximum 4096 entries per category) and expose p50/p95/max summaries without synchronizing on every decode step. The benchmark explicitly synchronizes at collection boundaries.

- [ ] **Step 5: Run the full focused QR suite and dispatch test.**

  ```bash
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust.py source/ShadowKV/tests/test_query_robust_triton.py
  ```

- [ ] **Step 6: Commit the integration.**

  ```bash
  git add source/ShadowKV/models/query_robust_cache.py source/ShadowKV/models/streaming_cache.py source/ShadowKV/tests/test_query_robust_triton.py
  git commit -m "perf: dispatch QR cache through fused router"
  ```

### Task 4: Add reusable QR summary workspace and checked compile backend

**Files:**
- Modify: `source/ShadowKV/models/query_robust.py:373-455`
- Modify: `source/ShadowKV/models/query_robust_cache.py:35-160`
- Create: `source/ShadowKV/tests/test_query_robust_summary_backend.py`

**Interfaces:**
- Add `QueryRobustSummaryWorkspace(device, dtype=torch.float32)` with `build(keys, vertices, *, scale, solver_iters, solver_lr, uniform_p) -> QueryRobustSummary`.
- Add `build_query_robust_page_summaries(..., workspace=None)` while preserving all existing positional/keyword calls.
- Add `QUERY_ROBUST_SUMMARY_BACKEND=eager|compile`, default `eager`; compilation is used only when explicitly requested and available.

- [ ] **Step 1: Write tests for eager workspace reuse and compile parity before implementing the new backend.**

  Assert that two builds with the same shape return the same values as the current function, certificate fields are finite, and compiled/eager summary landmarks, bias, epsilon, dual, and gap are within `2e-2`. Add a shape-change test proving the workspace safely grows/reallocates rather than reusing incompatible storage.

- [ ] **Step 2: Run the backend tests and verify RED because the workspace/backend interfaces do not exist.**

- [ ] **Step 3: Implement workspace-owned flattened buffers and use them for the repeated `flat_keys`, `flat_vertices`, `logits`, `lambda_logits`, and solver intermediates.**

  Do not mutate caller tensors or the frozen vertex asset. Return cloned/view-safe summary fields so the next build cannot overwrite a previous metadata cache entry.

- [ ] **Step 4: Implement an internal tensor-only compiled solver entry point.**

  Cache compiled callables by `(pages, page_size, kv_heads, head_dim, vertices, solver_iters, uniform_p)`, warm each callable before timing, and catch compilation/runtime failures to the eager path. Do not compile the dataclass wrapper or dynamic asset validation.

- [ ] **Step 5: Connect `StreamingQueryRobustCache._build_blocks` to the selected backend and record summary-build samples.**

- [ ] **Step 6: Run focused reference and backend tests.**

  ```bash
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust.py source/ShadowKV/tests/test_query_robust_triton.py source/ShadowKV/tests/test_query_robust_summary_backend.py
  ```

- [ ] **Step 7: Commit the summary backend.**

  ```bash
  git add source/ShadowKV/models/query_robust.py source/ShadowKV/models/query_robust_cache.py source/ShadowKV/tests/test_query_robust_summary_backend.py
  git commit -m "perf: add reusable and compiled QR summary backend"
  ```

### Task 5: Fix B200 runner configuration and add local benchmark entry point

**Files:**
- Modify: `run_b200_ruler.sh:1-120`
- Modify: `README_B200.md:80-125`
- Create: `benchmark_qr_b200.sh`
- Test: `source/ShadowKV/tests/test_query_robust_runner_config.py`

**Interfaces:**
- `QUERY_ROBUST_ROUTER_BACKEND` and `QUERY_ROBUST_SUMMARY_BACKEND` are forwarded to the model process.
- `QUERY_ROBUST_OFFLOAD=0` is the B200 default; `QUERY_ROBUST_OFFLOAD=1` remains a separate UVA comparison.
- Quest uses GPU-resident K/V when `QUEST_OFFLOAD=0`; offload mode forces `dense_layers=0`.

- [ ] **Step 1: Write a shell/config test that parses the runner and asserts the default QR path does not request offload, and that Quest never sends `dense_layers>0` with offload.**

- [ ] **Step 2: Run the config test and observe RED against the current hard-coded Quest `--streaming_offload` and fixed QR defaults.**

- [ ] **Step 3: Add environment-controlled argument arrays while keeping the existing 128K/100-sample protocol.**

- [ ] **Step 4: Add `benchmark_qr_b200.sh` to call `eval_acc.py --runtime_out` with a local Qwen3 model, fixed dataset row, warmup 8, measured decode 128 tokens, backend settings, and JSON output.**

- [ ] **Step 5: Document exact eager-vs-compile and GPU-vs-offload commands and make clear that this local benchmark is a preflight, not a B200 claim.**

- [ ] **Step 6: Run shell syntax and config tests, then commit.**

  ```bash
  bash -n run_b200_ruler.sh benchmark_qr_b200.sh
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust_runner_config.py
  git add run_b200_ruler.sh benchmark_qr_b200.sh README_B200.md source/ShadowKV/tests/test_query_robust_runner_config.py
  git commit -m "test: make QR and Quest B200 modes explicit"
  ```

### Task 6: Build the native Modal B200 benchmark harness

**Files:**
- Create: `source/ShadowKV/tools/modal_qr_benchmark.py`
- Create at run time: `remote_artifacts/modal_qr_benchmark_YYYYMMDD_HHMMSS/`
- Modify: `README_B200.md:125-180`

**Interfaces:**
- Modal local entry point accepts `--model-dir`, `--context`, `--decode-tokens`, `--warmup`, `--steps`, `--summary-backend`, and `--router-backend`.
- Remote function prints `QR_B200_ENV_JSON=...`, `QR_B200_METRICS_JSONL_BEGIN`, one JSON object per measured step/update, `QR_B200_METRICS_JSONL_END`, and `QR_B200_SUMMARY_JSON=...`.

- [ ] **Step 1: Add a local entrypoint that packages the repository's `model/Qwen3-4B-Instruct-2507` directory and source into the Modal image without reading or printing credentials.**

- [ ] **Step 2: Build the image with Python 3.12, PyTorch 2.11 CUDA 13.0, Triton, Transformers, NumPy, FlashAttention, and the B200-built ShadowKV extension.**

  The remote preflight must print only versions and paths, assert compute capability at least `10.0`, and fail before the model run if `flash_attn` or `kernels.shadowkv` cannot import.

- [ ] **Step 3: Run the same Qwen3 `LLM` path as `eval_acc.py`, use a real 131072-token local prompt from the bundled RULER data, warm up eight decode steps, then measure 128 decode steps.**

  Record each synchronized decode step, classify update steps from cache counters, collect `torch.cuda.max_memory_allocated`, `max_memory_reserved`, current allocated/reserved, and compute p50/p95/max latency and tokens/s.

- [ ] **Step 4: Run two isolated B200 configurations with all other variables fixed.**

  Primary: `router=triton`, `summary=compile`, GPU-resident QR. Comparison: `router=torch`, `summary=eager`, GPU-resident QR. Run offload only as a third explicitly labeled comparison if the primary image has enough time and quota.

- [ ] **Step 5: Save full timestamped Modal logs, exit code, environment JSON, metrics JSONL, and parsed summary under `remote_artifacts/`.**

- [ ] **Step 6: Commit the harness and documentation, excluding runtime logs and model weights.**

  ```bash
  git add source/ShadowKV/tools/modal_qr_benchmark.py README_B200.md
  git commit -m "bench: add native B200 QR latency harness"
  ```

### Task 7: Verification and completion audit

**Files:**
- Verify: all files changed by Tasks 1-6
- Verify: `remote_artifacts/modal_qr_benchmark_YYYYMMDD_HHMMSS/`

- [ ] **Step 1: Run local focused tests, syntax checks, diff checks, and credential scans.**

  ```bash
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m pytest -q source/ShadowKV/tests/test_query_robust.py source/ShadowKV/tests/test_query_robust_triton.py source/ShadowKV/tests/test_query_robust_summary_backend.py source/ShadowKV/tests/test_query_robust_runner_config.py
  PYTHONPATH=source/ShadowKV /venv/main/bin/python -m py_compile source/ShadowKV/models/query_robust.py source/ShadowKV/models/query_robust_triton.py source/ShadowKV/models/query_robust_cache.py source/ShadowKV/models/streaming_cache.py source/ShadowKV/tools/modal_qr_benchmark.py
  git diff --check
  ! rg -n --hidden -g '!model/**' -g '!wheels/**' -g '!.git/**' '(hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ak-[A-Za-z0-9]{20,}|as-[A-Za-z0-9]{20,})' .
  ```

  The credential scan must return no matches.

- [ ] **Step 2: Run the Modal harness and verify the remote log contains B200 environment markers, model-load success, and metric markers.**

- [ ] **Step 3: Re-run the optimized and eager reference configurations on the same B200 and compare parity on generated tokens, top-k route IDs, summary certificates, and metric distributions.**

- [ ] **Step 4: Only after all gates pass, report the exact commit, Modal run directory, p50/p95/max decode and update latency, HBM peak, and throughput.**
