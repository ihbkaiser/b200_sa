# Query-Robust Maximum-Performance Design

## Goal

Accelerate the Qwen3 Query-Robust 128K runtime while preserving the current
QR route semantics, then measure the resulting implementation on a native
Modal B200 with decode latency, update spikes, HBM peak, and throughput.

## Scope and non-goals

This design covers the QR page-summary builder, decode-time page scorer,
streaming cache integration, and a model-level B200 benchmark harness. It does
not change the QR vertex asset, solver hyperparameters, page size, sparse
budget, attention math, or RULER dataset protocol. Quest changes are limited to
the campaign configuration needed to keep its comparison runnable; Quest is
not part of the QR kernel redesign.

## Behaviour contract

The optimized path must preserve these observable rules:

- Summary construction continues to solve the same 24-step FP32 dual problem
  with `solver_lr=0.25`, unless a benchmark explicitly selects another value.
- Runtime landmarks remain the BF16 tensor used by the current reference
  scorer; `bias`, `epsilon`, and validity remain FP32/bool metadata.
- For each valid page, the score is
  `max_head(max_g(scale * dot(query[h,g], landmark[page,h]) + bias[page,h] +
  alpha * epsilon[page,h]))`.
- A page with any invalid KV-head metadata returns positive infinity, matching
  the reference implementation.
- The cache keeps one shared page order per request, uses the existing
  `torch.topk` ordering/tie semantics, and gathers the exact K/V rows selected
  by those page IDs.
- Non-CUDA, unsupported-shape, and explicitly selected reference paths remain
  available and return the current reference result.

Parity is accepted when invalid-page masks and selected top-k page IDs match
the reference on deterministic and randomized fixtures, and valid score values
are within `atol=2e-2, rtol=2e-2` for BF16 landmark inputs. Certificate
diagnostics must remain finite and their absolute difference must be at most
`2e-2` on the same fixtures.

## Recommended architecture

### Fused decode scorer

Add `models/query_robust_triton.py` with a Triton kernel specialized for the
runtime's contiguous candidate range. One program processes a tile of pages
for one batch item, loops over the eight local KV heads and four GQA groups,
accumulates the dot product in FP32, applies `bias` and `alpha * epsilon`,
reduces over groups and heads, and writes one score per page. Validity is
reduced in the same kernel. The wrapper receives `first_page`/`last_page`, so
the hot path no longer allocates `torch.arange` or materializes four
`index_select` tensors plus the `einsum` output.

The cache calls this kernel only for CUDA decode with contiguous page ranges.
The reference scorer remains the fallback and test oracle. `torch.topk` stays
separate initially so its existing tie behavior is unchanged; benchmark data
will determine whether replacing it is worthwhile.

### Summary build backend

Keep the existing FP32 solver as the canonical implementation. Add a reusable
workspace for the fixed page-batch shape so repeated streaming flushes reuse
the flattened key/vertex/logit/solver buffers where possible. Add an optional
compiled backend for the pure tensor solver body, warmed before measurement and
falling back to eager PyTorch if compilation is unavailable. The compiled path
must be opt-in until parity tests show the same summary contract and certificate
thresholds; the benchmark records which backend ran.

The implementation must not run summary work on a background stream without an
explicit dependency event: metadata must be fully visible before page scoring.

### Benchmark harness

Add a Modal B200 runner that packages the repository source, vendored QR asset,
and a local model checkpoint path or an explicitly configured remote model
source. It must fail clearly if the Qwen3 checkpoint, flash-attention, or the
ShadowKV CUDA extension is unavailable. The run emits environment JSON and
JSONL metrics with:

- GPU name, compute capability, PyTorch/CUDA/Triton/flash-attn versions;
- QR backend, solver settings, page size, budget, context, generation length;
- decode latency p50/p95/max and tokens/s after warmup;
- per-update summary-build latency p50/p95/max;
- HBM allocated/reserved/peak before and after generation;
- router, top-k, gather, and attention timing where instrumentation is
  available;
- output path, exit code, and a reproducible run label.

Run GPU-resident QR as the primary B200 configuration and CPU-offload QR as a
separate comparison. The benchmark must not label a synthetic cache smoke as a
model-level 128K result.

## Files and interfaces

- Create `source/ShadowKV/models/query_robust_triton.py` with
  `query_robust_page_scores(...) -> torch.Tensor` and a strict CUDA wrapper.
- Modify `source/ShadowKV/models/query_robust_cache.py` to select the fused
  scorer only for the supported contiguous decode path and to expose backend
  statistics.
- Modify `source/ShadowKV/models/query_robust.py` to expose the reusable
  summary workspace/backend without changing the canonical reference API.
- Add focused parity/dispatch tests under `source/ShadowKV/tests/`.
- Add `source/ShadowKV/tools/modal_qr_benchmark.py` and a local wrapper that
  preserve full Modal logs and machine-readable markers.
- Modify `run_b200_ruler.sh` only to make QR/Quest backend and offload choices
  explicit and to remove the known Quest dense/offload contradiction.
- Update `README_B200.md` with exact benchmark commands and interpretation
  rules, including the fact that B200 metrics are required for performance
  claims.

## Verification gates

1. Focused reference tests pass before and after optimization.
2. New Triton parity tests fail before the kernel exists and pass afterward.
3. CPU/reference fallback remains importable without Triton or CUDA.
4. `git diff --check`, syntax checks, and credential scans pass.
5. A native B200 run imports the target CUDA extension and flash-attention,
   loads Qwen3 locally, executes a short 128K model-level decode, and writes
   all required metrics. If model assets are not available in Modal, the run
   remains incomplete rather than being replaced by a synthetic claim.

