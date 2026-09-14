# Plan: Qwen3 Query-Robust 128K reference path

## Goal

Port the Query-Robust page-summary and QR-vertex routing logic from
`ihb-sparse` into this repository's existing `ShadowKV` streaming cache, and
vendor the frozen Qwen3-4B-Instruct-2507 128K QR vertex artifact. The result
must be selectable from the existing RULER runner, remain offline-compatible,
and be validated locally plus on a Modal B200 before it is pushed to
`ihbkaiser/b200_sa`.

## Architecture and constraints

- Keep the existing `StreamingBlockCache` lifecycle and exact K/V gather path.
- Add a first-class `StreamingQueryRobustCache`; do not add method-name
  branches inside the generic cache hot path.
- Implement the QR reference math in PyTorch FP32 for solver/scoring, store
  landmarks in BF16, and use a page/block size of 8 by default for parity with
  the current B200 Quest campaign.
- Route pages with the source method's formula:
  `scale * q·landmark + bias + alpha * epsilon`, reducing a GQA group by max.
- Validate artifact shape, dtype, metadata, padding, model fingerprint, and
  SHA-256 before use. No model weights or credentials are added to GitHub.
- The port is a correctness/reference implementation. It must not claim the
  source repo's optimized Triton throughput until a separate kernel exists.
- Support the existing runtime's batch-size-one streaming contract explicitly.

## Spec path

The source specification is the public `query-robust-paper-b200` implementation
and its tests in `/workspace/ihb_sparse_src`, especially:

- `src/sparsevllm/engine/cache_manager/query_robust.py`
- `tests/test_query_robust.py`
- `remote_artifacts/modal_20260913_145407_qwen3_qr_128k_final/`

## Implementation tasks

### 1. Freeze the source artifact and repository contract

- Copy the Qwen3 `qwen3_4b_qr_vertices_m32_128k.pt` artifact from the source
  branch into a tracked, model-specific artifact directory.
- Add a provenance manifest, validation report, and checksum without retaining
  private calibration paths.
- Add artifact-loading tests for valid metadata, bad shape/dtype/hash, and
  padded-vertex semantics.
- Run the new artifact tests first and confirm they fail before implementation.

### 2. Add the pure-PyTorch Query-Robust math

- Add `models/query_robust.py` with typed summary/asset dataclasses, asset
  validation, the minimax dual solver, batched page-summary construction, and
  reference page scoring.
- Keep all numerical reductions in FP32 and certify the BF16 landmark that the
  runtime actually stores.
- Add tests for the convex-hull certificate, uniform baseline, batched-vs-
  single-page equivalence, GQA scoring, invalid-page handling, and solver input
  validation.
- Run the focused suite in red before adding the implementation, then green.

### 3. Integrate the cache lifecycle

- Add `StreamingQueryRobustCache` as a `StreamingBlockCache` subclass.
- Build summaries when sealed blocks become retrievable, invalidate metadata
  when cache pages are reused, and keep exact prefix/recent tokens unchanged.
- Implement page selection through QR scores and reuse the inherited exact K/V
  gather so the selected IDs affect actual attention output.
- Register `query_robust`/`qr` in `LLM.init_kv_cache` and add explicit CLI
  arguments for the artifact, solver settings, score alpha, and page size.
- Add lifecycle tests proving page metadata is built, selected pages are
  correct, stale metadata is invalidated, and selected K/V matches the exact
  backing cache.

### 4. Add the offline runner and documentation

- Add a `query_robust` method to `run_b200_ruler.sh` with Qwen3-only artifact
  discovery and the same 128K/100-sample dataset contract.
- Add a standalone artifact validator and a small CPU smoke command that does
  not require model weights or network access.
- Document that the frozen QR vertices are calibration-only and that RULER
  quality/throughput claims require a completed campaign.

### 5. Verification ladder

- Run syntax/import checks and the focused unit tests.
- Run the full source unit suite that is compatible with the current Python
  environment, recording skips caused by unavailable optional GPU extensions.
- Run a real Modal B200 smoke with a tiny synthetic config and the vendored
  Qwen3 artifact; verify CUDA execution, metadata construction, routing, and
  exact gather output. Preserve logs and JSON markers under a stable run dir.
- If a local Qwen3 checkpoint is available in the Modal environment, run a
  short model-level decode. Otherwise report that model-level 128K RULER was
  not run rather than inferring it from the synthetic test.
- Run credential scans, artifact checksum validation, and `git diff --check`.

### 6. Commit and publish

- Commit with a Conventional Commit message after fresh verification.
- Fetch the remote and push the validated branch; fast-forward `main` only if
  the remote has not diverged and the verification artifacts are complete.
- Report the exact commit/branch, tests, Modal result, and any limitations.
