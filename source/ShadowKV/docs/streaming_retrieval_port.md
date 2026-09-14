# Streaming retrieval port: ParisKV and RetroInfer

Reference snapshot: 2026-09-04.

## Scope and invariants

This port deliberately excludes ClusterKV.  Released ShadowKV remains an
unchanged, prefill-indexed baseline.  New implementations use separate method
names and must satisfy the following lifecycle:

1. chronological tokens are grouped without resetting at the prefill/decode
   boundary;
2. a partial prompt tail is exact until decode completes its block;
3. a complete block is summarized exactly once;
4. blocks inside the rolling recent window remain exact and are ineligible for
   retrieval, preventing duplicate attention;
5. after leaving the recent window, the block joins the same candidate pool and
   competes under the same dynamic token budget as prefill blocks.

For block size 8, a 1025-token prompt has a one-token tail and needs seven
decode tokens to seal it.  A 1029-token prompt has a five-token tail and needs
three; this is determined by `length % block_size`.

`models/streaming_blocks.py` is the scorer-independent state machine.
`models/streaming_cache.py` is the common exact-KV backing store. It now has a
GPU reference mode and a pinned-CPU mode with one independently implemented
fused UVA K/V gather. The latter keeps two small GPU attention buffers: each
step preserves the current top-k order, copies overlapping blocks from the
preceding buffer, and reads only replacements plus exact prefix/recent rows
over PCIe. Appending one decode KV pair is also one fused CUDA launch. Router
latency remains method-specific and must be measured separately.

## ParisKV

- Paper: Yanlin Qi et al., *ParisKV: Fast and Drift-Robust KV-Cache Retrieval
  for Long-Context LLMs*, arXiv:2602.07721v3.
- Official code inspected at `amy-77/ParisKV`, commit
  `db7ad7f59ebe5670f9f2e1092b1b689802dda7c5`.
- Native lifecycle: Sink, Retrieval, Local, Update Buffer.  On a buffer update,
  the oldest local tokens are encoded/offloaded and buffered tokens become the
  new local region.
- Router: normalize plus SRHT; split into subspaces; analytic sign-pattern
  centroids; collision-vote candidates; 4-bit direction-code inner-product
  reranking; fetch only final top-k full KV.

The official repository contained no `LICENSE` or copying terms at the
inspected commit.  Therefore no source or CUDA kernel is copied here.  Any
ParisKV adapter must be an independent implementation from the published
algorithm, and must be named as a reference implementation until matched
against the authors' output.

## RetroInfer

- Paper: Zhiqiang Chen et al., *RetroInfer: A Vector Storage Engine for
  Scalable Long-Context LLM Inference*, arXiv:2505.02922v3 / PVLDB 2026.
- Official code inspected at `microsoft/RetrievalAttention`, commit
  `75829e630122d4ea6f568dcd001405698bc2db84`.
- License: MIT.
- Native lifecycle: a steady region and an update segment; each full update
  segment is clustered and appended to the wave index.
- Router: segmented spherical k-means; exact retrieval zone; centroid/value-sum
  estimation zone; exact steady zone.  The estimation contribution must be
  merged into both the attention numerator and denominator, not treated as a
  normal top-k cache.

MIT-licensed clustering or transfer kernels may be adapted only with the
Microsoft copyright and MIT notice retained.  A top-centroid-only
implementation must not be called RetroInfer because it omits the defining
estimation zone.

## Implemented method names

- `quest_streaming`: exact Quest page bounds with incremental page metadata;
  this is the only Quest runtime.
- `exact_block_lse_streaming` and `exact_block_max_streaming`: offline routing
  controls that read every candidate key and rank eight-token blocks by exact
  log-sum-exp or exact maximum. They share the streaming cache accounting but
  are oracles, not deployable methods.
- `adaptive_centroid_lse[_prefix4]`: historical ShadowKV-backed experiment,
  unchanged.
- `adaptive_centroid_lse_streaming[_prefix4]`: standalone append-only router
  with exact post-RoPE KV on GPU or pinned CPU. The canonical query-mean
  variant builds a 1--8-center mean-gap path and independently selects each
  block order by fixed-price rate--distortion with lambda 1.25; it optionally
  fuses decode LSE in Triton.
- `pariskv_official`: external authors' Qwen runtime pinned to the inspected
  commit. Both quality and speed use its native PolarANN collision voting,
  packed 4-bit reranking, CPU offload, UVA gather, and sparse attention. This
  repository supplies only the RULER dataset/metric adapter.
- `retroinfer_reference_streaming`: segmented spherical k-means, exact
  retrieval zone, centroid/value-sum estimation zone, exact steady zone, and
  append-only 1024-token update segments.  Its absolute budget is translated
  into a cluster count using the configured average cluster size; actual exact
  token traffic is therefore measured separately.

RetroInfer retains the explicit `reference_streaming` suffix. ParisKV does not:
its only registered path is now the authors' native external implementation.

## Source map in this worktree

- `repro/shadowkv/eval_pariskv_official.py`: RULER adapter around the external
  authors' native runtime.
- `repro/shadowkv/smoke_pariskv_official.py`: native runtime and latency smoke.
- `ShadowKV/models/offload_gather.py` and
  `ShadowKV/kernels/streaming_uva_gather.cu`: independent shared offload path.
- `ShadowKV/models/adaptive_centroid_triton.py`: fused canonical two-slot
  decode scorer; its variable cache length is explicitly excluded from Triton
  specialization so a long generation does not JIT a kernel per sealed block.
  PyTorch remains the reference backend.
- `ShadowKV/models/adaptive_centroid_incremental.py`: CUDA-graph replay of the
  exact 127-bipartition block-8 update; it changes launch scheduling, not the
  optimizer or its allocation threshold.
- `ShadowKV/models/retroinfer_reference.py`: clustering and tripartite merge math.
- `ShadowKV/models/retroinfer_triton.py`: MIT-licensed author clustering
  primitives adapted to return the token assignments consumed by this harness.
- `ShadowKV/models/retroinfer_streaming_cache.py`: incremental RetroInfer correctness
  cache.
- `ShadowKV/tests/test_retroinfer_reference.py`: method-level invariants.
- `ShadowKV/tests/test_streaming_blocks.py` and
  `ShadowKV/tests/test_streaming_caches.py`:
  lifecycle, prompt/decode boundary, disjointness and dense-limit invariants.

No ParisKV source was copied because the inspected repository has no license.
The RetroInfer reference was written independently from the paper/API
contract; the Microsoft MIT notice is recorded in the module and must remain
if official kernels are later adapted.

## Required validation before results

1. CPU lifecycle/unit tests.
2. One-GPU synthetic generation crossing a block boundary.
3. Dense equality when the dynamic budget covers every active block.
4. Audit that prefix/recent/retrieved position sets are disjoint.
5. Stamp method name, block size, recent tokens, prefix tokens, and exact versus
   compressed backing store.
6. For RetroInfer, compare selected indices or attention outputs against
   their official implementation on an identical captured Q/K/V tensor before
   running benchmark cells.

Current gates: ParisKV's six-tier collision votes, BF16 4-bit reranker scores,
and selected sets match the inspected author code exactly on the synthetic
tensor gate.  RetroInfer assignments and cluster sizes match exactly, while
centroids/value sums match within BF16 accumulation tolerance.  Retrieving all
RetroInfer clusters is numerically equal to dense attention, as is estimating
clusters whose keys are constant.  These are correctness gates, not evidence
of production efficiency or benchmark quality. The additional 2026-09-05
initial capacity gate completed Qwen3 128K/512 on A5000 for ours (15.99 GiB
peak) and native ParisKV (17.62 GiB peak). After exact graph replay, fused append and
temporal block reuse, isolated 32K/512 runs measure Qwen3 at 34.53 ms/token
median (35.64 mean over 96 tokens) and Llama-3.2 at 26.19 ms/token median
(27.01 mean). Real 128K gates complete at 35.22 ms/token and 16.07 GiB for
Qwen3, and 26.87 ms/token and 12.79 GiB for Llama-3.2. These remain engineering
gates rather than a cross-runtime paper speed claim.
