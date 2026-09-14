# Qwen3 128K engineered streaming retrieval

Measured on machine 4 (`sashimi`), one RTX 4090, Qwen3-4B-Instruct-2507,
one real 128K RULER CWE prompt, nominal budget 4096, 32 prefix tokens and 32
recent tokens.  K and V remain in pinned CPU memory.  Both streaming methods
maintain ping-pong sparse working sets on GPU and fetch only replacement
blocks through the fused UVA gather.  FlashInfer RMSNorm, one full warm-up
pass, one measured prefill and 32 measured decode tokens were used.

| Method | Version | Prefill (s) | Decode median (ms) | Decode mean (ms) | Maximum step (ms) | Peak GPU (GiB) |
|---|---|---:|---:|---:|---:|---:|
| Quest-streaming | PyTorch broadcast router | 45.81 | 50.39 | 50.22 | 52.94 | 15.86 |
| Quest-streaming | fused min/max Triton router | 45.87 | 35.56 | 35.38 | 38.55 | 15.80 |
| Ours, fixed center price 1.5 | synchronous sealing | 48.99 | 35.66 | 55.24 | 194.79 | 18.74 |
| Ours, fixed center price 1.5 | amortized sealing | 49.16 | 53.10 | 51.49 | 77.44 | 18.66 |
| Released ShadowKV-CPU | SVD K reconstruction | 52.10 | 30.96 | 31.19 | 33.40 | 14.30 |

The fused Quest router performs the same page-min/max upper bound and the
same top-k selection.  Against an FP32 reference its maximum score error was
`7.63e-6` and top-256 overlap was 100%.  A standalone 8192-page router
microbenchmark on an RTX A5000 improved from 0.911 ms to 0.064 ms per layer
(14.3x); end-to-end decode improved by 29.6% (50.22 to 35.38 ms).

Ours already used fused INT8 routing and temporal whole-block reuse.  Its
remaining periodic cost was construction of the full 1..8 self-K hierarchy
for every layer whenever eight generated tokens sealed a block.  The new
scheduler uses the exact recent window to spread the unchanged construction
over the following eight steps.  It lowers mean latency by 6.8% and the worst
step by 60.2%, but moves work onto ordinary steps; this is why the median
increases.  Eliminating rather than smoothing this cost requires a batched or
fused hierarchy builder and is separate from K/V offload and gather.

The remaining gap to ShadowKV is expected: ShadowKV reconstructs selected K
from a GPU-resident low-rank representation and transfers primarily V.  The
two methods here deliberately retain exact K and V in CPU memory, so changed
blocks require both tensors to cross the host-device boundary.

Relevant commits: `5374f289` (fused Quest router) and `68a8c6bb` (amortized
adaptive-centroid sealing).  Raw results are stored under
`/storage/nbao/shadowkv_results/latency_128k_engineered_20260910` on machine 4.
