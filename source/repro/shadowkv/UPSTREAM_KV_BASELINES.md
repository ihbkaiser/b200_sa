# PQCache, MagicPIG and InfLLM protocol

This protocol prevents an adapter that merely shares a method name from being
reported as the paper implementation.  Every accuracy cell is stamped with the
main-repository commit and the exact author-checkout commit.

## Pinned author sources

| Method | Repository | Commit | License in checkout |
|---|---|---|---|
| PQCache | `HugoZHL/PQCache` | `0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828` | none found; no source copied |
| MagicPIG | `Infini-AI-Lab/MagicPIG` | `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb` | Apache-2.0 |
| InfLLM | `thunlp/InfLLM` | `12b70798f56e56ebb23c53c7018091a3f540a028` | MIT |

`setup_upstream_kv_baselines.sh` fetches those commits.  Models, datasets and
Python environments are machine-local and are never transferred through git.

## Primary accuracy table

- Models: Qwen3-4B-Instruct-2507 and Llama-3.2-3B-Instruct.
- Data: the same 13 RULER tasks, 100 examples per task.
- Lengths: 32,768 and 65,536 input tokens.
- Nominal active-attention budget: `B = context_length / 32`, hence 1,024 and
  2,048 tokens.
- Same tokenizer, weights, greedy decoding, task prompts and scorer for every
  common-forward row.
- No label/query calibration and no task-specific tuning.

### Model-family scope

The released PQCache README lists Llama-3.1-8B-Instruct and
Mistral-7B-Instruct-v0.2 as supported models.  The released MagicPIG
HuggingFace path says that only Llama models are supported and demonstrates
Llama-3.1-8B-Instruct.  Consequently, Llama-3.2-3B is a related-family but
different-size/version transfer for both methods, while Qwen3 is an explicit
common-adapter extrapolation outside their released model scope.  These rows
are valuable transfer tests but must not be described as bit-exact reproduction
of the authors' reported model experiments.  Results are therefore always
broken down by model; a cross-model macro alone is not reported.

The primary table uses a matched exact-region protocol. In every method,
`B=L/32` denotes the remote/routed target; 32 prefix and 32 recent tokens are
exact and outside `B`:

| Method | Routed/target budget | Mandatory exact region outside that budget | Effective active-token accounting |
|---|---:|---:|---:|
| Ours (canonical reference) | hard `B` routed | 32 prefix + 32 recent | `B + 64` |
| PQCache | hard `B` retrieved | 32 prefix + 32 recent | `B + 64` |
| MagicPIG | stochastic remote target `B` | 32 prefix + 32 recent | realized remote count + 64; mean and maximum mandatory |
| InfLLM | hard `B` in complete 128-token units | 32 initial + 32 local/recent | `B + 64` |

Thus the fixed cap is 1,088 at 32K and 2,112 at 64K for ours, PQCache and
InfLLM. MagicPIG is compared using its measured realized count, never by
relabeling its target as a cap. `UPSTREAM_MATCHED_EXACT_REGIONS=1` is stamped
in every primary cell and changes its cell key, so earlier author-structure
pilots cannot be mixed into this table.

- **PQCache:** author PQ dimensions (two subvectors, six bits, 64 centroids),
  Euclidean clustering and per-query-head approximate softmax followed by GQA
  summation. The matched lane asks that router for `B` remote tokens, then adds
  32 exact prefix and 32 exact recent tokens, so its cap is `B+64`. Codes
  currently occupy one byte rather than a
  packed six-bit representation.  The common adapter has exact CPU-pinned KV
  but not the author's LFU GPU block cache.  It also uses `kmeans-gpu` for ten
  iterations on at most 8,192 training keys; the released production path uses
  parallel sklearn KMeans on the full key set.  Both use the released
  compressor initialization seed `PQCACHE_SEED=4321`, recorded in every cell.
  Thus this row reuses the author
  PQ lookup/ranking structure but is **not** bit-exact author-native PQCache.
  Runtime additionally reports a separately labelled `PQCache native` lane.
  That lane dynamically imports the released multiprocessing compressor,
  FP16 CPU backing, LFU GPU block cache and asynchronous transfer schedule;
  only the post-RoPE tensor bridge and explicit `B+32+32` ratio mapping are
  ours. The released model path supports Llama/Mistral, so Qwen is explicitly
  labelled a tensor-ABI compatibility port and uses the released Llama prefill
  timing coefficients. It is not folded into the primary common-forward row.
  PQCache's `pq_search.py` has an unused eager import of its bundled SparQ
  experiment package. The bridge supplies the two unused imported symbols so
  native PQ does not require WandB/torchaudio; no executed PQ path is replaced.
- **MagicPIG:** author `K=10`, `L=210` LSH and native CPU importance-sampling
  attention, with 32 sink/prefix and 32 local/recent tokens in the matched
  lane. LSH has no fixed token budget: `B` is a target operating point, and the
  realized sampled remote KV count plus 64 mandatory tokens must be reported.
  The primary table disables
  author dense layers so they cannot silently buy extra attention.  Native
  runtime is a separate row because the common adapter builds each layer's
  table synchronously instead of using the author's overlap schedule.
  The adapter constructs the random LSH hyperplanes under the released HF RULER seed
  `MAGICPIG_SEED=43` (overridable), restores the
  model RNG afterwards, and includes the seed in each cell key.
  The released `lsh` and `sparse_attention_cpu` setup files hard-code
  AVX-512F.  MagicPIG is therefore scheduled only on AVX-512F hosts; setting
  `UPSTREAM_KV_METHODS=pqcache,infllm` installs/runs the portable subset on an
  AVX2-only host without substituting a different MagicPIG implementation.
- **InfLLM:** author 128-token memory units, four representatives per unit and
  ContextManager. The matched lane retrieves exactly `B/128` complete memory
  units and adds 32 initial plus 32 local/recent tokens. The released Triton
  auxiliary-score kernel uses a 64-row tile that is unsafe for Qwen's
  four-query-head GQA group with a 32-token local window. The matched lane
  retains the official Triton attention output (`fattn=true`) and evaluates
  only that auxiliary representative-score expression in Torch for short
  tiles. Author-native settings use the released Triton path throughout. The kernel does
  not mask query rows introduced by its 64-row padding.  The adapter guards
  those rows by setting their log-normalizers to `+inf`; direct Torch-vs-Triton
  tests then agree to at most 0.00391 in the auxiliary score and below 0.001
  relative error in attention output for tested query lengths 1--128.
  At the 32K primary budget this is concretely `32 init + 32 local +
  8 * 128 retrieved = 1088` active tokens. The released Llama/Mistral InfLLM
  configurations instead use `128 init + 4096 local + 16 * 128 retrieved =
  6272`. Thus the matched row is a fair exact-region stress test with eight
  retrievable memory units, not a reproduction of the author's much larger
  operating point.  Any native-budget secondary row must display its 6.125x
  active-token cost next to accuracy.

Native paper settings may be reported as secondary rows, never substituted for
the primary matched table.

### 48GB machine-3 completion path

InfLLM-64K exceeds a 24GB card during matched common-forward prefill (the
measured Llama process reached 22.92/23.55 GiB before a further 1,016 MiB
allocation).
`run_m3_upstream_kv_32_64.sh` therefore runs the 26 InfLLM-64K accuracy cells
on the 48GB L40 machine. Runtime remains locked by default even after those
local cells finish: that host cannot by itself prove the independent M1/M2/M4
queues are complete.
MagicPIG-64K is assigned to machine 2; it remains an opt-in recovery
lane on machine 3 because its released CPU kernel benefits from AVX-512/BF16.
Before launching, the script idempotently builds/verifies all 13 tasks x 100
examples at both 32K and 64K for each tokenizer, then preflights both methods
for both models. It refuses a busy GPU pool rather than killing the
owner's reservation. Once it has accepted an idle pool, an exit trap restores
`serve_122B.sh` after successful completion, setup/runtime failure, or an
interrupt.

Launch it only after explicitly stopping the owner's reservation process; the
launcher deliberately refuses to kill an existing GPU process.  The result
directory, queue and state are resumable, so rerunning the same command does
not repeat audited cells:

```bash
BASE=/home/zhufangzhou/workspace/sheruifeng/baonn/baonn
CODE=$BASE/shadowkv-research
cd "$CODE"
git pull --ff-only origin shadowkv-paris-retro-streaming
tmux new-session -d -s upstream_kv_m3 \
  "cd '$CODE' && bash repro/shadowkv/run_m3_upstream_kv_32_64.sh \
   > '$BASE/results_upstream_kv_matched_x32_l32_20260912/driver_m3.log' 2>&1"
```

When machine 2 runs the 26 MagicPIG-64K cells, set
`M3_UPSTREAM_ACCURACY_METHODS=infllm` on the machine-3 launcher. Duplicate
MagicPIG accuracy work is suppressed. Machine 2 is launched by
`run_m2_magicpig_64.sh`; it never schedules InfLLM-64K because the matched
24GB-card smoke above fails during prefill.

Only after the merged compact report proves 156/156 accuracy cells complete,
zero failures and zero duplicate inputs, stop the holder and rerun the same
resumable launcher with `M3_ENABLE_RUNTIME=1`. It skips the 26 completed local
cells, then measures all 12 common-forward rows plus four separately labelled
PQCache-native rows:

```bash
M3_ENABLE_RUNTIME=1 bash repro/shadowkv/run_m3_upstream_kv_32_64.sh
```

Inspect the live process, per-cell state and partial output without modifying
the pool:

```bash
BASE=/home/zhufangzhou/workspace/sheruifeng/baonn/baonn
ROOT=$BASE/results_upstream_kv_matched_x32_l32_20260912
tmux ls | grep upstream_kv_m3
pgrep -af 'run_upstream_kv_latency|run_pool.sh|run_cell.sh'
for state in done running failed; do
  printf '%s=' "$state"
  find "$ROOT/.state_m3/$state" -maxdepth 1 -type f 2>/dev/null | wc -l
done
tail -n 30 "$ROOT/driver_m3.log"
```

When the launcher has accepted an idle pool, its exit trap starts the owner's
`serve_122B.sh` reservation on success, failure, or interruption.  If it exits
with status 2, the pool was never accepted and the existing reservation was
left untouched.

## Runtime table

Run one fresh process and one method on one otherwise-idle GPU.  Use one real
RULER prompt, not random token IDs.  Report:

1. end-to-end prefill time, including summary/index construction and transfers;
   the timer stops only after each cache's readiness barrier, so asynchronous
   work cannot leak into the discarded first decode step;
2. steady-state decode p50, mean and p95 after discarding the first lazy/JIT
   step; the clean matrix measures 128 subsequent steps so it crosses one
   complete 128-token InfLLM memory unit and sixteen 8-token streaming seals;
   maximum latency is also reported so periodic maintenance is not hidden by
   the median;
3. peak CUDA allocated and reserved memory;
4. peak host RSS and, where available, pinned-memory use;
5. realized active/sample count and index/metadata footprint.

The rendered runtime table carries the realized active mean/cap, observed
maximum, and the router/index configuration beside latency.  This is needed
especially for MagicPIG: `B` is a stochastic target, so its realized mean and
maximum cannot be replaced by the nominal target when comparing runtime.

Accuracy completes before the clean runtime matrix starts. Warm-up and measured passes use the same prompt.  Common-forward runtime and
author-native runtime are different lanes.  A native kernel result is not mixed
with a PyTorch-reference result without an explicit label.  The launcher is
row-resumable: an existing real-text row is skipped only when it contains at
least 128 measured decode steps; older 32/64-step diagnostics are rerun.

## Commands

Distributed result directories remain on their originating hosts.  Do not use
the SSH control channel to copy raw generations or traffic artifacts.  Export a
compact, provenance-bearing status summary on each host and merge only those
small JSON files:

```bash
$PY repro/shadowkv/report_upstream_kv_campaign.py \
  --expected-samples 100 --require-matched-exact-regions \
  --export-summary-json "$SHADOWKV_RESULTS_ROOT" > host-summary.json

$PY repro/shadowkv/report_upstream_kv_campaign.py \
  --expected-samples 100 --require-matched-exact-regions \
  --summary-input m1-summary.json --summary-input m2-summary.json \
  --summary-input m3-summary.json --summary-input m4-summary.json
```

The summary contains only the final scalar score, score count, active-token
accounting and provenance fields for each cell.  It never contains prompts,
generated text, KV tensors or per-step traffic traces.  Final strict reporting
adds `--require-complete` and the clean runtime JSONL file.

```bash
source repro/shadowkv/env_m1.sh
repro/shadowkv/setup_upstream_kv_baselines.sh

$PY repro/shadowkv/gen_upstream_kv_queue.py > /tmp/upstream-kv.queue
NUM_SAMPLES=100 SHADOWKV_POOL_GPUS=1,2 \
  repro/shadowkv/run_pool.sh /tmp/upstream-kv.queue /tmp/upstream-kv-state

$PY repro/shadowkv/status_upstream_kv_campaign.py "$SHADOWKV_RESULTS_ROOT" \
  --state /tmp/upstream-kv-state
```
