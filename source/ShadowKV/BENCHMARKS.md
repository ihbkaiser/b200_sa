# Local benchmark adapter

`test/eval_acc.py` can evaluate `ruler/<task>`, `longbench/<task>`,
`longbench-v2`, `aime25`, `math500`, and `gpqa[/subset]` without importing
code from a sibling repository.

The generation and scoring contract was checked against
`NVIDIA/kvpress@71640b4` (2026-09-07):

| dataset | rows | decoding | max new tokens | primary metric |
|---|---:|---|---:|---|
| LongBench-v2 | 503 | greedy | 16 | exact A/B/C/D accuracy; also difficulty and length breakdowns |
| AIME25 | 30 | greedy | 32000 | last balanced `boxed{}` exact match |
| MATH-500 | 500 | greedy | 4096 | first balanced `boxed{}` exact match |
| GPQA-Diamond | 198 | greedy | 16384 | extracted A/B/C/D accuracy |

Current upstream KVPress does not register GPQA. The last row is therefore the
workspace's existing zero-shot GPQA-Diamond extension, not an upstream claim.

LongBench-v2 overflow follows the official evaluator and ParisKV: tokenize the
complete rendered prompt and, when it exceeds `--datalen`, retain equal-size
prefix and suffix token spans. This keeps the question and choices at the end;
it is not KVPress's context-prefix truncation. Results at a finite cap must be
reported as `middle-truncated @ <cap>`, not as full-context evaluation. For an
untruncated claim, evaluate and report only rows whose complete prompt fits the
model's native context window, including the retained row count.

Models are always supplied with `--model_name`. The non-RULER datasets are
external and are located through these environment variables:

```bash
export SHADOWKV_LONGBENCH_PATH=/path/to/LongBench/snapshot
export SHADOWKV_LONGBENCH_V2_PATH=/path/to/LongBench-v2
export SHADOWKV_AIME25_PATH=/path/to/aime25/dataset-directory
export SHADOWKV_MATH500_PATH=/path/to/MATH500
export SHADOWKV_GPQA_PATH=/path/to/gpqa/dataset-directory
```

The expected layouts are:

```text
LongBench/<task>/test-*.parquet
LongBench-v2/test-*.parquet
AIME/test-*.parquet
MATH500/test-*.parquet
GPQA/diamond/test-*.parquet
```

RULER is kept inside this checkout at
`data/ruler/data/<model-template>/<length>/<task>/validation.jsonl` because its
files are tokenizer-specific. LongBench, AIME and GPQA use the prompt fields
stored in their parquet files and the self-contained scorers in
`data/benchmark_metrics.py`.

Download the three public processed datasets on each machine (datasets are not
copied between hosts):

```bash
source repro/shadowkv/env_m3.sh  # or env_m1/env_m2/env_m4
$PY repro/shadowkv/prepare_kvpress_benchmarks.py \
  --output-root "$SHADOWKV_BASE/datasets"
```

For GPQA, prepare the licensed source separately with
`repro/gpqa/build_gpqa_dataset.py`, then point `SHADOWKV_GPQA_PATH` at its
output. Validate all adapters without a GPU:

```bash
$PY repro/shadowkv/preflight_benchmark_data.py --skip-ruler
```

Example:

```bash
cd ShadowKV
CUDA_VISIBLE_DEVICES=0 python test/eval_acc.py \
  --model_name /path/to/model --datalen 32768 --method full \
  --dataset_name longbench/narrativeqa --num_samples 100 \
  --out_root /path/to/results
```

The four new benchmark dataset names are passed directly:

```bash
CUDA_VISIBLE_DEVICES=0 $PY ShadowKV/test/eval_acc.py \
  --model_name "$SHADOWKV_QWEN3_PATH" --datalen 131072 --method full \
  --dataset_name longbench-v2 --num_samples -1 --out_root /path/to/results

CUDA_VISIBLE_DEVICES=0 $PY ShadowKV/test/eval_acc.py \
  --model_name "$SHADOWKV_QWEN3_PATH" --datalen 32768 --method full \
  --dataset_name aime25,math500,gpqa/diamond --num_samples -1 \
  --out_root /path/to/results
```

The pool runner uses filesystem-safe task keys `longbench-v2`, `aime25`,
`math500`, and `gpqa-diamond`; bare legacy names still resolve to RULER. Set
`NUM_SAMPLES=-1` for the complete datasets rather than the pool's short default.

Do not pass `--max_new_tokens` unless intentionally overriding KVPress: each
dataset row already carries the value in the table. ShadowKV decoding is greedy
(`temperature=0`). LongBench-v2 stores subgroup metadata in every output row;
report it with:

```bash
$PY repro/shadowkv/summarize_benchmark_jsonl.py /path/to/cell.jsonl
```
