#!/usr/bin/env bash
# Matched Llama end-to-end latency cells for the four sparse methods on real
# RULER prompts. Run one process at a time on an otherwise idle GPU.
set -euo pipefail

: "${SHADOWKV_DIR:?source the machine environment first}"
: "${PARISKV_AUTHOR_ROOT:?set PARISKV_AUTHOR_ROOT to the ParisKV checkout}"

GPU=${1:?usage: run_latency_typical_32_64.sh <gpu> <cell>...}
shift
[ "$#" -gt 0 ] || { echo "provide cells as <length>:<task>" >&2; exit 2; }

ROOT=${LATENCY_ROOT:-${SHADOWKV_RESULTS_ROOT%/}/latency_typical_32_64_20260911}
mkdir -p "$ROOT/logs"
OUT="$ROOT/results_gpu${GPU}.jsonl"

run_common() {
  local length=$1 task=$2 method=$3 budget=$4
  shift 4
  local key="llama32_${length}_${task}_${method}_b${budget}"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/repro/shadowkv/bench_latency.py" \
    --model_key llama32 --datalen "$length" --ruler_task "$task" --real_text \
    --method "$method" --sparse_budget "$budget" \
    --prefix_tokens 32 --recent_tokens 32 \
    --prefill_reps 1 --warmup_passes 1 --decode_steps 33 \
    --out "$OUT" "$@" 2>&1 | tee "$ROOT/logs/${key}.log"
}

for cell in "$@"; do
  length=${cell%%:*}
  task=${cell#*:}
  case "$length" in
    32768) budget=1024; outliers=24 ;;
    65536) budget=2048; outliers=48 ;;
    *) echo "unsupported length: $length" >&2; exit 2 ;;
  esac

  # Production settings shared with the 100-sample quality campaign.
  SHADOWKV_CENTER_PLACEMENT=self \
  SHADOWKV_CENTER_ALLOCATION=tail_absolute_rate_distortion \
  SHADOWKV_ABSOLUTE_RD_PENALTY=1.5 \
    run_common "$length" "$task" \
      adaptive_centroid_lse_streaming_prefix4_querymean "$budget" \
      --chunk_size 8 --streaming_offload --streaming_gather_backend uva \
      --streaming_router_backend triton

  run_common "$length" "$task" quest_streaming "$budget" \
    --page_size 8 --dense_layers 0 --streaming_offload \
    --streaming_gather_backend uva --streaming_router_backend triton

  run_common "$length" "$task" shadowkv_cpu "$budget" \
    --rank 160 --chunk_size 4 --shadow_outlier_chunks "$outliers"

  run_common "$length" "$task" pariskv_author_common "$budget" \
    --chunk_size 8 --streaming_offload --streaming_gather_backend uva \
    --pariskv_author_root "$PARISKV_AUTHOR_ROOT"
done

echo "LATENCY_SLICE_DONE gpu=$GPU out=$OUT"
