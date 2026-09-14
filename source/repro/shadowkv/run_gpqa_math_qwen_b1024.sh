#!/usr/bin/env bash
# Qwen reasoning campaign for the three directly comparable streaming methods.
#
# Usage (after sourcing env_m1.sh):
#   run_gpqa_math_qwen_b1024.sh gpqa 1 /storage/baonn/results-root
#   run_gpqa_math_qwen_b1024.sh math500 2 /storage/baonn/results-root
#
# Each task first runs a two-example smoke for every method.  A full task is
# opened only if all three smoke cells exit successfully.
set -euo pipefail

: "${CODE:?source repro/shadowkv/env_m1.sh first}"
: "${PY:?source repro/shadowkv/env_m1.sh first}"
: "${PARISKV_AUTHOR_ROOT:?source repro/shadowkv/env_m1.sh first}"

TASK=${1:?task must be gpqa or math500}
GPU=${2:?GPU id is required}
BASE_ROOT=${3:?persistent result root is required}

case "$TASK" in
  gpqa)
    MAX_NEW_TOKENS=16384
    ;;
  math500)
    MAX_NEW_TOKENS=4096
    ;;
  *)
    echo "unsupported task: $TASK (expected gpqa or math500)" >&2
    exit 2
    ;;
esac

methods=(
  adaptive_centroid_lse_streaming_prefix4_querymean
  pariskv_author_common
  quest_streaming
)

export STREAMING_OFFLOAD=1
export STREAMING_GATHER_BACKEND=uva
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32
export UPSTREAM_MATCHED_EXACT_REGIONS=1
export SHADOWKV_MAX_NEW_TOKENS=$MAX_NEW_TOKENS
export SHADOWKV_GENERATION_TEMPERATURE=0.6
# temp .6 / top-p .9 / no top-k is the protocol the predecessor project ran every
# reasoning benchmark under (s2-ttt repro/reasoning/run_reasoning_pool.sh). This
# script briefly used .95/20; numbers produced under that are not comparable to
# AIME or to any earlier reasoning table.
export SHADOWKV_GENERATION_TOP_P=0.9
export SHADOWKV_GENERATION_TOP_K=-1
export SHADOWKV_GENERATION_SEED=0

mkdir -p "$BASE_ROOT/_logs"

run_phase() {
  local phase=$1 samples=$2 method log
  export SHADOWKV_RESULTS_ROOT="$BASE_ROOT/$phase"
  for method in "${methods[@]}"; do
    log="$BASE_ROOT/_logs/${TASK}_${phase}_${method}_gpu${GPU}.log"
    echo "$(date -Is) START task=$TASK phase=$phase method=$method gpu=$GPU" | tee "$log"
    "$CODE/repro/shadowkv/run_cell.sh" \
      qwen3 32768 "$TASK" "$method" 1024 160 8 8 0 "$samples" "$GPU" \
      >>"$log" 2>&1
    echo "$(date -Is) DONE task=$TASK phase=$phase method=$method gpu=$GPU" | tee -a "$log"
  done
}

run_phase smoke 2
touch "$BASE_ROOT/${TASK}_SMOKE_OK"
run_phase full -1
touch "$BASE_ROOT/${TASK}_COMPLETE"

