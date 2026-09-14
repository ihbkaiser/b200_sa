#!/usr/bin/env bash
# Compare the current router, mean-placement/minimax, and official ParisKV
# at the common 1/32 budget on the four hard RULER tasks.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env_m1.sh"
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${COMPARE32_ROOT:-/storage/baonn/compare32k_default_mean_paris_20260910}
STATE="$ROOT/.state"
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${COMPARE32_GPUS:-3,4,5,6}
TASKS=(cwe qa_1 qa_2 niah_multikey_3)
METHODS=(default mean paris)

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#TASKS[@]} * ${#METHODS[@]} )) index task method key
  exec 8>"$STATE/claim.lock"; flock 8
  for ((index=0; index<total; index++)); do
    method=${METHODS[$(( index % ${#METHODS[@]} ))]}
    task=${TASKS[$(( index / ${#METHODS[@]} ))]}
    key="${task}_${method}"
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ] \
       && [ ! -e "$STATE/failed/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s\n' "$key" "$task" "$method"
      return 0
    fi
  done
  return 1
}

gpu_free() {
  local output
  output=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
  [[ "$output" =~ ^[[:space:]]*$ ]]
}

run_one() {
  local gpu=$1 task=$2 variant=$3
  export SHADOWKV_RESULTS_ROOT="$ROOT"
  export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
  export STREAMING_OFFLOAD=0 STREAMING_GATHER_BACKEND=auto
  export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  case "$variant" in
    default)
      export ADAPTIVE_LSE_EXTRA_FRACTION=0.25 STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=self SHADOWKV_SELF_LSE_COST=max_gap
      export SHADOWKV_CENTER_ALLOCATION=tail_cvar
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export SHADOWKV_TAIL_CVAR_FRACTION=0.25
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0.25
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean \
        1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
      ;;
    mean)
      export ADAPTIVE_LSE_EXTRA_FRACTION=0.25 STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=self SHADOWKV_SELF_LSE_COST=mean_gap
      export SHADOWKV_CENTER_ALLOCATION=self_lse
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean \
        1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
      ;;
    paris)
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        pariskv_author_common 1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
      ;;
  esac
}

worker() {
  local gpu=$1 job key task variant log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task variant <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (run_one "$gpu" "$task" "$variant") >"$log" 2>&1; then
      mv "$STATE/running/$key" "$STATE/done/$key"
    else
      mv "$STATE/running/$key" "$STATE/failed/$key"
      echo "[gpu$gpu] FAILED $key -- $log"
    fi
  done
}

IFS=',' read -ra gpu_list <<< "$GPUS"
pids=()
for gpu in "${gpu_list[@]}"; do worker "$gpu" & pids+=("$!"); done
echo "cells=12 samples=$NUM_SAMPLES budget=1024 tasks=${TASKS[*]} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "32K three-method comparison complete"
