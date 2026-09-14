#!/usr/bin/env bash
# Distributed 128K sweep for the two center objectives at rbar={1.75,2.5}.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${CENTER128_ROOT:-${SHADOWKV_RESULTS_ROOT%/}/center_sweep128k_hard_20260910}
STATE="$ROOT/.state"
NUM_SAMPLES=${NUM_SAMPLES:-30}
read -ra TASKS <<< "${CENTER128_TASKS:-cwe qa_1 qa_2 niah_multikey_3}"
METHODS=(default mean)
CENTERS=(1.75 2.5)
GPUS=${CENTER128_GPUS:-$SHADOWKV_POOL_GPUS}

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#TASKS[@]} * ${#METHODS[@]} * ${#CENTERS[@]} ))
  local index rem task variant center key
  exec 8>"$STATE/claim.lock"; flock 8
  for ((index=0; index<total; index++)); do
    rem=$index
    variant=${METHODS[$(( rem % ${#METHODS[@]} ))]}; rem=$(( rem / ${#METHODS[@]} ))
    center=${CENTERS[$(( rem % ${#CENTERS[@]} ))]}; rem=$(( rem / ${#CENTERS[@]} ))
    task=${TASKS[$rem]}
    key="${task}_${variant}_r${center}"
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ] \
       && [ ! -e "$STATE/failed/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s\n' "$key" "$task" "$variant" "$center"
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
  local gpu=$1 task=$2 variant=$3 center=$4 extra
  extra=$($PY -c "print(float('$center') - 1.0)")
  export SHADOWKV_RESULTS_ROOT="$ROOT"
  export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
  export STREAMING_OFFLOAD=1 STREAMING_GATHER_BACKEND=uva
  export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export ADAPTIVE_LSE_EXTRA_FRACTION="$extra" STREAMING_MAX_COMPONENTS=8
  export SHADOWKV_CENTER_PLACEMENT=self
  export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
  export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
  case "$variant" in
    default)
      export SHADOWKV_SELF_LSE_COST=max_gap
      export SHADOWKV_CENTER_ALLOCATION=tail_cvar
      export SHADOWKV_TAIL_CVAR_FRACTION=0.25
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0.25
      ;;
    mean)
      export SHADOWKV_SELF_LSE_COST=mean_gap
      export SHADOWKV_CENTER_ALLOCATION=self_lse
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0
      ;;
  esac
  "$HERE/run_cell.sh" qwen3 131072 "$task" \
    adaptive_centroid_lse_streaming_prefix4_querymean \
    4096 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
}

worker() {
  local gpu=$1 job key task variant center log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task variant center <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (run_one "$gpu" "$task" "$variant" "$center") >"$log" 2>&1; then
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
echo "cells=$(( ${#TASKS[@]} * 4 )) samples=$NUM_SAMPLES tasks=${TASKS[*]} centers=${CENTERS[*]} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "128K center sweep complete"
