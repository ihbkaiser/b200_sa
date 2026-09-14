#!/usr/bin/env bash
# Sweep a task-blind absolute price per center.  Unlike the fixed-rbar
# campaigns, the realized mean number of centers is an output, not an input.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${FREE_RD_ROOT:-${SHADOWKV_RESULTS_ROOT%/}/free_rate_adaptive_32k_20260910}
STATE="$ROOT/.state"
NUM_SAMPLES=${NUM_SAMPLES:-5}
read -ra TASKS <<< "${FREE_RD_TASKS:-cwe qa_1 qa_2 niah_multikey_3}"
read -ra PENALTIES <<< "${FREE_RD_PENALTIES:-0.001 0.003 0.01 0.03 0.1 0.3 1.0}"
GPUS=${FREE_RD_GPUS:-$SHADOWKV_POOL_GPUS}
ALLOCATOR=${FREE_RD_ALLOCATOR:-tail_absolute_rate_distortion}

mkdir -p "$STATE"/{done,running,failed} "$ROOT"/{_logs,_allocation}
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#TASKS[@]} * ${#PENALTIES[@]} ))
  local index rem task penalty key
  exec 8>"$STATE/claim.lock"; flock 8
  for ((index=0; index<total; index++)); do
    rem=$index
    penalty=${PENALTIES[$(( rem % ${#PENALTIES[@]} ))]}
    rem=$(( rem / ${#PENALTIES[@]} ))
    task=${TASKS[$rem]}
    key="${task}_lambda${penalty}"
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ] \
       && [ ! -e "$STATE/failed/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s\n' "$key" "$task" "$penalty"
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
  local gpu=$1 task=$2 penalty=$3 key=$4
  export SHADOWKV_RESULTS_ROOT="$ROOT"
  export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
  export STREAMING_OFFLOAD=0 STREAMING_GATHER_BACKEND=auto
  export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # No fixed quota.  Keeping x=0 in the filename makes that explicit; packed
  # storage grows on demand to the order selected by the Lagrangian.
  export ADAPTIVE_LSE_EXTRA_FRACTION=0 STREAMING_MAX_COMPONENTS=8
  export SHADOWKV_CENTER_PLACEMENT=self
  export SHADOWKV_SELF_LSE_COST=max_gap
  export SHADOWKV_CENTER_ALLOCATION="$ALLOCATOR"
  if [ "$ALLOCATOR" = tail_distortion_threshold ]; then
    export SHADOWKV_DISTORTION_TOLERANCE="$penalty"
  elif [ "$ALLOCATOR" = tail_group_distortion_target ]; then
    export SHADOWKV_GROUP_DISTORTION_TARGET="$penalty"
  elif [ "$ALLOCATOR" = tail_demand_adaptive_rate_distortion ]; then
    # In a demand-adaptive sweep, the queue value is the layer-demand cutoff;
    # the conservative/aggressive prices are campaign-level constants.
    export SHADOWKV_DEMAND_THRESHOLD="$penalty"
  else
    export SHADOWKV_ABSOLUTE_RD_PENALTY="$penalty"
  fi
  export SHADOWKV_TAIL_CVAR_FRACTION=0.25
  export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0.25
  export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
  export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
  export STREAMING_ALLOCATION_TRACE_OUT="$ROOT/_allocation/${key}.jsonl"
  "$HERE/run_cell.sh" qwen3 32768 "$task" \
    adaptive_centroid_lse_streaming_prefix4_querymean \
    1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
}

worker() {
  local gpu=$1 job key task penalty log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task penalty <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (run_one "$gpu" "$task" "$penalty" "$key") >"$log" 2>&1; then
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
echo "cells=$(( ${#TASKS[@]} * ${#PENALTIES[@]} )) samples=$NUM_SAMPLES allocator=$ALLOCATOR tasks=${TASKS[*]} penalties=${PENALTIES[*]} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "free-rate 32K campaign complete"
