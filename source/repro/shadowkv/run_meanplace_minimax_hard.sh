#!/usr/bin/env bash
# Mean Jensen-gap placement + global minimax center allocation.
#
# This is a deliberately narrow factorial: query-mean GQA, block size 8,
# adaptive r in [1,8], no token reranking, and mean center budgets supplied by
# MEAN_MINIMAX_CENTERS.  Machine-specific task/length subsets let independent
# hosts share the campaign without a shared filesystem.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${MEAN_MINIMAX_ROOT:-${SHADOWKV_RESULTS_ROOT%/}/meanplace_minimax_hard_20260910}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
read -ra tasks <<< "${MEAN_MINIMAX_TASKS:-cwe qa_1 qa_2 niah_multikey_3}"
read -ra lengths <<< "${MEAN_MINIMAX_LENGTHS:-32768 131072}"
read -ra centers <<< "${MEAN_MINIMAX_CENTERS:-1.25 1.75}"
GPUS=${SHADOWKV_POOL_GPUS:-1,2,3,4,5,6,7}

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#tasks[@]} * ${#lengths[@]} * ${#centers[@]} ))
  local index rem task length center budget key
  exec 8>"$STATE/claim.lock"; flock 8
  index=0
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    length=${lengths[$(( rem % ${#lengths[@]} ))]}; rem=$(( rem / ${#lengths[@]} ))
    center=${centers[$rem]}
    case "$length" in
      32768) budget=1024 ;;
      131072) budget=4096 ;;
      *) echo "unsupported length $length; expected 32768 or 131072" >&2; return 2 ;;
    esac
    key="${task}_l${length}_b${budget}_r${center}_oneshot"
    # A failed cell is terminal for this invocation.  Retrying it blindly can
    # hide an OOM/host error behind an infinite loop and starve later cells.
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ] \
       && [ ! -e "$STATE/failed/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s %s\n' "$key" "$task" "$length" "$budget" "$center"
      return 0
    fi
    index=$((index + 1))
  done
  return 1
}

gpu_free() {
  local output
  output=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
  [[ "$output" =~ ^[[:space:]]*$ ]]
}

worker() {
  local gpu=$1 job key task length budget center extra offload backend log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task length budget center <<< "$job"
    extra=$($PY -c "print(float('$center') - 1.0)")
    if [ "$length" = 131072 ]; then
      offload=1; backend=uva
    else
      offload=0; backend=auto
    fi
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION="$extra"
      export STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=self
      export SHADOWKV_SELF_LSE_COST=mean_gap
      export SHADOWKV_CENTER_ALLOCATION=self_lse
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
      export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      export STREAMING_OFFLOAD="$offload" STREAMING_GATHER_BACKEND="$backend"
      "$HERE/run_cell.sh" qwen3 "$length" "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean \
        "$budget" 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
    ) >"$log" 2>&1; then
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
echo "cells=$(( ${#tasks[@]} * ${#lengths[@]} * ${#centers[@]} )) samples=$NUM_SAMPLES tasks=${tasks[*]} lengths=${lengths[*]} centers=${centers[*]} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "mean-placement/minimax-allocation campaign complete"
