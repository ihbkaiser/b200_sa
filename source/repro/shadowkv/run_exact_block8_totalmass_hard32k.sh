#!/usr/bin/env bash
# Offline ceiling for block-8 routing on the same hard 32K grid as the
# robust-placement campaign.  This oracle scans all keys and ranks complete
# blocks by exact total attention mass.  The rerank2b control first admits 2B
# tokens in exact-mass blocks, then retains the best B individual tokens by
# exact total mass.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi

ROOT=${EXACT_BLOCK8_ROOT:-/storage/baonn/exact_block8_totalmass_hard32k_20260908}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${SHADOWKV_POOL_GPUS:-0,1,2,3,4,5,6,7}
QUERY_GROUP_MEAN=${EXACT_QUERY_GROUP_MEAN:-0}
GROUP_REDUCE=${EXACT_GROUP_REDUCE:-sum}
REFINEMENTS=${EXACT_REFINEMENTS:-"oneshot rerank2b"}
TOTAL_SLOTS=${EXACT_TOTAL_SLOTS:-1}
SLOT_START=${EXACT_SLOT_START:-0}
SLOT_COUNT=${EXACT_SLOT_COUNT:-1}
if (( SLOT_START < 0 || SLOT_COUNT < 1 || SLOT_START + SLOT_COUNT > TOTAL_SLOTS )); then
  echo "invalid slot range: start=$SLOT_START count=$SLOT_COUNT total=$TOTAL_SLOTS" >&2
  exit 2
fi
if [[ "$QUERY_GROUP_MEAN" != 0 && "$QUERY_GROUP_MEAN" != 1 ]]; then
  echo "EXACT_QUERY_GROUP_MEAN must be 0 or 1" >&2
  exit 2
fi
if [[ "$GROUP_REDUCE" != max && "$GROUP_REDUCE" != sum ]]; then
  echo "EXACT_GROUP_REDUCE must be max or sum" >&2
  exit 2
fi

tasks=(cwe qa_1 qa_2 niah_multikey_3)
budgets=(512 2048)
read -r -a refinements <<< "$REFINEMENTS"

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }
if [ ! -f "$STATE/next" ]; then printf '0\n' > "$STATE/next"; fi

claim() {
  local total=$(( ${#tasks[@]} * ${#budgets[@]} * ${#refinements[@]} ))
  local index task budget refinement key rem
  exec 8>"$STATE/claim.lock"; flock 8
  index=$(<"$STATE/next")
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    budget=${budgets[$(( rem % ${#budgets[@]} ))]}; rem=$(( rem / ${#budgets[@]} ))
    refinement=${refinements[$(( rem % ${#refinements[@]} ))]}
    key="${task}_b${budget}_${refinement}"
    printf '%s\n' $(( index + 1 )) > "$STATE/next"
    if (( index % TOTAL_SLOTS < SLOT_START || index % TOTAL_SLOTS >= SLOT_START + SLOT_COUNT )); then
      index=$(( index + 1)); continue
    fi
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s\n' "$key" "$task" "$budget" "$refinement"
      return 0
    fi
    index=$(( index + 1 ))
  done
  return 1
}

gpu_free() {
  local output
  output=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
  [[ "$output" =~ ^[[:space:]]*$ ]]
}

worker() {
  local gpu=$1 job key task budget refinement log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task budget refinement <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export SHADOWKV_EXACT_QUERY_GROUP_MEAN="$QUERY_GROUP_MEAN"
      if [ "$refinement" = rerank2b ]; then
        export STREAMING_REFINE_FACTOR=2 STREAMING_REFINE_TOKENS=1
      else
        export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      fi
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        exact_block_lse_softmax_streaming "$budget" 160 8 16 0 \
        "$NUM_SAMPLES" "$gpu" "$GROUP_REDUCE"
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
echo "cells=$(( ${#tasks[@]} * ${#budgets[@]} * ${#refinements[@]} )) samples=$NUM_SAMPLES gqa=$([[ $QUERY_GROUP_MEAN == 1 ]] && echo qmean || echo $GROUP_REDUCE) refinements=$REFINEMENTS slots=${SLOT_START}..$((SLOT_START+SLOT_COUNT-1))/$TOTAL_SLOTS gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "exact block-8 total-mass campaign complete"
