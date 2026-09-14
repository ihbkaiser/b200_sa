#!/usr/bin/env bash
# Unified self-K rate--distortion campaign.
#
# A single parameter-free distortion, mean fractional within-block mass loss,
# drives both the nested 1..8 center path and the Lagrangian rate allocation.
# No future query, trim/CVaR fraction, dispersion correction, or sequential
# cross-block water filling is used.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi

ROOT=${UNIFIED_RD_ROOT:-/storage/baonn/unified_mass_rd_hard32k_20260908}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${SHADOWKV_POOL_GPUS:-0,1,2,3,4,5,6,7}
TOTAL_SLOTS=${UNIFIED_RD_TOTAL_SLOTS:-1}
SLOT_START=${UNIFIED_RD_SLOT_START:-0}
SLOT_COUNT=${UNIFIED_RD_SLOT_COUNT:-1}
REFINEMENTS=${UNIFIED_RD_REFINEMENTS:-oneshot,rerank2b}

tasks=(cwe qa_1 qa_2 niah_multikey_3)
centers=(1.25 1.5)
IFS=',' read -ra refinements <<< "$REFINEMENTS"

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }
if [ ! -f "$STATE/next" ]; then printf '0\n' > "$STATE/next"; fi

claim() {
  local total=$(( ${#tasks[@]} * ${#centers[@]} * ${#refinements[@]} ))
  local index rem task center refinement key
  exec 8>"$STATE/claim.lock"; flock 8
  index=$(<"$STATE/next")
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    center=${centers[$(( rem % ${#centers[@]} ))]}; rem=$(( rem / ${#centers[@]} ))
    refinement=${refinements[$rem]}
    key="${task}_b512_r${center}_${refinement}"
    printf '%s\n' $(( index + 1 )) > "$STATE/next"
    if (( index % TOTAL_SLOTS < SLOT_START || index % TOTAL_SLOTS >= SLOT_START + SLOT_COUNT )); then
      index=$(( index + 1)); continue
    fi
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s\n' "$key" "$task" "$center" "$refinement"
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
  local gpu=$1 job key task center refinement extra log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task center refinement <<< "$job"
    extra=$($PY -c "print(float('$center') - 1.0)")
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION="$extra"
      export STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=self
      export SHADOWKV_CENTER_ALLOCATION=rate_distortion
      export SHADOWKV_SELF_LSE_COST=mean_relative
      export ADAPTIVE_LSE_TEMPERATURES=0.25
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=16
      export STREAMING_ALLOCATION_OUT="$ROOT/_allocation/${key}.jsonl"
      if [ "$refinement" = rerank2b ]; then
        export STREAMING_REFINE_FACTOR=2 STREAMING_REFINE_TOKENS=1
      else
        export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      fi
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean \
        512 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
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
echo "cells=$(( ${#tasks[@]} * ${#centers[@]} * ${#refinements[@]} )) samples=$NUM_SAMPLES gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "unified mass rate--distortion campaign complete"
