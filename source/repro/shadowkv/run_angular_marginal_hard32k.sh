#!/usr/bin/env bash
# Controlled test of cosine-driven center allocation.  Placement, router,
# budgets and decoding match the robust-trimmed/tail-CVaR hard-32K baseline;
# only the cross-block allocation objective changes to angular-radius decrease.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi

ROOT=${ANGULAR_MARGINAL_ROOT:-/storage/baonn/angular_marginal_hard32k_20260909}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${SHADOWKV_POOL_GPUS:-0,1,2,3,4,5,6,7}
TOTAL_SLOTS=${ANGULAR_MARGINAL_TOTAL_SLOTS:-1}
SLOT_START=${ANGULAR_MARGINAL_SLOT_START:-0}
SLOT_COUNT=${ANGULAR_MARGINAL_SLOT_COUNT:-1}
if (( SLOT_START < 0 || SLOT_COUNT < 1 || SLOT_START + SLOT_COUNT > TOTAL_SLOTS )); then
  echo "invalid slot range: start=$SLOT_START count=$SLOT_COUNT total=$TOTAL_SLOTS" >&2
  exit 2
fi

tasks=(cwe qa_1 qa_2 niah_multikey_3)
centers=(1.25 1.5)
refinements=(oneshot rerank2b)

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#tasks[@]} * ${#centers[@]} * ${#refinements[@]} ))
  local index=0 task center refinement key rem
  exec 8>"$STATE/claim.lock"; flock 8
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    center=${centers[$(( rem % ${#centers[@]} ))]}; rem=$(( rem / ${#centers[@]} ))
    refinement=${refinements[$(( rem % ${#refinements[@]} ))]}
    key="${task}_b512_r${center}_${refinement}"
    if (( index % TOTAL_SLOTS >= SLOT_START && index % TOTAL_SLOTS < SLOT_START + SLOT_COUNT )); then
      if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
        : > "$STATE/running/$key"
        printf '%s %s %s %s\n' "$key" "$task" "$center" "$refinement"
        return 0
      fi
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
      export SHADOWKV_CENTER_PLACEMENT=robust_trimmed
      export SHADOWKV_CENTER_ALLOCATION=angular_marginal
      export SHADOWKV_ROBUST_QUERY_SOURCE=self_k
      export SHADOWKV_ROBUST_TRIM_FRACTION=0.25
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=16
      if [ "$refinement" = rerank2b ]; then
        export STREAMING_REFINE_FACTOR=2 STREAMING_REFINE_TOKENS=1
      else
        export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      fi
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        adaptive_centroid_lse_streaming_prefix4 512 160 8 16 0 \
        "$NUM_SAMPLES" "$gpu" max
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
echo "cells=16 samples=$NUM_SAMPLES slots=${SLOT_START}..$((SLOT_START+SLOT_COUNT-1))/$TOTAL_SLOTS gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "angular-marginal hard-32K campaign complete"
