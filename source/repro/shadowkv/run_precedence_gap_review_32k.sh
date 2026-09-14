#!/usr/bin/env bash
# Factorial review of two independent adaptive-centroid choices:
#   (1) concavified marginal water filling versus complete-path
#       Lagrangian rate--distortion bundles;
#   (2) residual Jensen-gap correction gamma=0.25 versus gamma=1.
#
# The scientific controls are the validated self-K r=1..8 path, block 8,
# mean 2.5 centers/block, INT8 centers, qmean GQA, no alpha, and one-shot
# retrieval.  Sharding is deterministic across independent machines.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${PRECEDENCE_GAP_ROOT:-/storage/baonn/precedence_gap_review_32k_20260910}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-5}
GPUS=${SHADOWKV_POOL_GPUS:-0,1,2,3}
TOTAL_SLOTS=${TOTAL_SLOTS:-1}
SLOT_START=${SLOT_START:-0}
SLOT_COUNT=${SLOT_COUNT:-1}

read -ra tasks <<< "${PRECEDENCE_GAP_TASKS:-cwe qa_1 qa_2 niah_multikey_3}"
read -ra budgets <<< "${PRECEDENCE_GAP_BUDGETS:-512 2048}"
# label:allocator:gamma
read -ra configs <<< "${PRECEDENCE_GAP_CONFIGS:-marginal_g025:tail_cvar:0.25 bundle_g025:tail_rate_distortion:0.25 marginal_g1:tail_cvar:1 bundle_g1:tail_rate_distortion:1}"

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }
[ -f "$STATE/next" ] || printf '0\n' > "$STATE/next"

claim() {
  local total=$(( ${#tasks[@]} * ${#budgets[@]} * ${#configs[@]} ))
  local index rem task budget spec label allocator gamma key
  exec 8>"$STATE/claim.lock"; flock 8
  index=$(<"$STATE/next")
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    budget=${budgets[$(( rem % ${#budgets[@]} ))]}; rem=$(( rem / ${#budgets[@]} ))
    spec=${configs[$rem]}; IFS=: read -r label allocator gamma <<< "$spec"
    key="${label}_${task}_b${budget}"
    printf '%s\n' $((index + 1)) > "$STATE/next"
    if (( index % TOTAL_SLOTS < SLOT_START || index % TOTAL_SLOTS >= SLOT_START + SLOT_COUNT )); then
      index=$((index + 1)); continue
    fi
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s %s %s\n' "$key" "$task" "$budget" "$label" "$allocator" "$gamma"
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
  local gpu=$1 job key task budget label allocator gamma log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task budget label allocator gamma <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION=1.5
      export STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=self
      export SHADOWKV_CENTER_ALLOCATION="$allocator"
      export SHADOWKV_SELF_LSE_COST=max_gap
      export ADAPTIVE_LSE_TEMPERATURES=1
      export SHADOWKV_TAIL_CVAR_FRACTION=0.25
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE="$gamma"
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
      export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      export STREAMING_ALLOCATION_OUT="$ROOT/_allocation/${key}.jsonl"
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
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
echo "cells=$(( ${#tasks[@]} * ${#budgets[@]} * ${#configs[@]} )) samples=$NUM_SAMPLES mean_r=2.5 slots=${SLOT_START}/${TOTAL_SLOTS} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "precedence/gap review complete"
