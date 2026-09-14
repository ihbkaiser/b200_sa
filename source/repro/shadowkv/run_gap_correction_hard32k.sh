#!/usr/bin/env bash
# End-to-end screen for variable-resolution routing-bias correction.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
# env_m1 documents the canonical checkout; this campaign deliberately runs
# the isolated research worktree containing the correction under test.
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

ROOT=${GAP_CORRECTION_ROOT:-/storage/baonn/gap_correction_hard32k_20260909}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${SHADOWKV_POOL_GPUS:-1,2,3,4,5,6,7}
read -ra tasks <<< "${GAP_TASKS:-cwe qa_1 qa_2 niah_multikey_3}"
budgets=(${GAP_BUDGETS:-512})
refinements=(${GAP_REFINEMENTS:-oneshot})
QUERY_GROUP_MEAN=${GAP_QUERY_GROUP_MEAN:-1}
if [[ "$QUERY_GROUP_MEAN" != 0 && "$QUERY_GROUP_MEAN" != 1 ]]; then
  echo "GAP_QUERY_GROUP_MEAN must be 0 or 1" >&2
  exit 2
fi
CENTER_PLACEMENT=${GAP_CENTER_PLACEMENT:-robust_trimmed}
CENTER_ALLOCATION=${GAP_CENTER_ALLOCATION:-tail_cvar}
SELF_LSE_COST=${GAP_SELF_LSE_COST:-max_gap}
SELF_LSE_COST_BETA=${GAP_SELF_LSE_COST_BETA:-4}
EXTRA_FRACTION=${GAP_EXTRA_FRACTION:-0.25}
CENTER_BITS=${GAP_CENTER_BITS:-16}
DISPERSION_CORRECTION=${GAP_DISPERSION_CORRECTION:-0}
MODEL_KEY=${GAP_MODEL_KEY:-qwen3}
DATALEN=${GAP_DATALEN:-32768}
QANCHOR_WEIGHT=${GAP_QANCHOR_WEIGHT:-0.25}
OFFLOAD=${GAP_OFFLOAD:-0}
method=adaptive_centroid_lse_streaming_prefix4
if [ "$QUERY_GROUP_MEAN" = 1 ]; then
  method=${method}_querymean
fi
# name:placement-trim:allocation-tail:score-correction
read -ra configs <<< "${GAP_CONFIGS:-base:0.25:0.25:0 corr25:0.25:0.25:0.25 tuned50:0:0.5:0.5}"

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }
[ -f "$STATE/next" ] || printf '0\n' > "$STATE/next"

claim() {
  local total=$(( ${#tasks[@]} * ${#budgets[@]} * ${#refinements[@]} * ${#configs[@]} ))
  local index rem task budget refinement spec name trim tail correction key
  exec 8>"$STATE/claim.lock"; flock 8
  index=$(<"$STATE/next")
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    budget=${budgets[$(( rem % ${#budgets[@]} ))]}; rem=$(( rem / ${#budgets[@]} ))
    refinement=${refinements[$(( rem % ${#refinements[@]} ))]}; rem=$(( rem / ${#refinements[@]} ))
    spec=${configs[$rem]}; IFS=: read -r name trim tail correction <<< "$spec"
    key="${name}_${task}_b${budget}_${refinement}"
    printf '%s\n' $((index + 1)) > "$STATE/next"
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s %s %s %s %s\n' \
        "$key" "$task" "$budget" "$refinement" "$name" "$trim" "$tail" "$correction"
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
  local gpu=$1 job key task budget refinement name trim tail correction log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task budget refinement name trim tail correction <<< "$job"
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION="$EXTRA_FRACTION"
      export STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT="$CENTER_PLACEMENT"
      export SHADOWKV_CENTER_ALLOCATION="$CENTER_ALLOCATION"
      export SHADOWKV_QANCHOR_WEIGHT="$QANCHOR_WEIGHT"
      export SHADOWKV_SELF_LSE_COST="$SELF_LSE_COST"
      export SHADOWKV_SELF_LSE_COST_BETA="$SELF_LSE_COST_BETA"
      export SHADOWKV_ROBUST_TRIM_FRACTION="$trim"
      export SHADOWKV_TAIL_CVAR_FRACTION="$tail"
      export SHADOWKV_TAIL_GAP_CORRECTION_SCALE="$correction"
      export SHADOWKV_CENTER_DISPERSION_CORRECTION="$DISPERSION_CORRECTION"
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS="$CENTER_BITS"
      export STREAMING_OFFLOAD="$OFFLOAD"
      if [ "$refinement" = rerank2b ]; then
        export STREAMING_REFINE_FACTOR=2 STREAMING_REFINE_TOKENS=1
      else
        export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      fi
      "$HERE/run_cell.sh" "$MODEL_KEY" "$DATALEN" "$task" \
        "$method" "$budget" 160 8 16 0 \
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
MEAN_R=$(awk "BEGIN { print 1 + $EXTRA_FRACTION }")
echo "cells=$(( ${#tasks[@]} * ${#budgets[@]} * ${#refinements[@]} * ${#configs[@]} )) samples=$NUM_SAMPLES model=$MODEL_KEY length=$DATALEN gqa=$([[ $QUERY_GROUP_MEAN == 1 ]] && echo qmean || echo max) placement=$CENTER_PLACEMENT allocation=$CENTER_ALLOCATION mean_r=$MEAN_R center_bits=$CENTER_BITS dispersion=$DISPERSION_CORRECTION gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "gap-correction campaign complete"
