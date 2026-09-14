#!/usr/bin/env bash
# Promotion grid for the density-weighted center allocator on hard 128K RULER.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi

ROOT=${DENSITY_TAIL_128_ROOT:-/storage/baonn/density_tailcvar_hard128k_20260908}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-30}
GPUS=${SHADOWKV_POOL_GPUS:-1,2,3,4,5,6,7}
CENTER_ALLOCATION=${CENTER_ALLOCATION:-density_tail_cvar}
DENSITY_WEIGHT_MODE=${DENSITY_WEIGHT_MODE:-symmetric}
DENSITY_SHRINKAGE=${DENSITY_SHRINKAGE:-0.5}
DENSITY_HEAD_TRUST_POWER=${DENSITY_HEAD_TRUST_POWER:-0}
DENSITY_VALUE_POWER=${DENSITY_VALUE_POWER:-0}
DENSITY_SIGNAL_MODE=${DENSITY_SIGNAL_MODE:-contrast}
if [[ "$DENSITY_SIGNAL_MODE" != contrast && "$DENSITY_SIGNAL_MODE" != absolute_contrast && "$DENSITY_SIGNAL_MODE" != absolute_max ]]; then
  echo "DENSITY_SIGNAL_MODE must be contrast, absolute_contrast, or absolute_max" >&2
  exit 2
fi
if [[ "$CENTER_ALLOCATION" != density_tail_cvar && "$CENTER_ALLOCATION" != tail_cvar ]]; then
  echo "CENTER_ALLOCATION must be density_tail_cvar or tail_cvar" >&2
  exit 2
fi
if [[ "$DENSITY_WEIGHT_MODE" != symmetric && "$DENSITY_WEIGHT_MODE" != up_only ]]; then
  echo "DENSITY_WEIGHT_MODE must be symmetric or up_only" >&2
  exit 2
fi
TOTAL_SLOTS=${DENSITY_TAIL_TOTAL_SLOTS:-1}
SLOT_START=${DENSITY_TAIL_SLOT_START:-0}
SLOT_COUNT=${DENSITY_TAIL_SLOT_COUNT:-1}
if (( SLOT_START < 0 || SLOT_COUNT < 1 || SLOT_START + SLOT_COUNT > TOTAL_SLOTS )); then
  echo "invalid slot range: start=$SLOT_START count=$SLOT_COUNT total=$TOTAL_SLOTS" >&2
  exit 2
fi

tasks=(cwe niah_multikey_3)
budgets=(512 2048 4096)
centers=(1.25 1.5)
refinements=(oneshot rerank2b)

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

claim() {
  local total=$(( ${#tasks[@]} * ${#budgets[@]} * ${#centers[@]} * ${#refinements[@]} ))
  local index task budget center refinement key rem
  exec 8>"$STATE/claim.lock"; flock 8
  # Marker-backed rescanning permits disjoint slots to start or resume at
  # different times without a leading slot consuming a shared cursor.
  index=0
  while (( index < total )); do
    rem=$index
    task=${tasks[$(( rem % ${#tasks[@]} ))]}; rem=$(( rem / ${#tasks[@]} ))
    budget=${budgets[$(( rem % ${#budgets[@]} ))]}; rem=$(( rem / ${#budgets[@]} ))
    center=${centers[$(( rem % ${#centers[@]} ))]}; rem=$(( rem / ${#centers[@]} ))
    refinement=${refinements[$(( rem % ${#refinements[@]} ))]}
    key="${task}_b${budget}_r${center}_${refinement}"
    if (( index % TOTAL_SLOTS < SLOT_START || index % TOTAL_SLOTS >= SLOT_START + SLOT_COUNT )); then
      index=$(( index + 1)); continue
    fi
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s %s\n' "$key" "$task" "$budget" "$center" "$refinement"
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
  local gpu=$1 job key task budget center refinement extra log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task budget center refinement <<< "$job"
    extra=$($PY -c "print(float('$center') - 1.0)")
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION="$extra"
      export STREAMING_MAX_COMPONENTS=8
      export SHADOWKV_CENTER_PLACEMENT=robust_trimmed
      export SHADOWKV_CENTER_ALLOCATION="$CENTER_ALLOCATION"
      export SHADOWKV_ROBUST_TRIM_FRACTION=0.25
      export SHADOWKV_TAIL_CVAR_FRACTION=0.25
      export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
      export SHADOWKV_DENSITY_PAGE_TOKENS=1024
      export SHADOWKV_DENSITY_REFERENCE_COUNT=64
      export SHADOWKV_DENSITY_KAPPA=8
      export SHADOWKV_DENSITY_SHRINKAGE="$DENSITY_SHRINKAGE"
      export SHADOWKV_DENSITY_HEAD_TRUST_POWER="$DENSITY_HEAD_TRUST_POWER"
      export SHADOWKV_DENSITY_VALUE_POWER="$DENSITY_VALUE_POWER"
      export SHADOWKV_DENSITY_WEIGHT_MODE="$DENSITY_WEIGHT_MODE"
      export SHADOWKV_DENSITY_SIGNAL_MODE="$DENSITY_SIGNAL_MODE"
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
      export STREAMING_OFFLOAD=1 STREAMING_GATHER_BACKEND=uva
      if [ "$refinement" = rerank2b ]; then
        export STREAMING_REFINE_FACTOR=2 STREAMING_REFINE_TOKENS=1
      else
        export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      fi
      "$HERE/run_cell.sh" qwen3 131072 "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean "$budget" 160 8 16 0 \
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
echo "cells=24 samples=$NUM_SAMPLES allocation=$CENTER_ALLOCATION density_signal_mode=$DENSITY_SIGNAL_MODE density_weight_mode=$DENSITY_WEIGHT_MODE density_shrinkage=$DENSITY_SHRINKAGE density_head_trust_power=$DENSITY_HEAD_TRUST_POWER density_value_power=$DENSITY_VALUE_POWER slots=${SLOT_START}..$((SLOT_START+SLOT_COUNT-1))/$TOTAL_SLOTS gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "density-tail hard-128K campaign complete"
