#!/usr/bin/env bash
# Fill the nine missing RULER tasks for every row in the reported 32K table.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(hostname -s)" in
  sepc810*) source "$HERE/env_m2.sh" ;;
  sashimi*) source "$HERE/env_m4.sh" ;;
  *) source "$HERE/env_m1.sh" ;;
esac
export CODE=$(cd "$HERE/../.." && pwd)
export SHADOWKV_DIR="$CODE/ShadowKV"

WAIT_ROOT=${FULL32_WAIT_ROOT:-}
[ -n "$WAIT_ROOT" ] || { echo "FULL32_WAIT_ROOT is required" >&2; exit 2; }
ROOT=${FULL32_ROOT:-}
if [ -z "$ROOT" ]; then
  ROOT="${SHADOWKV_RESULTS_ROOT%/}/full_tasks32k_20260910"
fi
STATE="$ROOT/.state"
NUM_SAMPLES=${NUM_SAMPLES:-30}
TASK_SPEC=${FULL32_TASKS:-}
[ -n "$TASK_SPEC" ] || { echo "FULL32_TASKS is required" >&2; exit 2; }
read -ra TASKS <<< "$TASK_SPEC"
GPUS=${FULL32_GPUS:-$SHADOWKV_POOL_GPUS}
CONFIGS=(default:1.25 mean:1.25 paris:0 default:1.5 mean:1.5 \
         default:1.75 mean:1.75 default:2.0 mean:2.0 default:2.5 mean:2.5)

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "pool already running: $STATE"; exit 1; }

expected_128=${FULL32_WAIT_CELLS:-0}
if (( expected_128 <= 0 )); then
  expected_128=${#TASKS[@]}
fi
while true; do
  done_n=$(find "$WAIT_ROOT/.state/done" -type f 2>/dev/null | wc -l)
  failed_n=$(find "$WAIT_ROOT/.state/failed" -type f 2>/dev/null | wc -l)
  running_n=$(find "$WAIT_ROOT/.state/running" -type f 2>/dev/null | wc -l)
  if (( done_n + failed_n >= expected_128 && running_n == 0 )); then break; fi
  echo "waiting for local 128K sweep: done=$done_n failed=$failed_n running=$running_n expected=$expected_128"
  sleep 30
done

claim() {
  local total=$(( ${#TASKS[@]} * ${#CONFIGS[@]} )) index task config variant center key
  exec 8>"$STATE/claim.lock"; flock 8
  for ((index=0; index<total; index++)); do
    config=${CONFIGS[$(( index % ${#CONFIGS[@]} ))]}
    task=${TASKS[$(( index / ${#CONFIGS[@]} ))]}
    variant=${config%%:*}; center=${config#*:}
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
  export SHADOWKV_RESULTS_ROOT="$ROOT"
  export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
  export STREAMING_OFFLOAD=0 STREAMING_GATHER_BACKEND=auto
  export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if [ "$variant" = paris ]; then
    "$HERE/run_cell.sh" qwen3 32768 "$task" \
      pariskv_author_common 1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
    return
  fi

  extra=$($PY -c "print(float('$center') - 1.0)")
  export ADAPTIVE_LSE_EXTRA_FRACTION="$extra" STREAMING_MAX_COMPONENTS=8
  export SHADOWKV_CENTER_PLACEMENT=self
  export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
  export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=8
  if [ "$variant" = default ]; then
    export SHADOWKV_SELF_LSE_COST=max_gap SHADOWKV_CENTER_ALLOCATION=tail_cvar
    export SHADOWKV_TAIL_CVAR_FRACTION=0.25 SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0.25
  else
    export SHADOWKV_SELF_LSE_COST=mean_gap SHADOWKV_CENTER_ALLOCATION=self_lse
    export SHADOWKV_TAIL_GAP_CORRECTION_SCALE=0
  fi
  "$HERE/run_cell.sh" qwen3 32768 "$task" \
    adaptive_centroid_lse_streaming_prefix4_querymean \
    1024 160 8 16 0 "$NUM_SAMPLES" "$gpu" max
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
echo "cells=$(( ${#TASKS[@]} * ${#CONFIGS[@]} )) samples=$NUM_SAMPLES tasks=${TASKS[*]} gpus=$GPUS root=$ROOT"
wait "${pids[@]}"
echo "full 32K task fill complete"
