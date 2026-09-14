#!/usr/bin/env bash
# One-shot adaptive-center matrix: block size S versus mean center budget rbar.
# All cells use B=512, prefix32/recent32, and no exact reranking.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi

ROOT=${CENTER_BLOCK_MATRIX_ROOT:-/storage/baonn/center_block_matrix_oneshot_20260908}
STATE=$ROOT/.state
NUM_SAMPLES=${NUM_SAMPLES:-10}
GPUS=${SHADOWKV_POOL_GPUS:-0,1,2,3,4,5,6,7}
TOTAL_SLOTS=${MATRIX_TOTAL_SLOTS:-1}
SLOT_START=${MATRIX_SLOT_START:-0}
SLOT_COUNT=${MATRIX_SLOT_COUNT:-1}
if (( SLOT_START < 0 || SLOT_COUNT < 1 || SLOT_START + SLOT_COUNT > TOTAL_SLOTS )); then
  echo "invalid matrix slot range: start=$SLOT_START count=$SLOT_COUNT total=$TOTAL_SLOTS" >&2
  exit 2
fi

tasks=(cwe qa_1 qa_2 niah_multikey_3)
configs=(
  "1 1"
  "2 1" "2 1.25" "2 1.5" "2 2"
  "4 1" "4 1.25" "4 1.5" "4 2" "4 3" "4 4"
  "8 1" "8 1.25" "8 1.5" "8 2" "8 3" "8 4" "8 6" "8 8"
)

mkdir -p "$STATE"/{done,running,failed} "$ROOT/_logs"
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "matrix pool already running: $STATE"; exit 1; }
if [ ! -f "$STATE/next" ]; then printf '0\n' > "$STATE/next"; fi

claim() {
  local index total=$(( ${#tasks[@]} * ${#configs[@]} ))
  exec 8>"$STATE/claim.lock"
  flock 8
  index=$(<"$STATE/next")
  while (( index < total )); do
    local config_index=$(( index / ${#tasks[@]} ))
    local task_index=$(( index % ${#tasks[@]} ))
    local block centers task key
    read -r block centers <<< "${configs[$config_index]}"
    task=${tasks[$task_index]}
    key="${task}_s${block}_r${centers}"
    printf '%s\n' $(( index + 1 )) > "$STATE/next"
    if (( index % TOTAL_SLOTS < SLOT_START || index % TOTAL_SLOTS >= SLOT_START + SLOT_COUNT )); then
      index=$(( index + 1 ))
      continue
    fi
    if [ ! -e "$STATE/done/$key" ] && [ ! -e "$STATE/running/$key" ]; then
      : > "$STATE/running/$key"
      printf '%s %s %s %s\n' "$key" "$task" "$block" "$centers"
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
  local gpu=$1 job key task block centers extra log
  while true; do
    gpu_free "$gpu" || { sleep 20; continue; }
    job=$(claim) || return 0
    read -r key task block centers <<< "$job"
    extra=$("$PY" -c "print(float('$centers') - 1.0)")
    log="$ROOT/_logs/${key}_gpu${gpu}.log"
    echo "[gpu$gpu] $key"
    if (
      export SHADOWKV_RESULTS_ROOT="$ROOT"
      export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
      export ADAPTIVE_LSE_EXTRA_FRACTION="$extra"
      export SHADOWKV_CENTER_PLACEMENT=residual_path
      export SHADOWKV_CENTER_ALLOCATION=residual_marginal
      export STREAMING_REFINE_FACTOR=1 STREAMING_REFINE_TOKENS=0
      export STREAMING_MAX_COMPONENTS="$block"
      export STREAMING_COMPACT_METADATA=1 STREAMING_CENTER_BITS=16
      "$HERE/run_cell.sh" qwen3 32768 "$task" \
        adaptive_centroid_lse_streaming_prefix4 512 160 "$block" 16 0 \
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
echo "matrix cells=$(( ${#tasks[@]} * ${#configs[@]} )) samples=$NUM_SAMPLES gpus=$GPUS slots=${SLOT_START}..$((SLOT_START + SLOT_COUNT - 1))/$TOTAL_SLOTS root=$ROOT"
wait "${pids[@]}"
echo "matrix complete"
