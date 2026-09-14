#!/usr/bin/env bash
# Hot-add one GPU worker to an existing claim pool without interrupting the
# workers already running there.
#
#   source repro/shadowkv/env_m1.sh
#   repro/shadowkv/run_worker.sh <queue_file> <state_root> <gpu_id>
#
# The shared claim lock prevents duplicate cells.  A per-GPU lock prevents two
# worker processes from targeting the same card.  The GPU must also be present
# in <state_root>/gpus.txt; removing it retires this worker after its cell.
set -uo pipefail

: "${SHADOWKV_RESULTS_ROOT:?source the machine environment first}"
: "${CUDA_DEVICE_ORDER:?must be PCI_BUS_ID}"
[ "$CUDA_DEVICE_ORDER" = "PCI_BUS_ID" ] || {
  echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID"
  exit 1
}

QUEUE=$(readlink -f "$1")
STATE=$(readlink -f "$2")
GPU=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[ -s "$QUEUE" ] || { echo "queue file missing or empty: $QUEUE"; exit 1; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $GPU"; exit 1; }
mkdir -p "$STATE"/{done,running,failed} "$SHADOWKV_RESULTS_ROOT/_logs"

exec 8>"$STATE/gpu-${GPU}.lock"
flock -n 8 || { echo "GPU$GPU already has a hot-add worker"; exit 1; }

gpu_free() {
  local out
  out=$(nvidia-smi --id="$GPU" --query-compute-apps=pid \
    --format=csv,noheader 2>/dev/null) || return 1
  [[ "$out" =~ ^[[:space:]]*$ ]]
}

claim_next() {
  local line cell
  while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    # shellcheck disable=SC2086
    if [ "$(echo "$line" | awk '{print $4}')" = m51 ]; then
      cell=$("$PY" "$HERE/cell_key.py" $line max \
        "${SHADOWKV_M51_VARIANT:-v2}" 2>/dev/null | head -1)
    else
      cell=$("$PY" "$HERE/cell_key.py" $line 2>/dev/null | head -1)
    fi
    [ -n "$cell" ] || continue
    [ -e "$STATE/done/$cell" ] && continue
    [ -e "$STATE/running/$cell" ] && continue
    [ -e "$STATE/failed/$cell" ] && continue
    : > "$STATE/running/$cell"
    echo "$line"
    return 0
  done < "$QUEUE"
  return 1
}

echo "[hot-add] queue=$QUEUE state=$STATE gpu=$GPU"
while true; do
  grep -qx "$GPU" "$STATE/gpus.txt" 2>/dev/null || {
    echo "[gpu$GPU] retired from roster"
    exit 0
  }

  if ! gpu_free; then
    sleep 30
    continue
  fi

  line=$(flock "$STATE/claim.lock" bash -c \
    "$(declare -f claim_next); STATE='$STATE' PY='$PY' QUEUE='$QUEUE' HERE='$HERE' claim_next")
  if [ -z "$line" ]; then
    echo "[gpu$GPU] queue drained"
    exit 0
  fi

  # shellcheck disable=SC2086
  set -- $line
  if [ "$4" = m51 ]; then
    cell=$("$PY" "$HERE/cell_key.py" "$@" max \
      "${SHADOWKV_M51_VARIANT:-v2}")
  else
    cell=$("$PY" "$HERE/cell_key.py" "$@")
  fi
  log="$SHADOWKV_RESULTS_ROOT/_logs/${cell}_gpu${GPU}.log"
  echo "[gpu$GPU] $cell"
  if "$HERE/run_cell.sh" "$@" "${NUM_SAMPLES:-96}" "$GPU" > "$log" 2>&1; then
    mv "$STATE/running/$cell" "$STATE/done/$cell"
  else
    mv "$STATE/running/$cell" "$STATE/failed/$cell"
    echo "[gpu$GPU] FAILED $cell -- see $log"
  fi
done
