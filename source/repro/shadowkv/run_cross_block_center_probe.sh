#!/usr/bin/env bash
# Query-free cross-block/multi-scale center diagnostic on one trace.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${SHADOWKV_ENV_FILE:-} ]]; then
  source "$SHADOWKV_ENV_FILE"
else
  case $(hostname -s) in
    sepc810*) source "$HERE/env_m2.sh" ;;
    sashimi*) source "$HERE/env_m4.sh" ;;
    *) source "$HERE/env_m1.sh" ;;
  esac
fi

TASK=${1:?usage: run_cross_block_center_probe.sh TASK SAMPLE GPU [OUTPUT]}
SAMPLE=${2:?usage: run_cross_block_center_probe.sh TASK SAMPLE GPU [OUTPUT]}
GPU=${3:?usage: run_cross_block_center_probe.sh TASK SAMPLE GPU [OUTPUT]}
OUTPUT=${4:-/storage/baonn/selfk_cross_campaign_20260907/screen32k/${TASK}_s${SAMPLE}}
if (( $# >= 4 )); then
  shift 4
else
  shift "$#"
fi
DATA_ROOT=${SHADOWKV_RULER32K_ROOT:-$SHADOWKV_DIR/data/ruler/data/qwen/32768}
DATA=$DATA_ROOT/$TASK/validation.jsonl
M1_DATA=$SHADOWKV_DIR/data/ruler/data/qwen/32768/$TASK/validation.jsonl
if [[ ! -f "$DATA" && -f "$M1_DATA" ]]; then
  DATA=$M1_DATA
fi

[[ -f "$DATA" ]] || { echo "missing RULER file: $DATA" >&2; exit 1; }
mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

exec "$PY" "$HERE/diagnose_router_mass_full_trace.py" \
  --model "$SHADOWKV_QWEN3_PATH" \
  --dataset "$DATA" \
  --sample-index "$SAMPLE" \
  --budgets 512 \
  --block-size 8 \
  --prefix-tokens 32 \
  --recent-tokens 32 \
  --extra-fraction 0.25 \
  --center-allocation-probe \
  --cross-block-center-probe \
  "$@" \
  --output "$OUTPUT"
