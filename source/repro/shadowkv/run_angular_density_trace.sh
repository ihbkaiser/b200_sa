#!/usr/bin/env bash
# Screen macro angular-density / block-local center objectives on one trace.
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

TASK=${1:?usage: run_angular_density_trace.sh TASK SAMPLE GPU OUTPUT [extra args]}
SAMPLE=${2:?usage: run_angular_density_trace.sh TASK SAMPLE GPU OUTPUT [extra args]}
GPU=${3:?usage: run_angular_density_trace.sh TASK SAMPLE GPU OUTPUT [extra args]}
OUTPUT=${4:?usage: run_angular_density_trace.sh TASK SAMPLE GPU OUTPUT [extra args]}
shift 4

DATA=${SHADOWKV_DIR}/data/ruler/data/qwen/32768/${TASK}/validation.jsonl
if [[ ! -f "$DATA" ]]; then
  for root in "${SHADOWKV_DIR:-}" "$CODE/ShadowKV"; do
    candidate=${root}/data/ruler/data/qwen/32768/${TASK}/validation.jsonl
    if [[ -f "$candidate" ]]; then DATA=$candidate; break; fi
  done
fi
[[ -f "$DATA" ]] || { echo "missing dataset: $DATA" >&2; exit 1; }

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
  --angular-density-probe \
  --macro-page-tokens 1024 \
  --macro-proxy-count 64 \
  --no-dispersion \
  --router-query-group-mean \
  --refine-factors 2.0 \
  --output "$OUTPUT" \
  "$@"
