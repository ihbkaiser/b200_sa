#!/usr/bin/env bash
# One independently writable shard of the causal prompt-query-mass pilot.
set -euo pipefail

TASK=${1:?task}
START=${2:?sample start}
STOP=${3:?sample stop}
GPU=${4:?gpu}
LENGTH=${5:-32768}
BUDGET=${6:-512}
MEAN_COMPONENTS=${7:-1.5}
ROOT=${8:-/storage/baonn/qmass_e2e_20260909}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${SHADOWKV_DIR:-}" ] || [ -z "${PY:-}" ]; then
  source "$HERE/env_m1.sh"
fi
export CODE=${QMASS_CODE_ROOT:-$(cd "$HERE/../.." && pwd)}
export SHADOWKV_DIR=$CODE/ShadowKV
export SHADOWKV_RESULTS_ROOT="$ROOT/${TASK}_s${START}_${STOP}"
export SHADOWKV_CENTER_PLACEMENT=${SHADOWKV_QMASS_PLACEMENT:-qmass_bank}
export SHADOWKV_CENTER_ALLOCATION=marginal
export SHADOWKV_QUERY_MASS_BANK_SIZE=${SHADOWKV_QUERY_MASS_BANK_SIZE:-32}
export SHADOWKV_QUERY_MASS_WINDOW=${SHADOWKV_QUERY_MASS_WINDOW:-512}
export ADAPTIVE_LSE_EXTRA_FRACTION=$(
  "$PY" -c "print(float('$MEAN_COMPONENTS') - 1.0)"
)
export STREAMING_MAX_COMPONENTS=8
export STREAMING_COMPACT_METADATA=${STREAMING_COMPACT_METADATA:-1}
export STREAMING_CENTER_BITS=${STREAMING_CENTER_BITS:-8}
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32
export STREAMING_OFFLOAD=1
export STREAMING_GATHER_BACKEND=torch
export STREAMING_REFINE_FACTOR=${STREAMING_REFINE_FACTOR:-1}
export STREAMING_REFINE_TOKENS=${STREAMING_REFINE_TOKENS:-0}
export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
export SHADOWKV_SAMPLE_START=$START
export SHADOWKV_SAMPLE_STOP=$STOP

MODEL_KEY=${QMASS_MODEL_KEY:-qwen3}
"$HERE/run_cell.sh" "$MODEL_KEY" "$LENGTH" "$TASK" \
  adaptive_centroid_lse_streaming_prefix4_querymean \
  "$BUDGET" 160 8 16 0 "$STOP" "$GPU" max
