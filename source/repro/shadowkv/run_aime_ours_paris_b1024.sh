#!/usr/bin/env bash
# AIME25 parity run for ours, ParisKV common-forward, and Quest streaming.
# One seed per GPU; the two methods run sequentially on that GPU so each seed
# sees identical hardware conditions.  Generation matches the established
# KVPress Qwen campaign: temperature=.6, top-p=.9, no top-k, 32768-token cap.
set -euo pipefail

: "${CODE:?source repro/shadowkv/env_m1.sh first}"
: "${PY:?source repro/shadowkv/env_m1.sh first}"
: "${PARISKV_AUTHOR_ROOT:?source repro/shadowkv/env_m1.sh first}"

BASE_ROOT=${AIME_RESULTS_ROOT:-/storage/baonn/aime25_ours_paris_b1024_20260912}
mkdir -p "$BASE_ROOT/_logs"

export STREAMING_OFFLOAD=1
export STREAMING_GATHER_BACKEND=uva
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32
export UPSTREAM_MATCHED_EXACT_REGIONS=1
export SHADOWKV_MAX_NEW_TOKENS=32768
export SHADOWKV_GENERATION_TEMPERATURE=0.6
export SHADOWKV_GENERATION_TOP_P=0.9
export SHADOWKV_GENERATION_TOP_K=-1

methods=(
  adaptive_centroid_lse_streaming_prefix4_querymean
  pariskv_author_common
  quest_streaming
)

IFS=',' read -r -a gpu_ids <<< "${AIME_GPU_IDS:-0,1,2,3}"
if (( ${#gpu_ids[@]} != 4 )); then
  echo "AIME_GPU_IDS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi

for seed in 0 1 2 3; do
  (
    gpu=${gpu_ids[$seed]}
    export SHADOWKV_GENERATION_SEED=$seed
    export SHADOWKV_RESULTS_ROOT="$BASE_ROOT/seed$seed"
    for method in "${methods[@]}"; do
      log="$BASE_ROOT/_logs/qwen3_${method}_s${seed}_gpu${gpu}.log"
      "$CODE/repro/shadowkv/run_cell.sh" \
        qwen3 32768 aime25 "$method" 1024 160 8 8 0 -1 "$gpu" \
        >"$log" 2>&1
    done
  ) &
done
wait
touch "$BASE_ROOT/COMPLETE"
