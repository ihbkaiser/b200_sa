#!/usr/bin/env bash
# M2 (4x RTX4500 Ada 24GB): run the 26 MagicPIG-64K accuracy cells.
# InfLLM-64K is intentionally excluded: a matched Llama smoke reached 22.92
# GiB and then failed a further 1,016 MiB prefill allocation on a 23.55 GiB
# card.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
source "$HERE/env_m2.sh"
cd "$CODE"

ROOT=${M2_UPSTREAM_RESULTS_ROOT:-/home/nbnguyen/results_upstream_kv_matched_x32_l32_20260912_m2}
QUEUE=$ROOT/queue_magicpig64.txt
STATE=$ROOT/.state_m2
mkdir -p "$ROOT"

export SHADOWKV_RESULTS_ROOT=$ROOT
export SHADOWKV_UPSTREAM_ROOT=${SHADOWKV_UPSTREAM_ROOT:-/home/nbnguyen/upstream-kv-methods}
export PQCACHE_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/PQCache
export MAGICPIG_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/MagicPIG
export INFLLM_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/InfLLM
export UPSTREAM_KV_METHODS=magicpig
export UPSTREAM_MATCHED_EXACT_REGIONS=1
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32

bash "$HERE/setup_upstream_kv_baselines.sh" \
  2>&1 | tee "$ROOT/setup_m2.log"

for spec in \
  "$SHADOWKV_LLAMA32_PATH llama-3" \
  "$SHADOWKV_QWEN3_PATH qwen"; do
  # shellcheck disable=SC2086
  set -- $spec
  "$PY" "$HERE/preflight.py" --model "$1" --template "$2" \
    --datalen 65536 --method magicpig_author_common --sparse_budget 2048 \
    --page_size 8 --chunk_size 8
done

awk '$2 == 65536 && $4 == "magicpig_author_common"' \
  "$HERE/queue_upstream_kv_32_64_20260911.txt" > "$QUEUE.tmp"
rows=$(wc -l < "$QUEUE.tmp")
[[ $rows -eq 26 ]] || {
  echo "MagicPIG-64K queue has $rows cells, expected 26" >&2
  exit 5
}
printf '# M2 MagicPIG-64K: official AVX-512 CPU kernels\n' \
  | cat - "$QUEUE.tmp" > "$QUEUE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SHADOWKV_POOL_GPUS=1,2,3,4
export NUM_SAMPLES=100
bash "$HERE/run_pool.sh" "$QUEUE" "$STATE" 2>&1 | tee "$ROOT/pool_m2.log"
