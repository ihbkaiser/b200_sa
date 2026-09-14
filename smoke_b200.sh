#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
MODEL=${MODEL_PATH:-$ROOT/model/Qwen3-4B-Instruct-2507}
DATA=${SHADOWKV_RULER_DATA_ROOT:-$ROOT/source/ShadowKV/data/ruler/data}
RESULTS=${RESULTS_ROOT:-$ROOT/results/smoke}

export PYTHONPATH="$ROOT/source/ShadowKV${PYTHONPATH:+:$PYTHONPATH}"
export SHADOWKV_RULER_DATA_ROOT="$DATA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RESULTS"

for method in quest_streaming shadowkv_cpu; do
  if [ "$method" = quest_streaming ]; then
    extra=(--page_size 8 --dense_layers 2 --group_reduce max --quest_prefix_tokens 32
      --streaming_recent_tokens 32 --streaming_update_interval 32 --streaming_offload
      --streaming_gather_backend auto)
    budget=(--sparse_budget 128)
  else
    extra=(--rank 160 --chunk_size 4 --shadow_outlier_chunks 3 --streaming_recent_tokens 32)
    budget=(--sparse_budget 128)
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" \
    "$ROOT/source/ShadowKV/test/eval_acc.py" \
    --model_name "$MODEL" --datalen 4096 --method "$method" \
    --dataset_name ruler/niah_single_1 --num_samples 1 "${budget[@]}" \
    "${extra[@]}" --out_root "$RESULTS/$method"
done
echo "[done] smoke results written to $RESULTS"
