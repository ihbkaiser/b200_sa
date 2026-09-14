#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
MODEL_NAME=${MODEL_NAME:-qwen3}
case "$MODEL_NAME" in
  qwen3)
    MODEL_DEFAULT="$ROOT/model/Qwen3-4B-Instruct-2507"
    MODEL_DIR=qwen
    ;;
  deepseek-r1-distill-llama-8b|deepseek)
    MODEL_DEFAULT="$ROOT/model/DeepSeek-R1-Distill-Llama-8B"
    MODEL_DIR=llama-3
    ;;
  *) echo "unsupported MODEL_NAME: $MODEL_NAME (use qwen3 or deepseek)" >&2; exit 2 ;;
esac
MODEL=${MODEL_PATH:-$MODEL_DEFAULT}
DATA=${SHADOWKV_RULER_DATA_ROOT:-$ROOT/source/ShadowKV/data/ruler/data}
RESULTS=${RESULTS_ROOT:-$ROOT/results/qwen3_128k_100}
GPU=${CUDA_VISIBLE_DEVICES:-0}
METHODS=${METHODS:-quest_streaming,shadowkv_cpu}

TASKS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multivalue,ruler/niah_multiquery,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2"

[ -d "$MODEL" ] || { echo "model directory not found: $MODEL" >&2; exit 2; }
[ -d "$DATA/$MODEL_DIR/131072" ] || { echo "RULER data not found: $DATA/$MODEL_DIR/131072" >&2; exit 2; }
if [ "$MODEL_DIR" = llama-3 ] && ! compgen -G "$MODEL/model-*.safetensors" >/dev/null; then
  echo "DeepSeek raw shards are missing; run ./restore_deepseek.sh first" >&2
  exit 2
fi
compgen -G "$ROOT/source/ShadowKV/kernels/*.so" >/dev/null || {
  echo "ShadowKV extension missing; run ./prepare_b200.sh first" >&2; exit 2;
}

export PYTHONPATH="$ROOT/source/ShadowKV${PYTHONPATH:+:$PYTHONPATH}"
export SHADOWKV_RULER_DATA_ROOT="$DATA"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export QUEST_PREFIX_TOKENS=${QUEST_PREFIX_TOKENS:-32}
export STREAMING_RECENT_TOKENS=${STREAMING_RECENT_TOKENS:-256}
export STREAMING_UPDATE_INTERVAL=${STREAMING_UPDATE_INTERVAL:-256}

mkdir -p "$RESULTS"
echo "[run] model   : $MODEL"
echo "[run] model   : $MODEL_NAME"
echo "[run] data    : $DATA/$MODEL_DIR/131072"
echo "[run] samples : 100/task"
echo "[run] tasks   : 13 RULER tasks"
echo "[run] methods : $METHODS"
echo "[run] gpu     : $GPU"

run_quest() {
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/source/ShadowKV/test/eval_acc.py" \
    --model_name "$MODEL" \
    --datalen 131072 \
    --method quest_streaming \
    --dataset_name "$TASKS" \
    --num_samples 100 \
    --sparse_budget 4096 \
    --page_size 8 \
    --dense_layers 2 \
    --group_reduce max \
    --quest_prefix_tokens "$QUEST_PREFIX_TOKENS" \
    --streaming_recent_tokens "$STREAMING_RECENT_TOKENS" \
    --streaming_update_interval "$STREAMING_UPDATE_INTERVAL" \
    --streaming_offload \
    --streaming_gather_backend auto \
    --out_root "$RESULTS/quest_streaming"
}

run_shadowkv_cpu() {
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/source/ShadowKV/test/eval_acc.py" \
    --model_name "$MODEL" \
    --datalen 131072 \
    --method shadowkv_cpu \
    --dataset_name "$TASKS" \
    --num_samples 100 \
    --sparse_budget 4096 \
    --rank 160 \
    --chunk_size 4 \
    --shadow_outlier_chunks 96 \
    --streaming_recent_tokens "$STREAMING_RECENT_TOKENS" \
    --out_root "$RESULTS/shadowkv_cpu"
}

run_shadowkv_gpu() {
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/source/ShadowKV/test/eval_acc.py" \
    --model_name "$MODEL" \
    --datalen 131072 \
    --method shadowkv \
    --dataset_name "$TASKS" \
    --num_samples 100 \
    --sparse_budget 4096 \
    --rank 160 \
    --chunk_size 4 \
    --shadow_outlier_chunks 96 \
    --streaming_recent_tokens "$STREAMING_RECENT_TOKENS" \
    --out_root "$RESULTS/shadowkv"
}

IFS=',' read -r -a requested_methods <<< "$METHODS"
for method in "${requested_methods[@]}"; do
  case "$method" in
    quest_streaming) run_quest ;;
    shadowkv_cpu) run_shadowkv_cpu ;;
    shadowkv) run_shadowkv_gpu ;;
    *) echo "unsupported method: $method" >&2; exit 2 ;;
  esac
done

echo "[done] results written to $RESULTS"
