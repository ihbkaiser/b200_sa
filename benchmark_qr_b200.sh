#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
MODEL=${MODEL_PATH:-$ROOT/model/Qwen3-4B-Instruct-2507}
DATA=${SHADOWKV_RULER_DATA_ROOT:-$ROOT/source/ShadowKV/data/ruler/data}
DATASET=${QR_BENCHMARK_DATASET:-ruler/niah_single_1}
CONTEXT=${QR_CONTEXT_LENGTH:-131072}
WARMUP=${QR_RUNTIME_WARMUP:-8}
STEPS=${QR_RUNTIME_STEPS:-128}
RESULTS=${RESULTS_ROOT:-$ROOT/results/qr_runtime}
OUT=${QR_RUNTIME_OUT:-$RESULTS/query_robust_runtime.json}
ASSET=${QUERY_ROBUST_VERTICES_PATH:-$ROOT/source/ShadowKV/artifacts/query_robust/qwen3_4b_128k/qwen3_4b_qr_vertices_m32_128k.pt}

[ -d "$MODEL" ] || { echo "model directory not found: $MODEL" >&2; exit 2; }
[ -d "$DATA/qwen/131072" ] || { echo "Qwen RULER data not found: $DATA/qwen/131072" >&2; exit 2; }
[ -f "$ASSET" ] || { echo "QR vertex asset not found: $ASSET" >&2; exit 2; }
compgen -G "$ROOT/source/ShadowKV/kernels/*.so" >/dev/null || {
  echo "ShadowKV extension missing; run ./prepare_b200.sh first" >&2
  exit 2
}

export PYTHONPATH="$ROOT/source/ShadowKV${PYTHONPATH:+:$PYTHONPATH}"
export SHADOWKV_RULER_DATA_ROOT="$DATA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export QUERY_ROBUST_ROUTER_BACKEND=${QUERY_ROBUST_ROUTER_BACKEND:-triton}
export QUERY_ROBUST_SUMMARY_BACKEND=${QUERY_ROBUST_SUMMARY_BACKEND:-compile}
export SHADOWKV_RUNTIME_TIMINGS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

QR_OFFLOAD_ARGS=()
if [ "${QUERY_ROBUST_OFFLOAD:-0}" = 1 ]; then
  QR_OFFLOAD_ARGS=(--streaming_offload --streaming_gather_backend auto)
fi

mkdir -p "$(dirname "$OUT")"
echo "[benchmark] model=$MODEL context=$CONTEXT dataset=$DATASET warmup=$WARMUP steps=$STEPS"
echo "[benchmark] router=$QUERY_ROBUST_ROUTER_BACKEND summary=$QUERY_ROBUST_SUMMARY_BACKEND offload=${QUERY_ROBUST_OFFLOAD:-0}"

exec CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" \
  "$ROOT/source/ShadowKV/test/eval_acc.py" \
  --model_name "$MODEL" \
  --datalen "$CONTEXT" \
  --method query_robust \
  --dataset_name "$DATASET" \
  --num_samples 1 \
  --sparse_budget 4096 \
  --page_size 8 \
  --dense_layers 0 \
  --group_reduce max \
  --quest_prefix_tokens 32 \
  --streaming_recent_tokens 256 \
  --streaming_update_interval 256 \
  --query_robust_vertices_path "$ASSET" \
  --query_robust_model_fingerprint cdbee75f17c01a7cc42f958dc650907174af0554 \
  --query_robust_vertices_sha256 189b839536e53dac532b032504311b803438d0a968a5aed1db90223f0e76fd68 \
  --query_robust_num_vertices 32 \
  --query_robust_solver_iters 24 \
  --query_robust_solver_lr 0.25 \
  --query_robust_score_alpha 1.0 \
  --runtime_warmup "$WARMUP" \
  --runtime_steps "$STEPS" \
  --runtime_out "$OUT" \
  "${QR_OFFLOAD_ARGS[@]}"

