#!/usr/bin/env bash
# Does the sparsity actually bite?
#
# A method that silently falls back to dense attention still scores plausibly.
# The only cheap proof is the predictions themselves: run the same cell at a
# generous and at a punishing budget and diff the raw output strings.
#
#   * predictions identical to full attention at a tiny budget  -> not biting
#   * predictions identical between two budgets                 -> flag swallowed
#
#   source repro/shadowkv/env_m1.sh
#   repro/shadowkv/bite_check.sh <model_key> <datalen> <task> <num_samples> <gpu>
set -euo pipefail

: "${SHADOWKV_DIR:?source repro/shadowkv/env_m1.sh first}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # resolve before any cd

MODEL_KEY=$1; DATALEN=$2; TASK=$3; NUM_SAMPLES=$4; GPU=$5

case "$MODEL_KEY" in
  llama32) MODEL_PATH=$SHADOWKV_LLAMA32_PATH ;;
  qwen3)   MODEL_PATH=$SHADOWKV_QWEN3_PATH ;;
  *) echo "unknown model_key '$MODEL_KEY'"; exit 1 ;;
esac

OUT=$SHADOWKV_RESULTS_ROOT/_bite/$MODEL_KEY
mkdir -p "$OUT"
cd "$SHADOWKV_DIR"

run() {  # run <method> <extra args...>
  local method=$1; shift
  echo "  [run] $method $*"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$PY" test/eval_acc.py \
    --model_name "$MODEL_PATH" --datalen "$DATALEN" --method "$method" \
    --dataset_name "ruler/$TASK" --num_samples "$NUM_SAMPLES" \
    --out_root "$OUT" "$@" > /dev/null 2>&1
}

echo "[bite] $MODEL_KEY $TASK @ $DATALEN, $NUM_SAMPLES samples, gpu $GPU"
run full
run quest_streaming --sparse_budget 1024 --page_size 16 --dense_layers 2
run quest_streaming --sparse_budget 128  --page_size 16 --dense_layers 2
run shadowkv --sparse_budget 1024 --rank 160 --chunk_size 8
run shadowkv --sparse_budget 128  --rank 160 --chunk_size 8

MODEL_DIR=$(basename "${MODEL_PATH%/}")
"$PY" "$HERE/diff_predictions.py" \
  --dir "$OUT/$MODEL_DIR/ruler" --task "$TASK" --datalen "$DATALEN"
