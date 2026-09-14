#!/usr/bin/env bash
# Build RULER for one model at one or more lengths, using ShadowKV's own
# generators (the source corpora ship with the repo, so this runs offline).
#
#   repro/shadowkv/build_ruler.sh <model_path> <template: llama-3|qwen> <num_samples> <len> [len...]
#
# Output: ShadowKV/data/ruler/data/<template>/<len>/<task>/validation.jsonl
# The per-model directory matters: sample lengths are trimmed with that model's
# tokenizer, so llama-3 and qwen data are not interchangeable.
set -euo pipefail

: "${SHADOWKV_DIR:?source repro/shadowkv/env_m1.sh first}"
: "${PY:?source repro/shadowkv/env_m1.sh first}"

MODEL_PATH=$1; TEMPLATE=$2; NUM_SAMPLES=$3; shift 3
LENGTHS=("$@")
[ ${#LENGTHS[@]} -gt 0 ] || { echo "no lengths given"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "model path not found: $MODEL_PATH"; exit 1; }

# The 13 RULER tasks. ShadowKV's own create_dataset.sh drops cwe and
# niah_multikey_3; we keep the full set so the same data can back other tables.
TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery
  vt cwe fwe qa_1 qa_2
)

cd "$SHADOWKV_DIR/data/ruler"

for LEN in "${LENGTHS[@]}"; do
  OUT="data/${TEMPLATE}/${LEN}"
  mkdir -p "$OUT"
  for TASK in "${TASKS[@]}"; do
    FILE="${OUT}/${TASK}/validation.jsonl"
    LINES=0
    [ -f "$FILE" ] && LINES=$(wc -l < "$FILE")
    if [ "$LINES" -eq "$NUM_SAMPLES" ]; then
      echo "[skip] ${TEMPLATE}/${LEN}/${TASK} already built"
      continue
    fi
    [ "$LINES" -eq 0 ] || echo "[rebuild] partial file has ${LINES}/${NUM_SAMPLES} lines: $FILE"
    echo "[build] ${TEMPLATE}/${LEN}/${TASK}"
    "$PY" prepare.py \
      --save_dir "${OUT}/" \
      --task "${TASK}" \
      --tokenizer_path "${MODEL_PATH}" \
      --tokenizer_type hf \
      --max_seq_length "${LEN}" \
      --model_template_type "${TEMPLATE}" \
      --num_samples "${NUM_SAMPLES}"
    [ -f "$FILE" ] || { echo "ERROR: generator did not create $FILE"; exit 1; }
    LINES=$(wc -l < "$FILE")
    [ "$LINES" -eq "$NUM_SAMPLES" ] || {
      echo "ERROR: expected ${NUM_SAMPLES} lines, got ${LINES}: $FILE"
      exit 1
    }
  done
done

echo
echo "=== built ==="
for LEN in "${LENGTHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    f="data/${TEMPLATE}/${LEN}/${TASK}/validation.jsonl"
    [ -s "$f" ] && printf "%-42s %s lines\n" "$f" "$(wc -l < "$f")"
  done
done
