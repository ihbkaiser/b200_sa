#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL="$ROOT/model/Qwen3-4B-Instruct-2507"
DEEPSEEK="$ROOT/model/DeepSeek-R1-Distill-Llama-8B"
DATA_ROOT="$ROOT/source/ShadowKV/data/ruler/data"
DATA_QWEN="$DATA_ROOT/qwen/131072"
DATA_LLAMA="$DATA_ROOT/llama-3/131072"
TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

for file in config.json tokenizer.json tokenizer_config.json \
  model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors; do
  [ -s "$MODEL/$file" ] || { echo "missing model file: $file" >&2; exit 1; }
done

[ ! -e "$MODEL/model-00001-of-00003.safetensors.incomplete" ] || exit 1
[ -d "$ROOT/source/ShadowKV/3rdparty/cutlass/include" ] || exit 1

for file in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
  generation_config.json; do
  [ -s "$DEEPSEEK/$file" ] || { echo "missing DeepSeek metadata: $file" >&2; exit 1; }
done
for file in model-00001-of-000002.safetensors.zst model-00002-of-000002.safetensors.zst; do
  [ -s "$DEEPSEEK/$file" ] || {
    [ -s "$DEEPSEEK/${file%.zst}" ] || { echo "missing DeepSeek shard: $file" >&2; exit 1; }
  }
done

for model_dir in qwen llama-3; do
  for task in "${TASKS[@]}"; do
    file="$DATA_ROOT/$model_dir/131072/$task/validation.jsonl"
    [ "$(wc -l < "$file")" -eq 100 ] || {
      echo "bad RULER count: $model_dir/$task" >&2; exit 1;
    }
  done
done

echo "bundle verification: qwen=OK deepseek=OK ruler-qwen=13x100 ruler-llama=13x100 cutlass=OK"
