#!/usr/bin/env bash
# Isolated common-forward runtime matrix for PQCache, MagicPIG and InfLLM.
# Run on an otherwise idle machine: bench_latency.py rejects a busy target GPU,
# and each row is a new Python process so CUDA allocator peaks do not leak.
set -euo pipefail

GPU=${1:?usage: run_upstream_kv_latency.sh <idle physical GPU> [output.jsonl]}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd "$HERE/../.." && pwd)
OUT=${2:-${SHADOWKV_RESULTS_ROOT:?}/upstream_kv_latency.jsonl}

: "${PY:?source the machine environment first}"
: "${PQCACHE_AUTHOR_ROOT:?run setup_upstream_kv_baselines.sh first}"
: "${MAGICPIG_AUTHOR_ROOT:?run setup_upstream_kv_baselines.sh first}"
: "${INFLLM_AUTHOR_ROOT:?run setup_upstream_kv_baselines.sh first}"

# Headline timing follows the same B + 32 prefix + 32 recent accounting as
# the accuracy campaign. Override only for a separately labelled native lane.
export UPSTREAM_MATCHED_EXACT_REGIONS=${UPSTREAM_MATCHED_EXACT_REGIONS:-1}
export QUEST_PREFIX_TOKENS=${QUEST_PREFIX_TOKENS:-32}
export STREAMING_RECENT_TOKENS=${STREAMING_RECENT_TOKENS:-32}

mkdir -p "$(dirname "$OUT")"
touch "$OUT"

row_complete() {
  "$PY" - "$OUT" "$1" "$2" "$3" <<'PY'
import json, sys
path, model, length, method = sys.argv[1:]
valid = False
with open(path) as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            row.get("model") == model
            and row.get("datalen") == int(length)
            and row.get("method") == method
            and row.get("real_text") is True
            and row.get("n_decode", 0) >= 128
        ):
            valid = True
sys.exit(0 if valid else 1)
PY
}

IFS=',' read -r -a METHODS <<< "${UPSTREAM_KV_METHODS:-pqcache,magicpig,infllm}"

for datalen in 32768 65536; do
  budget=$((datalen / 32))
  for model in llama32 qwen3; do
    for short_method in "${METHODS[@]}"; do
      case "$short_method" in
        pqcache) method=pqcache_author_common ;;
        magicpig) method=magicpig_author_common ;;
        infllm) method=infllm_author_common ;;
        *) echo "unknown upstream method: $short_method" >&2; exit 2 ;;
      esac
      if row_complete "$model" "$datalen" "$method"; then
        echo "[skip-valid] $model $datalen $method"
        continue
      fi
      echo "[latency] $model $datalen $method B=$budget"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/bench_latency.py" \
        --model_key "$model" --datalen "$datalen" --method "$method" \
        --sparse_budget "$budget" --chunk_size 8 --page_size 8 \
        --real_text --ruler_task niah_multikey_2 \
        --warmup_passes 1 --prefill_reps 2 --decode_steps 129 \
        --out "$OUT"
    done
  done
done

# The complete released PQCache runtime is a separately labelled secondary
# lane.  It includes the author multiprocessing compressor, LFU GPU cache and
# asynchronous H2D schedule and therefore must never replace the common-forward
# accuracy/runtime row silently.
if [[ ${PQCACHE_NATIVE_RUNTIME:-0} == 1 ]]; then
  for datalen in 32768 65536; do
    budget=$((datalen / 32))
    for model in llama32 qwen3; do
      method=pqcache_author_native
      if row_complete "$model" "$datalen" "$method"; then
        echo "[skip-valid] $model $datalen $method"
        continue
      fi
      echo "[latency-native] $model $datalen $method B=$budget"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/bench_latency.py" \
        --model_key "$model" --datalen "$datalen" --method "$method" \
        --sparse_budget "$budget" --chunk_size 8 --page_size 8 \
        --real_text --ruler_task niah_multikey_2 \
        --warmup_passes 1 --prefill_reps 2 --decode_steps 129 \
        --out "$OUT"
    done
  done
fi

echo "UPSTREAM_KV_LATENCY_COMPLETE rows=$(wc -l < "$OUT") out=$OUT"
