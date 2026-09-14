#!/usr/bin/env bash
# Clean task/model sensitivity sweep on one machine-4 RTX 4090.  Rows are
# deliberately sequential: CPU offload traffic from another process would
# otherwise contaminate both decode latency and host RSS.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/env_m4.sh"
cd "$CODE"

GPU=${1:?usage: run_m4_latency_tasks_qwen.sh <idle-gpu>}
ROOT=${LATENCY_ROOT:-/storage/nbao/shadowkv_results/latency_tasks_qwen_adaptive_seal_m4_20260911}
OUT=$ROOT/results.jsonl
mkdir -p "$ROOT/logs"
touch "$OUT"

export SHADOWKV_RMSNORM_BACKEND=flashinfer
export SHADOWKV_CENTER_PLACEMENT=self
export SHADOWKV_CENTER_ALLOCATION=tail_absolute_rate_distortion
export SHADOWKV_SELF_LSE_COST=mean_gap
export SHADOWKV_TAIL_CVAR_FRACTION=0.25
export SHADOWKV_ABSOLUTE_RD_PENALTY=1.5
export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
run_one() {
    local model=$1 length=$2 task=$3 method=$4 budget=$5
    shift 5
    local tag="${model}_${length}_${task}_${method}_b${budget}"
    if "$PY" - "$OUT" "$model" "$length" "$task" "$method" <<'PY'
import json, sys
path, model, length, task, method = sys.argv[1:]
for line in open(path):
    row = json.loads(line)
    if (row.get("model") == model and row.get("datalen") == int(length)
            and row.get("ruler_task") == task
            and row.get("method") == method):
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "[skip] $tag"
        return
    fi
    echo "[start] $tag $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/bench_latency.py" \
        --model_key "$model" --datalen "$length" \
        --ruler_task "$task" --real_text \
        --method "$method" --sparse_budget "$budget" \
        --prefix_tokens 32 --recent_tokens 32 \
        --warmup_passes 1 --prefill_reps 1 --decode_steps 33 \
        --out "$OUT" "$@" > "$ROOT/logs/${tag}.log" 2>&1
    echo "[done] $tag $(date -u +%FT%TZ)"
}

run_matrix_row() {
    local model=$1 length=$2 task=$3
    local budget=$((length / 32))
    local outliers
    outliers=$("$PY" "$HERE/outlier_policy.py" "$length" 4)
    run_one "$model" "$length" "$task" \
        adaptive_centroid_lse_streaming_prefix4_querymean "$budget" \
        --chunk_size 8 --streaming_offload \
        --streaming_gather_backend uva --streaming_router_backend triton
    run_one "$model" "$length" "$task" quest_streaming "$budget" \
        --page_size 8 --dense_layers 0 --streaming_offload \
        --streaming_gather_backend uva --streaming_router_backend triton
    run_one "$model" "$length" "$task" shadowkv_cpu "$budget" \
        --rank 160 --chunk_size 4 --shadow_outlier_chunks "$outliers"
    run_one "$model" "$length" "$task" pariskv_author_common "$budget" \
        --chunk_size 8 --streaming_offload \
        --streaming_gather_backend uva \
        --pariskv_author_root "$PARISKV_AUTHOR_ROOT"
}

# Task sensitivity at the middle context length.
for model in llama32 qwen3; do
    for task in cwe qa_2 niah_multikey_3; do
        run_matrix_row "$model" 65536 "$task"
    done
done

# Qwen scaling endpoints; 64K is already covered above.
for length in 32768 131072; do
    run_matrix_row qwen3 "$length" niah_multikey_3
done

echo "LATENCY_TASK_QWEN_DONE out=$OUT rows=$(wc -l < "$OUT")"
