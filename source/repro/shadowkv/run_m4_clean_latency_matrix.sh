#!/usr/bin/env bash
# Isolated latency matrix on one machine-4 RTX 4090. Every row is a fresh
# process; methods never share a GPU or overlap in time.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/env_m4.sh"
cd "$CODE"

GPU=${1:?usage: run_m4_clean_latency_matrix.sh <idle-gpu>}
TASK=${LATENCY_TASK:-niah_multikey_3}
ROOT=${LATENCY_ROOT:-/storage/nbao/shadowkv_results/clean_latency_m4_20260911}
OUT=$ROOT/results.jsonl
mkdir -p "$ROOT/logs"

if [[ -s "$OUT" && ${LATENCY_RESUME:-0} != 1 ]]; then
    echo "results already exist: $OUT (set LATENCY_RESUME=1 to resume)" >&2
    exit 2
fi
touch "$OUT"

export SHADOWKV_RMSNORM_BACKEND=flashinfer
export SHADOWKV_CENTER_PLACEMENT=self
export SHADOWKV_CENTER_ALLOCATION=tail_absolute_rate_distortion
export SHADOWKV_SELF_LSE_COST=mean_gap
export SHADOWKV_TAIL_CVAR_FRACTION=0.25
export SHADOWKV_ABSOLUTE_RD_PENALTY=1.5
export SHADOWKV_CENTER_DISPERSION_CORRECTION=0

run_one() {
    local length=$1 method=$2 budget=$3
    shift 3
    local tag="${length}_${method}_b${budget}"
    if grep -q "\"datalen\": $length.*\"method\": \"$method\"" "$OUT"; then
        echo "[skip] $tag"
        return
    fi
    echo "[start] $tag $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/bench_latency.py" \
        --model_key llama32 --datalen "$length" --ruler_task "$TASK" --real_text \
        --method "$method" --sparse_budget "$budget" \
        --prefix_tokens 32 --recent_tokens 32 \
        --warmup_passes 1 --prefill_reps 3 --decode_steps 65 \
        --out "$OUT" "$@" 2>&1 | tee "$ROOT/logs/${tag}.log"
    echo "[done] $tag $(date -u +%FT%TZ)"
}

for length in 32768 65536 131072; do
    budget=$((length / 32))
    outliers=$("$PY" "$HERE/outlier_policy.py" "$length" 4)
    run_one "$length" adaptive_centroid_lse_streaming_prefix4_querymean "$budget" \
        --chunk_size 8 --streaming_offload --streaming_gather_backend uva \
        --streaming_router_backend triton
    run_one "$length" quest_streaming "$budget" \
        --page_size 8 --dense_layers 0 --streaming_offload \
        --streaming_gather_backend uva --streaming_router_backend triton
    run_one "$length" shadowkv_cpu "$budget" \
        --rank 160 --chunk_size 4 --shadow_outlier_chunks "$outliers"
    run_one "$length" pariskv_author_common "$budget" \
        --chunk_size 8 --streaming_offload --streaming_gather_backend uva \
        --pariskv_author_root "$PARISKV_AUTHOR_ROOT"
done

"$PY" "$HERE/status_latency_matrix.py" "$OUT"
echo "LATENCY_MATRIX_DONE out=$OUT"
