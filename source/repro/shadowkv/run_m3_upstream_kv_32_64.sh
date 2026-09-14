#!/usr/bin/env bash
# M3 (8xL40 48GB): run the 26 InfLLM-64K accuracy cells that cannot fit on
# the 24GB hosts, then measure all three methods on one AVX-512 host. MagicPIG-64K is
# already assigned to machine 2 in the matched campaign; it remains an
# opt-in accuracy lane here for recovery.
# Restore the owner's reservation workload after the pool drains.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
source "$HERE/env_m3.sh"
cd "$CODE"

ROOT=${M3_UPSTREAM_RESULTS_ROOT:-$SHADOWKV_BASE/results_upstream_kv_matched_x32_l32_20260912}
QUEUE=$ROOT/queue_infllm_magicpig64.txt
STATE=$ROOT/.state_m3
ACCURACY_METHODS=${M3_UPSTREAM_ACCURACY_METHODS:-infllm}
ENABLE_RUNTIME=${M3_ENABLE_RUNTIME:-0}
HOLDER_DIR=${M3_HOLDER_DIR:-$HOME/workspace/sheruifeng/utils/qwen3.5-deploy}
mkdir -p "$ROOT"

# Never kill or steal the owner's reservation implicitly.
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
[[ $busy -eq 0 ]] || {
  echo "M3 has $busy GPU process(es); stop the reservation explicitly first" >&2
  exit 2
}

restore_holder() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -f "$HOLDER_DIR/serve_122B.sh" ]]; then
    echo "[restore] starting $HOLDER_DIR/serve_122B.sh (campaign rc=$rc)"
    cd "$HOLDER_DIR" || exit 6
    set +u
    source venv/bin/activate
    set -u
    exec bash serve_122B.sh
  fi
  exit "$rc"
}
# Install only after the busy-GPU refusal above: if the owner's holder is
# already alive, a rejected campaign must not try to launch a second copy.
trap restore_holder EXIT
trap 'exit 130' INT TERM

export SHADOWKV_RESULTS_ROOT=$ROOT
export SHADOWKV_UPSTREAM_ROOT=$SHADOWKV_BASE/upstream-kv-methods
export PQCACHE_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/PQCache
export MAGICPIG_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/MagicPIG
export INFLLM_AUTHOR_ROOT=$SHADOWKV_UPSTREAM_ROOT/InfLLM
export UPSTREAM_KV_METHODS=pqcache,magicpig,infllm
export UPSTREAM_MATCHED_EXACT_REGIONS=1
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32

# Do not assume that the earlier 128K setup also created the tokenizer-specific
# 32K/64K corpora.  The builder is idempotent and runs offline once punkt and
# the models are present; every file is checked for exactly 100 rows.
{
  bash "$HERE/build_ruler.sh" "$SHADOWKV_LLAMA32_PATH" llama-3 100 32768 65536
  bash "$HERE/build_ruler.sh" "$SHADOWKV_QWEN3_PATH" qwen 100 32768 65536
} 2>&1 | tee "$ROOT/data_m3.log" || exit 3

bash "$HERE/setup_upstream_kv_baselines.sh" 2>&1 | tee "$ROOT/setup_m3.log" || exit 3

for spec in \
  "$SHADOWKV_LLAMA32_PATH llama-3" \
  "$SHADOWKV_QWEN3_PATH qwen"; do
  # shellcheck disable=SC2086
  set -- $spec
  "$PY" "$HERE/preflight.py" --model "$1" --template "$2" \
    --datalen 65536 --method infllm_author_common --sparse_budget 2048 \
    --page_size 8 --chunk_size 8 || exit 3
  "$PY" "$HERE/preflight.py" --model "$1" --template "$2" \
    --datalen 65536 --method magicpig_author_common --sparse_budget 2048 \
    --page_size 8 --chunk_size 8 || exit 3
done

"$PY" "$HERE/preflight.py" --model "$SHADOWKV_LLAMA32_PATH" \
  --template llama-3 --datalen 65536 --method pqcache_author_native \
  --sparse_budget 2048 --page_size 8 --chunk_size 8 || exit 3

case ",$ACCURACY_METHODS," in
  ,infllm,)
    expected_rows=26 ;;
  ,magicpig,)
    expected_rows=26 ;;
  ,infllm,magicpig,|,magicpig,infllm,)
    expected_rows=52 ;;
  *)
    echo "unsupported M3_UPSTREAM_ACCURACY_METHODS=$ACCURACY_METHODS" >&2
    exit 5 ;;
esac
awk -v mode="$ACCURACY_METHODS" '$2 == 65536 && (
  (mode == "infllm" && $4 == "infllm_author_common") ||
  (mode == "magicpig" && $4 == "magicpig_author_common") ||
  ((mode == "infllm,magicpig" || mode == "magicpig,infllm") &&
   ($4 == "infllm_author_common" || $4 == "magicpig_author_common"))
)' "$HERE/queue_upstream_kv_32_64_20260911.txt" > "$QUEUE.tmp"
rows=$(wc -l < "$QUEUE.tmp")
[[ $rows -eq $expected_rows ]] || {
  echo "$ACCURACY_METHODS 64K queue has $rows cells, expected $expected_rows" >&2
  rm -f "$QUEUE.tmp"
  exit 5
}
printf '# M3 %s-64K: capacity plus same-host runtime routing\n' "$ACCURACY_METHODS" \
  | cat - "$QUEUE.tmp" > "$QUEUE"
rm -f "$QUEUE.tmp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SHADOWKV_POOL_GPUS=0,1,2,3,4,5,6,7
export NUM_SAMPLES=100
bash "$HERE/run_pool.sh" "$QUEUE" "$STATE" 2>&1 | tee "$ROOT/pool_m3.log"
pool_rc=${PIPESTATUS[0]}

done_count=$(find "$STATE/done" -maxdepth 1 -type f 2>/dev/null | wc -l)
running_count=$(find "$STATE/running" -maxdepth 1 -type f 2>/dev/null | wc -l)
failed_count=$(find "$STATE/failed" -maxdepth 1 -type f 2>/dev/null | wc -l)
{
  printf 'done=%s\n' "$done_count"
  printf 'running=%s\n' "$running_count"
  printf 'failed=%s\n' "$failed_count"
} | tee "$ROOT/final_audit.txt"

# Accuracy is the gating experiment.  Only a fully successful/resumed accuracy
# queue is allowed to consume the clean machine for runtime measurements.
if [[ $pool_rc -ne 0 || $done_count -ne $expected_rows \
      || $running_count -ne 0 || $failed_count -ne 0 ]]; then
  echo "accuracy gate failed: pool_rc=$pool_rc done=$done_count/$expected_rows " \
       "running=$running_count failed=$failed_count; runtime not started" >&2
  exit 7
fi

# This host cannot prove that the independent M1/M2/M4 queues are complete.
# Accuracy-only is therefore the safe default.  The controller may set the
# explicit unlock only after the merged report says 156/156 complete and zero
# failures/duplicates.  Rerunning this resumable launcher then skips all 26
# local cells and measures runtime on the clean M3 host.
if [[ $ENABLE_RUNTIME == 1 ]]; then
  export PQCACHE_NATIVE_RUNTIME=${PQCACHE_NATIVE_RUNTIME:-1}
  bash "$HERE/run_upstream_kv_latency.sh" 0 "$ROOT/latency_m3.jsonl" \
    2>&1 | tee "$ROOT/latency_m3.log" || exit 4
else
  echo "local accuracy complete; runtime remains locked pending merged 156/156 audit"
fi

exit "$pool_rc"
