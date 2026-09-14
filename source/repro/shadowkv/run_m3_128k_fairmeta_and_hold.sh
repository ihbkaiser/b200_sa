#!/usr/bin/env bash
# Run the complete machine-3 128K fair-metadata campaign, then restore the
# owner's 122B reservation workload. The reservation is deliberately started
# after the pool drains even if individual cells failed: the audit file keeps
# the failure count, while the GPUs must not be left unclaimed overnight.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
source "$HERE/env_m3.sh"
cd "$CODE"

ROOT=${M3_128K_RESULTS_ROOT:-$SHADOWKV_BASE/results_shadowkv/ruler128_fairmeta_20260911}
QUEUE=$ROOT/queue.txt
STATE=$ROOT/.state
HOLDER_DIR=${M3_HOLDER_DIR:-$HOME/workspace/sheruifeng/utils/qwen3.5-deploy}
HOLDER_SCRIPT=${M3_HOLDER_SCRIPT:-serve_122B.sh}
mkdir -p "$ROOT"

test -f "$HOLDER_DIR/$HOLDER_SCRIPT" || {
  echo "missing reservation script: $HOLDER_DIR/$HOLDER_SCRIPT" >&2
  exit 2
}

# Refuse an incomplete local dataset before opening eight workers.
for template in llama-3 qwen; do
  data_root=$SHADOWKV_DIR/data/ruler/data/$template/131072
  complete=0
  for file in "$data_root"/*/validation.jsonl; do
    [[ -f "$file" ]] || continue
    [[ $(wc -l < "$file") -eq 100 ]] && ((complete += 1))
  done
  [[ $complete -eq 13 ]] || {
    echo "incomplete RULER data: $template has $complete/13 tasks" >&2
    exit 2
  }
done

"$PY" - <<'PY'
import fast_hadamard_transform, flash_attn, flashinfer, torch
from ShadowKV.kernels import shadowkv
assert torch.cuda.is_available()
print("dependencies ready", torch.__version__, flash_attn.__version__, flashinfer.__version__)
PY

tmp_queue=$QUEUE.tmp.$$
{
  "$PY" "$HERE/gen_fairmeta_32x_queue.py" \
    --model llama32 --datalens 131072 --budgets 4096,8192 --include-full
  "$PY" "$HERE/gen_fairmeta_32x_queue.py" \
    --model qwen3 --datalens 131072 --budgets 4096,8192 --include-full
} | "$PY" "$HERE/interleave_queue.py" > "$tmp_queue"
rows=$(grep -cvE '^[[:space:]]*(#|$)' "$tmp_queue")
[[ $rows -eq 234 ]] || {
  echo "queue has $rows cells, expected 234" >&2
  rm -f "$tmp_queue"
  exit 2
}
mv "$tmp_queue" "$QUEUE"

export SHADOWKV_RESULTS_ROOT=$ROOT
export SHADOWKV_POOL_GPUS=0,1,2,3,4,5,6,7
export NUM_SAMPLES=100
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=32
export STREAMING_OFFLOAD=1
export STREAMING_GATHER_BACKEND=uva

# Pin the canonical method rather than inheriting an old probe environment.
export SHADOWKV_CENTER_PLACEMENT=self
export SHADOWKV_CENTER_ALLOCATION=tail_absolute_rate_distortion
export SHADOWKV_SELF_LSE_COST=mean_gap
export SHADOWKV_TAIL_CVAR_FRACTION=0.25
export SHADOWKV_ABSOLUTE_RD_PENALTY=1.5
export SHADOWKV_CENTER_DISPERSION_CORRECTION=0
export STREAMING_MAX_COMPONENTS=8
export STREAMING_COMPACT_METADATA=1
export STREAMING_CENTER_BITS=8
export STREAMING_REFINE_FACTOR=1
export STREAMING_ROUTER_BACKEND=torch
export SHADOWKV_RMSNORM_BACKEND=torch
unset SHADOWKV_HEAD_ALLOC_MODE SHADOWKV_HEAD_ALLOC_TAU
unset SHADOWKV_HEAD_ALLOC_REGULARIZER STREAMING_REFINE_TOKENS

echo "campaign root=$ROOT queue_cells=$rows gpus=$SHADOWKV_POOL_GPUS"
"$HERE/run_pool.sh" "$QUEUE" "$STATE"
pool_rc=$?

done_count=$(find "$STATE/done" -type f 2>/dev/null | wc -l)
failed_count=$(find "$STATE/failed" -type f 2>/dev/null | wc -l)
running_count=$(find "$STATE/running" -type f 2>/dev/null | wc -l)
{
  echo "pool_rc=$pool_rc"
  echo "done=$done_count"
  echo "failed=$failed_count"
  echo "running=$running_count"
  echo "expected=234"
  date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ"
} | tee "$ROOT/final_audit.txt"

echo "campaign drained; starting reservation workload from $HOLDER_DIR"
cd "$HOLDER_DIR" || exit 3
set +u
source venv/bin/activate
set -u
exec bash "$HOLDER_SCRIPT"
