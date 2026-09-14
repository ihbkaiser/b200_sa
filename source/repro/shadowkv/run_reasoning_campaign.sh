#!/usr/bin/env bash
# MATH500 / AIME25 / GPQA-diamond for the six methods, on m3's eight L40s.
#
#   source repro/shadowkv/env_m3.sh
#   repro/shadowkv/run_reasoning_campaign.sh smoke   # 4 examples per cell
#   repro/shadowkv/run_reasoning_campaign.sh full    # the whole matrix
#
# To write the queue and stop before the pool, pass a second argument:
#   repro/shadowkv/run_reasoning_campaign.sh full queue
# Do NOT do that by passing an empty SHADOWKV_POOL_GPUS: run_pool seeds its
# roster file from that variable, so an empty one leaves an empty gpus.txt in
# .state and the NEXT launch inherits it and starts zero workers.
#
# Two phases, and the order is not optional.  These cells are priced by
# generation, not by prompt, and the generation ceiling of the default matrix
# is 74M tokens -- hundreds of GPU-hours if nothing stops early.  How much of
# that ceiling the model actually spends is a property of the model and the
# benchmark that no amount of reading the code will tell you, so the smoke
# phase runs two examples of every cell and `report_reasoning.py --cost` turns
# the result into an estimate for the full matrix.  Read it before committing
# the cards.
#
# Tuning knobs, all of them environment variables read here:
#   REASONING_TASKS    math500,aime25,gpqa
#   REASONING_METHODS  comma-separated; add `full` back for the dense ceiling
#   REASONING_SEEDS    task:count pairs, e.g. aime25:4,math500:1,gpqa:1
#   REASONING_DATALEN  sequence scale the budget is matched against (32768)
#   REASONING_BUDGETS  one or more absolute budgets, primary one first
set -uo pipefail

: "${SHADOWKV_MACHINE:?source repro/shadowkv/env_mX.sh first}"
: "${SHADOWKV_RESULTS_ROOT:?source repro/shadowkv/env_mX.sh first}"
: "${PY:?source repro/shadowkv/env_mX.sh first}"

PHASE=${1:?usage: run_reasoning_campaign.sh smoke|full [queue]}
case "$PHASE" in smoke|full) ;; *) echo "phase must be smoke or full" >&2; exit 2 ;; esac
QUEUE_ONLY=${2:-}
case "$QUEUE_ONLY" in ""|queue) ;; *) echo "second argument, if given, must be 'queue'" >&2; exit 2 ;; esac

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
cd "$CODE"

ROOT=${REASONING_ROOT:-$SHADOWKV_RESULTS_ROOT/reasoning_$(date +%Y%m%d)}
ROOT=$ROOT/$PHASE
mkdir -p "$ROOT"

# --- the protocol, as one block -------------------------------------------
# temperature .6 / top-p .9 / NO top-k is a sacred constant of this project
# (PROJECT.md 4): it is what every reasoning table the predecessor produced was
# measured under, and the 12/09 campaign's .95/20 on GPQA and MATH500 made
# those two numbers unmergeable with AIME and with every earlier table. It is
# pinned here, not defaulted, and the run refuses to start if it is contradicted.
for pair in "SHADOWKV_GENERATION_TEMPERATURE:0.6" "SHADOWKV_GENERATION_TOP_P:0.9" \
            "SHADOWKV_GENERATION_TOP_K:-1"; do
  var=${pair%%:*} want=${pair#*:} have=${!var:-}
  if [ -n "$have" ] && [ "$have" != "$want" ]; then
    echo "ERROR: $var is '$have' in the environment; this campaign is $want." >&2
    echo "Numbers measured under a different sampler do not merge with any" >&2
    echo "reasoning table this project has published. Unset it, or run by hand." >&2
    exit 2
  fi
  export "$var=$want"
done
# max_new_tokens is left UNSET on purpose: each benchmark carries its own in
# the dataset (32000 / 4096 / 16384) and eval_acc reserves exactly that.
unset SHADOWKV_MAX_NEW_TOKENS

export UPSTREAM_MATCHED_EXACT_REGIONS=1
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=256
export STREAMING_UPDATE_INTERVAL=256
# Whole benchmark, always: these have fixed published sizes (500 / 30 / 198)
# and a truncated one is not the benchmark. The smoke phase is the exception.
# A smoke cell's wall time includes loading the model, so its cost per sample
# is an upper bound that shrinks as the sample count rises. Four is enough for
# the bound to be useful and still cheap; raise it for a tighter estimate.
export NUM_SAMPLES=$([ "$PHASE" = smoke ] && echo "${REASONING_SMOKE_SAMPLES:-4}" || echo -1)

QUEUE=$ROOT/queue_$SHADOWKV_MACHINE.txt
# A queue is a live document -- the pool re-reads it on every claim, so cells
# are moved and reordered while it runs. Regenerating on restart would undo
# that silently, and the only symptom would be work redone.
if [ -s "$QUEUE" ] && [ "${REASONING_REGEN:-0}" != 1 ]; then
  echo "[campaign] dùng queue sẵn có $QUEUE (REASONING_REGEN=1 để sinh lại)"
else
  # ShadowKV is NOT in the roster either, and cannot be put back: its landmark
  # basis is an SVD of the prompt and these prompts are shorter than the
  # budget, so it never leaves exact warm-up. gen_reasoning_queue.py refuses it
  # with the reason.
  # `full` is NOT in the default roster: the dense ceiling for these three
  # benchmarks is already measured, and it is the single most expensive cell
  # here. Put it back through REASONING_METHODS if the existing dense numbers
  # turn out not to merge -- they only do if they were taken at temperature .6
  # / top-p .9 / no top-k AND against kvpress_aime25_local.
  GEN=("$PY" "$HERE/gen_reasoning_queue.py" --out "$QUEUE"
       --tasks "${REASONING_TASKS:-math500,aime25,gpqa}"
       --datalen "${REASONING_DATALEN:-32768}"
       --methods "${REASONING_METHODS:-adaptive_centroid_lse_streaming_prefix4_querymean,quest_streaming,pariskv_author_common,retroinfer_author_common}"
       --budgets "${REASONING_BUDGETS:-1024,512}")
  # The smoke exists to price the run, and cost is bucketed by (task, method):
  # a second budget doubles it and measures nothing new. One draw, one budget.
  if [ "$PHASE" = smoke ]; then
    GEN+=(--budgets "${REASONING_BUDGETS:-1024,512}")
    GEN[${#GEN[@]}-1]=${GEN[${#GEN[@]}-1]%%,*}
  fi
  # One draw of everything is all a smoke needs; seeds are a full-phase concern.
  if [ "$PHASE" = smoke ]; then
    GEN+=(--seeds "math500:1,aime25:1,gpqa:1")
  elif [ -n "${REASONING_SEEDS:-}" ]; then
    GEN+=(--seeds "$REASONING_SEEDS")
  fi
  "${GEN[@]}" || exit 1
fi
[ -s "$QUEUE" ] || { echo "empty queue: $QUEUE" >&2; exit 1; }

# RetroInfer needs the authors' compiled kernels; a machine without them would
# claim those cells and fail on every one.
if grep -q retroinfer_author_common "$QUEUE"; then
  if ! "$PY" - <<'CHECK'
import sys
try:
    import torch  # noqa: F401   -- the extensions link libc10, load it first
    import retroinfer_kernels  # noqa: F401
    from weighted_flash_decoding import weighted_flash_decoding  # noqa: F401
except Exception as error:                      # pragma: no cover
    sys.exit(f"retroinfer kernels unavailable: {error}")
CHECK
  then
    echo "ERROR: queue holds RetroInfer cells but $SHADOWKV_MACHINE cannot import its kernels." >&2
    echo "Build them (PROJECT.md 3b step 4), or drop the method from REASONING_METHODS." >&2
    exit 2
  fi
fi

# The three datasets ship in the repo (repro/shadowkv/data/reasoning) and every
# env_m*.sh resolves them there, so a missing one means a stale checkout. Check
# before the pool starts: on a machine with HF_HUB_OFFLINE=1 a wrong path fails
# only after the model has loaded, one cell at a time.
for pair in "SHADOWKV_MATH500_PATH:math500" "SHADOWKV_AIME25_PATH:aime25" \
            "SHADOWKV_GPQA_PATH:gpqa"; do
  var=${pair%%:*} task=${pair##*:}
  grep -q "$task-s" "$QUEUE" || continue
  path=${!var:-}
  [ -n "$path" ] && [ -e "$path" ] && continue
  echo "ERROR: $task is queued but $var is '${path:-unset}', which is not on disk." >&2
  echo "git pull -- the three benchmarks are vendored under" >&2
  echo "repro/shadowkv/data/reasoning/ and env_m*.sh points there." >&2
  exit 2
done

# These cells are long generations, not long prompts: host memory is not the
# constraint the RULER pool caps, so lift its ceiling out of the way.
export SHADOWKV_MAX_LONG_CELLS=${SHADOWKV_MAX_LONG_CELLS:-8}

echo "[campaign] $SHADOWKV_MACHINE  phase=$PHASE  root=$ROOT  \
cells=$(grep -vc '^#' "$QUEUE")  samples=$NUM_SAMPLES  gpus=$SHADOWKV_POOL_GPUS"
if [ "$QUEUE_ONLY" = queue ]; then
  echo "[campaign] queue written, pool not started"
  exit 0
fi
SHADOWKV_RESULTS_ROOT=$ROOT exec "$HERE/run_pool.sh" "$QUEUE" "$ROOT/.state"
