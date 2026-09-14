#!/usr/bin/env bash
# RULER 32K/64K/128K for the five methods, on whichever machine runs it.
#
#   source repro/shadowkv/env_m1.sh   # or env_m2.sh / env_m4.sh
#   repro/shadowkv/run_ruler_five_campaign.sh
#
# Each machine generates its own shard from the same generator and the same
# arguments, so the three shards partition the matrix without any file being
# copied between hosts.  Stop a machine by killing its pool first, then the
# running cell (the other order lets the pool claim again).
set -uo pipefail

: "${SHADOWKV_MACHINE:?source repro/shadowkv/env_mX.sh first}"
: "${SHADOWKV_RESULTS_ROOT:?source repro/shadowkv/env_mX.sh first}"
: "${PY:?source repro/shadowkv/env_mX.sh first}"

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
cd "$CODE"

ROOT=${RULER5_ROOT:-$SHADOWKV_RESULTS_ROOT/ruler_five_20260913}
mkdir -p "$ROOT"

# --- the protocol, as one block -------------------------------------------
# Every knob that changes a number lives here rather than in a default, so a
# cell run by hand on any machine reproduces a cell run by the pool.
export UPSTREAM_MATCHED_EXACT_REGIONS=1
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=256
export STREAMING_UPDATE_INTERVAL=256
# RULER ships exactly 100 examples per task per length; the pool's own default
# of 96 would silently discard four of them.
export NUM_SAMPLES=${NUM_SAMPLES:-100}
# Left at their run_cell.sh defaults on purpose, and named here so the record
# is explicit: STREAMING_ROUTER_BACKEND=triton, STREAMING_OFFLOAD=1,
# STREAMING_GATHER_REUSE=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

QUEUE=$ROOT/queue_$SHADOWKV_MACHINE.txt
# Generate the shards once.  A queue is a live document -- the pool re-reads it
# on every claim, so cells get moved between machines while the campaign runs to
# rebalance it.  Regenerating on every start would silently undo that, and the
# only symptom would be machines quietly redoing each other's work.
if [ -s "$QUEUE" ] && [ "${RULER5_REGEN:-0}" != 1 ]; then
  echo "[campaign] dùng shard sẵn có $QUEUE (RULER5_REGEN=1 để sinh lại)"
else
  "$PY" "$HERE/gen_ruler_five_queue.py" --out-prefix "$ROOT/queue" \
    --retroinfer-host m1 || exit 1
fi
[ -s "$QUEUE" ] || { echo "no shard for $SHADOWKV_MACHINE in $ROOT" >&2; exit 1; }

# RetroInfer needs the authors' compiled kernels.  A machine without them must
# not claim those cells and fail 39 times in a row.
if grep -q retroinfer_author_common "$QUEUE"; then
  if ! "$PY" - <<'CHECK'
import sys
try:
    # torch first: the extensions link against libc10 and cannot load alone.
    import torch  # noqa: F401
    import retroinfer_kernels  # noqa: F401
    from weighted_flash_decoding import weighted_flash_decoding  # noqa: F401
except Exception as error:                      # pragma: no cover
    sys.exit(f"retroinfer kernels unavailable: {error}")
CHECK
  then
    echo "ERROR: $SHADOWKV_MACHINE holds RetroInfer cells but cannot import its kernels." >&2
    echo "Build them, or regenerate with --retroinfer-host pointing elsewhere." >&2
    exit 2
  fi
fi

echo "[campaign] $SHADOWKV_MACHINE  root=$ROOT  cells=$(grep -vc '^#' "$QUEUE")  gpus=$SHADOWKV_POOL_GPUS"
SHADOWKV_RESULTS_ROOT=$ROOT exec "$HERE/run_pool.sh" "$QUEUE" "$ROOT/.state"
