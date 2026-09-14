#!/usr/bin/env bash
# LongBench-v2 at 128K for the five methods, on one machine.
#
#   source repro/shadowkv/env_m3.sh
#   repro/shadowkv/run_longbench_v2_campaign.sh
#   LBV2_BUDGETS=4096,2048,1024,512 repro/shadowkv/run_longbench_v2_campaign.sh
#   repro/shadowkv/run_longbench_v2_campaign.sh queue   # write the queue, stop
#
# A budget sweep EXTENDS the campaign rather than replacing it: new lines are
# appended to the existing queue, cells already in done/ are skipped by the
# pool, so the 4096 cells already paid for are not run again. `full` is emitted
# once for the whole sweep -- dense attention does not read the budget.
#
# Separate from the RULER campaign on purpose: LongBench-v2 ships 180 short and
# 215 medium examples, so a cell must take ALL of them.  The RULER launcher
# pins NUM_SAMPLES=100, which would silently truncate both bands.
set -uo pipefail

: "${SHADOWKV_MACHINE:?source repro/shadowkv/env_mX.sh first}"
: "${SHADOWKV_RESULTS_ROOT:?source repro/shadowkv/env_mX.sh first}"
: "${PY:?source repro/shadowkv/env_mX.sh first}"

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CODE=$(cd -- "$HERE/../.." && pwd -P)
cd "$CODE"

ROOT=${LBV2_ROOT:-$SHADOWKV_RESULTS_ROOT/longbench_v2_20260914}
mkdir -p "$ROOT"

export UPSTREAM_MATCHED_EXACT_REGIONS=1
export QUEST_PREFIX_TOKENS=32
export STREAMING_RECENT_TOKENS=256
export STREAMING_UPDATE_INTERVAL=256
# -1 = every example in the band.  180 short, 215 medium (PROJECT.md 5).
export NUM_SAMPLES=${NUM_SAMPLES:--1}

QUEUE_ONLY=${1:-}
case "$QUEUE_ONLY" in ""|queue) ;; *) echo "argument, if given, must be 'queue'" >&2; exit 2 ;; esac

QUEUE=$ROOT/queue_$SHADOWKV_MACHINE.txt
GEN=("$PY" "$HERE/gen_longbench_v2_queue.py")
[ -n "${LBV2_BUDGETS:-}" ] && GEN+=(--budgets "$LBV2_BUDGETS")
if [ -s "$QUEUE" ]; then
  # Append, never overwrite. A queue is a live document -- the pool re-reads it
  # on every claim and cells get reordered or moved between machines while it
  # runs -- so regenerating would silently undo that. Appending only the lines
  # that are not already there adds a sweep without touching what is running.
  TMPQ=$(mktemp) && "${GEN[@]}" --out "$TMPQ" >/dev/null || exit 1
  added=0
  while read -r new; do
    case "$new" in ""|\#*) continue ;; esac
    grep -qxF "$new" "$QUEUE" || { printf '%s\n' "$new" >> "$QUEUE"; added=$((added + 1)); }
  done < "$TMPQ"
  rm -f "$TMPQ"
  echo "[lbv2] queue sẵn có $QUEUE — thêm $added dòng mới"
else
  "${GEN[@]}" --out "$QUEUE" || exit 1
fi

if grep -q retroinfer_author_common "$QUEUE"; then
  if ! "$PY" - <<'CHECK'
import sys
try:
    import torch  # noqa: F401  (the extensions link against libc10)
    import retroinfer_kernels  # noqa: F401
    from weighted_flash_decoding import weighted_flash_decoding  # noqa: F401
except Exception as error:                      # pragma: no cover
    sys.exit(f"retroinfer kernels unavailable: {error}")
CHECK
  then
    echo "ERROR: $SHADOWKV_MACHINE thiếu kernel RetroInfer." >&2
    exit 2
  fi
fi

echo "[lbv2] $SHADOWKV_MACHINE root=$ROOT cells=$(grep -vc '^#' "$QUEUE") gpus=$SHADOWKV_POOL_GPUS"
if [ "$QUEUE_ONLY" = queue ]; then
  echo "[lbv2] queue written, pool not started"
  exit 0
fi
SHADOWKV_RESULTS_ROOT=$ROOT exec "$HERE/run_pool.sh" "$QUEUE" "$ROOT/.state"
