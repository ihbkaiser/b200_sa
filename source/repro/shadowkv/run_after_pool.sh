#!/usr/bin/env bash
# Wait for a pool to finish, then run something. For leaving the machine.
#
#   source repro/shadowkv/env_m3.sh
#   R=$SHADOWKV_RESULTS_ROOT/reasoning_$(date +%Y%m%d)/full
#   setsid nohup repro/shadowkv/run_after_pool.sh \
#     --wait "$R/.state" "$R/queue_m3.txt" \
#     --wait "$L/.state" "$L/queue_m3.txt" \
#     -- repro/shadowkv/run_longbench_v2_campaign.sh > $R/../chain.log 2>&1 &
#
# `setsid` matters: without it the chain dies with the SSH session, and the
# whole point is to leave.
#
# Repeat --wait for every pool that must be finished first. Waiting on only one
# is a trap when the command starts a pool in a state dir some OTHER pool still
# holds: run_pool takes an exclusive flock there and the new one refuses, so
# the chain dies at 3am having done nothing.
#
# The wait condition is the pool's STATE, not its process. A pool process can
# outlive its work -- a worker parked on a card that never frees used to sleep
# forever with an empty queue -- and waiting on the process would then wait for
# ever. What actually matters is that nothing is running and nothing is left to
# claim, which the markers answer exactly:
#
#     running/ is empty   AND   done + failed >= queue lines
#
# Failed cells count as settled on purpose. A campaign that ends with three
# failures is finished; blocking the next one behind them would mean coming
# back to a machine that did nothing all night.
set -uo pipefail

USAGE="usage: run_after_pool.sh --wait <state_dir> <queue_file> [--wait ...] -- <command...>"
STATES=() QUEUES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wait) [ $# -ge 3 ] || { echo "$USAGE" >&2; exit 2; }
            STATES+=("$2"); QUEUES+=("$3"); shift 3 ;;
    --)     shift; break ;;
    *)      echo "$USAGE" >&2; exit 2 ;;
  esac
done
[ ${#STATES[@]} -gt 0 ] || { echo "$USAGE" >&2; exit 2; }
[ $# -gt 0 ] || { echo "no command to run afterwards" >&2; exit 2; }

INTERVAL=${RUN_AFTER_INTERVAL:-60}
# A ceiling, so a wedged pool eventually reports rather than waits silently.
DEADLINE=$(( $(date +%s) + ${RUN_AFTER_MAX_HOURS:-48} * 3600 ))

for i in "${!STATES[@]}"; do
  [ -d "${STATES[$i]}" ] || { echo "no such state dir: ${STATES[$i]}" >&2; exit 2; }
  [ -s "${QUEUES[$i]}" ] || { echo "no such queue: ${QUEUES[$i]}" >&2; exit 2; }
  echo "$(date -Is) [chain] waiting on ${STATES[$i]}"
done
echo "$(date -Is) [chain] then: $*"

while true; do
  pending="" report=""
  for i in "${!STATES[@]}"; do
    state=${STATES[$i]}
    running=$(ls "$state/running" 2>/dev/null | wc -l)
    done_n=$(ls "$state/done" 2>/dev/null | wc -l)
    failed=$(ls "$state/failed" 2>/dev/null | wc -l)
    # re-read the queue every pass: it is a live document and may have grown
    queued=$(grep -vc '^[[:space:]]*\(#\|$\)' "${QUEUES[$i]}")
    report="$report $(basename "$(dirname "$state")"):$((done_n + failed))+${running}r/$queued"
    if [ "$running" -ne 0 ] || [ $((done_n + failed)) -lt "$queued" ]; then
      pending="$pending $state"
    fi
  done
  if [ -z "$pending" ]; then
    echo "$(date -Is) [chain] all pools settled:$report"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -Is) [chain] GIVING UP after the time limit --$report." \
         "Nothing was started." >&2
    exit 1
  fi
  sleep "$INTERVAL"
done

echo "$(date -Is) [chain] starting: $*"
exec "$@"
