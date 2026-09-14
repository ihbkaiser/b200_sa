#!/usr/bin/env bash
# Claim-pool: one cell per GPU, cells claimed under flock from a queue file that
# is re-read on every claim, so the queue can be edited while the pool runs.
#
#   source repro/shadowkv/env_m1.sh
#   repro/shadowkv/run_pool.sh <queue_file> [state_root]
#
# Live controls (no restart needed):
#   * edit the queue file  -> changes take effect at the next claim
#   * edit <state>/gpus.txt -> deleting a line retires that GPU after its
#     current cell finishes; emptying the file drains the pool cleanly
#
# To stop: kill the POOL first, then the running cell (the other order either
# lets the pool claim again, or orphans a finished cell with no marker).
set -uo pipefail

: "${SHADOWKV_RESULTS_ROOT:?source repro/shadowkv/env_m1.sh first}"
: "${CUDA_DEVICE_ORDER:?must be PCI_BUS_ID -- nvidia-smi and CUDA disagree otherwise}"
[ "$CUDA_DEVICE_ORDER" = "PCI_BUS_ID" ] || { echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID"; exit 1; }

QUEUE=$(readlink -f "$1")
STATE=${2:-$SHADOWKV_RESULTS_ROOT/.state}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[ -s "$QUEUE" ] || { echo "queue file missing or empty: $QUEUE"; exit 1; }

mkdir -p "$STATE"/{done,running,failed} "$SHADOWKV_RESULTS_ROOT/_logs"

# One pool per state dir. Two pools sharing one would each see a card as free
# and put two cells on it.
exec 9>"$STATE/pool.lock"
flock -n 9 || { echo "another pool holds $STATE/pool.lock -- refusing to start"; exit 1; }

# GPU roster: a live file, seeded from the launch list.
if [ ! -s "$STATE/gpus.txt" ]; then
  tr ',' '\n' <<< "${SHADOWKV_POOL_GPUS}" | sed '/^$/d' > "$STATE/gpus.txt"
fi
# An empty roster starts zero workers and the pool exits at once, printing
# nothing that says why -- it reads as "the campaign finished instantly". That
# happened for real: an earlier invocation with SHADOWKV_POOL_GPUS="" left an
# empty gpus.txt behind, and the next launch inherited it. Emptying the file
# WHILE a pool runs is the documented way to drain it; finding it empty at
# startup is a mistake, so say so instead of exiting silently.
if [ ! -s "$STATE/gpus.txt" ]; then
  echo "empty GPU roster: SHADOWKV_POOL_GPUS is unset and $STATE/gpus.txt is empty." >&2
  echo "  Delete that file and relaunch, or set SHADOWKV_POOL_GPUS." >&2
  exit 1
fi
# --- other pools on this host ---------------------------------------------
# The flock above stops two pools sharing one state dir. It does nothing about
# two pools with DIFFERENT state dirs, which have exactly the same problem:
# each sees a card as free the moment the other's cell exits, and both put a
# cell on it. The window is real -- it is precisely how a second campaign gets
# started "on the idle cards" while the first one still has work.
#
# A pool that has already drained its queue is harmless: it can never claim
# again, its workers only exit. So the test is not "is another pool alive" but
# "can another pool with an overlapping roster still claim". Comparing counts
# is enough and needs no cell_key: a drained queue has every line accounted
# for in done, running or failed.
REGISTRY=${TMPDIR:-/tmp}/shadowkv_pools_$(id -u)
mkdir -p "$REGISTRY"
for entry in "$REGISTRY"/*; do
  [ -e "$entry" ] || continue
  # shellcheck disable=SC1090
  other_pid=$(sed -n 1p "$entry"); other_state=$(sed -n 2p "$entry")
  other_queue=$(sed -n 3p "$entry"); other_gpus=$(sed -n 4p "$entry")
  if ! kill -0 "$other_pid" 2>/dev/null; then rm -f "$entry"; continue; fi
  [ "$other_state" = "$STATE" ] && continue
  overlap=$(comm -12 <(sort -u "$STATE/gpus.txt") \
                     <(tr ',' '\n' <<< "$other_gpus" | sed '/^$/d' | sort -u) | tr '\n' ' ')
  [ -z "${overlap// /}" ] && continue
  queued=$(grep -vc '^[[:space:]]*\(#\|$\)' "$other_queue" 2>/dev/null || echo 0)
  settled=$(( $(ls "$other_state/done" 2>/dev/null | wc -l) \
            + $(ls "$other_state/running" 2>/dev/null | wc -l) \
            + $(ls "$other_state/failed" 2>/dev/null | wc -l) ))
  if [ "$queued" -gt "$settled" ]; then
    echo "REFUSING: pool $other_pid ($other_state) shares GPU(s) $overlap and still has" >&2
    echo "  $((queued - settled)) of $queued cell(s) to claim. Two pools racing for one" >&2
    echo "  card put two cells on it. Wait for it to drain, or take those GPUs out of" >&2
    echo "  its roster ($other_state/gpus.txt) before starting this one." >&2
    exit 1
  fi
  echo "[pool] pool $other_pid on GPU(s)$overlap has drained its queue -- proceeding"
done
printf '%s\n%s\n%s\n%s\n' "$$" "$STATE" "$QUEUE" \
  "$(tr '\n' ',' < "$STATE/gpus.txt")" > "$REGISTRY/$$"
trap 'rm -f "$REGISTRY/$$"' EXIT

# Host-memory ceiling on concurrent long-context cells (see claim_next).
LONG_TOKENS=${SHADOWKV_LONG_TOKENS:-131072}
MAX_LONG=${SHADOWKV_MAX_LONG_CELLS:-4}
echo "[pool] queue=$QUEUE state=$STATE roster=$(tr '\n' ',' < "$STATE/gpus.txt") \
long>=${LONG_TOKENS} capped at ${MAX_LONG}"

gpu_free() {
  # Binary: any compute process at all means busy. A card we cannot read is
  # busy too -- we never claim a card we cannot see clearly.
  local id=$1 out
  out=$(nvidia-smi --id="$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
  [[ "$out" =~ ^[[:space:]]*$ ]] || return 1
  return 0
}

claim_next() {
  # echoes a queue line, or nothing. Called under the claim lock.
  # With PROBE=1 it answers "is there anything left to claim" WITHOUT taking
  # it: same scan, same skip rules, no marker written. One definition of
  # claimable, because two would drift.
  local line cell
  while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    # shellcheck disable=SC2086
    # m51 cells carry the campaign's implementation variant in their name, so
    # the marker matches the file run_cell.sh will write
    if [ "$(echo "$line" | awk '{print $4}')" = m51 ]; then
      cell=$("$PY" "$HERE/cell_key.py" $line max "${SHADOWKV_M51_VARIANT:-v2}" 2>/dev/null | head -1)
    else
      cell=$("$PY" "$HERE/cell_key.py" $line 2>/dev/null | head -1)
    fi
    [ -n "$cell" ] || continue
    [ -e "$STATE/done/$cell" ] && continue
    [ -e "$STATE/running/$cell" ] && continue
    [ -e "$STATE/failed/$cell" ] && continue
    # A long-context cell pins host memory for its whole KV cache -- about
    # 19 GB at 128K for Qwen3-4B -- so the ceiling is the machine's RAM, not
    # its GPU count.  Seven of them at once does not fit in 188 GB.  Skip this
    # line rather than the whole claim: a shorter cell further down the queue
    # is still runnable, which is what keeps the spare cards busy.
    if [ "$(echo "$line" | awk '{print $2}')" -ge "$LONG_TOKENS" ]; then
      running_long=$(ls "$STATE/running" 2>/dev/null \
        | awk -F_ -v t="$LONG_TOKENS" '$2+0 >= t' | wc -l)
      [ "$running_long" -ge "$MAX_LONG" ] && continue
    fi
    [ "${PROBE:-0}" = 1 ] || : > "$STATE/running/$cell"
    echo "$line"
    return 0
  done < "$QUEUE"
  return 1
}

worker() {
  local gpu=$1 line cell
  while true; do
    grep -qx "$gpu" "$STATE/gpus.txt" 2>/dev/null || { echo "[gpu$gpu] retired from roster"; return 0; }

    if ! gpu_free "$gpu"; then
      # A worker parked on a card that never frees would sleep forever, because
      # the only place it learns the queue is drained is after the gpu_free
      # check. That leaves the pool alive with nothing to do -- and looking, in
      # pgrep, exactly like a second pool racing the first.
      if ! flock "$STATE/claim.lock" bash -c "$(declare -f claim_next); PROBE=1 STATE='$STATE' PY='$PY' QUEUE='$QUEUE' HERE='$HERE' LONG_TOKENS='$LONG_TOKENS' MAX_LONG='$MAX_LONG' claim_next" | grep -q .; then
        echo "[gpu$gpu] queue drained while card busy"; return 0
      fi
      sleep 30; continue
    fi

    # flock -c runs /bin/sh; declare -f emits bash syntax, so ask for bash explicitly
    line=$(flock "$STATE/claim.lock" bash -c "$(declare -f claim_next); STATE='$STATE' PY='$PY' QUEUE='$QUEUE' HERE='$HERE' LONG_TOKENS='$LONG_TOKENS' MAX_LONG='$MAX_LONG' claim_next")
    if [ -z "$line" ]; then echo "[gpu$gpu] queue drained"; return 0; fi

    # shellcheck disable=SC2086
    set -- $line
    if [ "$4" = m51 ]; then
      cell=$("$PY" "$HERE/cell_key.py" "$@" max "${SHADOWKV_M51_VARIANT:-v2}")
    else
      cell=$("$PY" "$HERE/cell_key.py" "$@")
    fi
    local log="$SHADOWKV_RESULTS_ROOT/_logs/${cell}_gpu${gpu}.log"
    echo "[gpu$gpu] $cell"
    if "$HERE/run_cell.sh" "$@" "${NUM_SAMPLES:-96}" "$gpu" > "$log" 2>&1; then
      mv "$STATE/running/$cell" "$STATE/done/$cell"
    else
      mv "$STATE/running/$cell" "$STATE/failed/$cell"
      echo "[gpu$gpu] FAILED $cell -- see $log"
    fi
  done
}

: > "$STATE/claim.lock"
pids=()
while read -r gpu; do
  [ -z "$gpu" ] && continue
  worker "$gpu" &
  pids+=($!)
done < "$STATE/gpus.txt"

echo "[pool] ${#pids[@]} worker(s) started; pool pid $$"
wait "${pids[@]}"
echo "[pool] all workers finished"
