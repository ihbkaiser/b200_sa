#!/usr/bin/env bash
# Latency sweep: prefill and one decode step, every method, on ONE idle GPU,
# strictly one process at a time. Sharing a card moves latency by 3-5x, so the
# sequential run is the point, not an inefficiency to optimise away.
#
#   source repro/shadowkv/env_m1.sh
#   repro/shadowkv/bench_sweep.sh <gpu> [out.jsonl]
set -uo pipefail

: "${SHADOWKV_DIR:?source repro/shadowkv/env_m1.sh first}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU=${1:?usage: bench_sweep.sh <gpu> [out.jsonl]}
OUT=${2:-$SHADOWKV_RESULTS_ROOT/_bench/latency.jsonl}
mkdir -p "$(dirname "$OUT")"

# Budgets are token-matched: shadowkv b2048 attends 2464, while streaming
# Quest adds 32 recent tokens, so its matched sparse budget is 2432.
run() {
  echo "--- $* ---"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/bench_latency.py" --out "$OUT" "$@" \
    || echo "    FAILED: $*"
}

for MODEL in llama32 qwen3; do
  for LEN in 32768 65536; do
    run --model_key $MODEL --datalen $LEN --method full
    run --model_key $MODEL --datalen $LEN --method quest_streaming --sparse_budget 2048
    run --model_key $MODEL --datalen $LEN --method quest_streaming --sparse_budget 2432
    run --model_key $MODEL --datalen $LEN --method shadowkv --sparse_budget 2048
  done
done

echo
echo "=== $OUT ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
seen = {}
for r in rows:                     # last write wins, so a rerun replaces
    seen[(r["model"], r["datalen"], r["method"], r["sparse_budget"])] = r
for model in sorted({r["model"] for r in seen.values()}):
    for L in sorted({r["datalen"] for r in seen.values() if r["model"] == model}):
        sub = [r for r in seen.values() if r["model"] == model and r["datalen"] == L]
        base = next((r for r in sub if r["method"] == "full"), None)
        print(f"\n{model} @ {L//1024}K")
        print(f"  {'method':<16} {'prefill':>9} {'':>7} {'decode/step':>12} {'':>7} {'peak':>8}")
        for r in sorted(sub, key=lambda r: (r["method"], r["sparse_budget"])):
            name = r["method"] + ("" if r["method"] == "full" else f" b{r['sparse_budget']}")
            ps = f"x{r['prefill_s']/base['prefill_s']:.2f}" if base else ""
            ds = f"x{r['decode_ms']/base['decode_ms']:.2f}" if base else ""
            print(f"  {name:<16} {r['prefill_s']:8.3f}s {ps:>7} {r['decode_ms']:11.2f}ms {ds:>7} "
                  f"{r['peak_gib']:7.2f}G")
PYEOF
