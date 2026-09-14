#!/usr/bin/env python
"""
How many tokens does each method ACTUALLY attend to at a given --sparse_budget?

The flag does not mean the same thing across methods, so matching the flag
value across a table does not match the compute or the information the methods
get. ShadowKV adds a context-scaled always-kept outlier region
(outlier_chunk * chunk_size) and a local window on top of the budget; Quest
adds only the trailing partial page.

Use this to pick budgets that match on attended tokens before building a table.

  source repro/shadowkv/env_m1.sh
  $PY repro/shadowkv/budget_audit.py --datalen 8192 --budgets 128,512,1024,2048
"""

import argparse
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))

import torch  # noqa: E402
from models.kv_cache import ShadowKVCache  # noqa: E402
from models.quest_streaming_cache import StreamingQuestCache  # noqa: E402
from outlier_policy import shadow_outlier_chunks  # noqa: E402

MODELS = {
    "llama32": dict(hidden_size=3072, num_attention_heads=24, num_key_value_heads=8,
                    num_hidden_layers=2),
    "qwen3": dict(hidden_size=2560, num_attention_heads=32, num_key_value_heads=8,
                  num_hidden_layers=2, head_dim=128),
}


def shadowkv_attended(cfg, datalen, budget, chunk_size):
    outliers = shadow_outlier_chunks(datalen, chunk_size)
    cache = ShadowKVCache(cfg, max_length=datalen + 2048, sparse_budget=budget,
                          chunk_size=chunk_size, rank=16,
                          outlier_chunk=outliers)
    k = torch.zeros(1, cfg.num_key_value_heads, datalen, cache.head_dim,
                    device="cuda", dtype=torch.bfloat16)
    for layer in range(cfg.num_hidden_layers):
        cache.get_svd(k, layer)
        cache.prefill_kv_cache(k, layer, k, k[:, :, -1:])
    out = (cache.sparse_end, cache.prefill_local, cache.outlier_chunk * cache.chunk_size)
    del cache, k
    torch.cuda.empty_cache()
    return out


def quest_attended(cfg, datalen, budget, page_size):
    cache = StreamingQuestCache(
        cfg, max_length=datalen + 2048, sparse_budget=budget,
        page_size=page_size, recent_tokens=32,
    )
    k = torch.zeros(1, cfg.num_key_value_heads, datalen, cache.head_dim,
                    device="cuda", dtype=torch.bfloat16)
    for layer in range(cfg.num_hidden_layers):
        cache.prefill_kv_cache(k, layer, k)
    state = cache.block_state[-1]
    exact = sum(end - start for start, end in state.exact_ranges)
    out = (budget + exact, exact, 0)
    del cache, k
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32", choices=sorted(MODELS))
    ap.add_argument("--datalen", type=int, default=8192)
    ap.add_argument("--budgets", default="128,512,1024,2048")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--page_size", type=int, default=16)
    args = ap.parse_args()

    cfg = types.SimpleNamespace(**MODELS[args.model])
    budgets = [int(b) for b in args.budgets.split(",") if b]

    print(f"=== attended tokens per KV head, {args.model} @ prefill {args.datalen} ===")
    print(f"{'flag':>8} | {'quest_streaming':>26} | {'shadowkv':>34}")
    print(f"{'budget':>8} | {'attended':>10} {'= budget+exact':>15} | "
          f"{'attended':>10} {'= budget+outlier+local':>23}")
    rows = []
    for budget in budgets:
        if budget >= args.datalen:
            print(f"{budget:>8} | skipped: budget >= datalen")
            continue
        q_end, q_local, _ = quest_attended(cfg, args.datalen, budget, args.page_size)
        s_end, s_local, s_out = shadowkv_attended(cfg, args.datalen, budget, args.chunk_size)
        rows.append((budget, q_end, s_end))
        print(f"{budget:>8} | {q_end:>10} {f'{budget}+{q_local}':>15} | "
              f"{s_end:>10} {f'{budget}+{s_out}+{s_local}':>23}")

    print("\nMatched-budget suggestions (equal attended tokens):")
    for budget, q_end, s_end in rows:
        # The streaming baseline adds the same exact recent/tail region at any
        # sparse budget, so subtract it before solving for the matched flag.
        need = s_end - q_local
        need -= need % args.page_size
        print(f"  shadowkv b{budget} attends {s_end} -> pair it with quest_streaming b{need} "
              f"(not quest_streaming b{budget}, which attends {q_end})")


if __name__ == "__main__":
    main()
