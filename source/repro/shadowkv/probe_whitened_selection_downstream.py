#!/usr/bin/env python3
"""Does a better token score buy a better attention output? The end of the idea.

Recall of the true top-B is a proxy. What a sparse method actually produces is
an attention output computed over the B tokens it selected, and a selector can
miss half the top-B and still be exact if what it missed carried no mass. So
this measures the thing itself:

  * mass  -- the share of the true softmax weight that lands inside the
             selected set;
  * error -- ||o_selected - o_full|| / ||o_full||, the attention output the
             layer would actually pass on.

with an ORACLE row that selects by the true score. The oracle is the ceiling a
budget imposes no matter how good the compressor is, so the distance from a row
to the oracle is the part a better score can still recover -- and the distance
from the oracle to zero is the part it never can.

Same compressors and same bits-per-token axis as
probe_token_compressor_rate.py, plain basis against query-whitened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_token_compressor_rate import (  # noqa: E402
    lowrank_scores, pq_scores, pq_scores_allocated, psd_sqrt,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--datalen", type=int, default=32768)
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--fit-queries", type=int, default=256)
    ap.add_argument("--eval-queries", type=int, default=32)
    ap.add_argument("--budget", type=int, default=0, help="0 = datalen/32")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    wanted = {int(x) for x in args.layers.split(",") if x}
    budget = args.budget or args.datalen // 32
    from models import choose_model_class

    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            if index == args.sample_index:
                row = json.loads(line)
                break

    llm = choose_model_class(args.model)(
        model_name=args.model, batch_size=1, device="cuda:0",
        max_length=args.datalen + 2048, attn_mode="full", dtype=torch.bfloat16,
    )
    captured: dict[int, dict[str, torch.Tensor]] = {}
    counter = {"rope": 0, "pre": 0}
    original_rope = llm.apply_rotary_pos_emb
    original_pre = llm.pre_attention_compute

    def wrapped_pre(*a, **k):
        query, key, value = original_pre(*a, **k)
        layer = counter["pre"] % llm.num_layers
        counter["pre"] += 1
        if layer in wanted and query.shape[-2] > 1:
            captured.setdefault(layer, {})["v"] = value[0].detach().float()
        return query, key, value

    def wrapped_rope(query, key, position_ids):
        query, key = original_rope(query, key, position_ids)
        layer = counter["rope"] % llm.num_layers
        counter["rope"] += 1
        if layer in wanted and query.shape[-2] > 1:
            captured.setdefault(layer, {})["q"] = query[0].detach().float()
            captured[layer]["k"] = key[0].detach().float()
        return query, key

    llm.pre_attention_compute = wrapped_pre
    llm.apply_rotary_pos_emb = wrapped_rope
    ids = torch.tensor(llm.tokenizer.encode(row["input"], add_special_tokens=False),
                       device="cuda:0", dtype=torch.long)[None][:, : args.datalen]
    llm.prefill(ids)
    llm.pre_attention_compute = original_pre
    llm.apply_rotary_pos_emb = original_rope

    plans = [("pq  m2 b6", 12), ("pq  m4 b8", 32), ("pq  m8 b8", 64),
             ("wf  m8 t12", 12), ("wf  m8 t32", 32), ("wf  m8 t64", 64),
             ("lr  r8 b8", 64)]
    rows = {(n, b): [0.0, 0.0, 0] for n, _ in plans for b in ("plain", "white")}
    rows[("ORACLE", "-")] = [0.0, 0.0, 0]

    for layer in sorted(captured):
        q_all, k_all, v_all = (captured[layer]["q"], captured[layer]["k"],
                               captured[layer]["v"])
        heads_q, _, dim = q_all.shape
        heads_kv = k_all.shape[0]
        group = heads_q // heads_kv
        scale = dim ** -0.5
        for head in range(heads_kv):
            keys, values = k_all[head], v_all[head]
            grp = slice(head * group, (head + 1) * group)
            fit = q_all[grp, -args.fit_queries - args.eval_queries:
                        -args.eval_queries].reshape(-1, dim)
            held = q_all[grp, -args.eval_queries:].reshape(-1, dim)
            root, inverse = psd_sqrt(fit.T @ fit / fit.shape[0])
            keys_w, held_w = keys @ root, held @ inverse

            logits = (held @ keys.T) * scale
            weight = torch.softmax(logits, dim=1)
            output = weight @ values                                  # [Q, D]
            norm = output.norm(dim=1).clamp_min(1e-9)

            def evaluate(index):
                """exact attention restricted to the selected tokens"""
                picked = logits.gather(1, index)
                partial = torch.softmax(picked, dim=1)
                gathered = values[index]                              # [Q, B, D]
                approx = (partial.unsqueeze(1) @ gathered).squeeze(1)
                mass = weight.gather(1, index).sum(1).mean().item()
                err = ((approx - output).norm(dim=1) / norm).mean().item()
                return mass, err

            mass, err = evaluate(logits.topk(budget, dim=1).indices)
            slot = rows[("ORACLE", "-")]
            slot[0] += mass; slot[1] += err; slot[2] += 1

            for basis, kk, qq in (("plain", keys, held), ("white", keys_w, held_w)):
                eig = torch.linalg.eigh(kk.T @ kk / kk.shape[0])[1].flip(1)
                for name, _bits in plans:
                    kind, a, b = name.split()
                    if kind == "pq":
                        est = pq_scores(kk, qq, int(a[1:]), int(b[1:]))
                    elif kind == "wf":
                        est = pq_scores_allocated(kk @ eig, qq @ eig,
                                                  int(a[1:]), int(b[1:]))
                    else:
                        est = lowrank_scores(kk, qq, eig, int(a[1:]))
                    mass, err = evaluate(est.topk(budget, dim=1).indices)
                    slot = rows[(name, basis)]
                    slot[0] += mass; slot[1] += err; slot[2] += 1

    heads = rows[("ORACLE", "-")][2]
    print(f"\nheads {heads}  layers {sorted(captured)}  T={args.datalen}  "
          f"budget B={budget}  ({100 * budget / args.datalen:.1f}% of tokens)")
    om, oe, on = rows[("ORACLE", "-")]
    print(f"\n{'compressor':<11}{'bits':>6} | {'mass plain':>11}{'mass white':>11}"
          f" | {'err plain':>10}{'err white':>10}")
    print(f"{'ORACLE':<11}{'-':>6} | {om/on:>11.4f}{'':>11} | {oe/on:>10.4f}")
    for name, bits in plans:
        pm, pe, n = rows[(name, "plain")]
        wm, we, _ = rows[(name, "white")]
        print(f"{name:<11}{bits:>6} | {pm/n:>11.4f}{wm/n:>11.4f}"
              f" | {pe/n:>10.4f}{we/n:>10.4f}")
    print("\n  mass  = share of the true softmax weight inside the selected set")
    print("  err   = ||o_selected - o_full|| / ||o_full||, what the layer passes on")
    print("  ORACLE selects by the true score: the budget's own ceiling.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, handle, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
