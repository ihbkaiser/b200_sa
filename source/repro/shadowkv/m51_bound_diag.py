#!/usr/bin/env python
"""Why lowering M51's coverage does not buy what it looks like it should.

The stopping rule is  sigmoid(bound_over_unloaded - true_mass_loaded) <= 1-coverage.
Coverage therefore enters only as a THRESHOLD IN LOG SPACE: the rule stops when

    bound_unresolved - retained <= logit(1 - coverage)

and logit(1-c) spans just [-2.94, 0] nats over c in [0.50, 0.95]. Whatever slack
the bound carries per block competes against that same axis. So this script
measures the two things that decide the trade:

  * ORACLE depth -- blocks needed to hold `coverage` of the TRUE softmax mass,
    ranked by true mass. This is what coverage would cost with a perfect bound.
  * M51 depth    -- blocks the rule actually loads. The ratio is bound slack,
    expressed in the only unit that matters here (fraction of the cache read).

and the one that decides whether the failure is graceful or a cliff:

  * the RANK, in the bound's order, of the block that truly holds the most mass.
    If that rank is small, cutting depth drops low-mass blocks and accuracy
    degrades smoothly. If it is large, the top block survives only because the
    depth is large, and any cut throws the answer away outright.

Run per task: contrast a task the bound can resolve against one it cannot.
"""

import argparse, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))
import torch  # noqa: E402
from bench_latency import real_ruler_prompt, MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="qwen3")
    ap.add_argument("--datalen", type=int, default=8192)
    ap.add_argument("--task", default="niah_multikey_3")
    ap.add_argument("--m51_anchors", default=None)
    ap.add_argument("--m51_anchors_n", type=int, default=8)
    ap.add_argument("--m51_leaf", type=int, default=8)
    ap.add_argument("--m51_pq_mode", default="offline")
    ap.add_argument("--coverages", default="0.50,0.60,0.70,0.80,0.90,0.95")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    covs = [float(c) for c in args.coverages.split(",")]
    model_path = os.environ[MODELS[args.model_key]]
    from models import choose_model_class
    LLM = choose_model_class(model_path)
    llm = LLM(model_name=model_path, batch_size=1, device="cuda:0",
              max_length=args.datalen + 2048, attn_mode="m51",
              dtype=torch.bfloat16, sparse_budget=0, rank=160, chunk_size=8,
              m51_anchors=args.m51_anchors, m51_coverage=0.90,
              m51_leaf=args.m51_leaf, m51_anchors_n=args.m51_anchors_n,
              m51_pq_mode=args.m51_pq_mode, m51_group_select="per_query",
              m51_decode_mode="dense_mask", m51_load_batch=64, dense_layers=0)

    ids = real_ruler_prompt(llm, args.datalen, args.model_key, args.task)
    logits = llm.prefill(ids)
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    llm.kv_cache.H2D()

    cache = llm.kv_cache
    captured = []
    orig = type(cache).decode_attend

    def spy(self, layer_idx, query_states):
        captured.append((layer_idx, query_states.detach().clone()))
        return orig(self, layer_idx, query_states)

    type(cache).decode_attend = spy
    llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
    type(cache).decode_attend = orig

    L, ls = cache.n_leaves, cache.leaf_size
    H, G, D = cache.num_key_value_heads, cache.num_key_value_groups, cache.head_dim
    acc = {c: {"m51": [], "oracle": [], "got": []} for c in covs}
    slack, top1_rank, top1_frac = [], [], []

    with torch.inference_mode():
        for layer_idx, qs in captured:
            q = qs.view(H, G, D).to(torch.float32) * cache.scale
            upper = cache._upper_bounds(layer_idx, q)                       # [H,G,L]
            _, order_g, rev, _ = cache._order_and_tail(upper)
            kc = cache.k_cache[layer_idx][0, :, :L * ls].to(torch.float32)
            lg = torch.einsum('hgd,hnd->hgn', q, kc)
            true_mass = torch.logsumexp(lg.view(H, G, L, ls), dim=-1)       # [H,G,L]
            total = torch.logsumexp(true_mass, dim=-1, keepdim=True)

            # slack of the bound, per block, in nats
            slack.append((upper - true_mass).flatten())

            # where the truly-heaviest block sits in the BOUND's order
            best = true_mass.argmax(dim=-1)                                 # [H,G]
            rank = torch.empty_like(order_g)
            rank.scatter_(-1, order_g, torch.arange(L, device=q.device).expand_as(order_g))
            top1_rank.append(torch.gather(rank, -1, best.unsqueeze(-1)).squeeze(-1).flatten())
            top1_frac.append(torch.exp(true_mass.amax(-1, keepdim=True) - total).flatten())

            # depths
            ms = torch.gather(true_mass, -1, order_g)
            retained = torch.logcumsumexp(ms, dim=-1)
            unres = torch.cat([rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
            oracle_sorted = torch.sort(true_mass, dim=-1, descending=True).values
            oracle_ret = torch.logcumsumexp(oracle_sorted, dim=-1)
            oracle_frac = torch.exp(oracle_ret - total)

            for c in covs:
                ok = torch.sigmoid(unres - retained) <= (1.0 - c)
                k51 = torch.where(ok.any(-1), ok.float().argmax(-1) + 1,
                                  torch.full_like(ok[..., 0].long(), L))
                ok_o = oracle_frac >= c
                ko = torch.where(ok_o.any(-1), ok_o.float().argmax(-1) + 1,
                                 torch.full_like(ok_o[..., 0].long(), L))
                acc[c]["m51"].append(k51.flatten().float())
                acc[c]["oracle"].append(ko.flatten().float())
                # what the nominal coverage ACTUALLY buys: the true mass fraction
                # held by the prefix the rule stopped at. If this sits far above
                # the nominal target, the knob is not the thing setting the depth.
                got = torch.exp(torch.gather(retained, -1, (k51 - 1).unsqueeze(-1)).squeeze(-1)
                                - total.squeeze(-1))
                acc[c]["got"].append(got.flatten())

    slack = torch.cat(slack)
    top1_rank = torch.cat(top1_rank).float()
    top1_frac = torch.cat(top1_frac)
    rec = {"model": args.model_key, "datalen": args.datalen, "task": args.task,
           "L": L, "leaf": ls, "layers": len(captured),
           "slack_nats_median": float(slack.median()),
           "slack_nats_p90": float(slack.quantile(0.90)),
           "top1_mass_frac_median": float(top1_frac.median()),
           "top1_bound_rank_median": float(top1_rank.median()),
           "top1_bound_rank_p90": float(top1_rank.quantile(0.90)),
           "top1_bound_rank_p99": float(top1_rank.quantile(0.99)),
           "cov": {}}
    for c in covs:
        m = torch.cat(acc[c]["m51"]); o = torch.cat(acc[c]["oracle"])
        g = torch.cat(acc[c]["got"])
        rec["cov"][f"{c:.2f}"] = {
            "m51_depth_pct": float(m.mean()) / L * 100,
            "oracle_depth_pct": float(o.mean()) / L * 100,
            "slack_ratio": float(m.mean()) / max(float(o.mean()), 1e-9),
            "top1_kept_pct": float((top1_rank < m).float().mean()) * 100,
            "mass_actually_retained": float(g.mean()),
        }

    print(json.dumps(rec, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
