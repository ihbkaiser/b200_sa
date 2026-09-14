#!/usr/bin/env python
"""Selection quality of M51, Quest and ShadowKV on ONE axis: error vs budget.

Every knob swept so far -- coverage, the envelope quantile, the envelope family --
moves M51 along a single curve. So the question is not where on its curve M51
sits, it is where its curve sits against the baselines'. This measures exactly
that, with all three selecting from the same prefill, scored by the quantity
RULER accuracy reads: the relative error of the attention OUTPUT against full
attention.

Faithful to each method's actual rule:
  quest_streaming pages of 16, score = sum_d max(q_d*min_d, q_d*max_d), max over the
             GQA group, fixed top-k. One selection per KV head.
  shadowkv   chunks of 8, landmark = chunk mean of the roped keys, softmax over
             chunks then max over the group, fixed top-k, PLUS the 48 outlier
             chunks whose keys sit furthest from their own landmark, which
             ShadowKV always keeps and which count against its budget here.
  m51_share  M51's routed-PQ bound, max over the group, fixed top-k. Selecting
             once per KV head is what makes its cost equal its budget, the same
             way Quest's and ShadowKV's do.
  m51_perq   M51's bound per query head. Cheaper per head, but a GQA group
             shares one cache, so the budget charged is the UNION.
  oracle     top-k blocks by true mass, per query head. The floor.

Budget is counted in TOKENS, so a method with 16-token pages and one with
8-token blocks are compared at the same cache traffic rather than the same k.
"""

import argparse, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))
import torch  # noqa: E402
from bench_latency import real_ruler_prompt, MODELS  # noqa: E402

OUTLIER_CHUNK = 48          # ShadowKV's default, models/kv_cache.py:138


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="qwen3")
    ap.add_argument("--datalen", type=int, default=8192)
    ap.add_argument("--task", default="niah_multikey_2")
    ap.add_argument("--m51_anchors", required=True)
    ap.add_argument("--m51_anchors_n", type=int, default=8)
    ap.add_argument("--m51_leaf", type=int, default=8)
    ap.add_argument("--m51_eta_q", type=float, default=0.999)
    ap.add_argument("--pq_subdim", type=int, default=8)
    ap.add_argument("--pq_mode", default="offline")
    ap.add_argument("--pq_warm_iters", type=int, default=2)
    ap.add_argument("--budgets", default="256,512,1024,2048,4096")
    ap.add_argument("--quest_page", type=int, default=16)
    ap.add_argument("--sk_chunk", type=int, default=8)
    ap.add_argument("--layers", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    model_path = os.environ[MODELS[args.model_key]]
    from models import choose_model_class
    LLM = choose_model_class(model_path)
    llm = LLM(model_name=model_path, batch_size=1, device="cuda:0",
              max_length=args.datalen + 2048, attn_mode="m51",
              dtype=torch.bfloat16, sparse_budget=0, rank=160, chunk_size=8,
              m51_anchors=args.m51_anchors, m51_coverage=0.90,
              m51_leaf=args.m51_leaf, m51_anchors_n=args.m51_anchors_n,
              m51_pq_mode=args.pq_mode, m51_pq_warm_iters=args.pq_warm_iters, m51_group_select="per_query",
              m51_decode_mode="dense_mask", m51_load_batch=64, dense_layers=0,
              m51_eta_q=args.m51_eta_q, m51_pq_subdim=args.pq_subdim)

    ids = real_ruler_prompt(llm, args.datalen, args.model_key, args.task)
    logits = llm.prefill(ids)
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    llm.kv_cache.H2D()
    cache = llm.kv_cache

    cap = []
    orig = type(cache).decode_attend
    type(cache).decode_attend = lambda s, li, q: (cap.append((li, q.detach().clone())),
                                                 orig(s, li, q))[1]
    llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
    type(cache).decode_attend = orig

    L, ls = cache.n_leaves, cache.leaf_size
    H, G, D = cache.num_key_value_heads, cache.num_key_value_groups, cache.head_dim
    N = L * ls
    want = set(int(x) for x in args.layers.split(",") if x) or None
    # quest_streaming_p8 is not a configuration Quest ships (page 8 doubles its metadata to
    # 12.5% of KV bytes). It is here to separate SCORING from GRANULARITY: M51
    # ranks 8-token blocks, Quest 16-token pages, and at a fixed token budget the
    # finer unit is an advantage on its own.
    # m51_lin drops the envelope from the RANKING. eta exists to make the score
    # an upper bound; for choosing which blocks to read, a bound is the wrong
    # object -- it promotes whatever is most uncertain, not what is heaviest.
    # m51_sm copies ShadowKV's group reduction (softmax over blocks, then max),
    # which normalises each query head before the heads are compared.
    # m51_out copies ShadowKV's other half: always keep the blocks whose keys sit
    # furthest from their own summary, which is exactly where a mean-like score
    # is least trustworthy. Those blocks are charged against the budget.
    # m51_min: every anchor yields a valid bound, so the MINIMUM over anchors is
    # tighter than the one the nearest-L2 route happens to pick. Routing was
    # chosen to make the scan cost O(1) in M; this measures what that cost buys.
    METH = ["oracle", "quest_streaming", "quest_streaming_p8", "shadowkv", "m51_share", "m51_perq",
            "m51_lin", "m51_sm", "m51_out", "m51_lin_out", "m51_min", "m51_minsm", "m51_min2", "m51_min4"]
    acc = {m: {b: {"err": [], "tok": []} for b in budgets} for m in METH}
    sc = 1.0 / (D ** 0.5)

    with torch.inference_mode():
        for layer_idx, qq in cap:
            if want is not None and layer_idx not in want:
                continue
            keys = cache.k_cache[layer_idx][0, :, :N].float()                # [H,N,D]
            vals = cache.v_cache[layer_idx][0, :, :N].float()
            qs = qq.view(H, G, D).float()
            lgt = torch.einsum('hgd,hnd->hgn', qs * sc, keys)
            o_full = torch.einsum('hgn,hnd->hgd', torch.softmax(lgt, -1), vals)
            onorm = o_full.norm(dim=-1).clamp(min=1e-6)

            def score_err(tokmask):
                p = torch.softmax(lgt.masked_fill(~tokmask, float('-inf')), -1)
                o = torch.einsum('hgn,hnd->hgd', p, vals)
                return ((o - o_full).norm(dim=-1) / onorm).flatten()

            def blocks_to_tokens(sel, bs):     # sel [H,G,nb] bool -> [H,G,N]
                return sel.unsqueeze(-1).expand(*sel.shape, bs).reshape(*sel.shape[:-1], -1)

            # ---- oracle: true block mass, per query head
            tm = torch.logsumexp(lgt.view(H, G, L, ls), -1)
            ord_o = tm.argsort(-1, descending=True)

            # ---- streaming Quest
            P = args.quest_page
            np_ = N // P
            pg = keys[:, :np_ * P].view(H, np_, P, D)
            pmin, pmax = pg.amin(2), pg.amax(2)                              # [H,np,D]
            qe = (qs * sc).unsqueeze(2)                                      # [H,G,1,D]
            qsc = torch.maximum(qe * pmin.unsqueeze(1),
                                qe * pmax.unsqueeze(1)).sum(-1)              # [H,G,np]
            ord_q = qsc.amax(1).argsort(-1, descending=True)                 # [H,np]

            P8 = 8
            np8 = N // P8
            pg8 = keys[:, :np8 * P8].view(H, np8, P8, D)
            p8min, p8max = pg8.amin(2), pg8.amax(2)
            q8 = torch.maximum(qe * p8min.unsqueeze(1), qe * p8max.unsqueeze(1)).sum(-1)
            ord_q8 = q8.amax(1).argsort(-1, descending=True)                 # [H,np8]

            # ---- shadowkv
            C = args.sk_chunk
            nc = N // C
            ch = keys[:, :nc * C].view(H, nc, C, D)
            lm = ch.mean(2)                                                  # [H,nc,D]
            cos = torch.nn.functional.cosine_similarity(
                lm.unsqueeze(2).expand(-1, -1, C, -1), ch, dim=-1)           # [H,nc,C]
            out_idx = cos.min(-1).values.topk(OUTLIER_CHUNK, largest=False).indices
            is_out = torch.zeros(H, nc, dtype=torch.bool, device=keys.device)
            is_out.scatter_(1, out_idx, True)
            ca = torch.softmax(torch.einsum('hgd,hcd->hgc', qs * sc, lm), -1)
            ca = ca.masked_fill(is_out.unsqueeze(1), float('-inf'))          # outliers are not ranked
            ord_s = ca.amax(1).argsort(-1, descending=True)                  # [H,nc]

            # ---- m51
            up = cache._upper_bounds(layer_idx, qs * sc)                     # [H,G,L]
            ord_m = up.argsort(-1, descending=True)
            ord_ms = up.amax(1).argsort(-1, descending=True)                 # [H,L]

            # the same bound with the envelope removed: off + q.PQ(g)
            qsc_ = qs * sc
            anc = cache.anchors[layer_idx]
            rt = torch.cdist(qsc_, anc).argmin(dim=-1)                       # [H,G]
            eta_r = cache.eta[layer_idx][
                torch.arange(H, device=keys.device).unsqueeze(1), rt,
                torch.arange(G, device=keys.device).unsqueeze(0)]            # [H,G,L]
            lin_m = up - eta_r
            ord_l = lin_m.amax(1).argsort(-1, descending=True)
            ord_sm = torch.softmax(lin_m, dim=-1).amax(1).argsort(-1, descending=True)

            # bounds from EVERY anchor, not just the routed one
            cb_ = cache.codebook[layer_idx]                                   # [H,S,C,sub]
            qsub = (qsc_).view(H, G, cache.n_sub, cache.pq_subdim)
            lut = torch.einsum('hgsp,hscp->hgsc', qsub, cb_)                  # [H,G,S,C]
            cd_all = cache.codes[layer_idx].long()                            # [H,S,M,L]
            Mn = cache.n_anchors
            sc_all = torch.stack([
                torch.gather(lut[:, :, sub], 2,
                             cd_all[:, sub].reshape(H, 1, Mn * L)
                             .expand(H, G, Mn * L)).view(H, G, Mn, L)
                for sub in range(cache.n_sub)], 0).sum(0)                     # [H,G,M,L]
            up_all = (sc_all + cache.offsets[layer_idx].unsqueeze(1)
                      + cache.eta[layer_idx].permute(0, 2, 1, 3))             # [H,G,M,L]
            up_min = up_all.amin(dim=2)                                       # [H,G,L]
            ord_mn = up_min.amax(1).argsort(-1, descending=True)
            ord_mnsm = torch.softmax(up_min, dim=-1).amax(1).argsort(-1, descending=True)
            # the same, restricted to the n nearest anchors -- the decode scan
            # cost is linear in n, so this is what a cheaper version would rank with
            near = torch.cdist(qsc_, anc).argsort(dim=-1)                     # [H,G,M]
            hq = torch.arange(H, device=keys.device).view(H, 1, 1)
            gq = torch.arange(G, device=keys.device).view(1, G, 1)
            ord_n = {}
            for nn in (2, 4):
                sub_up = up_all[hq, gq, near[..., :nn]].amin(dim=2)
                ord_n[nn] = torch.softmax(sub_up, dim=-1).amax(1).argsort(-1, descending=True)
            del up_all, sc_all

            # outlier blocks: min cosine between a block's keys and its routed tangent
            ghat_r = cache.codebook[layer_idx]                               # decode via codes
            cds = cache.codes[layer_idx].transpose(1, 2)[
                torch.arange(H, device=keys.device).unsqueeze(1), rt].long()  # [H,G,S,L]
            gh = torch.cat([torch.gather(
                    ghat_r[:, sub].unsqueeze(1).expand(H, G, cache.pq_codes, cache.pq_subdim), 2,
                    cds[:, :, sub].unsqueeze(-1).unsqueeze(-1)
                    .expand(H, G, L, cache.pq_subdim)).squeeze(2)
                 if False else torch.gather(
                    ghat_r[:, sub], 1,
                    cds[:, :, sub].reshape(H, G * L).unsqueeze(-1)
                    .expand(H, G * L, cache.pq_subdim)).view(H, G, L, cache.pq_subdim)
                 for sub in range(cache.n_sub)], dim=-1)                      # [H,G,L,D]
            lv = keys.view(H, L, ls, D)
            cosm = torch.nn.functional.cosine_similarity(
                gh.unsqueeze(3), lv.unsqueeze(1), dim=-1).amin(-1)            # [H,G,L]
            out_m = cosm.amin(1).topk(OUTLIER_CHUNK, dim=-1, largest=False).indices
            is_out_m = torch.zeros(H, L, dtype=torch.bool, device=keys.device)
            is_out_m.scatter_(1, out_m, True)

            ar_L = torch.arange(L, device=keys.device)
            for b in budgets:
                # oracle / m51_perq: per query head, k blocks each
                k = max(1, b // ls)
                rk = torch.empty_like(ord_o); rk.scatter_(-1, ord_o, ar_L.expand_as(ord_o))
                acc["oracle"][b]["err"].append(score_err(blocks_to_tokens(rk < k, ls)))
                acc["oracle"][b]["tok"].append(torch.full((H * G,), float(k * ls)))

                rk = torch.empty_like(ord_m); rk.scatter_(-1, ord_m, ar_L.expand_as(ord_m))
                sel = rk < k
                acc["m51_perq"][b]["err"].append(score_err(blocks_to_tokens(sel, ls)))
                # the group shares one cache: charge the union, per KV head
                acc["m51_perq"][b]["tok"].append(
                    sel.any(1).sum(-1).float().repeat_interleave(G) * ls)

                # m51_share: one order per KV head -> cost == budget
                selh = torch.zeros(H, L, dtype=torch.bool, device=keys.device)
                selh.scatter_(1, ord_ms[:, :k], True)
                acc["m51_share"][b]["err"].append(
                    score_err(blocks_to_tokens(selh.unsqueeze(1).expand(H, G, L), ls)))
                acc["m51_share"][b]["tok"].append(torch.full((H * G,), float(k * ls)))

                for nm, o in (("m51_lin", ord_l), ("m51_sm", ord_sm),
                              ("m51_min", ord_mn), ("m51_minsm", ord_mnsm),
                              ("m51_min2", ord_n[2]), ("m51_min4", ord_n[4])):
                    sh = torch.zeros(H, L, dtype=torch.bool, device=keys.device)
                    sh.scatter_(1, o[:, :k], True)
                    acc[nm][b]["err"].append(
                        score_err(blocks_to_tokens(sh.unsqueeze(1).expand(H, G, L), ls)))
                    acc[nm][b]["tok"].append(torch.full((H * G,), float(k * ls)))

                ko = max(0, (b - OUTLIER_CHUNK * ls) // ls)
                for nm, o in (("m51_out", ord_ms), ("m51_lin_out", ord_l)):
                    sh = is_out_m.clone()
                    if ko:
                        # walk the order and take the first ko entries that are
                        # not already held as outliers
                        free = ~is_out_m.gather(1, o)
                        take = free & (free.cumsum(-1) <= ko)
                        sh = sh | torch.zeros_like(sh).scatter_(1, o, take)
                    acc[nm][b]["err"].append(
                        score_err(blocks_to_tokens(sh.unsqueeze(1).expand(H, G, L), ls)))
                    acc[nm][b]["tok"].append(sh.sum(-1).float().repeat_interleave(G) * ls)

                kq = max(1, b // P)
                selq = torch.zeros(H, np_, dtype=torch.bool, device=keys.device)
                selq.scatter_(1, ord_q[:, :kq], True)
                m = torch.zeros(H, N, dtype=torch.bool, device=keys.device)
                m[:, :np_ * P] = blocks_to_tokens(selq, P)
                acc["quest_streaming"][b]["err"].append(score_err(m.unsqueeze(1).expand(H, G, N)))
                acc["quest_streaming"][b]["tok"].append(torch.full((H * G,), float(kq * P)))

                kq8 = max(1, b // P8)
                selq8 = torch.zeros(H, np8, dtype=torch.bool, device=keys.device)
                selq8.scatter_(1, ord_q8[:, :kq8], True)
                m = torch.zeros(H, N, dtype=torch.bool, device=keys.device)
                m[:, :np8 * P8] = blocks_to_tokens(selq8, P8)
                acc["quest_streaming_p8"][b]["err"].append(score_err(m.unsqueeze(1).expand(H, G, N)))
                acc["quest_streaming_p8"][b]["tok"].append(torch.full((H * G,), float(kq8 * P8)))

                ks = max(0, (b - OUTLIER_CHUNK * C) // C)
                sels = is_out.clone()
                if ks:
                    sels.scatter_(1, ord_s[:, :ks], True)
                m = torch.zeros(H, N, dtype=torch.bool, device=keys.device)
                m[:, :nc * C] = blocks_to_tokens(sels, C)
                acc["shadowkv"][b]["err"].append(score_err(m.unsqueeze(1).expand(H, G, N)))
                acc["shadowkv"][b]["tok"].append(
                    sels.sum(-1).float().repeat_interleave(G) * C)
            del keys, vals, lgt, o_full

    rec = {"tag": args.tag, "model": args.model_key, "datalen": args.datalen,
           "task": args.task, "M": cache.n_anchors, "leaf": ls,
           "eta_q": args.m51_eta_q, "pq_mode": args.pq_mode, "methods": {}}
    for m in METH:
        rec["methods"][m] = {str(b): {
            "tokens": float(torch.cat(acc[m][b]["tok"]).mean()),
            "tok_frac": float(torch.cat(acc[m][b]["tok"]).mean()) / N,
            "relerr_median": float(torch.cat(acc[m][b]["err"]).median()),
            "relerr_p90": float(torch.cat(acc[m][b]["err"]).quantile(0.9)),
        } for b in budgets}
    print(json.dumps(rec, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
