#!/usr/bin/env python
"""Second envelope sweep: the two knobs the first lab showed I had not touched.

Lab 1 killed the obvious ideas. Scaling the envelope by ||q-u||^2 made it WORSE
(a ratio-quantile is set by the queries nearest the anchor, which then inflates
it for every far query); the certified diameter bound is 180 nats loose; and a
key-free one-shot rule needs a LOWER bound whose PQ term swamps it. What lab 1
did surface is that today's envelope is not a bound at all -- it is violated on
19.5% of decode blocks -- and yet the heaviest block survives 97.9% of the time.

If the envelope is already a heuristic score, then the quantile it is fitted at
is a free knob, and nobody has swept it. That is knob one. Knob two is the
metadata budget: M51 spends 4.30% of KV bytes against Quest's 6.25%, so there is
room for more anchors -- and a bigger leaf pays for them outright, since every
per-block term halves when the block doubles.

The metric is no longer top1-kept, which saturates. It is the RELATIVE ERROR OF
THE ATTENTION OUTPUT against full attention -- the quantity RULER accuracy
actually reads.
"""

import argparse, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))
import torch  # noqa: E402
from bench_latency import real_ruler_prompt, MODELS  # noqa: E402


def qtile(x, q):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    out = torch.empty(flat.shape[1], device=x.device, dtype=torch.float32)
    step = max(1, (1 << 22) // max(1, n))
    for i in range(0, flat.shape[1], step):
        out[i:i + step] = torch.quantile(flat[:, i:i + step].float(), q, dim=0)
    return out.view(x.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="qwen3")
    ap.add_argument("--datalen", type=int, default=8192)
    ap.add_argument("--task", default="niah_multikey_2")
    ap.add_argument("--m51_anchors", required=True)
    ap.add_argument("--m51_anchors_n", type=int, default=8)
    ap.add_argument("--m51_leaf", type=int, default=8)
    ap.add_argument("--pq_subdim", type=int, default=8)
    ap.add_argument("--quantiles", default="0.90,0.95,0.99,0.999")
    ap.add_argument("--coverages", default="0.70,0.80,0.90,0.95")
    ap.add_argument("--layers", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qs_list = [float(x) for x in args.quantiles.split(",")]
    covs = [float(c) for c in args.coverages.split(",")]
    model_path = os.environ[MODELS[args.model_key]]
    from models import choose_model_class
    LLM = choose_model_class(model_path)
    llm = LLM(model_name=model_path, batch_size=1, device="cuda:0",
              max_length=args.datalen + 2048, attn_mode="m51",
              dtype=torch.bfloat16, sparse_budget=0, rank=160, chunk_size=8,
              m51_anchors=args.m51_anchors, m51_coverage=0.90,
              m51_leaf=args.m51_leaf, m51_anchors_n=args.m51_anchors_n,
              m51_pq_mode="per_context", m51_group_select="per_query",
              m51_decode_mode="dense_mask", m51_load_batch=64, dense_layers=0,
              m51_pq_subdim=args.pq_subdim)

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
    H, G, D, M = (cache.num_key_value_heads, cache.num_key_value_groups,
                  cache.head_dim, cache.n_anchors)
    S = cache.n_sub
    want = set(int(x) for x in args.layers.split(",") if x) or None

    keys_ = [(f, q) for f in ("const", "affine") for q in qs_list]
    st = {f"{f}@{q}": {"viol": [], "depth": {c: [] for c in covs},
                       "relerr": {c: [] for c in covs}} for f, q in keys_}
    oracle = {c: [] for c in covs}
    oracle_rel = {c: [] for c in covs}

    with torch.inference_mode():
        for layer_idx, qq in cap:
            if want is not None and layer_idx not in want:
                continue
            keys = cache.k_cache[layer_idx][0, :, :L * ls].float()
            vals = cache.v_cache[layer_idx][0, :, :L * ls].float()
            leaves = keys.view(H, L, ls, D)
            anchors = cache.anchors[layer_idx]

            al = torch.einsum('hmd,hltd->hmlt', anchors, leaves)
            w = torch.softmax(al, dim=-1)
            g_ex = torch.einsum('hmlt,hltd->hmld', w, leaves)
            offs = torch.logsumexp(al, dim=-1) - torch.einsum('hmd,hmld->hml', anchors, g_ex)
            del al, w
            codes = cache.codes[layer_idx].long()
            cb = cache.codebook[layer_idx]
            ghat = torch.empty_like(g_ex)
            for sub in range(S):
                sl = slice(sub * cache.pq_subdim, (sub + 1) * cache.pq_subdim)
                pick = codes[..., sub].reshape(H, M * L)
                ghat[..., sl] = torch.gather(
                    cb[:, sub], 1, pick.unsqueeze(-1).expand(-1, -1, cache.pq_subdim)
                ).view(H, M, L, cache.pq_subdim)
            del g_ex

            calib = cache.calib[layer_idx]
            Nc = calib.shape[2]
            d2c = torch.cdist(calib.reshape(H, G * Nc, D), anchors) ** 2
            route_c = d2c.argmin(dim=-1)
            env = {k: torch.zeros(H, M, G, L, device=keys.device) for k in st}
            beta = torch.zeros(H, M, G, L, device=keys.device)
            for h in range(H):
                qh = calib[h].reshape(G * Nc, D)
                tm = torch.logsumexp((qh @ keys[h].T).view(G * Nc, L, ls), dim=-1)
                for m in range(M):
                    lin = offs[h, m].unsqueeze(0) + qh @ ghat[h, m].T
                    err = (tm - lin).view(G, Nc, L)
                    dd = d2c[h, :, m].view(G, Nc)
                    for g in range(G):
                        s = route_c[h].view(G, Nc)[g]
                        e = err[g][s] if bool(s.any()) else err[g]
                        x = dd[g][s] if bool(s.any()) else dd[g]
                        # affine: least-squares slope on d^2, then a quantile of
                        # the residual. Unlike a ratio-quantile this cannot be
                        # hijacked by the queries sitting on top of the anchor.
                        xc = x - x.mean()
                        var = float((xc * xc).sum())
                        b = ((xc.unsqueeze(-1) * (e - e.mean(0))).sum(0) / var) if var > 1e-9 \
                            else torch.zeros(L, device=keys.device)
                        b = b.clamp(min=0)
                        beta[h, m, g] = b
                        res = e - x.unsqueeze(-1) * b
                        for qv in qs_list:
                            env[f"const@{qv}"][h, m, g] = qtile(e, qv)
                            env[f"affine@{qv}"][h, m, g] = qtile(res, qv)
                    del err
                del tm

            q = qq.view(H, G, D).float() * cache.scale
            d2 = torch.cdist(q, anchors) ** 2
            route = d2.argmin(dim=-1)
            hh = torch.arange(H, device=q.device).unsqueeze(1)
            gg = torch.arange(G, device=q.device).unsqueeze(0)
            dmin = d2.gather(-1, route.unsqueeze(-1)).squeeze(-1)
            lin = offs[hh, route] + torch.einsum('hgd,hgld->hgl', q, ghat[hh, route])
            lgt = torch.einsum('hgd,hnd->hgn', q, keys)
            true_mass = torch.logsumexp(lgt.view(H, G, L, ls), dim=-1)
            total = torch.logsumexp(true_mass, -1, keepdim=True)
            p_full = torch.softmax(lgt, dim=-1)
            o_full = torch.einsum('hgn,hnd->hgd', p_full, vals)
            onorm = o_full.norm(dim=-1).clamp(min=1e-6)

            def relerr(loaded):                    # loaded: [H,G,L] bool
                mask = loaded.unsqueeze(-1).expand(H, G, L, ls).reshape(H, G, L * ls)
                p = torch.softmax(lgt.masked_fill(~mask, float('-inf')), dim=-1)
                o = torch.einsum('hgn,hnd->hgd', p, vals)
                return ((o - o_full).norm(dim=-1) / onorm).flatten()

            os_ = torch.sort(true_mass, -1, descending=True)
            of = torch.exp(torch.logcumsumexp(os_.values, -1) - total)
            rk_o = torch.empty_like(os_.indices)
            rk_o.scatter_(-1, os_.indices, torch.arange(L, device=q.device).expand_as(os_.indices))
            for c in covs:
                ok = of >= c
                k = torch.where(ok.any(-1), ok.float().argmax(-1) + 1,
                                torch.full_like(ok[..., 0].long(), L))
                oracle[c].append(k.flatten().float())
                oracle_rel[c].append(relerr(rk_o < k.unsqueeze(-1)))

            for name, e in env.items():
                fam = name.split("@")[0]
                upper = lin + e[hh, route, gg] + (beta[hh, route, gg] * dmin[..., None]
                                                 if fam == "affine" else 0.0)
                st[name]["viol"].append((upper < true_mass).flatten())
                order = upper.argsort(-1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(-1, order, torch.arange(L, device=q.device).expand_as(order))
                us = torch.gather(upper, -1, order)
                rev = torch.flip(torch.logcumsumexp(torch.flip(us, [-1]), -1), [-1])
                unres = torch.cat([rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
                ret = torch.logcumsumexp(torch.gather(true_mass, -1, order), -1)
                for c in covs:
                    ok = torch.sigmoid(unres - ret) <= (1.0 - c)
                    k = torch.where(ok.any(-1), ok.float().argmax(-1) + 1,
                                    torch.full_like(ok[..., 0].long(), L))
                    st[name]["depth"][c].append(k.flatten().float())
                    st[name]["relerr"][c].append(relerr(rank < k.unsqueeze(-1)))
            del keys, vals, leaves, ghat, offs, lin, lgt, true_mass, env, beta

    block_kv = ls * D * 4
    meta_const = (M * S * 6 / 8 + M * 2 + M * G * 2) / block_kv
    meta_affine = (M * S * 6 / 8 + M * 2 + M * G * 4) / block_kv
    rec = {"tag": args.tag, "model": args.model_key, "datalen": args.datalen,
           "task": args.task, "M": M, "leaf": ls, "L": L, "pq_subdim": cache.pq_subdim,
           "quest_meta_frac": 2 * D * 2 / (16 * D * 4),
           "oracle": {f"{c:.2f}": {
               "depth_pct": float(torch.cat(oracle[c]).mean()) / L * 100,
               "relerr_median": float(torch.cat(oracle_rel[c]).median())} for c in covs},
           "variants": {}}
    for name, s in st.items():
        fam = name.split("@")[0]
        rec["variants"][name] = {
            "meta_frac_of_kv": meta_const if fam == "const" else meta_affine,
            "violation_pct": float(torch.cat(s["viol"]).float().mean()) * 100,
            "cov": {f"{c:.2f}": {
                "depth_pct": float(torch.cat(s["depth"][c]).mean()) / L * 100,
                "relerr_median": float(torch.cat(s["relerr"][c]).median()),
                "relerr_p90": float(torch.cat(s["relerr"][c]).quantile(0.9)),
            } for c in covs}}
    print(json.dumps(rec, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
