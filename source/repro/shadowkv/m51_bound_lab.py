#!/usr/bin/env python
"""Bench of ENVELOPE variants for M51, evaluated post-hoc from one prefill.

The diagnostic (m51_bound_diag.py) established where M51's cost comes from: the
bound ranks blocks well (the heaviest block is rank 0 in the median case) but
over-states their VALUE by ~1 nat median / ~3.5 nats p90, and the tail logsumexp
of that over-statement is what sets the load depth. The coverage knob spans only
2.94 nats across its whole range, so it cannot move against that.

So this sweeps the envelope, which is the term that carries the slack:

  v0_const   eta[h,m,g,l]                 -- what ships today: one q99.9 constant
                                             per (anchor, group, block).
  v1_quad    c[h,m,g,l] * ||q-u_m||^2     -- same storage, but scaled by how far
                                             the query actually is from its anchor.
                                             The tangent gap IS quadratic in that
                                             distance, so a constant has to cover
                                             the worst query in the cell.
  v2_quad_l  c[h,l]     * ||q-u_m||^2     -- drops the (anchor, group) axes: 1
                                             scalar per block instead of M*G.
  v3_quad_a  c[h,m,g]   * ||q-u_m||^2     -- drops the block axis instead, to
                                             separate which axis is load-bearing.
  v4_cert    (diam_l^2/8) * ||q-u_m||^2   -- CERTIFIED, no calibration set at all:
                                             logsumexp's Hessian is Cov_p(k), and
                                             Cov of a variable of diameter d is
                                             <= d^2/4 I (Popoviciu), so the second
                                             order term is <= diam^2/8 * ||q-u||^2.
                                             PQ error is covered by ||q||*||g-ghat||.

Reported for each: slack, load depth at each coverage, whether the heaviest block
survives the cut, metadata bytes (against Quest's 6.25% of KV bytes), and the
VIOLATION rate -- how often the "upper" bound is not one, which is the price of
the empirical variants and is zero by construction for v4.

Also reports the depth of a ONE-SHOT rule that never reads a key: stopping on
lse(upper over unloaded) - lse(lower over loaded), where lower is the tangent
minus the PQ error. That is the variant whose decode is shaped like Quest's --
one selection, one gather, no interleaved rounds.
"""

import argparse, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))
import torch  # noqa: E402
from bench_latency import real_ruler_prompt, MODELS  # noqa: E402

Q = 0.999


def quantile_last(x, q):
    """q-quantile along dim 0 of a [N, ...] tensor, chunked to dodge the
    2^24-element limit torch.quantile carries."""
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
    ap.add_argument("--coverages", default="0.70,0.80,0.90,0.95")
    ap.add_argument("--layers", default="", help="comma list; default every layer")
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
              m51_pq_mode="offline", m51_group_select="per_query",
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
    H, G, D, M = (cache.num_key_value_heads, cache.num_key_value_groups,
                  cache.head_dim, cache.n_anchors)
    S, Cq = cache.n_sub, cache.pq_codes
    want = set(int(x) for x in args.layers.split(",") if x) or None

    VAR = ["v0_const", "v1_quad", "v2_quad_l", "v3_quad_a", "v4_cert"]
    stat = {v: {"slack": [], "viol": [], "top1_rank": [],
                "depth": {c: [] for c in covs}, "oneshot": {c: [] for c in covs},
                "kept": {c: [] for c in covs}} for v in VAR}
    oracle = {c: [] for c in covs}

    with torch.inference_mode():
        for layer_idx, qs in captured:
            if want is not None and layer_idx not in want:
                continue
            keys = cache.k_cache[layer_idx][0, :, :L * ls].float()          # [H, L*ls, D]
            leaves = keys.view(H, L, ls, D)
            anchors = cache.anchors[layer_idx]                               # [H, M, D]

            # exact tangents, as prefill builds them
            al = torch.einsum('hmd,hltd->hmlt', anchors, leaves)
            w = torch.softmax(al, dim=-1)
            g_exact = torch.einsum('hmlt,hltd->hmld', w, leaves)             # [H,M,L,D]
            offs = torch.logsumexp(al, dim=-1) - torch.einsum('hmd,hmld->hml', anchors, g_exact)
            del al, w

            # PQ reconstruction of the tangents, from what is actually stored
            codes = cache.codes[layer_idx].long()                            # [H,M,L,S]
            cb = cache.codebook[layer_idx]                                   # [H,S,C,sub]
            ghat = torch.empty_like(g_exact)
            for sub in range(S):
                sl = slice(sub * cache.pq_subdim, (sub + 1) * cache.pq_subdim)
                pick = codes[..., sub].reshape(H, M * L)
                ghat[..., sl] = torch.gather(
                    cb[:, sub], 1, pick.unsqueeze(-1).expand(-1, -1, cache.pq_subdim)
                ).view(H, M, L, cache.pq_subdim)
            r_pq = (g_exact - ghat).norm(dim=-1)                             # [H,M,L]
            del g_exact

            # certified curvature: half the squared diameter of the block's keys, /4
            with torch.no_grad():
                dif = leaves.unsqueeze(3) - leaves.unsqueeze(2)              # [H,L,ls,ls,D]
                diam2 = (dif * dif).sum(-1).amax(dim=(-1, -2))               # [H,L]
                del dif
            c_cert = diam2 / 8.0

            # ---- fit the empirical envelopes on the CALIBRATION queries ----
            calib = cache.calib[layer_idx]                                   # [H,G,Nc,D]
            Nc = calib.shape[2]
            d2c = torch.cdist(calib.reshape(H, G * Nc, D), anchors) ** 2     # [H,G*Nc,M]
            route_c = d2c.argmin(dim=-1)                                     # [H,G*Nc]
            e_const = torch.zeros(H, M, G, L, device=keys.device)
            e_quad = torch.zeros(H, M, G, L, device=keys.device)
            for h in range(H):
                qh = calib[h].reshape(G * Nc, D)
                tm = torch.logsumexp((qh @ keys[h].T).view(G * Nc, L, ls), dim=-1)
                for m in range(M):
                    lin = offs[h, m].unsqueeze(0) + qh @ ghat[h, m].T        # [G*Nc, L]
                    err = (tm - lin).view(G, Nc, L)
                    dd = d2c[h, :, m].view(G, Nc).clamp(min=1e-6)
                    rat = err / dd.unsqueeze(-1)
                    sel = (route_c[h] == m).view(G, Nc)
                    for g in range(G):
                        s = sel[g]
                        ec = err[g][s] if bool(s.any()) else err[g]
                        rc = rat[g][s] if bool(s.any()) else rat[g]
                        e_const[h, m, g] = quantile_last(ec, Q)
                        e_quad[h, m, g] = quantile_last(rc, Q)
                    del err, rat
                    # v2 pools every (anchor, group) into one scalar per block,
                    # v3 pools every block into one scalar per (anchor, group)
                del tm
            # v2 pools every (anchor, group) into one scalar per block, v3 pools
            # every block into one scalar per (anchor, group). Pool the RAW
            # ratios: a q99.9 of q99.9s is not a q99.9.
            e_l = torch.zeros(H, L, device=keys.device)
            e_a = torch.zeros(H, M, G, device=keys.device)
            for h in range(H):
                qh = calib[h].reshape(G * Nc, D)
                tm = torch.logsumexp((qh @ keys[h].T).view(G * Nc, L, ls), dim=-1)
                acc = []
                for m in range(M):
                    lin = offs[h, m].unsqueeze(0) + qh @ ghat[h, m].T
                    err = (tm - lin).view(G, Nc, L)
                    dd = d2c[h, :, m].view(G, Nc).clamp(min=1e-6)
                    rat = err / dd.unsqueeze(-1)
                    sel = (route_c[h] == m).view(G, Nc)
                    for g in range(G):
                        s = sel[g]
                        rc = rat[g][s] if bool(s.any()) else rat[g]
                        acc.append(rc)
                        e_a[h, m, g] = torch.quantile(rc.reshape(-1).float(), Q)
                    del err, rat
                e_l[h] = quantile_last(torch.cat(acc, 0), Q)
                del tm, acc

            # ---- evaluate on the DECODE queries ----
            q = qs.view(H, G, D).float() * cache.scale
            d2 = torch.cdist(q, anchors) ** 2                                # [H,G,M]
            route = d2.argmin(dim=-1)                                        # [H,G]
            hh = torch.arange(H, device=q.device).unsqueeze(1)
            gg = torch.arange(G, device=q.device).unsqueeze(0)
            dmin = d2.gather(-1, route.unsqueeze(-1)).squeeze(-1)            # [H,G]
            qn = q.norm(dim=-1)                                              # [H,G]

            lin = (offs[hh, route] +
                   torch.einsum('hgd,hgld->hgl', q, ghat[hh, route]))        # [H,G,L]
            true_mass = torch.logsumexp(
                torch.einsum('hgd,hnd->hgn', q, keys).view(H, G, L, ls), dim=-1)
            total = torch.logsumexp(true_mass, -1, keepdim=True)
            lower = lin - qn[..., None] * r_pq[hh, route]

            env = {
                "v0_const": e_const[hh, route, gg],
                "v1_quad": e_quad[hh, route, gg] * dmin[..., None],
                "v2_quad_l": e_l.unsqueeze(1).expand(H, G, L) * dmin[..., None],
                "v3_quad_a": e_a[hh, route, gg].unsqueeze(-1) * dmin[..., None],
                "v4_cert": (c_cert.unsqueeze(1).expand(H, G, L) * dmin[..., None]
                            + qn[..., None] * r_pq[hh, route]),
            }
            os_ = torch.sort(true_mass, -1, descending=True).values
            of = torch.exp(torch.logcumsumexp(os_, -1) - total)
            for c in covs:
                ok = of >= c
                oracle[c].append(torch.where(ok.any(-1), ok.float().argmax(-1) + 1,
                                             torch.full_like(ok[..., 0].long(), L)).flatten().float())

            for v, e in env.items():
                upper = lin + e
                st = stat[v]
                st["slack"].append((upper - true_mass).flatten())
                st["viol"].append((upper < true_mass).flatten())
                order = upper.argsort(-1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(-1, order, torch.arange(L, device=q.device).expand_as(order))
                t1 = torch.gather(rank, -1, true_mass.argmax(-1, keepdim=True)).squeeze(-1)
                st["top1_rank"].append(t1.flatten().float())
                us = torch.gather(upper, -1, order)
                rev = torch.flip(torch.logcumsumexp(torch.flip(us, [-1]), -1), [-1])
                unres = torch.cat([rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
                ret_true = torch.logcumsumexp(torch.gather(true_mass, -1, order), -1)
                ret_low = torch.logcumsumexp(torch.gather(lower, -1, order), -1)
                for c in covs:
                    eps = 1.0 - c
                    for key, ret in (("depth", ret_true), ("oneshot", ret_low)):
                        ok = torch.sigmoid(unres - ret) <= eps
                        k = torch.where(ok.any(-1), ok.float().argmax(-1) + 1,
                                        torch.full_like(ok[..., 0].long(), L))
                        st[key][c].append(k.flatten().float())
                    st["kept"][c].append(
                        (t1.flatten() < st["depth"][c][-1].long()).float())
            del keys, leaves, ghat, offs, lin, true_mass, env

    block_kv = ls * D * 4
    code_b = M * S * 6 / 8
    meta = {"v0_const": code_b + M * 2 + M * G * 2,
            "v1_quad": code_b + M * 2 + M * G * 2,
            "v2_quad_l": code_b + M * 2 + 2,
            "v3_quad_a": code_b + M * 2,
            "v4_cert": code_b + M * 2 + 2 + M * 2}

    rec = {"model": args.model_key, "datalen": args.datalen, "task": args.task,
           "M": M, "leaf": ls, "L": L, "layers": len(captured),
           "quest_meta_frac": 2 * D * 2 / (16 * D * 4), "variants": {}}
    for c in covs:
        rec.setdefault("oracle_depth_pct", {})[f"{c:.2f}"] = \
            float(torch.cat(oracle[c]).mean()) / L * 100
    for v in VAR:
        st = stat[v]
        sl = torch.cat(st["slack"])
        rec["variants"][v] = {
            "meta_frac_of_kv": meta[v] / block_kv,
            "slack_median": float(sl.median()), "slack_p90": float(sl.quantile(0.9)),
            "violation_pct": float(torch.cat(st["viol"]).float().mean()) * 100,
            "top1_rank_median": float(torch.cat(st["top1_rank"]).median()),
            "top1_rank_p90": float(torch.cat(st["top1_rank"]).quantile(0.9)),
            "cov": {f"{c:.2f}": {
                "depth_pct": float(torch.cat(st["depth"][c]).mean()) / L * 100,
                "oneshot_pct": float(torch.cat(st["oneshot"][c]).mean()) / L * 100,
                "top1_kept_pct": float(torch.cat(st["kept"][c]).mean()) * 100,
            } for c in covs},
        }
    print(json.dumps(rec, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
