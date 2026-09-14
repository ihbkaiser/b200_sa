#!/usr/bin/env python3
"""Is the key spectrum steeper when measured in the metric the query induces?

The claim under test.  A token-level score is s_i = q^T k_i, so the error a
compression of k_i makes is q^T e_i, and averaged over the query distribution
that is

    E_q[(q^T e)^2] = e^T M_q e,      M_q = E[q q^T].

So the right distortion on keys is NOT the Euclidean one every low-rank method
minimises -- it is the M_q-weighted one.  Equivalently, whiten first: with
W = M_q^{1/2} and k~ = W k, we have q^T k = (W^{-1} q)^T k~ and W^{-1}q is
isotropic in distribution, so plain Euclidean distortion on k~ IS the score
distortion.  Loki compresses along the spectrum of K, which is where the keys
happen to vary; this compresses along the spectrum of W K, which is where the
keys vary IN THE DIRECTIONS QUERIES LOOK.

M_q is the SECOND MOMENT, not the covariance.  Queries live in a narrow cone,
so their mean is large and it carries most of the interaction; centring would
throw away precisely the dominant term.

Measured here, per (layer, kv head):

  * ranks needed for 90 / 95 / 99 % of the energy, plain vs whitened;
  * the honest version of the same question -- relative score error and
    top-k retrieval recall against rank r, on HELD-OUT queries, for the two
    bases at equal metadata (r numbers per token);
  * drift: the same evaluated with decode-time queries while W was fitted on
    prefill queries, which is the one assumption the design cannot avoid.

  $PY repro/shadowkv/probe_query_whitened_key_spectrum.py \
      --model $SHADOWKV_QWEN3_PATH --datalen 32768 \
      --dataset /storage/baonn/ruler_shadowkv/data/qwen/32768/niah_multikey_3/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def energy_rank(eigenvalues: torch.Tensor, fraction: float) -> int:
    total = eigenvalues.sum()
    cumulative = torch.cumsum(eigenvalues, 0) / total
    return int((cumulative < fraction).sum().item()) + 1


def psd_sqrt(matrix: torch.Tensor, floor: float = 1e-10):
    values, vectors = torch.linalg.eigh(matrix.double())
    values = values.clamp_min(floor)
    root = (vectors * values.sqrt()) @ vectors.T
    inverse = (vectors * values.rsqrt()) @ vectors.T
    return root, inverse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--datalen", type=int, default=32768)
    ap.add_argument("--layers", default="0,8,16,24,31")
    ap.add_argument("--fit-queries", type=int, default=256,
                    help="prefill positions used to fit M_q")
    ap.add_argument("--eval-queries", type=int, default=64,
                    help="held-out prefill positions used to score")
    ap.add_argument("--decode-tokens", type=int, default=32,
                    help="decode steps captured for the drift check")
    ap.add_argument("--ranks", default="8,16,24,32,48,64")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    wanted = {int(x) for x in args.layers.split(",") if x}
    ranks = [int(x) for x in args.ranks.split(",") if x]

    from models import choose_model_class
    from models.tensor_op import sample_token

    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            if index == args.sample_index:
                row = json.loads(line)
                break

    llm = choose_model_class(args.model)(
        model_name=args.model, batch_size=1, device="cuda:0",
        max_length=args.datalen + 2048, attn_mode="full",
        dtype=torch.bfloat16,
    )

    # post-RoPE q and k, which is what the score actually uses
    captured: dict[int, dict[str, torch.Tensor]] = {}
    decode_q: dict[int, list[torch.Tensor]] = {}
    counter = {"layer": 0}
    original_rope = llm.apply_rotary_pos_emb

    def wrapped_rope(query, key, position_ids):
        query, key = original_rope(query, key, position_ids)
        layer = counter["layer"] % llm.num_layers
        counter["layer"] += 1
        if layer in wanted:
            if query.shape[-2] > 1:                      # prefill
                captured[layer] = {
                    "q": query[0].detach().float().cpu(),   # [Hq, T, D]
                    "k": key[0].detach().float().cpu(),     # [Hkv, T, D]
                }
            else:                                        # decode
                decode_q.setdefault(layer, []).append(
                    query[0, :, 0].detach().float().cpu())  # [Hq, D]
        return query, key

    llm.apply_rotary_pos_emb = wrapped_rope
    input_ids = torch.tensor(
        llm.tokenizer.encode(row["input"], add_special_tokens=False),
        device="cuda:0", dtype=torch.long,
    )[None][:, : args.datalen]
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    for _ in range(args.decode_tokens):
        logits = llm.inference(token, llm.get_ctx(token))
        token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    llm.apply_rotary_pos_emb = original_rope

    results = []
    for layer in sorted(captured):
        q_all = captured[layer]["q"]                     # [Hq, T, D]
        k_all = captured[layer]["k"]                     # [Hkv, T, D]
        heads_q, _, dim = q_all.shape
        heads_kv = k_all.shape[0]
        group = heads_q // heads_kv
        drift = torch.stack(decode_q[layer], 0) if layer in decode_q else None

        for head in range(heads_kv):
            keys = k_all[head].double()                                  # [T,D]
            # every query head in the group reads this kv head
            grp = slice(head * group, (head + 1) * group)
            fit = q_all[grp, -args.fit_queries - args.eval_queries:
                        -args.eval_queries].reshape(-1, dim).double()
            held = q_all[grp, -args.eval_queries:].reshape(-1, dim).double()
            later = (drift[:, grp].reshape(-1, dim).double()
                     if drift is not None else None)

            m_q = fit.T @ fit / fit.shape[0]
            root, inverse = psd_sqrt(m_q)
            m_k = keys.T @ keys / keys.shape[0]
            m_kt = root @ m_k @ root

            plain_eig = torch.linalg.eigvalsh(m_k).flip(0).clamp_min(0)
            white_eig = torch.linalg.eigvalsh(m_kt).flip(0).clamp_min(0)
            # Energy is not discrimination. Whitening pulls the query-mean
            # direction to the front, and every token has a large component
            # along it -- that is a near-constant offset across i, so it costs
            # a rank but buys no ranking. Centring the keys removes exactly
            # that shared part, and the centred rank is the honest count.
            centred = keys - keys.mean(0, keepdim=True)
            m_kc = centred.T @ centred / centred.shape[0]
            white_c = torch.linalg.eigvalsh(root @ m_kc @ root).flip(0).clamp_min(0)
            plain_c = torch.linalg.eigvalsh(m_kc).flip(0).clamp_min(0)

            # the honest test: equal metadata, held-out queries
            _, vec_plain = torch.linalg.eigh(m_k)
            vec_plain = vec_plain.flip(1)
            _, vec_white = torch.linalg.eigh(m_kt)
            vec_white = vec_white.flip(1)
            keys_w = keys @ root                                        # k~

            def score_error(queries, rank):
                truth = queries @ keys.T                                # [Q,T]
                denominator = truth.square().mean()
                basis = vec_plain[:, :rank]
                approx = (queries @ basis) @ (keys @ basis).T
                err_plain = (approx - truth).square().mean() / denominator
                basis_w = vec_white[:, :rank]
                q_w = queries @ inverse                                 # W^-1 q
                approx_w = (q_w @ basis_w) @ (keys_w @ basis_w).T
                err_white = (approx_w - truth).square().mean() / denominator
                gold = truth.topk(args.topk, dim=1).indices
                def recall(estimate):
                    pick = estimate.topk(args.topk, dim=1).indices
                    hits = [len(set(a.tolist()) & set(b.tolist()))
                            for a, b in zip(gold, pick)]
                    return sum(hits) / (len(hits) * args.topk)
                return (err_plain.item(), err_white.item(),
                        recall(approx), recall(approx_w))

            entry = {
                "layer": layer, "head": head,
                "rank90_plain": energy_rank(plain_eig, 0.90),
                "rank90_white": energy_rank(white_eig, 0.90),
                "rank95_plain": energy_rank(plain_eig, 0.95),
                "rank95_white": energy_rank(white_eig, 0.95),
                "rank99_plain": energy_rank(plain_eig, 0.99),
                "rank99_white": energy_rank(white_eig, 0.99),
                "top1_share_white": float(white_eig[0] / white_eig.sum()),
                "rank90_plain_centred": energy_rank(plain_c, 0.90),
                "rank90_white_centred": energy_rank(white_c, 0.90),
                "rank95_white_centred": energy_rank(white_c, 0.95),
                "held": {r: score_error(held, r) for r in ranks},
            }
            if later is not None and later.shape[0]:
                entry["decode"] = {r: score_error(later, r) for r in ranks}
            results.append(entry)

    def show(tag, key):
        print(f"\n=== {tag}: relative score MSE and top-{args.topk} recall ===")
        print(f"{'rank':>5} | {'MSE plain':>10} {'MSE white':>10} "
              f"{'| recall plain':>15} {'recall white':>13}")
        for r in ranks:
            rows = [e[key][r] for e in results if key in e]
            if not rows:
                continue
            mp = sum(x[0] for x in rows) / len(rows)
            mw = sum(x[1] for x in rows) / len(rows)
            rp = sum(x[2] for x in rows) / len(rows)
            rw = sum(x[3] for x in rows) / len(rows)
            print(f"{r:>5} | {mp:>10.4f} {mw:>10.4f} | {rp:>13.3f} {rw:>13.3f}")

    print(f"heads measured: {len(results)}  (layers {sorted(captured)}, "
          f"D={captured[sorted(captured)[0]]['k'].shape[-1]})")
    print("\n=== rank for a fraction of the energy (mean over heads) ===")
    for fraction in (90, 95, 99):
        plain = sum(e[f"rank{fraction}_plain"] for e in results) / len(results)
        white = sum(e[f"rank{fraction}_white"] for e in results) / len(results)
        print(f"  {fraction}% energy: plain {plain:6.1f}   whitened {white:6.1f}")
    top1 = sum(e["top1_share_white"] for e in results) / len(results)
    pc = sum(e["rank90_plain_centred"] for e in results) / len(results)
    wc = sum(e["rank90_white_centred"] for e in results) / len(results)
    wc95 = sum(e["rank95_white_centred"] for e in results) / len(results)
    print(f"\n  top-1 whitened direction holds {top1:.1%} of the energy")
    print(f"  90% of CENTRED energy: plain {pc:6.1f}   whitened {wc:6.1f}")
    print(f"  95% of CENTRED energy:                  whitened {wc95:6.1f}")
    print("  (centred = keys minus their mean. The uncentred rank counts the "
          "shared\n   component every token has, which is an offset across i "
          "and ranks nothing.)")
    show("held-out prefill queries", "held")
    if any("decode" in e for e in results):
        show("decode queries (W fitted on prefill -- the drift test)", "decode")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=1, default=str)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
