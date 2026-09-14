#!/usr/bin/env python3
"""Token-level key compressors at equal bits: PQ vs low-rank, plain vs whitened.

The previous probe showed that compressing keys in the metric the query induces
-- whiten by M_q^{1/2} where M_q = E[q q^T], then compress Euclidean -- beats
the plain basis at every rank. That was measured with low-rank projection,
which is a poor use of bits: rank 32 at 8 bits a coordinate is 256 bits a token
against PQCache's 12.

So the question that decides whether any of it matters is not "is whitening a
better basis" but "does whitening help the compressor that is actually cheap".
This measures both families on one bits-per-token axis:

  * PQ: D split into m subvectors, 2^b codewords each, m*b bits;
  * low-rank: top-r directions, 8 bits a coordinate, 8r bits;

each in the plain basis and in the whitened one. Whitening costs nothing at
run time -- one matrix multiply per head applied to the query, not to the
tokens -- so if it lifts PQ at fixed bits it composes with what is already
deployed instead of replacing it.

The metric is recall of the true top-k against the estimate, on held-out
queries. Retrieval is a tail statistic: score MSE is reported too, and the two
do not agree, which is the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def kmeans(points: torch.Tensor, clusters: int, iterations: int = 12,
           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd on [N, d] -> (centroids [k, d], assignment [N]). GPU, float32."""
    generator = torch.Generator(device=points.device).manual_seed(seed)
    count = points.shape[0]
    pick = torch.randperm(count, generator=generator, device=points.device)[:clusters]
    centroids = points[pick].clone()
    assignment = torch.zeros(count, dtype=torch.long, device=points.device)
    for _ in range(iterations):
        distance = torch.cdist(points, centroids)
        assignment = distance.argmin(1)
        total = torch.zeros_like(centroids)
        total.index_add_(0, assignment, points)
        counts = torch.bincount(assignment, minlength=clusters).clamp_min(1)
        fresh = total / counts[:, None]
        empty = torch.bincount(assignment, minlength=clusters) == 0
        if empty.any():                       # keep an empty cell where it was
            fresh[empty] = centroids[empty]
        centroids = fresh
    return centroids, assignment


def pq_scores(keys: torch.Tensor, queries: torch.Tensor,
              subvectors: int, bits: int) -> torch.Tensor:
    """Estimated q.k for every (query, token) from a product-quantised key."""
    tokens, dim = keys.shape
    width = dim // subvectors
    clusters = 1 << bits
    table = queries.new_zeros(queries.shape[0], subvectors, clusters)
    codes = torch.zeros(tokens, subvectors, dtype=torch.long, device=keys.device)
    for s in range(subvectors):
        piece = keys[:, s * width:(s + 1) * width].contiguous()
        centroids, assignment = kmeans(piece, min(clusters, tokens), seed=s)
        codes[:, s] = assignment
        # q_s . C_s[c]  for every codeword: the usual asymmetric PQ lookup
        table[:, s, :centroids.shape[0]] = (
            queries[:, s * width:(s + 1) * width] @ centroids.T)
    out = queries.new_zeros(queries.shape[0], tokens)
    for s in range(subvectors):
        out += table[:, s, :].gather(1, codes[:, s].expand(queries.shape[0], -1))
    return out


def waterfill_bits(variance: torch.Tensor, budget: int, cap: int = 10) -> list[int]:
    """Greedy reverse water-filling: hand the next bit to the worst group.

    Each bit roughly quarters a group's squared error, so the marginal gain of
    a bit for group s is its current variance/4^b. Giving every group the same
    number of bits -- what plain PQ does -- is optimal only when the groups
    carry equal variance, which is exactly what whitening destroys. That is why
    whitening and uniform allocation fight each other.
    """
    groups = variance.shape[0]
    bits = [0] * groups
    residual = variance.clone()
    for _ in range(budget):
        order = torch.argsort(residual, descending=True)
        for index in order.tolist():
            if bits[index] < cap:
                bits[index] += 1
                residual[index] = variance[index] / (4.0 ** bits[index])
                break
        else:
            break
    return bits


def pq_scores_allocated(keys: torch.Tensor, queries: torch.Tensor,
                        subvectors: int, total_bits: int) -> torch.Tensor:
    """PQ whose bits per subvector come from water-filling on its variance."""
    tokens, dim = keys.shape
    width = dim // subvectors
    variance = torch.stack([
        keys[:, s * width:(s + 1) * width].var(0).sum() for s in range(subvectors)
    ])
    allocation = waterfill_bits(variance, total_bits)
    out = queries.new_zeros(queries.shape[0], tokens)
    for s, bits in enumerate(allocation):
        if bits <= 0:
            continue                      # a dead group costs nothing and says nothing
        piece = keys[:, s * width:(s + 1) * width].contiguous()
        centroids, assignment = kmeans(piece, min(1 << bits, tokens), seed=s)
        table = queries[:, s * width:(s + 1) * width] @ centroids.T
        out += table.gather(1, assignment.expand(queries.shape[0], -1))
    return out


def lowrank_scores(keys: torch.Tensor, queries: torch.Tensor, basis: torch.Tensor,
                   rank: int, bits: int = 8) -> torch.Tensor:
    """Top-rank projection with the coordinates uniformly quantised."""
    sub = basis[:, :rank]
    coefficient = keys @ sub                                    # [T, r]
    low = coefficient.min(0).values
    high = coefficient.max(0).values
    levels = (1 << bits) - 1
    step = ((high - low) / levels).clamp_min(1e-12)
    quantised = torch.round((coefficient - low) / step) * step + low
    return (queries @ sub) @ quantised.T


def psd_sqrt(matrix: torch.Tensor, floor: float = 1e-10):
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(floor)
    return ((vectors * values.sqrt()) @ vectors.T,
            (vectors * values.rsqrt()) @ vectors.T)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--datalen", type=int, default=32768)
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--fit-queries", type=int, default=256)
    ap.add_argument("--eval-queries", type=int, default=32)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    wanted = {int(x) for x in args.layers.split(",") if x}
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
    counter = {"layer": 0}
    original = llm.apply_rotary_pos_emb

    def wrapped(query, key, position_ids):
        query, key = original(query, key, position_ids)
        layer = counter["layer"] % llm.num_layers
        counter["layer"] += 1
        if layer in wanted and query.shape[-2] > 1:
            captured[layer] = {"q": query[0].detach().float(),
                               "k": key[0].detach().float()}
        return query, key

    llm.apply_rotary_pos_emb = wrapped
    ids = torch.tensor(llm.tokenizer.encode(row["input"], add_special_tokens=False),
                       device="cuda:0", dtype=torch.long)[None][:, : args.datalen]
    llm.prefill(ids)
    llm.apply_rotary_pos_emb = original

    # (name, bits per token, builder)
    # name, bits/token. "wf" = PQ in the eigen-ordered basis with the bits
    # water-filled across subvectors instead of shared out equally.
    plans = [
        ("pq  m2 b6", 12), ("pq  m4 b8", 32), ("pq  m8 b8", 64),
        ("pq m16 b8", 128), ("pq m32 b8", 256),
        ("wf  m8 t12", 12), ("wf  m8 t32", 32),
        ("wf  m8 t64", 64), ("wf m16 t128", 128), ("wf m16 t256", 256),
        ("lr  r8 b8", 64), ("lr r16 b8", 128), ("lr r32 b8", 256),
    ]
    totals = {(name, basis): [0.0, 0.0, 0]
              for name, _ in plans for basis in ("plain", "white")}

    for layer in sorted(captured):
        q_all, k_all = captured[layer]["q"], captured[layer]["k"]
        heads_q, _, dim = q_all.shape
        heads_kv = k_all.shape[0]
        group = heads_q // heads_kv
        for head in range(heads_kv):
            keys = k_all[head]
            grp = slice(head * group, (head + 1) * group)
            fit = q_all[grp, -args.fit_queries - args.eval_queries:
                        -args.eval_queries].reshape(-1, dim)
            held = q_all[grp, -args.eval_queries:].reshape(-1, dim)
            root, inverse = psd_sqrt(fit.T @ fit / fit.shape[0])
            keys_w, held_w = keys @ root, held @ inverse

            truth = held @ keys.T
            gold = truth.topk(args.topk, dim=1).indices
            denominator = truth.square().mean()

            def record(name, estimate):
                mse = ((estimate - truth).square().mean() / denominator).item()
                pick = estimate.topk(args.topk, dim=1).indices
                hit = (pick.unsqueeze(2) == gold.unsqueeze(1)).any(2).float().mean().item()
                return mse, hit

            for basis, kk, qq in (("plain", keys, held), ("white", keys_w, held_w)):
                eigenvectors = torch.linalg.eigh(kk.T @ kk / kk.shape[0])[1].flip(1)
                for name, _bits in plans:
                    kind, a, b = name.split()
                    if kind == "pq":
                        estimate = pq_scores(kk, qq, int(a[1:]), int(b[1:]))
                    elif kind == "wf":
                        # water-filling only means anything once the axes are
                        # ordered by how much they carry, so rotate first
                        estimate = pq_scores_allocated(
                            kk @ eigenvectors, qq @ eigenvectors,
                            int(a[1:]), int(b[1:]))
                    else:
                        estimate = lowrank_scores(kk, qq, eigenvectors, int(a[1:]))
                    mse, hit = record(name, estimate)
                    slot = totals[(name, basis)]
                    slot[0] += mse; slot[1] += hit; slot[2] += 1

    heads = totals[(plans[0][0], "plain")][2]
    print(f"\nheads measured: {heads}  (layers {sorted(captured)}, "
          f"T={args.datalen}, top-{args.topk} of {args.datalen})")
    print(f"\n{'compressor':<11}{'bits/tok':>9} | {'recall plain':>13}"
          f"{'recall white':>14}{'  gain':>8} | {'MSE plain':>10}{'MSE white':>10}")
    for name, bits in plans:
        p_mse, p_hit, n = totals[(name, "plain")]
        w_mse, w_hit, _ = totals[(name, "white")]
        print(f"{name:<11}{bits:>9} | {p_hit/n:>13.3f}{w_hit/n:>14.3f}"
              f"{(w_hit - p_hit)/n:>+8.3f} | {p_mse/n:>10.4f}{w_mse/n:>10.4f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in totals.items()}, handle, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
