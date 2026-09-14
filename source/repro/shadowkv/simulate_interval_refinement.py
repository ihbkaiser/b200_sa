#!/usr/bin/env python3
"""Simulate query-time interval refinement for the block-8 LSE router.

This is deliberately an offline diagnostic.  It captures the first decode
query from the deployed streaming cache, reconstructs the complete nested
self-K partition path for every candidate block, and compares:

* the deployed packed INT8 router;
* the same static per-block component counts with exact FP32 summaries;
* query-time interval refinement with exactly the same total component reads.

The interval is deterministic for the exact centers used by this diagnostic:

    L = logsumexp_a(log n_a + q^T mu_a)
    U = logsumexp_a(log n_a + q^T mu_a + ||q|| rho_a)

where rho_a=max_{i in a} ||k_i-mu_a||.  Refinement reads the next level of a
nested hierarchy only for blocks whose interval can still affect the top-B
boundary.  No future-query label is used to select a refinement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def all_path_bounds(
    keys: torch.Tensor,
    query: torch.Tensor,
    assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic L/U arrays with shape [blocks, block_size]."""
    n_blocks, block_size, _ = keys.shape
    q = query.float()
    q_norm = q.norm()
    lower = []
    upper = []
    for order in range(block_size):
        labels = assignments[:, order].long()
        membership = F.one_hot(labels, num_classes=block_size).float()
        population = membership.sum(1)
        centers = torch.einsum("nsc,nsd->ncd", membership, keys.float())
        centers = centers / population.clamp_min(1)[..., None]
        token_center = centers.gather(
            1, labels[..., None].expand(-1, -1, keys.shape[-1])
        )
        token_radius = (keys.float() - token_center).norm(dim=-1)
        radius = torch.zeros_like(population)
        radius.scatter_reduce_(1, labels, token_radius, reduce="amax")
        valid = torch.arange(block_size, device=keys.device) <= order
        base = torch.einsum("d,ncd->nc", q, centers)
        base = base + population.clamp_min(1).log()
        base = base.masked_fill(~valid[None], float("-inf"))
        lower.append(torch.logsumexp(base, dim=-1))
        upper.append(torch.logsumexp(base + q_norm * radius, dim=-1))
    lower_path = torch.stack(lower, dim=-1)
    upper_path = torch.stack(upper, dim=-1)
    # Meet all levels seen so far.  This also removes harmless numerical
    # non-monotonicity from independently materialized centroids.
    lower_path = torch.cummax(lower_path, dim=-1).values
    upper_path = torch.cummin(upper_path, dim=-1).values
    return lower_path, upper_path


def refine_to_budget(
    lower: torch.Tensor,
    upper: torch.Tensor,
    select_blocks: int,
    total_components: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Width-first boundary refinement under a hard component-read budget."""
    n_blocks, max_order = lower.shape
    order = torch.ones(n_blocks, device=lower.device, dtype=torch.long)
    remaining = max(0, int(total_components) - n_blocks)
    steps = 0
    certified = False
    while remaining:
        col = order - 1
        lo = lower.gather(1, col[:, None]).squeeze(1)
        up = upper.gather(1, col[:, None]).squeeze(1)
        chosen = lo.topk(select_blocks).indices
        selected = torch.zeros(n_blocks, device=lower.device, dtype=torch.bool)
        selected[chosen] = True
        threshold = lo[chosen].amin()
        omitted_upper = up.masked_fill(selected, float("-inf")).amax()
        if omitted_upper <= threshold:
            certified = True
            break
        eligible = (up >= threshold) & (order < max_order)
        count = min(remaining, batch, int(eligible.sum().item()))
        if count == 0:
            break
        # An interval cannot influence top-B if U<threshold.  Among the rest,
        # the widest interval contains the most unresolved ranking evidence.
        priority = (up - lo).masked_fill(~eligible, float("-inf"))
        upgrade = priority.topk(count).indices
        order[upgrade] += 1
        remaining -= count
        steps += 1

    lo = lower.gather(1, (order - 1)[:, None]).squeeze(1)
    up = upper.gather(1, (order - 1)[:, None]).squeeze(1)
    chosen = lo.topk(select_blocks).indices
    selected = torch.zeros(n_blocks, device=lower.device, dtype=torch.bool)
    selected[chosen] = True
    certified = certified or (
        up.masked_fill(selected, float("-inf")).amax()
        <= lo[chosen].amin()
    )
    return chosen, order, bool(certified), steps


def refine_lucb_to_budget(
    lower: torch.Tensor,
    upper: torch.Tensor,
    select_blocks: int,
    total_components: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Batched top-k LUCB: refine boundary incumbents and challengers."""
    n_blocks, max_order = lower.shape
    order = torch.ones(n_blocks, device=lower.device, dtype=torch.long)
    remaining = max(0, int(total_components) - n_blocks)
    steps = 0
    certified = False
    while remaining:
        col = order - 1
        lo = lower.gather(1, col[:, None]).squeeze(1)
        up = upper.gather(1, col[:, None]).squeeze(1)
        chosen = lo.topk(select_blocks).indices
        selected = torch.zeros(n_blocks, device=lower.device, dtype=torch.bool)
        selected[chosen] = True
        threshold = lo[chosen].amin()
        omitted_upper = up.masked_fill(selected, float("-inf")).amax()
        if omitted_upper <= threshold:
            certified = True
            break

        refinable = order < max_order
        incumbent = selected & refinable
        challenger = (~selected) & refinable & (up >= threshold)
        count = min(remaining, batch, int((incumbent | challenger).sum()))
        if count == 0:
            break
        # LUCB resolves both sides of the decision boundary: low-L members of
        # the current top set and high-U non-members.  Batch the choices to
        # keep the offline simulation tractable without using future labels.
        n_challenger = min((count + 1) // 2, int(challenger.sum()))
        n_incumbent = min(count - n_challenger, int(incumbent.sum()))
        if n_challenger + n_incumbent < count:
            n_challenger = min(count - n_incumbent, int(challenger.sum()))
        upgrades = []
        if n_challenger:
            score = up.masked_fill(~challenger, float("-inf"))
            upgrades.append(score.topk(n_challenger).indices)
        if n_incumbent:
            score = (-lo).masked_fill(~incumbent, float("-inf"))
            upgrades.append(score.topk(n_incumbent).indices)
        if not upgrades:
            break
        upgrade = torch.cat(upgrades)
        order[upgrade] += 1
        remaining -= int(upgrade.numel())
        steps += 1

    lo = lower.gather(1, (order - 1)[:, None]).squeeze(1)
    up = upper.gather(1, (order - 1)[:, None]).squeeze(1)
    chosen = lo.topk(select_blocks).indices
    selected = torch.zeros(n_blocks, device=lower.device, dtype=torch.bool)
    selected[chosen] = True
    certified = certified or (
        up.masked_fill(selected, float("-inf")).amax()
        <= lo[chosen].amin()
    )
    return chosen, order, bool(certified), steps


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--datalen", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--refine-batch", type=int, default=32)
    args = ap.parse_args()

    from models import choose_model_class
    from models.centroid_router_cache import fit_agglomerative_self_lse_paths
    from models.tensor_op import sample_token

    row = load_row(args.dataset, args.sample_index)
    llm_class = choose_model_class(args.model)
    llm = llm_class(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=args.datalen + 2048,
        attn_mode="adaptive_centroid_lse_streaming_prefix4_querymean",
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=8,
        router_centroids=8,
        router_centroid_method="self_lse_adaptive_iso",
        router_split_fraction=0.0,
        self_lse_temperatures=(1.0,),
        quest_prefix_tokens=32,
        quest_recent_tokens=32,
        group_reduce="max",
        streaming_offload=True,
        streaming_gather_backend="torch",
        streaming_router_backend="torch",
        streaming_refine_factor=1.0,
        streaming_max_components=8,
        streaming_compact_metadata=True,
        streaming_center_bits=8,
    )
    cache = llm.kv_cache
    captures: dict[int, dict[str, object]] = {}
    original = cache._block_logits

    def wrapped(layer_idx, query_states, first_block, last_block):
        query, logits = original(layer_idx, query_states, first_block, last_block)
        if layer_idx not in captures:
            full = query_states.view(
                1, cache.num_key_value_heads, cache.num_key_value_groups,
                cache.incoming_q_len, cache.head_dim,
            )
            captures[layer_idx] = {
                "query_full": full.detach().float().cpu(),
                "query_router": query.detach().float().cpu(),
                "approx_logits": logits.detach().float().cpu(),
                "first": int(first_block), "last": int(last_block),
            }
        return query, logits

    cache._block_logits = wrapped
    tokens = (
        [int(value) for value in row["input_ids"]]
        if "input_ids" in row else
        llm.tokenizer.encode(row["input"], add_special_tokens=False)
    )
    input_ids = torch.tensor(tokens, device="cuda:0", dtype=torch.long)[None]
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    cache.H2D()
    llm.inference(token, llm.get_ctx(token))
    cache._block_logits = original

    select_blocks = args.budget // cache.block_size
    rows = []
    for layer_idx in sorted(captures):
        cap = captures[layer_idx]
        first, last = int(cap["first"]), int(cap["last"])
        n_blocks = last - first
        start, stop = first * cache.block_size, last * cache.block_size
        query_full = cap["query_full"].to("cuda:0")
        query_router = cap["query_router"].to("cuda:0")
        approx_logits = cap["approx_logits"].to("cuda:0").squeeze(3).squeeze(2)[0]
        production_count = cache.component_count[
            layer_idx, 0, :, first:last
        ].detach().long().to("cuda:0")
        layer_keys = cache.k_cache[
            layer_idx, 0, :, start:stop
        ].to("cuda:0", non_blocking=False).float().view(
            cache.num_key_value_heads, n_blocks, cache.block_size, cache.head_dim
        )
        _, assignment = fit_agglomerative_self_lse_paths(
            layer_keys,
            temperatures=(1.0,),
            batch_blocks=65536,
            cost_mode="mean_gap",
            cost_beta=4.0,
        )

        for head in range(cache.num_key_value_heads):
            keys = layer_keys[head]
            qfull = query_full[0, head, :, -1]
            qrouter = query_router[0, head, 0, -1] / math.sqrt(cache.head_dim)
            token_logits = torch.einsum("gd,nsd->gns", qfull, keys)
            token_logits = token_logits / math.sqrt(cache.head_dim)
            exact_lse = torch.logsumexp(token_logits, dim=-1)
            exact_probability = torch.softmax(exact_lse, dim=-1).mean(0)
            oracle = exact_probability.topk(select_blocks).indices

            lower, upper = all_path_bounds(keys, qrouter, assignment[head])
            qmean_oracle = lower[:, -1].topk(select_blocks).indices
            count = production_count[head].clamp(1, cache.block_size)
            static_score = lower.gather(1, (count - 1)[:, None]).squeeze(1)
            static = static_score.topk(select_blocks).indices
            deployed = approx_logits[head].topk(select_blocks).indices
            dynamic, dynamic_count, certified, rounds = refine_to_budget(
                lower, upper, select_blocks, int(count.sum()), args.refine_batch
            )
            lucb, lucb_count, lucb_certified, lucb_rounds = refine_lucb_to_budget(
                lower, upper, select_blocks, int(count.sum()), args.refine_batch
            )

            def mass(index: torch.Tensor) -> float:
                return float(exact_probability[index].sum().item())

            def overlap(index: torch.Tensor) -> float:
                return float(torch.isin(index, oracle).float().mean().item())

            rows.append({
                "layer": layer_idx,
                "head": head,
                "candidate_blocks": n_blocks,
                "components": int(count.sum()),
                "mean_r": float(count.float().mean()),
                "dynamic_mean_r": float(dynamic_count.float().mean()),
                "deployed_mass": mass(deployed),
                "static_fp32_mass": mass(static),
                "dynamic_mass": mass(dynamic),
                "oracle_mass": mass(oracle),
                "qmean_oracle_mass": mass(qmean_oracle),
                "deployed_overlap": overlap(deployed),
                "static_fp32_overlap": overlap(static),
                "dynamic_overlap": overlap(dynamic),
                "qmean_oracle_overlap": overlap(qmean_oracle),
                "lucb_mass": mass(lucb),
                "lucb_overlap": overlap(lucb),
                "certified": certified,
                "lucb_certified": lucb_certified,
                "refinement_rounds": rounds,
                "lucb_rounds": lucb_rounds,
                "dynamic_max_r": int(dynamic_count.max()),
                "lucb_max_r": int(lucb_count.max()),
                "dynamic_r1_fraction": float((dynamic_count == 1).float().mean()),
                "lucb_r1_fraction": float((lucb_count == 1).float().mean()),
                "initial_width_median": float((upper[:, 0] - lower[:, 0]).median()),
                "final_width_median": float(
                    (upper.gather(1, (dynamic_count - 1)[:, None]).squeeze(1)
                     - lower.gather(1, (dynamic_count - 1)[:, None]).squeeze(1)).median()
                ),
            })
        del layer_keys, assignment

    args.output.mkdir(parents=True, exist_ok=True)
    import csv
    with (args.output / "per_layer_head.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float, bool))]
    summary = {
        "sample": args.sample_index,
        "tokens": len(tokens),
        "budget": args.budget,
        "layer_heads": len(rows),
        "mean": {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in numeric if key not in {"layer", "head"}
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
