#!/usr/bin/env python3
"""Measure representation and ranking dilution on a real 128K decode step.

The model runs through the deployed CPU-offloaded streaming cache.  The first
decode query and the router logits are intercepted without changing selection.
After that step, exact post-RoPE keys in the CPU backing store are used to
measure, for every layer/head/block, the exact GQA attention mass, centroid
under-estimation, top-B misses, and query-free local/macro K/V descriptors.

This is an offline diagnostic.  Future-query quantities are labels only; all
candidate predictors written to the CSV are available at prefill time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman correlation without scipy; ties are negligible here."""
    rx = torch.empty_like(x, dtype=torch.float32)
    ry = torch.empty_like(y, dtype=torch.float32)
    rx[x.argsort()] = torch.arange(x.numel(), device=x.device).float()
    ry[y.argsort()] = torch.arange(y.numel(), device=y.device).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx @ ry / denom.clamp_min(1e-12)).item())


def macro_features(center: torch.Tensor, page_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagonal-Mahalanobis leverage and angular novelty per block."""
    n, dim = center.shape
    pad = (-n) % page_blocks
    if pad:
        center_pad = torch.cat((center, center[-1:].expand(pad, dim)), dim=0)
    else:
        center_pad = center
    page = center_pad.view(-1, page_blocks, dim)
    mean = page.mean(1, keepdim=True)
    residual = page - mean
    variance = residual.square().mean(1, keepdim=True).clamp_min(1e-6)
    leverage = (residual.square() / variance).mean(-1).sqrt().flatten()[:n]
    unit = torch.nn.functional.normalize(page, dim=-1, eps=1e-12)
    direction = torch.nn.functional.normalize(unit.mean(1), dim=-1, eps=1e-12)
    novelty = (1.0 - (unit * direction[:, None]).sum(-1)).flatten()[:n]
    return leverage, novelty


def optimal_scalar_gaps(scores: torch.Tensor) -> torch.Tensor:
    """Exact best r-group Jensen gap for scalar responses of block-8 keys.

    For a scalar query response, an optimum groups adjacent values after
    sorting: crossing two groups can only increase within-group convex spread.
    Enumerating the 2^(S-1) cut patterns is therefore exact and cheap for S=8.
    """
    ordered = scores.float().sort(dim=-1).values
    exact = torch.logsumexp(ordered, dim=-1)
    width = ordered.shape[-1]
    output = []
    boundaries = range(1, width)
    for groups in range(1, width + 1):
        values = []
        for cuts in itertools.combinations(boundaries, groups - 1):
            ends = (0, *cuts, width)
            terms = []
            for left, right in zip(ends[:-1], ends[1:]):
                count = right - left
                terms.append(
                    ordered[..., left:right].mean(-1) + math.log(count)
                )
            values.append(torch.logsumexp(torch.stack(terms, -1), dim=-1))
        best = torch.stack(values, -1).amax(-1)
        output.append((exact - best).clamp_min(0))
    return torch.stack(output, -1)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--datalen", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--mean-components", type=float, default=1.25)
    ap.add_argument(
        "--production-defaults",
        action="store_true",
        help=(
            "use the current query-mean production placement/allocation defaults "
            "instead of the historical robust-trimmed diagnostic settings"
        ),
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hard-per-head", type=int, default=32)
    args = ap.parse_args()

    if not args.production_defaults:
        # Historical configuration retained for reproducing the original
        # 128K dilution campaign.  New production diagnostics should use the
        # cache defaults and pass --production-defaults.
        os.environ["SHADOWKV_CENTER_PLACEMENT"] = "robust_trimmed"
        os.environ["SHADOWKV_CENTER_ALLOCATION"] = "tail_cvar"
        os.environ["SHADOWKV_ROBUST_TRIM_FRACTION"] = "0.25"
        os.environ["SHADOWKV_TAIL_CVAR_FRACTION"] = "0.25"
        os.environ["SHADOWKV_TAIL_GAP_CORRECTION_SCALE"] = "0.25"
        os.environ["SHADOWKV_CENTER_DISPERSION_CORRECTION"] = "0"

    from models import choose_model_class
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
        router_split_fraction=args.mean_components - 1.0,
        self_lse_temperatures=(1.0,),
        quest_prefix_tokens=32,
        quest_recent_tokens=32,
        group_reduce="max",
        streaming_offload=True,
        # The diagnostic needs correctness, not the optional UVA fast path;
        # torch avoids coupling the probe to a host-specific JIT toolchain.
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
                "first": int(first_block),
                "last": int(last_block),
            }
        return query, logits

    cache._block_logits = wrapped
    if "input_ids" in row:
        tokens = [int(value) for value in row["input_ids"]]
    else:
        tokens = llm.tokenizer.encode(row["input"], add_special_tokens=False)
    input_ids = torch.tensor(tokens, device="cuda:0", dtype=torch.long)[None]
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    cache.H2D()
    llm.inference(token, llm.get_ctx(token))
    cache._block_logits = original

    args.output.mkdir(parents=True, exist_ok=True)
    hard_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    select_blocks = args.budget // cache.block_size

    for layer_idx in sorted(captures):
        cap = captures[layer_idx]
        first, last = int(cap["first"]), int(cap["last"])
        n_blocks = last - first
        start, stop = first * cache.block_size, last * cache.block_size
        query_full = cap["query_full"].to("cuda:0")
        query_router = cap["query_router"].to("cuda:0")
        approx_logits = cap["approx_logits"].to("cuda:0").squeeze(3).squeeze(2)[0]
        counts = cache.component_count[layer_idx, 0, :, first:last].detach().cpu()

        for head in range(cache.num_key_value_heads):
            key = cache.k_cache[layer_idx, 0, head, start:stop].to(
                "cuda:0", non_blocking=False
            ).float().view(n_blocks, cache.block_size, cache.head_dim)
            value = cache.v_cache[layer_idx, 0, head, start:stop].to(
                "cuda:0", non_blocking=False
            ).float().view_as(key)
            q = query_full[0, head, :, -1]
            qr = query_router[0, head, 0, -1]
            token_logits = torch.einsum("gd,nsd->gns", q, key) / math.sqrt(cache.head_dim)
            exact_lse = torch.logsumexp(token_logits, dim=-1)
            exact_prob = torch.softmax(exact_lse, dim=-1).mean(0)
            qmean_token_logits = (
                torch.einsum("d,nsd->ns", qr, key) / math.sqrt(cache.head_dim)
            )
            qmean_lse = torch.logsumexp(qmean_token_logits, dim=-1)
            qmean_prob = torch.softmax(qmean_lse, dim=-1)
            approx = approx_logits[head]
            approx_prob = torch.softmax(approx, dim=-1)

            deployed = approx_prob.topk(select_blocks).indices
            oracle = exact_prob.topk(select_blocks).indices
            dep_mask = torch.zeros(n_blocks, device="cuda:0", dtype=torch.bool)
            ora_mask = torch.zeros_like(dep_mask)
            dep_mask[deployed] = True
            ora_mask[oracle] = True
            missed = ora_mask & ~dep_mask
            false_positive = dep_mask & ~ora_mask

            center = key.mean(1)
            residual = key - center[:, None]
            radius = residual.norm(dim=-1).amax(-1)
            rms = residual.square().mean(dim=(1, 2)).sqrt()
            unit = torch.nn.functional.normalize(key, dim=-1, eps=1e-12)
            angular = 1.0 - (
                unit.sum(1).square().sum(-1) - cache.block_size
            ) / float(cache.block_size * (cache.block_size - 1))
            vcenter = value.mean(1)
            vrms = (value - vcenter[:, None]).square().mean(dim=(1, 2)).sqrt()
            kv_coupling = torch.linalg.vector_norm(
                torch.einsum("nsd,nse->nde", residual, value - vcenter[:, None])
                / cache.block_size,
                dim=(-2, -1),
            )
            lev1, nov1 = macro_features(center, 1024 // cache.block_size)
            lev2, nov2 = macro_features(center, 2048 // cache.block_size)

            # Jensen summaries are lower bounds, so this is the exact fraction
            # of q-mean block mass hidden by representation dilution.
            gap = (qmean_lse - approx).clamp_min(0)
            fractional_loss = (1.0 - torch.exp(-gap)).clamp(0, 1)
            global_loss = qmean_prob * fractional_loss
            overlap = (dep_mask & ora_mask).sum().float() / select_blocks
            layer_rows.append({
                "layer": layer_idx,
                "head": head,
                "candidate_blocks": n_blocks,
                "deployed_exact_mass": float(exact_prob[dep_mask].sum()),
                "oracle_exact_mass": float(exact_prob[ora_mask].sum()),
                "topb_overlap": float(overlap),
                "mean_r": float(counts[head].float().mean()),
                "missed_mean_r": float(counts[head, missed.cpu()].float().mean()) if missed.any() else 0.0,
                "selected_mean_r": float(counts[head, dep_mask.cpu()].float().mean()),
                "rho_radius_loss": rank_correlation(radius, global_loss),
                "rho_rms_loss": rank_correlation(rms, global_loss),
                "rho_angular_loss": rank_correlation(angular, global_loss),
                "rho_vrms_loss": rank_correlation(vrms, global_loss),
                "rho_kvcoupling_loss": rank_correlation(kv_coupling, global_loss),
                "rho_macro1k_leverage_loss": rank_correlation(lev1, global_loss),
                "rho_macro2k_leverage_loss": rank_correlation(lev2, global_loss),
                "rho_macro1k_novelty_loss": rank_correlation(nov1, global_loss),
                "rho_macro2k_novelty_loss": rank_correlation(nov2, global_loss),
            })

            hard = torch.cat((
                global_loss.masked_fill(~missed, -1).topk(
                    min(args.hard_per_head, int(missed.sum())),
                ).indices if missed.any() else torch.empty(0, device="cuda:0", dtype=torch.long),
                (approx_prob - qmean_prob).masked_fill(~false_positive, -1).topk(
                    min(args.hard_per_head, int(false_positive.sum())),
                ).indices if false_positive.any() else torch.empty(0, device="cuda:0", dtype=torch.long),
            )).unique()
            oracle_gap_path = optimal_scalar_gaps(qmean_token_logits[hard])
            for block in hard.tolist():
                hard_offset = int((hard == block).nonzero(as_tuple=False)[0, 0])
                gap_path = oracle_gap_path[hard_offset]
                current_r = int(counts[head, block])
                retained = torch.exp(-gap_path)
                def required_order(target: float) -> int:
                    feasible = (retained >= target).nonzero(as_tuple=False)
                    return int(feasible[0, 0]) + 1 if feasible.numel() else cache.block_size
                hard_rows.append({
                    "layer": layer_idx, "head": head,
                    "block": first + block,
                    "kind": "miss" if bool(missed[block]) else "false_positive",
                    "r": current_r,
                    "exact_mass": float(exact_prob[block]),
                    "qmean_mass": float(qmean_prob[block]),
                    "router_mass": float(approx_prob[block]),
                    "jensen_gap": float(gap[block]),
                    "global_dilution_loss": float(global_loss[block]),
                    "oracle_same_r_gap": float(gap_path[current_r - 1]),
                    "placement_excess_gap": float(
                        (gap[block] - gap_path[current_r - 1]).clamp_min(0)
                    ),
                    "oracle_r90": required_order(0.90),
                    "oracle_r95": required_order(0.95),
                    "oracle_r99": required_order(0.99),
                    "radius": float(radius[block]), "rms": float(rms[block]),
                    "angular_dispersion": float(angular[block]),
                    "v_rms": float(vrms[block]),
                    "kv_coupling": float(kv_coupling[block]),
                    "macro1k_leverage": float(lev1[block]),
                    "macro2k_leverage": float(lev2[block]),
                    "macro1k_novelty": float(nov1[block]),
                    "macro2k_novelty": float(nov2[block]),
                })
            del key, value, token_logits

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv(args.output / "per_layer_head.csv", layer_rows)
    write_csv(args.output / "hard_blocks.csv", hard_rows)
    numeric = [key for key in layer_rows[0] if key not in {"layer", "head"}]
    summary = {
        "task": args.dataset.parent.name,
        "sample": args.sample_index,
        "tokens": len(tokens),
        "budget": args.budget,
        "mean_components_requested": args.mean_components,
        "layer_heads": len(layer_rows),
        "mean": {
            key: sum(float(row[key]) for row in layer_rows) / len(layer_rows)
            for key in numeric
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
