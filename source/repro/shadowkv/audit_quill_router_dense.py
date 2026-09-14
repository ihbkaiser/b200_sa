#!/usr/bin/env python3
"""Audit mean+QUILL-max routing against dense block relevance.

This runs the dense ShadowKV model only to capture the query trajectory and
full post-RoPE keys.  It then reproduces ShadowKV's fixed outlier blocks,
GQA reduction, local tail, and dynamic block budget offline.  Comparing the
hybrid against an oracle all-token max separates a bad mixed statistic from a
cache/SVD implementation bug.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from repro.certified_sparse.probe_hard_multikey_block8 import (  # noqa: E402
    capture_dense,
    find_text_spans,
    flatten_answer,
    load_ruler_row,
)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--quill-mask", type=Path, required=True)
    ap.add_argument("--max-decode-steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dynamic-budget", type=int, default=512)
    ap.add_argument("--outlier-blocks", type=int, default=48)
    ap.add_argument("--local-blocks", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    row = load_ruler_row(args.dataset, args.sample_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(row["input"], add_special_tokens=False)
    input_ids = torch.tensor(tokens, device="cuda:0", dtype=torch.long)[None]
    llm, queries, _captured, generated = capture_dense(
        args.model, input_ids, args.max_decode_steps
    )
    payload = torch.load(args.quill_mask, map_location="cpu", weights_only=False)
    mask_all = payload["exact_mask"]

    answer = flatten_answer(row["outputs"])
    answer_positions = {
        pos
        for start, stop in find_text_spans(tokenizer, tokens, answer)
        for pos in range(start, stop)
    }
    block = args.block_size
    pages = len(tokens) // block
    routed_pages = pages - args.local_blocks
    usable = pages * block
    evidence_blocks = sorted({p // block for p in answer_positions if p < usable})
    groups = llm.num_heads // llm.num_key_value_heads
    steps = queries.shape[1]
    dim = llm.head_dim
    query_all = queries.reshape(
        llm.num_layers, steps, llm.num_key_value_heads, groups, dim
    ).float() / math.sqrt(dim)
    select = args.dynamic_budget // block

    methods = ("mean", "quill_mixed_max", "oracle_all_max", "oracle_logmass")
    sums = {
        name: dict(instances=0, topmax_hits=0.0, topmass_hits=0.0,
                   mass_capture=0.0, evidence_mass_capture=0.0,
                   evidence_block_recall=0.0, true_topset_overlap=0.0)
        for name in methods
    }
    diagnostic = dict(
        eligible_instances=0,
        selected_exact_fraction=0.0,
        hybrid_mean_overlap=0.0,
        hybrid_oracle_max_overlap=0.0,
        quill_global_top_token_recall=0.0,
        hidden_better_token_fraction=0.0,
        hidden_better_token_gap=0.0,
        exact_minus_mean_median=0.0,
        exact_minus_mean_p90=0.0,
    )
    layer_rows = []

    for layer in range(llm.num_layers):
        key_layer = llm.kv_cache.k_cache[layer, 0, :, :usable].float()
        for head in range(llm.num_key_value_heads):
            chunks = key_layer[head].reshape(pages, block, dim)
            routed = chunks[:routed_pages]
            means = routed.mean(dim=1)
            cosine = F.cosine_similarity(means[:, None], routed, dim=-1)
            fixed_outlier = cosine.amin(dim=-1).topk(
                min(args.outlier_blocks, routed_pages), largest=False
            ).indices
            candidate = torch.ones(routed_pages, dtype=torch.bool, device="cuda")
            candidate[fixed_outlier] = False
            candidate_idx = candidate.nonzero().squeeze(-1)

            q = query_all[layer, :, head].to("cuda")  # [step, group, dim]
            token_logits = torch.einsum("agd,ptd->agpt", q, chunks)
            routed_logits = token_logits[:, :, :routed_pages]
            mean_logits = torch.einsum("agd,pd->agp", q, means)
            true_max = routed_logits.amax(dim=-1)
            true_logmass = torch.logsumexp(routed_logits, dim=-1)
            quill_mask = mask_all[layer, head, :routed_pages * block].to(
                "cuda"
            ).reshape(routed_pages, block)
            selected_exact = routed_logits.masked_fill(
                ~quill_mask[None, None], -torch.inf
            ).amax(dim=-1)
            mixed = torch.maximum(mean_logits, selected_exact)

            proxies = {
                "mean": mean_logits,
                "quill_mixed_max": mixed,
                "oracle_all_max": true_max,
                "oracle_logmass": true_logmass,
            }
            selected_by_method = {}
            for name, proxy in proxies.items():
                candidate_proxy = proxy[:, :, candidate_idx]
                # Exactly the ShadowKV reduction: normalize each GQA query
                # head over candidate blocks, then keep the strongest group.
                router = torch.softmax(candidate_proxy, dim=-1).amax(dim=1)
                local_sel = router.topk(min(select, candidate_idx.numel()), dim=-1).indices
                chosen = candidate_idx[local_sel]
                selected_by_method[name] = chosen

                instances = steps * groups
                chosen_g = chosen[:, None].expand(-1, groups, -1)
                topmax = true_max.argmax(dim=-1)
                topmass = true_logmass.argmax(dim=-1)
                retained_blocks = torch.zeros(
                    steps, routed_pages, dtype=torch.bool, device="cuda"
                )
                retained_blocks.scatter_(1, chosen, True)
                retained_blocks[:, fixed_outlier] = True
                retained_blocks_g = retained_blocks[:, None].expand(-1, groups, -1)
                topmax_hit = retained_blocks_g.gather(
                    2, topmax[..., None]
                ).squeeze(-1)
                topmass_hit = retained_blocks_g.gather(
                    2, topmass[..., None]
                ).squeeze(-1)
                retained_tokens = torch.zeros(
                    steps, pages, block, dtype=torch.bool, device="cuda"
                )
                retained_tokens[:, :routed_pages] = retained_blocks[:, :, None]
                retained_tokens[:, routed_pages:] = True
                flat_logits = token_logits.reshape(steps, groups, -1)
                flat_keep = retained_tokens.reshape(steps, 1, -1)
                captured = torch.exp(
                    torch.logsumexp(flat_logits.masked_fill(~flat_keep, -torch.inf), dim=-1)
                    - torch.logsumexp(flat_logits, dim=-1)
                )

                if evidence_blocks:
                    ev = torch.tensor(evidence_blocks, device="cuda")
                    ev_logits = token_logits[:, :, ev].reshape(steps, groups, -1)
                    ev_keep = retained_tokens[:, ev].reshape(steps, 1, -1)
                    ev_capture = torch.exp(
                        torch.logsumexp(ev_logits.masked_fill(~ev_keep, -torch.inf), dim=-1)
                        - torch.logsumexp(ev_logits, dim=-1)
                    )
                    ev_block_recall = retained_tokens[:, ev, 0].float().mean()
                else:
                    ev_capture = torch.ones_like(captured)
                    ev_block_recall = torch.tensor(float("nan"), device="cuda")

                true_order = candidate_idx[
                    true_logmass[:, :, candidate_idx].topk(
                        min(select, candidate_idx.numel()), dim=-1
                    ).indices
                ]
                overlap = (
                    (chosen_g[..., None] == true_order[..., None, :])
                    .any(dim=-1).float().sum(dim=-1) / chosen.shape[-1]
                )
                state = sums[name]
                state["instances"] += instances
                state["topmax_hits"] += float(topmax_hit.sum())
                state["topmass_hits"] += float(topmass_hit.sum())
                state["mass_capture"] += float(captured.sum())
                state["evidence_mass_capture"] += float(ev_capture.sum())
                state["evidence_block_recall"] += float(ev_block_recall) * instances
                state["true_topset_overlap"] += float(overlap.sum())

            mean_sel = selected_by_method["mean"]
            mix_sel = selected_by_method["quill_mixed_max"]
            oracle_sel = selected_by_method["oracle_all_max"]
            overlap_mean = (mix_sel[..., None] == mean_sel[:, None, :]).any(-1).float().mean()
            overlap_oracle = (mix_sel[..., None] == oracle_sel[:, None, :]).any(-1).float().mean()
            exact_block = quill_mask.any(dim=-1)
            selected_exact_fraction = exact_block[mix_sel].float().mean()

            flat_top = routed_logits.reshape(steps, groups, -1).argmax(dim=-1)
            top_block = flat_top // block
            top_offset = flat_top % block
            top_quill = quill_mask[top_block, top_offset]

            # The decisive asymmetric case: a mixed-selected block is promoted
            # by a QUILL key while some unselected token elsewhere has a larger
            # true logit than that promoting key.
            mix_g = mix_sel[:, None].expand(-1, groups, -1)
            mix_exact = selected_exact.gather(2, mix_g)
            unselected = torch.ones_like(true_max, dtype=torch.bool)
            unselected.scatter_(2, mix_g, False)
            hidden_best = routed_logits.masked_fill(
                ~unselected[..., None], -torch.inf
            ).amax(dim=(-2, -1))
            promoting = mix_exact.amax(dim=-1)
            hidden_better = hidden_best > promoting

            has_exact = torch.isfinite(selected_exact)
            delta = (selected_exact - mean_logits)[has_exact]
            weight = steps * groups
            diagnostic["eligible_instances"] += weight
            diagnostic["selected_exact_fraction"] += float(selected_exact_fraction) * weight
            diagnostic["hybrid_mean_overlap"] += float(overlap_mean) * weight
            diagnostic["hybrid_oracle_max_overlap"] += float(overlap_oracle) * weight
            diagnostic["quill_global_top_token_recall"] += float(top_quill.float().mean()) * weight
            diagnostic["hidden_better_token_fraction"] += float(hidden_better.float().mean()) * weight
            diagnostic["hidden_better_token_gap"] += float(
                (hidden_best - promoting)[hidden_better].mean() if hidden_better.any() else 0.0
            ) * weight
            diagnostic["exact_minus_mean_median"] += float(delta.median()) * weight
            diagnostic["exact_minus_mean_p90"] += float(delta.quantile(0.9)) * weight

            layer_rows.append({
                "layer": layer,
                "kv_head": head,
                "hybrid_mean_overlap": float(overlap_mean),
                "hybrid_oracle_max_overlap": float(overlap_oracle),
                "selected_blocks_with_exact": float(selected_exact_fraction),
                "quill_global_top_token_recall": float(top_quill.float().mean()),
                "exact_minus_mean_median": float(delta.median()),
                "exact_minus_mean_p90": float(delta.quantile(0.9)),
            })
        print(f"layer {layer + 1}/{llm.num_layers}", flush=True)

    rows = []
    for name, state in sums.items():
        n = state.pop("instances")
        rows.append({"method": name, "instances": n, **{k: v / n for k, v in state.items()}})
    n = diagnostic.pop("eligible_instances")
    diagnostic = {k: v / n for k, v in diagnostic.items()}
    summary = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "prompt_tokens": len(tokens),
        "decode_steps": steps,
        "generated": tokenizer.decode(generated, skip_special_tokens=True),
        "answer": answer,
        "methods": rows,
        "diagnostic": diagnostic,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "router_metrics.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(args.output / "per_layer_head.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
