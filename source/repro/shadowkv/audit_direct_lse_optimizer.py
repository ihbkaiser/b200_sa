#!/usr/bin/env python3
"""Exact block-8 direct-LSE partitioning with held-out decode evaluation.

For every block, enumerate all 127 unordered bipartitions.  Each partition is
scored by the residual MSE after the closed-form quadratic Jensen-gap
calibration ``alpha * ||q/sqrt(d)||^2``.  Calibration uses only prompt queries;
real dense decode queries are held out for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))
sys.path.insert(0, str(REPO))

from repro.certified_sparse.probe_hard_multikey_block8 import (  # noqa: E402
    find_text_spans,
    flatten_answer,
    load_ruler_row,
    stop_token,
)


def partition_masks(device: torch.device, block: int = 8) -> torch.Tensor:
    patterns = torch.arange(2 ** (block - 1) - 1, device=device)
    bits = torch.arange(block - 1, device=device)
    tail = ((patterns[:, None] >> bits[None]) & 1).bool()
    return torch.cat((torch.ones(len(patterns), 1, device=device, dtype=torch.bool), tail), 1)


@torch.inference_mode()
def capture_prompt_and_decode_queries(
    model_path: str,
    input_ids: torch.Tensor,
    calibration_tokens: int,
    max_decode_steps: int,
):
    from models import choose_model_class
    from models.tensor_op import sample_token

    llm_class = choose_model_class(model_path)
    llm = llm_class(
        model_name=model_path,
        batch_size=1,
        device="cuda:0",
        max_length=int(input_ids.shape[1]) + max_decode_steps + 8,
        attn_mode="full",
        dtype=torch.bfloat16,
        sparse_budget=0,
        rank=160,
        chunk_size=8,
    )
    prompt_queries = {}
    decode_queries = defaultdict(list)
    apply_calls = 0
    original = llm.apply_rotary_pos_emb

    def wrapped(q, k, position_ids):
        nonlocal apply_calls
        q_rot, k_rot = original(q, k, position_ids)
        layer = apply_calls % llm.num_layers
        apply_calls += 1
        if q.shape[-2] > 1:
            # Exclude the final prompt query: it is also the first query in the
            # decode trajectory and must remain held out.
            start = max(0, q.shape[-2] - calibration_tokens - 1)
            prompt_queries[layer] = q_rot[0, :, start:-1].detach().cpu().to(torch.float16)
            decode_queries[layer].append(q_rot[0, :, -1].detach().cpu().to(torch.float16))
        else:
            decode_queries[layer].append(q_rot[0, :, -1].detach().cpu().to(torch.float16))
        return q_rot, k_rot

    llm.apply_rotary_pos_emb = wrapped
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    generated = [int(token[0, 0])]
    llm.kv_cache.H2D()
    for _ in range(max_decode_steps - 1):
        logits = llm.inference(token, llm.get_ctx(token))
        token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
        generated.append(int(token[0, 0]))
        if stop_token(llm, token):
            break
    llm.apply_rotary_pos_emb = original

    prompt = torch.stack([prompt_queries[i] for i in range(llm.num_layers)])
    decode = torch.stack(
        [torch.stack(decode_queries[i]) for i in range(llm.num_layers)]
    )
    return llm, prompt, decode, generated


@torch.inference_mode()
def optimize_blocks(
    keys: torch.Tensor,
    calibration_q: torch.Tensor,
    masks: torch.Tensor,
    batch_blocks: int = 256,
):
    """Return calibrated r=1/r=2 summaries and empirical risks."""
    block, dim = keys.shape[-2:]
    mask_f = masks.float()
    count_a = mask_f.sum(-1)
    count_b = block - count_a
    t = calibration_q.square().sum(-1)
    t2_sum = t.square().sum().clamp_min(1e-12)

    outputs = defaultdict(list)
    for start in range(0, keys.shape[0], batch_blocks):
        x = keys[start : start + batch_blocks].float()
        total = x.sum(1)
        sum_a = torch.einsum("cs,bsd->bcd", mask_f, x)
        center_a = sum_a / count_a[None, :, None]
        center_b = (total[:, None] - sum_a) / count_b[None, :, None]
        centers = torch.stack((center_a, center_b), dim=2)
        counts = torch.stack((count_a, count_b), dim=-1)

        exact = torch.logsumexp(torch.einsum("qd,bsd->qbs", calibration_q, x), dim=-1)
        component = torch.einsum("qd,bcrd->qbcr", calibration_q, centers)
        lower_two = torch.logsumexp(component + counts.log()[None, None], dim=-1)
        gap_two = exact[:, :, None] - lower_two
        alpha_two = (gap_two * t[:, None, None]).sum(0) / t2_sum
        risk_two = (gap_two - t[:, None, None] * alpha_two[None]).square().mean(0)
        best = risk_two.argmin(-1)
        rows = torch.arange(x.shape[0], device=x.device)

        mean = x.mean(1)
        lower_one = torch.einsum("qd,bd->qb", calibration_q, mean) + math.log(block)
        gap_one = exact - lower_one
        alpha_one = (gap_one * t[:, None]).sum(0) / t2_sum
        risk_one = (gap_one - t[:, None] * alpha_one[None]).square().mean(0)

        chosen_mask = masks[best]
        pair_dist = torch.cdist(x, x)
        same_a = chosen_mask[:, :, None] & chosen_mask[:, None, :]
        same_b = (~chosen_mask)[:, :, None] & (~chosen_mask)[:, None, :]
        diam_a = pair_dist.masked_fill(~same_a, -torch.inf).amax((1, 2))
        diam_b = pair_dist.masked_fill(~same_b, -torch.inf).amax((1, 2))
        diameter_two = torch.stack((diam_a, diam_b), -1)
        diameter_one = pair_dist.amax((1, 2))

        outputs["mean"].append(mean)
        outputs["centers"].append(centers[rows, best])
        outputs["counts"].append(counts[best])
        outputs["alpha_one"].append(alpha_one)
        outputs["alpha_two"].append(alpha_two[rows, best])
        outputs["risk_one"].append(risk_one)
        outputs["risk_two"].append(risk_two[rows, best])
        outputs["diameter_one"].append(diameter_one)
        outputs["diameter_two"].append(diameter_two)

    return {name: torch.cat(parts) for name, parts in outputs.items()}


def calibrated_score(lower, alpha, t, upper):
    estimate = lower + t * alpha
    return torch.minimum(upper, estimate)


def select(proxy: torch.Tensor, count: int) -> torch.Tensor:
    return torch.softmax(proxy.float(), -1).amax(1).topk(count, -1).indices


def overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[..., None] == b[:, None, :]).any(-1).float().mean(-1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, default=5)
    ap.add_argument("--calibration-tokens", type=int, default=8)
    ap.add_argument("--max-decode-steps", type=int, default=40)
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
    llm, prompt_q, decode_q, generated = capture_prompt_and_decode_queries(
        args.model, input_ids, args.calibration_tokens, args.max_decode_steps
    )

    block = 8
    pages = len(tokens) // block
    routed_pages = pages - args.local_blocks
    usable = pages * block
    groups = llm.num_heads // llm.num_key_value_heads
    select_count = args.dynamic_budget // block
    masks = partition_masks(torch.device("cuda"), block)
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    variants = ("raw", "calibrated")
    stats = {
        (variant, fraction): defaultdict(list)
        for variant in variants for fraction in fractions
    }
    risk_gains = []

    for layer in range(llm.num_layers):
        key_layer = llm.kv_cache.k_cache[layer][0, :, :usable].float()
        for head in range(llm.num_key_value_heads):
            chunks = key_layer[head].reshape(pages, block, llm.head_dim)
            routed = chunks[:routed_pages]
            means = routed.mean(1)
            cosine = F.cosine_similarity(means[:, None], routed, dim=-1)
            fixed = cosine.amin(-1).topk(args.outlier_blocks, largest=False).indices
            candidate_mask = torch.ones(routed_pages, device="cuda", dtype=torch.bool)
            candidate_mask[fixed] = False
            candidate = candidate_mask.nonzero().squeeze(-1)
            keys = routed[candidate]

            cal = prompt_q[layer, :, :, :].reshape(
                llm.num_heads, args.calibration_tokens, llm.head_dim
            )
            cal = cal[head * groups : (head + 1) * groups].reshape(-1, llm.head_dim)
            cal = cal.to("cuda", dtype=torch.float32) / math.sqrt(llm.head_dim)
            fitted = optimize_blocks(keys, cal, masks)
            gain = fitted["risk_one"] - fitted["risk_two"]
            risk_gains.extend(gain.cpu().tolist())

            q = decode_q[layer, :, head * groups : (head + 1) * groups].to(
                "cuda", dtype=torch.float32
            ) / math.sqrt(llm.head_dim)
            t = q.square().sum(-1)
            exact = torch.logsumexp(torch.einsum("tgd,csd->tgcs", q, keys), -1)
            lower_one = torch.einsum("tgd,cd->tgc", q, fitted["mean"]) + math.log(block)
            lower_two = torch.logsumexp(
                torch.einsum("tgd,crd->tgcr", q, fitted["centers"])
                + fitted["counts"].log()[None, None],
                -1,
            )
            upper_one = lower_one + (
                t[:, :, None] * fitted["diameter_one"].square()[None, None] / 8.0
            )
            upper_two = torch.logsumexp(
                torch.einsum("tgd,crd->tgcr", q, fitted["centers"])
                + fitted["counts"].log()[None, None]
                + t[:, :, None, None]
                * fitted["diameter_two"].square()[None, None]
                / 8.0,
                -1,
            )
            score_one = calibrated_score(
                lower_one,
                fitted["alpha_one"][None, None],
                t[:, :, None],
                upper_one,
            )
            score_two = calibrated_score(
                lower_two,
                fitted["alpha_two"][None, None],
                t[:, :, None],
                upper_two,
            )
            exact_sel = select(exact, select_count)
            exact_total = torch.logsumexp(exact, -1)

            for fraction in fractions:
                n_split = round(fraction * keys.shape[0])
                split = torch.zeros_like(gain, dtype=torch.bool)
                if n_split:
                    split[gain.topk(n_split).indices] = True
                for variant in variants:
                    if variant == "raw":
                        proxy = torch.where(split[None, None], lower_two, lower_one)
                    else:
                        proxy = torch.where(split[None, None], score_two, score_one)
                    chosen = select(proxy, select_count)
                    state = stats[(variant, fraction)]
                    state["overlap"].extend(overlap(chosen, exact_sel).cpu().tolist())
                    gather = chosen[:, None].expand(-1, groups, -1)
                    mass = torch.exp(
                        torch.logsumexp(exact.gather(2, gather), -1) - exact_total
                    )
                    state["mass"].extend(mass.cpu().flatten().tolist())
                    state["bias"].append(float((exact - proxy).mean()))
                    state["mse"].append(float((exact - proxy).square().mean()))
                    state["selected_split"].extend(split[chosen].float().cpu().flatten().tolist())
        print(f"layer {layer + 1}/{llm.num_layers}", flush=True)

    rows = []
    for (variant, fraction), state in stats.items():
        rows.append(
            {
                "variant": variant,
                "split_fraction": fraction,
                "mean_centroids": 1 + fraction,
                "decode_bias_exact_minus_score": sum(state["bias"]) / len(state["bias"]),
                "decode_mse": sum(state["mse"]) / len(state["mse"]),
                "topset_overlap_with_exact": sum(state["overlap"]) / len(state["overlap"]),
                "candidate_mass_capture": sum(state["mass"]) / len(state["mass"]),
                "selected_blocks_that_are_split": sum(state["selected_split"]) / len(state["selected_split"]),
            }
        )
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "summary.csv", index=False)
    payload = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "calibration_source": f"last {args.calibration_tokens} prompt queries excluding final query",
        "decode_steps": int(decode_q.shape[1]),
        "positive_risk_gain_fraction": sum(x > 0 for x in risk_gains) / len(risk_gains),
        "risk_gain_median": float(torch.tensor(risk_gains).median()),
        "answer": flatten_answer(row["outputs"]),
        "dense_generated": tokenizer.decode(generated, skip_special_tokens=True),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
