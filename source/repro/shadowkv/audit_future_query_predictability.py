#!/usr/bin/env python3
"""Can prompt K,V predict how future dense queries use a sealed block?

Future full-attention decode queries are used *only* to construct labels.  All
candidate predictors are functions of prompt K,V (and the fixed output
projection) available when the block is sealed.  The experiment therefore
separates an offline oracle/audit from a deployable, query-free predictor.

The primary population is not every block-query pair: an irrelevant block has
an arbitrary within-block argmax.  At each dense decode query we first take the
blocks with largest exact attention mass under a fixed token budget.  Labels
inside a block are then aggregated over these relevant events, weighted by the
block's exact attention mass.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr


REPO = Path(__file__).resolve().parents[2]
SHADOW = REPO / "ShadowKV"
sys.path.insert(0, str(SHADOW))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_cwe_streaming import source_occurrences  # noqa: E402

BLOCK = 8
TOKEN_FEATURES = (
    "k_norm",
    "v_norm",
    "ov_norm",
    "k_residual",
    "v_residual",
    "k_isolation",
    "v_isolation",
    "keydiff_pool",
    "value_novelty_pool",
    "kv_norm_product",
    "k_leverage",
    "v_leverage",
    "kv_joint_leverage",
    "kv_joint_rff_iso",
)
BLOCK_FEATURES = (
    "k_dispersion",
    "v_dispersion",
    "k_effective_rank",
    "v_effective_rank",
    "k_norm_cv",
    "v_norm_cv",
    "k_isolation_max",
    "v_isolation_max",
    "joint_leverage_max",
    "joint_leverage_gap",
    "keydiff_range",
    "joint_rff_range",
)
TOKEN_CONTROLS = ("leftmost_position", "rightmost_position", "random_control")


def load_row(path: Path, index: int) -> dict[str, Any]:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"row {index} is absent from {path}")


def stop_token(llm, token: torch.Tensor) -> bool:
    token_id = int(token[0, 0].item())
    if token_id == llm.tokenizer.eos_token_id:
        return True
    decoded = llm.tokenizer.decode([token_id])
    return decoded in {"<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end|>"} or token_id in {
        151329,
        151336,
        151338,
    }


def parse_case(text: str) -> tuple[str, Path, int]:
    try:
        label, path, index = text.rsplit(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be LABEL:PATH:INDEX") from exc
    return label, Path(path), int(index)


@torch.inference_mode()
def build_model(model_path: str, max_length: int):
    from models import choose_model_class

    cls = choose_model_class(model_path)
    return cls(
        model_name=model_path,
        batch_size=1,
        device="cuda:0",
        max_length=max_length,
        attn_mode="full",
        dtype=torch.bfloat16,
        sparse_budget=0,
        rank=160,
        chunk_size=BLOCK,
    )


@torch.inference_mode()
def capture_dense(llm, input_ids: torch.Tensor, max_decode_steps: int):
    """Run dense greedy generation and retain its actual post-RoPE queries."""
    from models.tensor_op import sample_token

    captured: dict[int, list[torch.Tensor]] = defaultdict(list)
    apply_calls = 0
    original = llm.apply_rotary_pos_emb

    def wrapped(q, k, position_ids):
        nonlocal apply_calls
        q_rot, k_rot = original(q, k, position_ids)
        layer = apply_calls % llm.num_layers
        apply_calls += 1
        # The final prompt query produces generated token 0.  Subsequent
        # one-token forwards produce generated tokens 1, 2, ... .
        captured[layer].append(q_rot[0, :, -1].detach().cpu().to(torch.float16))
        return q_rot, k_rot

    llm.apply_rotary_pos_emb = wrapped
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    generated = [int(token[0, 0].item())]
    llm.kv_cache.H2D()
    for _ in range(max_decode_steps - 1):
        logits = llm.inference(token, llm.get_ctx(token))
        token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
        generated.append(int(token[0, 0].item()))
        if stop_token(llm, token):
            break
    llm.apply_rotary_pos_emb = original

    steps = len(generated)
    for layer in range(llm.num_layers):
        if len(captured[layer]) != steps:
            raise RuntimeError(
                f"layer {layer}: captured {len(captured[layer])}, expected {steps}"
            )
    queries = torch.stack(
        [torch.stack(captured[layer]) for layer in range(llm.num_layers)]
    )
    return queries, generated


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _effective_rank(points: torch.Tensor) -> torch.Tensor:
    centered = points.float() - points.float().mean(-2, keepdim=True)
    gram = centered @ centered.transpose(-2, -1)
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    prob = eigen / eigen.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum(-1))


def _leverage(gram: torch.Tensor, eta: float = 0.05) -> torch.Tensor:
    gram = 0.5 * (gram.float() + gram.float().transpose(-2, -1))
    n = gram.shape[-1]
    ridge = eta * torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1)
    ridge = ridge.clamp_min(1e-8)
    eye = torch.eye(n, device=gram.device, dtype=gram.dtype)
    operator = torch.linalg.solve(gram + ridge[..., None, None] * eye, gram)
    return torch.diagonal(operator, dim1=-2, dim2=-1)


def _pool_scores(
    keys: torch.Tensor,
    values: torch.Tensor,
    layer: int,
    head: int,
    pool: int,
    rff_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K-only KeyDiff, V novelty, and query-free joint RFF leverage."""
    total, dim = keys.shape
    keydiff = torch.empty(total, device=keys.device)
    vnovel = torch.empty(total, device=keys.device)
    joint = torch.empty(total, device=keys.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(173 + layer * 1009 + head * 7919)
    omega = torch.randn(rff_dim, dim, generator=generator).to(keys.device) / math.sqrt(dim)
    bias = torch.rand(rff_dim, generator=generator).to(keys.device) * (2 * math.pi)
    sign = (
        torch.randint(0, 2, (rff_dim, dim), generator=generator).float() * 2 - 1
    ).to(keys.device) / math.sqrt(dim)
    eye = torch.eye(rff_dim, device=keys.device)
    for start in range(0, total, pool):
        stop = min(start + pool, total)
        k = keys[start:stop].float()
        v = values[start:stop].float()
        ka = F.normalize(k, dim=-1).mean(0, keepdim=True)
        va = F.normalize(v, dim=-1).mean(0, keepdim=True)
        keydiff[start:stop] = -F.cosine_similarity(k, ka, dim=-1)
        vnovel[start:stop] = -F.cosine_similarity(v, va, dim=-1)
        z = math.sqrt(2.0 / rff_dim) * torch.cos(k @ omega.T + bias) * (v @ sign.T)
        covariance = z.T @ z
        ridge = 0.05 * z.square().sum() / max(len(z), 1)
        inverse = torch.linalg.inv(covariance + ridge.clamp_min(1e-8) * eye)
        joint[start:stop] = ((z @ inverse) * z).sum(-1)
    return keydiff, vnovel, joint


@torch.inference_mode()
def static_features(
    chunks_k: torch.Tensor,
    chunks_v: torch.Tensor,
    wo_metric: torch.Tensor,
    layer: int,
    head: int,
    pool: int,
    rff_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pages, block, dim = chunks_k.shape
    k = chunks_k.float()
    v = chunks_v.float()
    k_mean = k.mean(1, keepdim=True)
    v_mean = v.mean(1, keepdim=True)
    k_res = (k - k_mean).norm(dim=-1)
    v_res = (v - v_mean).norm(dim=-1)
    k_norm = k.norm(dim=-1)
    v_norm = v.norm(dim=-1)
    ov_norm = torch.einsum("psd,de,pse->ps", v, wo_metric, v).clamp_min(0).sqrt()

    eye = torch.eye(block, device=k.device, dtype=torch.bool)[None]
    k_cos = F.normalize(k, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    v_cos = F.normalize(v, dim=-1) @ F.normalize(v, dim=-1).transpose(-2, -1)
    k_isolation = 1 - k_cos.masked_fill(eye, -torch.inf).amax(-1)
    v_isolation = 1 - v_cos.masked_fill(eye, -torch.inf).amax(-1)

    k_flat, v_flat = k.flatten(0, 1), v.flatten(0, 1)
    keydiff, vnovel, joint_rff = _pool_scores(
        k_flat, v_flat, layer, head, pool, rff_dim
    )
    keydiff = keydiff.view(pages, block)
    vnovel = vnovel.view(pages, block)
    joint_rff = joint_rff.view(pages, block)

    k_gram = k @ k.transpose(-2, -1)
    v_gram = v @ v.transpose(-2, -1)
    k_lev = _leverage(k_gram)
    v_lev = _leverage(v_gram)
    joint_lev = _leverage(k_gram * v_gram)

    token = {
        "k_norm": k_norm,
        "v_norm": v_norm,
        "ov_norm": ov_norm,
        "k_residual": k_res,
        "v_residual": v_res,
        "k_isolation": k_isolation,
        "v_isolation": v_isolation,
        "keydiff_pool": keydiff,
        "value_novelty_pool": vnovel,
        "kv_norm_product": k_norm * v_norm,
        "k_leverage": k_lev,
        "v_leverage": v_lev,
        "kv_joint_leverage": joint_lev,
        "kv_joint_rff_iso": joint_rff,
        # Non-K,V controls expose a trivial fixed within-block position bias.
        "leftmost_position": -torch.arange(block, device=k.device).float()[None].expand(pages, -1),
        "rightmost_position": torch.arange(block, device=k.device).float()[None].expand(pages, -1),
        "random_control": torch.rand(
            pages,
            block,
            generator=torch.Generator(device="cpu").manual_seed(
                991 + layer * 1009 + head * 7919
            ),
        ).to(k.device),
    }
    tiny = 1e-8
    block_features = {
        "k_dispersion": k_res.square().mean(-1),
        "v_dispersion": v_res.square().mean(-1),
        "k_effective_rank": _effective_rank(k),
        "v_effective_rank": _effective_rank(v),
        "k_norm_cv": k_norm.std(-1) / k_norm.mean(-1).clamp_min(tiny),
        "v_norm_cv": v_norm.std(-1) / v_norm.mean(-1).clamp_min(tiny),
        "k_isolation_max": k_isolation.amax(-1),
        "v_isolation_max": v_isolation.amax(-1),
        "joint_leverage_max": joint_lev.amax(-1),
        "joint_leverage_gap": joint_lev.amax(-1) - joint_lev.median(-1).values,
        "keydiff_range": keydiff.amax(-1) - keydiff.amin(-1),
        "joint_rff_range": joint_rff.amax(-1) - joint_rff.amin(-1),
    }
    return token, block_features


def output_projection_metric(llm, layer: int, kv_head: int) -> torch.Tensor:
    """Average ||W_O,h v||^2 metric over GQA query heads sharing a KV head."""
    groups = llm.num_heads // llm.num_key_value_heads
    dim = llm.head_dim
    wo = llm.layers[layer].wo.float()
    metric = torch.zeros(dim, dim, device=wo.device)
    for query_head in range(kv_head * groups, (kv_head + 1) * groups):
        w = wo[:, query_head * dim : (query_head + 1) * dim]
        metric.add_(w.T @ w)
    return metric / groups


def shared_gqa_selection(proxy: torch.Tensor, count: int) -> torch.Tensor:
    """Return [T,P] block mask using the deployed softmax-then-GQA-max rule."""
    group_score = torch.softmax(proxy.float(), dim=-1).amax(dim=1)
    index = group_score.topk(count, dim=-1).indices
    selected = torch.zeros_like(group_score, dtype=torch.bool)
    selected.scatter_(-1, index, True)
    return selected


def router_metrics(
    proxy: torch.Tensor,
    exact_selected: torch.Tensor,
    block_mass: torch.Tensor,
    count: int,
    target_mass_by_block: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    selected = shared_gqa_selection(proxy, count)
    retained = (block_mass * selected[:, None]).sum(-1).mean()
    overlap = (selected & exact_selected).sum(-1).float().mean() / count
    if target_mass_by_block is None or float(target_mass_by_block.sum()) == 0:
        target_coverage = float("nan")
    else:
        target_coverage = float(
            (target_mass_by_block * selected[:, None]).sum()
            / target_mass_by_block.sum()
        )
    return float(retained), float(overlap), target_coverage


@torch.inference_mode()
def analyze_case(
    llm,
    case_label: str,
    dataset: Path,
    sample_index: int,
    max_decode_steps: int,
    budget: int,
    prefix_tokens: int,
    recent_tokens: int,
    layer_start_fraction: float,
    pool: int,
    rff_dim: int,
    records_per_head: int,
):
    row = load_row(dataset, sample_index)
    encoded = llm.tokenizer(
        row["input"], add_special_tokens=False, return_offsets_mapping=True
    )
    tokens = encoded.input_ids
    answers = [str(value).lower() for value in row.get("outputs", [])]
    try:
        _, _, _, target_positions, _ = source_occurrences(
            row["input"], encoded.offset_mapping, BLOCK, answers
        )
    except ValueError:
        target_positions = set()
    input_ids = torch.tensor(tokens, device="cuda:0", dtype=torch.long)[None]
    queries_cpu, generated = capture_dense(llm, input_ids, max_decode_steps)
    usable = len(tokens) // BLOCK * BLOCK
    first_page = math.ceil(prefix_tokens / BLOCK)
    last_page = (usable - recent_tokens) // BLOCK
    pages = max(0, last_page - first_page)
    if pages <= budget // BLOCK:
        raise ValueError("routed block population is no larger than the budget")

    layer_first = int(math.floor(llm.num_layers * layer_start_fraction))
    layers = list(range(layer_first, llm.num_layers))
    groups = llm.num_heads // llm.num_key_value_heads
    choose_blocks = budget // BLOCK
    target_token_mask = torch.zeros(pages * BLOCK, device="cuda", dtype=torch.bool)
    for position in target_positions:
        if first_page * BLOCK <= position < last_page * BLOCK:
            target_token_mask[position - first_page * BLOCK] = True
    head_rows: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []

    for layer in layers:
        key_layer = llm.kv_cache.k_cache[layer][
            0, :, first_page * BLOCK : last_page * BLOCK
        ]
        value_layer = llm.kv_cache.v_cache[layer][
            0, :, first_page * BLOCK : last_page * BLOCK
        ]
        for head in range(llm.num_key_value_heads):
            chunks_k = key_layer[head].reshape(pages, BLOCK, llm.head_dim)
            chunks_v = value_layer[head].reshape(pages, BLOCK, llm.head_dim)
            q = queries_cpu[layer, :, head * groups : (head + 1) * groups]
            q = q.to("cuda", dtype=torch.float32) / math.sqrt(llm.head_dim)
            logits = torch.einsum("tgd,psd->tgps", q, chunks_k.float())
            prob = torch.softmax(logits.flatten(-2), dim=-1).view_as(logits)
            block_mass = prob.sum(-1)
            target_mass_by_block = (
                prob
                * target_token_mask.view(pages, BLOCK)[None, None]
            ).sum(-1)
            exact_lse = torch.logsumexp(logits, dim=-1)
            selected_shared = shared_gqa_selection(exact_lse, choose_blocks)
            selected = selected_shared[:, None].expand_as(block_mass)
            event_weight = block_mass * selected
            cond = prob / block_mass[..., None].clamp_min(1e-30)
            cond_top1, cond_argmax = cond.max(-1)
            entropy = -(cond * cond.clamp_min(1e-30).log()).sum(-1) / math.log(BLOCK)

            block_weight = event_weight.sum((0, 1))
            relevant = block_weight > 0
            token_mass = (prob * selected[..., None]).sum((0, 1))
            share = token_mass / block_weight[:, None].clamp_min(1e-30)
            peak_onehot = F.one_hot(cond_argmax, num_classes=BLOCK).float()
            peak_vote = (peak_onehot * event_weight[..., None]).sum((0, 1))
            peak_vote = peak_vote / block_weight[:, None].clamp_min(1e-30)
            future_top1 = (cond_top1 * event_weight).sum((0, 1)) / block_weight.clamp_min(1e-30)
            future_entropy = (entropy * event_weight).sum((0, 1)) / block_weight.clamp_min(1e-30)
            hit_count = selected.sum((0, 1))

            token_feature, block_feature = static_features(
                chunks_k,
                chunks_v,
                output_projection_metric(llm, layer, head),
                layer,
                head,
                pool,
                rff_dim,
            )
            # A query-free Pareto score for blocks that are simultaneously
            # difficult in K-space and non-redundant in V-space.  Percentile
            # ranks remove layer/head scale without fitting any coefficients.
            def percentile_rank(score: torch.Tensor) -> torch.Tensor:
                order = score.argsort().argsort().float()
                return (order + 1.0) / max(len(score), 1)

            block_feature["kv_tail_pareto"] = (
                percentile_rank(block_feature["keydiff_range"])
                * percentile_rank(block_feature["v_effective_rank"])
            ).sqrt()
            target_by_token = target_token_mask.view(pages, BLOCK)
            target_blocks = target_by_token.any(-1)
            if target_blocks.any():
                all_token_mass = prob.sum((0, 1))
                all_share = all_token_mass / all_token_mass.sum(-1, keepdim=True).clamp_min(1e-30)
                future_winner = all_share.argmax(-1)
                target_rows = torch.arange(pages, device=q.device)[target_blocks]
                future_winner_is_target = target_by_token[
                    target_rows, future_winner[target_blocks]
                ].float().mean()
                target_token_attention = all_token_mass[target_by_token].sum()
                target_block_attention = all_token_mass[target_blocks].sum()
                for feature_name in (
                    "keydiff_pool",
                    "k_residual",
                    "k_isolation",
                    "value_novelty_pool",
                ):
                    feature_score = token_feature[feature_name]
                    feature_winner = feature_score.argmax(-1)
                    singleton_is_target = target_by_token[
                        target_rows, feature_winner[target_blocks]
                    ].float().mean()
                    selected_share = all_share[
                        target_rows, feature_winner[target_blocks]
                    ]
                    oracle_share = all_share[target_blocks].amax(-1)
                    top2_feature = feature_score.topk(2, dim=-1).values
                    confidence = top2_feature[:, 0] - top2_feature[:, 1]
                    n_enhanced = max(1, round(0.25 * pages))
                    enhanced = torch.zeros(pages, device=q.device, dtype=torch.bool)
                    enhanced[confidence.topk(n_enhanced).indices] = True
                    percentile = confidence.argsort().argsort().float() / max(pages - 1, 1)
                    head_rows.append(
                        {
                            "case": case_label,
                            "sample": sample_index,
                            "layer": layer,
                            "kv_head": head,
                            "kind": "target",
                            "feature": feature_name,
                            "future_steps": int(len(generated)),
                            "target_blocks": int(target_blocks.sum()),
                            "singleton_is_target_rate": float(singleton_is_target),
                            "target_blocks_enhanced_at_r1.25": float(
                                enhanced[target_blocks].float().mean()
                            ),
                            "target_confidence_percentile": float(
                                percentile[target_blocks].mean()
                            ),
                            "target_block_singleton_oracle_capture": float(
                                selected_share.sum() / oracle_share.sum().clamp_min(1e-30)
                            ),
                            "future_winner_is_target_rate": float(future_winner_is_target),
                            "target_token_share_inside_target_blocks": float(
                                target_token_attention / target_block_attention.clamp_min(1e-30)
                            ),
                        }
                    )
                # Which query-free statistic should receive the scarce extra
                # summaries?  Future target mass is the label only.  We test
                # both intrinsic block geometry and max/range/margin derived
                # from candidate token scores.
                allocation_scores: dict[str, torch.Tensor] = dict(block_feature)
                for feature_name in (
                    "keydiff_pool",
                    "k_residual",
                    "k_isolation",
                    "value_novelty_pool",
                    "v_isolation",
                ):
                    ordered = token_feature[feature_name].topk(2, dim=-1).values
                    allocation_scores[f"{feature_name}_max"] = ordered[:, 0]
                    allocation_scores[f"{feature_name}_margin"] = ordered[:, 0] - ordered[:, 1]
                    allocation_scores[f"{feature_name}_range"] = (
                        token_feature[feature_name].amax(-1)
                        - token_feature[feature_name].amin(-1)
                    )
                target_weight = target_mass_by_block.sum((0, 1))
                n_enhanced = max(1, round(0.25 * pages))
                target_np = target_blocks.cpu().numpy()
                for score_name, score in allocation_scores.items():
                    chosen = score.topk(n_enhanced).indices
                    chosen_mask = torch.zeros(pages, device=q.device, dtype=torch.bool)
                    chosen_mask[chosen] = True
                    head_rows.append(
                        {
                            "case": case_label,
                            "sample": sample_index,
                            "layer": layer,
                            "kv_head": head,
                            "kind": "allocation",
                            "feature": score_name,
                            "future_steps": int(len(generated)),
                            "target_block_auc": _binary_auc(
                                score.float().cpu().numpy(), target_np
                            ),
                            "top25_target_block_recall": float(
                                (chosen_mask & target_blocks).sum()
                                / target_blocks.sum().clamp_min(1)
                            ),
                            "top25_target_attention_recall": float(
                                target_weight[chosen_mask].sum()
                                / target_weight.sum().clamp_min(1e-30)
                            ),
                        }
                    )

            # Does a static K,V token improve a *block-level* summary?  The
            # routing unit remains the positional block.  A chosen token is a
            # singleton component and the other seven tokens are represented
            # by their arithmetic mean.  The 25% variants enhance only the
            # quarter of blocks with the largest static top1--top2 margin,
            # giving mean r=1.25; the 100% variants are a diagnostic ceiling.
            mean_lse = torch.einsum(
                "tgd,pd->tgp", q, chunks_k.float().mean(1)
            ) + math.log(BLOCK)
            oracle_mass, _, oracle_target = router_metrics(
                exact_lse,
                selected_shared,
                block_mass,
                choose_blocks,
                target_mass_by_block,
            )
            mean_mass, mean_overlap, mean_target = router_metrics(
                mean_lse,
                selected_shared,
                block_mass,
                choose_blocks,
                target_mass_by_block,
            )
            head_rows.append(
                {
                    "case": case_label,
                    "sample": sample_index,
                    "layer": layer,
                    "kv_head": head,
                    "kind": "router",
                    "feature": "mean_r1",
                    "future_steps": int(len(generated)),
                    "retained_mass": mean_mass,
                    "oracle_retained_mass": oracle_mass,
                    "oracle_router_fraction": mean_mass / max(oracle_mass, 1e-30),
                    "overlap_exact_router": mean_overlap,
                    "target_mass_coverage": mean_target,
                    "oracle_target_mass_coverage": oracle_target,
                }
            )
            for feature_name in (
                "keydiff_pool",
                "k_residual",
                "k_isolation",
                "value_novelty_pool",
            ):
                static_score = token_feature[feature_name]
                token_index = static_score.argmax(-1)
                row_index = torch.arange(pages, device=static_score.device)
                exact_key = chunks_k.float()[row_index, token_index]
                rest_mean = (
                    chunks_k.float().sum(1) - exact_key
                ) / (BLOCK - 1)
                exact_token_logit = torch.einsum("tgd,pd->tgp", q, exact_key)
                rest_logit = torch.einsum("tgd,pd->tgp", q, rest_mean) + math.log(BLOCK - 1)
                enhanced = torch.logaddexp(exact_token_logit, rest_logit)
                top2 = static_score.topk(2, dim=-1).values
                confidence = top2[:, 0] - top2[:, 1]
                for fraction in (0.25, 1.0):
                    if fraction == 1.0:
                        proxy = enhanced
                    else:
                        n_enhanced = max(1, round(fraction * pages))
                        use = torch.zeros(pages, device=q.device, dtype=torch.bool)
                        use[confidence.topk(n_enhanced).indices] = True
                        proxy = torch.where(use[None, None], enhanced, mean_lse)
                    retained, overlap_value, target_coverage = router_metrics(
                        proxy,
                        selected_shared,
                        block_mass,
                        choose_blocks,
                        target_mass_by_block,
                    )
                    head_rows.append(
                        {
                            "case": case_label,
                            "sample": sample_index,
                            "layer": layer,
                            "kv_head": head,
                            "kind": "router",
                            "feature": f"{feature_name}_r{1 + fraction:.2f}",
                            "future_steps": int(len(generated)),
                            "retained_mass": retained,
                            "oracle_retained_mass": oracle_mass,
                            "oracle_router_fraction": retained / max(oracle_mass, 1e-30),
                            "overlap_exact_router": overlap_value,
                            "target_mass_coverage": target_coverage,
                            "oracle_target_mass_coverage": oracle_target,
                        }
                    )
                # Decouple the token identity rule from the capacity allocator.
                # These remain K,V-only and keep exactly the same mean r=1.25.
                for allocator_name in (
                    "v_effective_rank",
                    "k_effective_rank",
                    "k_dispersion",
                    "keydiff_range",
                    "kv_tail_pareto",
                ):
                    allocation_score = block_feature[allocator_name]
                    n_enhanced = max(1, round(0.25 * pages))
                    use = torch.zeros(pages, device=q.device, dtype=torch.bool)
                    use[allocation_score.topk(n_enhanced).indices] = True
                    proxy = torch.where(use[None, None], enhanced, mean_lse)
                    retained, overlap_value, target_coverage = router_metrics(
                        proxy,
                        selected_shared,
                        block_mass,
                        choose_blocks,
                        target_mass_by_block,
                    )
                    head_rows.append(
                        {
                            "case": case_label,
                            "sample": sample_index,
                            "layer": layer,
                            "kv_head": head,
                            "kind": "router",
                            "feature": f"{feature_name}_by_{allocator_name}_r1.25",
                            "future_steps": int(len(generated)),
                            "retained_mass": retained,
                            "oracle_retained_mass": oracle_mass,
                            "oracle_router_fraction": retained / max(oracle_mass, 1e-30),
                            "overlap_exact_router": overlap_value,
                            "target_mass_coverage": target_coverage,
                            "oracle_target_mass_coverage": oracle_target,
                        }
                    )
            rel_idx = relevant.nonzero().squeeze(-1)
            weights = block_weight[rel_idx]
            oracle_idx = share[rel_idx].argmax(-1)
            oracle_best = share[rel_idx].amax(-1)
            vote_idx = peak_vote[rel_idx].argmax(-1)
            vote_stability = peak_vote[rel_idx].amax(-1)

            for name, score_all in token_feature.items():
                score = score_all[rel_idx]
                pred = score.argmax(-1)
                top2 = score.topk(2, dim=-1).indices
                rows = torch.arange(len(rel_idx), device=score.device)
                captured = share[rel_idx][rows, pred]
                weighted_hit = ((pred == oracle_idx).float() * weights).sum() / weights.sum()
                weighted_vote_hit = ((pred == vote_idx).float() * weights).sum() / weights.sum()
                weighted_top2 = (
                    (top2 == oracle_idx[:, None]).any(-1).float() * weights
                ).sum() / weights.sum()
                oracle_ratio = (captured * weights).sum() / (oracle_best * weights).sum()
                # Scores are effectively continuous here.  Two argsorts give
                # row-wise ranks without thousands of small scipy calls.
                score_rank = score.argsort(-1).argsort(-1).float()
                label_rank = share[rel_idx].argsort(-1).argsort(-1).float()
                score_rank = score_rank - score_rank.mean(-1, keepdim=True)
                label_rank = label_rank - label_rank.mean(-1, keepdim=True)
                rho = (score_rank * label_rank).sum(-1) / (
                    score_rank.square().sum(-1).sqrt()
                    * label_rank.square().sum(-1).sqrt()
                ).clamp_min(1e-12)
                head_rows.append(
                    {
                        "case": case_label,
                        "sample": sample_index,
                        "layer": layer,
                        "kv_head": head,
                        "kind": "token",
                        "feature": name,
                        "relevant_blocks": int(len(rel_idx)),
                        "future_steps": int(len(generated)),
                        "weighted_top1_hit": float(weighted_hit),
                        "weighted_top2_recall": float(weighted_top2),
                        "weighted_peak_vote_hit": float(weighted_vote_hit),
                        "oracle_mass_capture": float(oracle_ratio),
                        "within_block_spearman": float(
                            (rho * weights).sum() / weights.sum()
                        ),
                        "mean_future_top1": float(
                            (future_top1[rel_idx] * weights).sum() / weights.sum()
                        ),
                        "mean_future_mass_concentration": float(
                            (oracle_best * weights).sum() / weights.sum()
                        ),
                        "mean_peak_identity_stability": float(
                            (vote_stability * weights).sum() / weights.sum()
                        ),
                    }
                )

            top1_np = future_top1[rel_idx].float().cpu().numpy()
            peaked = top1_np >= 0.50
            uniform = top1_np <= 0.25
            decisive = peaked | uniform
            for name, score_all in block_feature.items():
                score_np = score_all[rel_idx].float().cpu().numpy()
                head_rows.append(
                    {
                        "case": case_label,
                        "sample": sample_index,
                        "layer": layer,
                        "kv_head": head,
                        "kind": "block",
                        "feature": name,
                        "relevant_blocks": int(len(rel_idx)),
                        "future_steps": int(len(generated)),
                        "peak_fraction": float(peaked.mean()),
                        "uniform_fraction": float(uniform.mean()),
                        "peak_spearman": _safe_spearman(score_np, top1_np),
                        "peak_vs_uniform_auc": _binary_auc(score_np[decisive], peaked[decisive]),
                        "mean_future_top1": float(
                            (future_top1[rel_idx] * weights).sum() / weights.sum()
                        ),
                        "mean_future_mass_concentration": float(
                            (oracle_best * weights).sum() / weights.sum()
                        ),
                        "mean_peak_identity_stability": float(
                            (vote_stability * weights).sum() / weights.sum()
                        ),
                    }
                )

            # A bounded raw sample supports held-out K,V-only linear probes.
            seed = 1000003 * sample_index + 1009 * layer + head
            generator = torch.Generator(device="cpu").manual_seed(seed)
            if len(rel_idx) > records_per_head:
                take = torch.randperm(len(rel_idx), generator=generator)[:records_per_head]
                chosen = rel_idx[take.to(rel_idx.device)]
            else:
                chosen = rel_idx
            for page in chosen.tolist():
                base = {
                    "case": case_label,
                    "sample": sample_index,
                    "layer": layer,
                    "kv_head": head,
                    "page": page + first_page,
                    "weight": float(block_weight[page]),
                    "hit_count": int(hit_count[page]),
                            "future_top1": float(future_top1[page]),
                            "future_entropy": float(future_entropy[page]),
                            "future_mass_concentration": float(share[page].amax()),
                            "future_peak_identity_stability": float(peak_vote[page].amax()),
                }
                block_records.append(
                    base | {name: float(value[page]) for name, value in block_feature.items()}
                )
                for position in range(BLOCK):
                    token_records.append(
                        base
                        | {
                            "position": position,
                            "future_mass_share": float(share[page, position]),
                            "future_peak_vote": float(peak_vote[page, position]),
                        }
                        | {
                            name: float(value[page, position])
                            for name, value in token_feature.items()
                        }
                    )

            del logits, prob, cond, token_feature, block_feature
        print(
            f"  {case_label}[{sample_index}] layer {layer + 1}/{llm.num_layers}",
            flush=True,
        )
    answer = llm.tokenizer.decode(generated, skip_special_tokens=True)
    meta = {
        "case": case_label,
        "dataset": str(dataset),
        "sample": sample_index,
        "prompt_tokens": len(tokens),
        "usable_routed_blocks": pages,
        "future_steps": len(generated),
        "generated_text": answer,
        "target_token_count": len(target_positions),
        "layers": layers,
    }
    return meta, head_rows, block_records, token_records


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.columns
        if column
        not in {"case", "sample", "layer", "kv_head", "kind", "feature"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    return metrics.groupby(["kind", "feature"], as_index=False)[numeric].mean()


def fit_ridge_probe(records: pd.DataFrame, features: tuple[str, ...], target: str):
    """Leave-one-case-out linear diagnostic; future q never enters X."""
    rows = []
    cases = sorted(records.case.unique())
    if len(cases) < 2:
        return pd.DataFrame()
    for heldout in cases:
        train = records[records.case != heldout]
        test = records[records.case == heldout].copy()
        x_train = train[list(features)].to_numpy(np.float64)
        x_test = test[list(features)].to_numpy(np.float64)
        mean = x_train.mean(0)
        std = x_train.std(0)
        std[std < 1e-8] = 1
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.column_stack((np.ones(len(x_train)), x_train))
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        y_train = train[target].to_numpy(np.float64)
        ridge = np.eye(x_train.shape[1]) * 1e-2
        ridge[0, 0] = 0
        beta = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
        test["prediction"] = x_test @ beta
        if target == "future_mass_share":
            group = ["case", "sample", "layer", "kv_head", "page"]
            pred_idx = test.groupby(group).prediction.idxmax()
            oracle_idx = test.groupby(group).future_mass_share.idxmax()
            pred_rows = test.loc[pred_idx].set_index(group)
            oracle_rows = test.loc[oracle_idx].set_index(group)
            aligned = pred_rows.join(
                oracle_rows[["future_mass_share"]].rename(columns={"future_mass_share": "oracle"})
            )
            weight = aligned.weight.to_numpy()
            hit = (pred_rows.position.to_numpy() == oracle_rows.position.to_numpy()).astype(float)
            rows.append(
                {
                    "heldout": heldout,
                    "target": target,
                    "weighted_top1_hit": float(np.average(hit, weights=weight)),
                    "oracle_mass_capture": float(
                        np.sum(weight * aligned.future_mass_share) / np.sum(weight * aligned.oracle)
                    ),
                }
            )
        else:
            rows.append(
                {
                    "heldout": heldout,
                    "target": target,
                    "spearman": _safe_spearman(
                        test.prediction.to_numpy(), test[target].to_numpy()
                    ),
                }
            )
    return pd.DataFrame(rows)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--case", action="append", type=parse_case, required=True)
    ap.add_argument("--max-decode-steps", type=int, default=32)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--prefix-tokens", type=int, default=32)
    ap.add_argument("--recent-tokens", type=int, default=32)
    ap.add_argument("--layer-start-fraction", type=float, default=2 / 3)
    ap.add_argument("--score-pool", type=int, default=1024)
    ap.add_argument("--rff-dim", type=int, default=64)
    ap.add_argument("--records-per-head", type=int, default=48)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.budget % BLOCK:
        raise ValueError("budget must be divisible by block size 8")

    max_prompt = 0
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for _, path, index in args.case:
        row = load_row(path, index)
        max_prompt = max(max_prompt, len(tokenizer.encode(row["input"], add_special_tokens=False)))
    del tokenizer
    llm = build_model(args.model, max_prompt + args.max_decode_steps + 8)

    args.output.mkdir(parents=True, exist_ok=True)
    all_meta, all_metrics, all_blocks, all_tokens = [], [], [], []
    for label, path, index in args.case:
        print(f"Running {args.model_label} {label}[{index}] ...", flush=True)
        meta, metrics, blocks, tokens = analyze_case(
            llm,
            label,
            path,
            index,
            args.max_decode_steps,
            args.budget,
            args.prefix_tokens,
            args.recent_tokens,
            args.layer_start_fraction,
            args.score_pool,
            args.rff_dim,
            args.records_per_head,
        )
        all_meta.append(meta)
        all_metrics.extend(metrics)
        all_blocks.extend(blocks)
        all_tokens.extend(tokens)
        gc.collect()
        torch.cuda.empty_cache()

    metrics = pd.DataFrame(all_metrics)
    blocks = pd.DataFrame(all_blocks)
    tokens = pd.DataFrame(all_tokens)
    metrics.to_csv(args.output / "per_head_metrics.csv", index=False)
    summarize(metrics).to_csv(args.output / "summary.csv", index=False)
    blocks.to_parquet(args.output / "block_records.parquet", index=False)
    tokens.to_parquet(args.output / "token_records.parquet", index=False)
    fit_ridge_probe(tokens, TOKEN_FEATURES, "future_mass_share").to_csv(
        args.output / "heldout_token_ridge.csv", index=False
    )
    fit_ridge_probe(blocks, BLOCK_FEATURES, "future_top1").to_csv(
        args.output / "heldout_peak_ridge.csv", index=False
    )
    with (args.output / "metadata.json").open("w") as handle:
        json.dump(
            {
                "model": args.model,
                "model_label": args.model_label,
                "block": BLOCK,
                "budget": args.budget,
                "prefix_tokens": args.prefix_tokens,
                "recent_tokens": args.recent_tokens,
                "max_decode_steps": args.max_decode_steps,
                "future_query_role": "labels only; no future query appears in predictors",
                "cases": all_meta,
            },
            handle,
            indent=2,
        )
    print("\nK,V-only feature summary")
    print(summarize(metrics).to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
