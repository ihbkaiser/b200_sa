#!/usr/bin/env python3
"""Test whether prefill K/V geometry identifies retrieval-sensitive heads.

Future attention is used only to construct held-out analysis labels.  Every
candidate predictor is available from the prefetched K/V tensors (and is
exported by ``diagnose_router_mass_full_trace.py`` before decode queries are
observed).  Leave-one-trace-out evaluation prevents a per-prompt oracle gate
from being mistaken for a deployable signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDS = ["sample", "layer", "kv_head"]
ORACLE = ["full_target_attention_mass", "full_source_attention_mass"]
ORACLE_SELECTIVITY = [
    "oracle_candidate_top1_mass",
    "oracle_candidate_top8_mass",
    "oracle_candidate_entropy",
    "oracle_candidate_effective_support",
]
CONTRAST = "ours_density3_robustplace_kde8_maxmed_shrink0p5_r1p25"
ABSOLUTE = (
    "ours_density3_robustplace_kde8_absmean_x_maxmed_shrink0p5_r1p25"
)


def rank01(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series.rank(method="average") - 1.0) / (len(series) - 1.0)


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.rank().corr(right.rank()))


def load_trace(directory: Path) -> pd.DataFrame | None:
    geometry_path = directory / "angular_density_head_geometry.csv"
    oracle_path = directory / "per_head_layer.csv"
    if not geometry_path.exists() or not oracle_path.exists():
        return None
    geometry = pd.read_csv(geometry_path)
    oracle_all = pd.read_csv(oracle_path)
    missing = set(IDS + ORACLE) - set(oracle_all.columns)
    if missing:
        raise ValueError(f"{oracle_path}: missing columns {sorted(missing)}")
    oracle = oracle_all[IDS + ORACLE].drop_duplicates(IDS)

    def method_columns(name: str, prefix: str) -> pd.DataFrame:
        frame = oracle_all[oracle_all.method == name]
        columns = [
            "total_mass_retained",
            "retained_target_attention_mass",
            "retained_source_attention_mass",
        ]
        if frame.empty:
            return pd.DataFrame(columns=IDS)
        return frame[IDS + columns].rename(
            columns={column: f"{prefix}_{column}" for column in columns}
        )

    joined = geometry.merge(oracle, on=IDS, validate="one_to_one")
    joined = joined.merge(
        method_columns(CONTRAST, "contrast"), on=IDS, how="left"
    ).merge(method_columns(ABSOLUTE, "absolute"), on=IDS, how="left")
    for quantity in (
        "total_mass_retained",
        "retained_target_attention_mass",
        "retained_source_attention_mass",
    ):
        joined[f"absolute_minus_contrast_{quantity}"] = (
            joined[f"absolute_{quantity}"] - joined[f"contrast_{quantity}"]
        )
    joined["trace"] = directory.name
    for column in ORACLE:
        joined[f"{column}_rank"] = joined.groupby("trace")[column].transform(
            rank01
        )
    joined["answer_role_rank"] = joined[
        [f"{column}_rank" for column in ORACLE]
    ].max(axis=1)
    if set(ORACLE_SELECTIVITY).issubset(joined.columns):
        for column in ORACLE_SELECTIVITY:
            joined[f"{column}_rank"] = joined.groupby("trace")[column].transform(
                rank01
            )
        joined["selective_read_rank"] = joined[
            "oracle_candidate_top8_mass_rank"
        ]
        # CWE has explicit source/answer labels; other RULER tasks use the
        # generic concentration label.  This choice is made per trace, never
        # per head, so it cannot leak an oracle gating decision.
        has_answer_annotation = bool(
            joined[ORACLE].to_numpy(float).max(initial=0.0) > 0.0
        )
        joined["semantic_role_rank"] = (
            joined["answer_role_rank"]
            if has_answer_annotation else joined["selective_read_rank"]
        )
    else:
        joined["semantic_role_rank"] = joined["answer_role_rank"]
    return joined


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[scale < 1e-8] = 1.0
    train = np.nan_to_num((train_x - mean) / scale)
    test = np.nan_to_num((test_x - mean) / scale)
    design = np.column_stack([np.ones(len(train)), train])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.column_stack([np.ones(len(test)), test]) @ coef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for geometry in sorted(args.root.glob("*/angular_density_head_geometry.csv")):
        frame = load_trace(geometry.parent)
        if frame is not None:
            frames.append(frame)
    if len(frames) < 2:
        raise SystemExit(f"need at least two completed traces under {args.root}")
    data = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "head_role_joined.csv", index=False)

    excluded_prefixes = (
        "absolute_", "contrast_", "full_", "retained_",
        "absolute_minus_",
    )
    features = [
        column for column in data.columns
        if column not in IDS + ["trace", "semantic_role_rank"]
        and not column.endswith("_rank")
        and not column.startswith(excluded_prefixes + ("oracle_",))
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    targets = {
        "semantic_role_rank": "semantic_role_rank",
        "absolute_density_total_delta": (
            "absolute_minus_contrast_total_mass_retained"
        ),
        "absolute_density_target_delta": (
            "absolute_minus_contrast_retained_target_attention_mass"
        ),
    }
    corr_rows = []
    for target_name, target in targets.items():
        for feature in features:
            corr_rows.append({
                "target": target_name,
                "feature": feature,
                "spearman": safe_corr(data[feature], data[target]),
            })
    correlations = pd.DataFrame(corr_rows).sort_values(
        ["target", "spearman"], ascending=[True, False]
    )
    correlations.to_csv(args.output / "feature_correlations.csv", index=False)

    subsets = {
        "k_only": [feature for feature in features if "value_" not in feature],
        "k_and_v": features,
        "compact": [
            feature for feature in (
                "self_risk1_q90", "self_gain12_q90",
                "density_contrast_q90", "density_absolute_q90",
                "mean_resultant_length", "key_norm_q90",
                "value_residual_q90", "local_angular_isolation_q90",
            ) if feature in features
        ],
    }
    cv_rows = []
    for subset_name, subset in subsets.items():
        if not subset:
            continue
        predictions = pd.Series(index=data.index, dtype=float)
        for held_out in sorted(data.trace.unique()):
            train = data.trace != held_out
            test = ~train
            predictions.loc[test] = ridge_predict(
                data.loc[train, subset].to_numpy(float),
                data.loc[train, "semantic_role_rank"].to_numpy(float),
                data.loc[test, subset].to_numpy(float),
            )
        for trace, group in data.assign(prediction=predictions).groupby("trace"):
            threshold = group.prediction.quantile(0.75)
            selected = group.prediction >= threshold
            true_top = group.semantic_role_rank >= 0.75
            cv_rows.append({
                "subset": subset_name,
                "trace": trace,
                "spearman": safe_corr(
                    group.prediction, group.semantic_role_rank
                ),
                "top_quartile_precision": float(true_top[selected].mean()),
                "top_quartile_recall": float(selected[true_top].mean()),
                "role_enrichment": float(
                    group.loc[selected, "semantic_role_rank"].mean()
                    / group.semantic_role_rank.mean()
                ),
                "features": len(subset),
            })
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(args.output / "leave_one_trace_out.csv", index=False)
    cv_summary = cv.groupby("subset", as_index=False).mean(numeric_only=True)
    cv_summary.to_csv(args.output / "leave_one_trace_out_summary.csv", index=False)

    result = {
        "traces": sorted(data.trace.unique().tolist()),
        "rows": len(data),
        "features": features,
        "cv": cv_summary.to_dict(orient="records"),
        "top_correlations": {
            target: correlations[correlations.target == target]
            .reindex(
                correlations[correlations.target == target].spearman.abs()
                .sort_values(ascending=False).index
            ).head(10).to_dict(orient="records")
            for target in targets
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["cv"], indent=2))


if __name__ == "__main__":
    main()
