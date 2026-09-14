#!/usr/bin/env python3
"""Visualize the complete self-K r=1..8 path for one audited block."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))

def reconstruct_keys(record: dict) -> torch.Tensor:
    """Recover an isometric key realization from norms and the Gram matrix."""
    norm = np.asarray(record["token_key_norm"], dtype=np.float64)
    cosine = np.asarray(record["pairwise_cosine"], dtype=np.float64)
    gram = norm[:, None] * cosine * norm[None, :]
    eigenvalue, eigenvector = np.linalg.eigh((gram + gram.T) / 2)
    key = eigenvector @ np.diag(np.sqrt(np.clip(eigenvalue, 0, None)))
    return torch.tensor(key, dtype=torch.float32)[None]


def visible_token(token: str) -> str:
    return token.replace(" ", "·").replace("\n", "↵") or "∅"


def partitions(labels: np.ndarray) -> list[list[int]]:
    return [
        np.flatnonzero(labels == label).astype(int).tolist()
        for label in np.unique(labels)
    ]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#d5d9df", width=3)
    draw.text((box[0] + 28, box[1] + 20), title, fill="#17202a", font=font(35, True))


def line_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, np.ndarray, str]],
    *,
    log_y: bool,
    y_label: str,
) -> None:
    left, top, right, bottom = box
    plot = (left + 115, top + 100, right - 35, bottom - 100)
    values = np.concatenate([item[1] for item in series])
    if log_y:
        values = np.log10(np.maximum(values, 1e-5))
        y_min = math.floor(float(values.min()))
        y_max = math.ceil(float(values.max()))
    else:
        y_min, y_max = 0.0, max(1.0, float(values.max()) * 1.1)
    if y_max <= y_min:
        y_max = y_min + 1

    def point(index: int, value: float) -> tuple[float, float]:
        x = plot[0] + index * (plot[2] - plot[0]) / 7
        transformed = math.log10(max(value, 1e-5)) if log_y else value
        y = plot[3] - (transformed - y_min) * (plot[3] - plot[1]) / (y_max - y_min)
        return x, y

    draw.line((plot[0], plot[1], plot[0], plot[3]), fill="#30343b", width=3)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#30343b", width=3)
    for index in range(8):
        x, _ = point(index, series[0][1][index])
        draw.line((x, plot[3], x, plot[3] + 8), fill="#30343b", width=2)
        draw.text((x, plot[3] + 16), str(index + 1), anchor="ma", fill="#30343b", font=font(23))
    for tick in range(int(y_min), int(y_max) + 1):
        value = 10**tick if log_y else tick
        _, y = point(0, value)
        draw.line((plot[0], y, plot[2], y), fill="#e4e7eb", width=2)
        label = f"{value:g}" if value < 1000 else f"{value:.0e}"
        draw.text((plot[0] - 15, y), label, anchor="rm", fill="#30343b", font=font(21))
    for name, data, color in series:
        points = [point(index, float(value)) for index, value in enumerate(data)]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)
    legend_x, legend_y = plot[0] + 20, plot[1] + 15
    for name, _, color in series:
        draw.line((legend_x, legend_y + 12, legend_x + 45, legend_y + 12), fill=color, width=6)
        draw.text((legend_x + 58, legend_y), name, fill="#30343b", font=font(22))
        legend_y += 34
    draw.text(((plot[0] + plot[2]) / 2, bottom - 40), "Number of centroids r", anchor="mm", fill="#30343b", font=font(25))
    draw.text((left + 25, (plot[1] + plot[3]) / 2), y_label, anchor="lm", fill="#30343b", font=font(22))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correction-scale", type=float, default=0.25)
    args = parser.parse_args()

    records = json.loads(args.input.read_text())
    record = records[args.record_index]
    # Use the path captured from the production robust-trimmed placement.
    # Re-fitting from the rounded Gram dump can change discrete merge ties and
    # must not be used to describe the deployed path.
    placement_risk = np.asarray(record["risk_path_r1_to_r8"])
    allocation_risk = np.asarray(record["allocation_risk_r1_to_r8"])
    labels = np.asarray(record["labels_path_r1_to_r8"], dtype=int)

    probability = np.asarray(
        json.loads(record["within_block_token_mass"]), dtype=np.float64
    )
    probability /= probability.sum()
    token_logit = np.log(np.maximum(probability, np.finfo(np.float64).tiny))
    peak = int(probability.argmax())
    tokens = [visible_token(token) for token in record["token_text"]]

    rows = []
    for r in range(1, 9):
        label = labels[r - 1]
        groups = partitions(label)
        raw_ratio = sum(
            len(group) * math.exp(float(token_logit[group].mean()))
            for group in groups
        )
        corrected_ratio = raw_ratio * math.exp(
            args.correction_scale * float(allocation_risk[r - 1])
        )
        peak_group = next(group for group in groups if peak in group)
        rows.append(
            {
                "r": r,
                "partition": " | ".join(
                    "{" + ",".join(map(str, group)) + "}" for group in groups
                ),
                "raw_mass_ratio": raw_ratio,
                "raw_log_error": math.log(raw_ratio),
                "corrected_mass_ratio": corrected_ratio,
                "corrected_log_error": math.log(corrected_ratio),
                "placement_risk": float(placement_risk[r - 1]),
                "tail_cvar_risk": float(allocation_risk[r - 1]),
                "peak_cluster_size": len(peak_group),
                "peak_cluster_true_mass": float(probability[peak_group].sum()),
            }
        )

    gain = np.maximum(allocation_risk[:-1] - allocation_risk[1:], 0)
    concave_gain = np.asarray(record["allocation_gain_r1_to_r7"])
    threshold = float(record["allocation_threshold"])
    allocated_r = int(record["components"])
    production_ratio = math.exp(float(record["signed_lse_error"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "partition_path.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    analysis = {
        "source": str(args.input),
        "record_index": args.record_index,
        "record": record,
        "current_objective": {
            "placement": "self-K robust-trimmed Jensen gap",
            "allocation": "top-25% tail-CVaR with concavified marginal gains",
            "correction_scale": args.correction_scale,
        },
        "peak_token_index": peak,
        "peak_token": record["token_text"][peak],
        "peak_probability": float(probability[peak]),
        "allocated_r": allocated_r,
        "allocation_threshold": threshold,
        "raw_marginal_gains": gain.tolist(),
        "concavified_marginal_gains": concave_gain.tolist(),
        "production_mass_ratio_at_allocated_r": production_ratio,
        "path": rows,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2)
    )

    # The figure is deliberately one concrete example.  This companion audit
    # checks whether its mechanism repeats across the other consequential
    # representation misses selected by diagnose_cwe_streaming.py.
    cohort = []
    for item in records:
        item_labels = np.asarray(item["labels_path_r1_to_r8"], dtype=int)
        item_probability = np.asarray(
            json.loads(item["within_block_token_mass"]), dtype=np.float64
        )
        item_probability /= item_probability.sum()
        item_logit = np.log(
            np.maximum(item_probability, np.finfo(np.float64).tiny)
        )
        item_peak = int(item_probability.argmax())
        item_ratios = []
        item_peak_sizes = []
        for stage in range(8):
            stage_labels = item_labels[stage]
            stage_groups = partitions(stage_labels)
            item_ratios.append(sum(
                len(group) * math.exp(float(item_logit[group].mean()))
                for group in stage_groups
            ))
            item_peak_sizes.append(int(np.sum(
                stage_labels == stage_labels[item_peak]
            )))

        def first_stage(values, predicate) -> int:
            return next(
                (index + 1 for index, value in enumerate(values) if predicate(value)),
                9,
            )

        item_allocated = int(item["components"])
        cohort.append(
            {
                "step": int(item["step"]),
                "layer": int(item["layer"]),
                "kv_head": int(item["kv_head"]),
                "block": int(item["block"]),
                "allocated_r": item_allocated,
                "r_for_50pct_raw_mass": first_stage(
                    item_ratios, lambda value: value >= 0.50
                ),
                "r_for_90pct_raw_mass": first_stage(
                    item_ratios, lambda value: value >= 0.90
                ),
                "r_to_isolate_live_peak": first_stage(
                    item_peak_sizes, lambda value: value == 1
                ),
                "raw_mass_ratio_at_allocated_r": item_ratios[
                    item_allocated - 1
                ],
                "live_peak_mass": float(item_probability[item_peak]),
                "live_top2_mass": float(np.sort(item_probability)[-2:].sum()),
                "exact_rank": int(item["exact_rank"]),
                "proxy_rank": int(item["proxy_rank"]),
            }
        )
    cohort_csv = args.output_dir / "cohort_records.csv"
    with cohort_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cohort[0]))
        writer.writeheader()
        writer.writerows(cohort)
    allocated = np.asarray([item["allocated_r"] for item in cohort])
    r50 = np.asarray([item["r_for_50pct_raw_mass"] for item in cohort])
    r90 = np.asarray([item["r_for_90pct_raw_mass"] for item in cohort])
    isolate = np.asarray([item["r_to_isolate_live_peak"] for item in cohort])
    allocated_ratio = np.asarray([
        item["raw_mass_ratio_at_allocated_r"] for item in cohort
    ])
    cohort_summary = {
        "selection": "top consequential representation misses; not all blocks",
        "n": len(cohort),
        "median_allocated_r": float(np.median(allocated)),
        "median_r_for_50pct_raw_mass": float(np.median(r50)),
        "median_r_for_90pct_raw_mass": float(np.median(r90)),
        "median_r_to_isolate_live_peak": float(np.median(isolate)),
        "fraction_allocated_below_r50": float(np.mean(allocated < r50)),
        "fraction_allocated_below_r90": float(np.mean(allocated < r90)),
        "fraction_needing_r8_for_90pct": float(np.mean(r90 == 8)),
        "fraction_needing_r8_to_isolate_peak": float(np.mean(isolate == 8)),
        "median_raw_mass_ratio_at_allocated_r": float(
            np.median(allocated_ratio)
        ),
        "fraction_below_10pct_raw_mass_at_allocated_r": float(
            np.mean(allocated_ratio <= 0.10)
        ),
    }
    (args.output_dir / "cohort_summary.json").write_text(
        json.dumps(cohort_summary, indent=2)
    )

    canvas = Image.new("RGB", (3200, 2400), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((1600, 45), "Qwen3-4B-Instruct-2507 · RULER CWE 32K · sample 14", anchor="ma", fill="#101820", font=font(48, True))
    draw.text(
        (1600, 105),
        f"step {record['step']} · layer {record['layer']} · KV head {record['kv_head']} · block {record['block']} · exact rank {record['exact_rank']} → proxy rank {record['proxy_rank']}",
        anchor="ma", fill="#34495e", font=font(29),
    )

    mass_box = (60, 170, 1550, 785)
    cosine_box = (1650, 170, 3140, 785)
    partition_box = (60, 825, 3140, 1535)
    recovery_box = (60, 1575, 1550, 2340)
    gain_box = (1650, 1575, 3140, 2340)
    panel(draw, mass_box, "True within-block attention")
    panel(draw, cosine_box, "Post-RoPE key cosine similarity")
    panel(draw, partition_box, "Current self-K nested partition path")
    panel(draw, recovery_box, "Dilution along the path")
    panel(draw, gain_box, "Why this block remains at r=1")

    # Attention bars.
    origin_x, origin_y = mass_box[0] + 100, mass_box[3] - 125
    chart_w, chart_h = mass_box[2] - mass_box[0] - 150, 430
    draw.line((origin_x, origin_y - chart_h, origin_x, origin_y), fill="#30343b", width=3)
    draw.line((origin_x, origin_y, origin_x + chart_w, origin_y), fill="#30343b", width=3)
    bar_w = chart_w / 8 * 0.68
    for index, value in enumerate(probability):
        x = origin_x + (index + 0.5) * chart_w / 8
        height = float(value) * chart_h
        color = "#e63946" if index == peak else "#2878b5"
        draw.rectangle((x - bar_w / 2, origin_y - height, x + bar_w / 2, origin_y), fill=color, outline="#20252b", width=2)
        draw.text((x, origin_y + 18), f"{index}: {tokens[index]}", anchor="ma", fill="#30343b", font=font(20))
        if value >= 5e-4:
            draw.text((x, origin_y - height - 12), f"{100 * value:.2f}%", anchor="ms", fill="#20252b", font=font(20, True))
    draw.text((mass_box[0] + 22, origin_y - chart_h / 2), "Mass (%)", anchor="lm", fill="#30343b", font=font(23))

    # Cosine heat map (blue negative, white zero, red positive).
    cosine = np.asarray(record["pairwise_cosine"], dtype=np.float64)
    cell = 57
    heat_x, heat_y = cosine_box[0] + 510, cosine_box[1] + 100
    for row in range(8):
        for column in range(8):
            value = float(cosine[row, column])
            if value >= 0:
                shade = int(255 - 150 * value)
                color = (255, shade, shade)
            else:
                shade = int(255 + 150 * value)
                color = (shade, shade, 255)
            box = (heat_x + column * cell, heat_y + row * cell, heat_x + (column + 1) * cell, heat_y + (row + 1) * cell)
            draw.rectangle(box, fill=color, outline="white", width=1)
            draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"{value:.2f}", anchor="mm", fill="#15191e", font=font(16))
    for index in range(8):
        draw.text((heat_x - 18, heat_y + (index + 0.5) * cell), str(index), anchor="rm", fill="#30343b", font=font(20))
        draw.text((heat_x + (index + 0.5) * cell, heat_y - 15), str(index), anchor="ms", fill="#30343b", font=font(20))
    draw.text((cosine_box[0] + 90, cosine_box[1] + 250), f"cos(k5, k6) = {cosine[5, 6]:.3f}\n\nThe target token and its\nfollowing delimiter look\nsimilar in key geometry,\nbut the live query assigns\nthem very different mass.", fill="#30343b", font=font(25), spacing=9)

    # Partition matrix.
    canonical = np.empty_like(labels)
    for row in range(8):
        for group in partitions(labels[row]):
            canonical[row, group] = min(group)
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
    matrix_x, matrix_y = partition_box[0] + 260, partition_box[1] + 100
    matrix_w, matrix_h = partition_box[2] - partition_box[0] - 330, 480
    cell_w, cell_h = matrix_w / 8, matrix_h / 8
    for row in range(8):
        draw.text((matrix_x - 25, matrix_y + (row + 0.5) * cell_h), f"r={row + 1}", anchor="rm", fill="#30343b", font=font(24, True if row + 1 == allocated_r else False))
        for column in range(8):
            value = int(canonical[row, column])
            box = (matrix_x + column * cell_w, matrix_y + row * cell_h, matrix_x + (column + 1) * cell_w, matrix_y + (row + 1) * cell_h)
            draw.rectangle(box, fill=palette[value], outline="white", width=3)
            draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"C{value}", anchor="mm", fill="white", font=font(24, True))
    for index, token in enumerate(tokens):
        draw.text((matrix_x + (index + 0.5) * cell_w, matrix_y + matrix_h + 22), f"{index}: {token}", anchor="ma", fill="#30343b", font=font(22))
    peak_x = matrix_x + peak * cell_w
    draw.rectangle((peak_x + 3, matrix_y + 3, peak_x + cell_w - 3, matrix_y + matrix_h - 3), outline="#ffe600", width=8)
    peak_isolated_r = next(
        r for r in range(1, 9)
        if rows[r - 1]["peak_cluster_size"] == 1
    )
    draw.text((partition_box[2] - 35, partition_box[3] - 35), f"Yellow outline: live-query peak. It is not isolated until r={peak_isolated_r}.", anchor="ra", fill="#5d4b00", font=font(25, True))

    # Recovery curve.
    r_axis = np.arange(1, 9)
    raw = 100 * np.asarray([row["raw_mass_ratio"] for row in rows])
    corrected = 100 * np.asarray([row["corrected_mass_ratio"] for row in rows])
    line_plot(
        draw,
        recovery_box,
        [("Raw centroid LSE", raw, "#2878b5"), ("After 0.25 correction", corrected, "#e07a1f")],
        log_y=True,
        y_label="Estimated / exact mass (%)",
    )
    draw.text((recovery_box[0] + 120, recovery_box[3] - 68), f"Production INT8 at allocated r={allocated_r}: {100 * production_ratio:.3f}%", fill="#d62728", font=font(24, True))

    # Allocation marginal gains.
    transition = np.arange(1, 8)
    chart = (gain_box[0] + 110, gain_box[1] + 105, gain_box[2] - 40, gain_box[3] - 125)
    y_max = max(float(gain.max()), threshold) * 1.18
    draw.line((chart[0], chart[1], chart[0], chart[3]), fill="#30343b", width=3)
    draw.line((chart[0], chart[3], chart[2], chart[3]), fill="#30343b", width=3)
    for tick in range(0, int(math.ceil(y_max)) + 1):
        y = chart[3] - tick / y_max * (chart[3] - chart[1])
        draw.line((chart[0], y, chart[2], y), fill="#e4e7eb", width=2)
        draw.text((chart[0] - 15, y), str(tick), anchor="rm", fill="#30343b", font=font(20))
    step_w = (chart[2] - chart[0]) / 7
    for index in range(7):
        center = chart[0] + (index + 0.5) * step_w
        raw_h = gain[index] / y_max * (chart[3] - chart[1])
        used_h = concave_gain[index] / y_max * (chart[3] - chart[1])
        draw.rectangle((center - 34, chart[3] - raw_h, center - 3, chart[3]), outline="#4c78a8", width=4)
        draw.rectangle((center + 3, chart[3] - used_h, center + 34, chart[3]), fill="#4c78a8")
        draw.text((center, chart[3] + 18), f"{index + 1}→{index + 2}", anchor="ma", fill="#30343b", font=font(20))
    threshold_y = chart[3] - threshold / y_max * (chart[3] - chart[1])
    draw.line((chart[0], threshold_y, chart[2], threshold_y), fill="#d62728", width=5)
    draw.text((chart[2] - 10, threshold_y - 12), f"global cutoff {threshold:.2f}", anchor="rs", fill="#d62728", font=font(23, True))
    draw.text((gain_box[0] + 115, gain_box[3] - 70), f"First gain {concave_gain[0]:.2f} < cutoff {threshold:.2f} ⇒ allocated r=1", fill="#8b1a1a", font=font(25, True))

    png = args.output_dir / "block_partition_dilution.png"
    pdf = args.output_dir / "block_partition_dilution.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)

    report = f"""# Block-level partition dilution audit

- Model/task: Qwen3-4B-Instruct-2507, RULER CWE 32K, sample 14.
- Location: decode step {record['step']}, layer {record['layer']}, KV head {record['kv_head']}, block {record['block']}.
- Retrieval failure: exact rank {record['exact_rank']}, production proxy rank {record['proxy_rank']}.
- The block contains `{record['source_words']}`; the lexical target token is at index 5 and the attention peak is index {peak} (`{visible_token(record['token_text'][peak])}`).
- True within-block mass: index 5 = {100 * probability[5]:.2f}%, index {peak} = {100 * probability[peak]:.2f}%, together = {100 * (probability[5] + probability[peak]):.2f}%.
- At r=1, raw centroid LSE retains only {100 * rows[0]['raw_mass_ratio']:.4f}% of the exact block mass. The correction raises this to {100 * rows[0]['corrected_mass_ratio']:.3f}%; the actual INT8 production score is {100 * production_ratio:.3f}%.
- The first center gain is {concave_gain[0]:.3f}, below the global {threshold:.3f} cutoff, so this exact-rank-1 block receives no upgrade under the mean-1.25 budget.
- At r=2 the path isolates the hot pair \{{5,6\}} from the cold tokens, but still merges the two hot positions. They contain {100 * (probability[5] + probability[6]):.2f}% of true mass, yet their centroid represents only {100 * rows[1]['raw_mass_ratio']:.2f}% of exact mass.
- From r=2 through r={peak_isolated_r - 1}, extra centers mostly split cold positions while indices 5 and 6 remain merged. The live peak is isolated only at r={peak_isolated_r}.

Across the 40 consequential representation misses selected by the audit (a deliberately failure-biased cohort), median allocated order is {cohort_summary['median_allocated_r']:.1f}, while median order needed for 50% and 90% raw mass recovery is {cohort_summary['median_r_for_50pct_raw_mass']:.1f} and {cohort_summary['median_r_for_90pct_raw_mass']:.1f}. The allocated order is below the 90%-recovery order in {100 * cohort_summary['fraction_allocated_below_r90']:.1f}% of these misses; {100 * cohort_summary['fraction_below_10pct_raw_mass_at_allocated_r']:.1f}% retain at most 10% raw mass at their allocated order.

## Immediate insight

The failure has two coupled parts. The global allocator under-prioritizes the block, but placement is also misordered: self-K spends later centers on directions that reduce its worst proxy risk while leaving the actual hot pair merged. More budget alone is therefore inefficient. A useful next objective must predict **which within-block contrast may be activated** and value separating that contrast before spending centers on cold directions.
"""
    (args.output_dir / "README.md").write_text(report)


if __name__ == "__main__":
    main()
