#!/usr/bin/env python3
"""Analyze RULER CWE outputs across a completed multi-method campaign.

The result JSONL stores one cumulative score vector per generated example.  This
script recovers the per-example scores, parses predicted source-list words, and
separates missing correct words from duplicate or off-target list entries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b")
SOURCE_RE = re.compile(r"(?:^|\s)\d+\.\s+([^\s]+)")


def method_name(stem: str) -> tuple[str, int | None]:
    match = re.search(r"_b(512|1024|2048)_", stem)
    budget = int(match.group(1)) if match else None
    if "_full" in stem:
        return "Full attention", budget
    if "_exact_block_lse_" in stem:
        return "Exact block-LSE", budget
    if "_exact_block_max_" in stem:
        return "Exact block-max", budget
    if "_adaptive_lse_" in stem:
        return "Ours", budget
    if "_quest_" in stem:
        return "Quest", budget
    if "_pariskv_" in stem:
        return "ParisKV", budget
    if "_retroinfer_" in stem:
        return "RetroInfer", budget
    raise ValueError(f"unknown method filename: {stem}")


def load_dataset(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_words(text: str) -> list[str]:
    before_question = text.rsplit("Question:", 1)[0]
    words = []
    for raw in SOURCE_RE.findall(before_question):
        clean = raw.strip(".,;:!?()[]{}\"'").lower()
        if clean:
            words.append(clean)
    return words


def predicted_source_words(text: str, vocabulary: set[str]) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if token.lower() in vocabulary]


def read_cell(path: Path) -> list[dict]:
    rows = []
    previous = 0
    with path.open() as handle:
        for sample, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            cumulative = row["correct"]
            if len(cumulative) <= previous:
                raise ValueError(f"non-growing cumulative score vector in {path}")
            score = float(cumulative[-1])
            prediction = row.get("prediction", [""])
            if isinstance(prediction, list):
                prediction = prediction[0] if prediction else ""
            rows.append({"sample": sample, "score": score, "prediction": str(prediction)})
            previous = len(cumulative)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True, help="flat directory tree of CWE JSONL cells")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-prefix", default="qwen3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cells.rglob(f"{args.model_prefix}_*_cwe_*.jsonl"))
    # Copies from different machines may overlap. Prefer one canonical path per basename.
    unique_files = {}
    for path in files:
        unique_files.setdefault(path.name, path)

    detail = []
    vectors: dict[tuple[str, int | None], dict[int, float]] = {}
    for path in unique_files.values():
        method, budget = method_name(path.stem)
        key = (method, budget)
        records = read_cell(path)
        if len(records) != len(dataset):
            raise ValueError(f"{path}: {len(records)} rows, dataset has {len(dataset)}")
        vectors[key] = {record["sample"]: record["score"] for record in records}
        for record, data_row in zip(records, dataset):
            source = source_words(data_row["input"])
            counts = Counter(source)
            vocabulary = set(counts)
            predicted = predicted_source_words(record["prediction"], vocabulary)
            unique = set(predicted)
            ground_truth = {word.lower() for word in data_row["outputs"]}
            correct_unique = ground_truth & unique
            wrong_unique = unique - ground_truth
            wrong_ranks = []
            ranked = {word: rank for rank, (word, _) in enumerate(counts.most_common(), 1)}
            for word in wrong_unique:
                wrong_ranks.append(ranked[word])
            detail.append(
                {
                    "model": args.model_prefix,
                    "method": method,
                    "budget": budget if budget is not None else "all",
                    "sample": record["sample"],
                    "score": record["score"],
                    "predicted_source_items": len(predicted),
                    "unique_source_items": len(unique),
                    "duplicate_source_items": len(predicted) - len(unique),
                    "correct_unique_items": len(correct_unique),
                    "wrong_unique_items": len(wrong_unique),
                    "best_wrong_frequency_rank": min(wrong_ranks) if wrong_ranks else "",
                    "ground_truth": " ".join(data_row["outputs"]),
                    "prediction": record["prediction"],
                }
            )

    summary = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in detail:
        grouped.setdefault((row["method"], str(row["budget"])), []).append(row)
    for (method, budget), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(rows)
        summary.append(
            {
                "method": method,
                "budget": budget,
                "samples": n,
                "accuracy_pct": 100 * sum(row["score"] for row in rows) / n,
                "mean_predicted_source_items": sum(row["predicted_source_items"] for row in rows) / n,
                "mean_unique_source_items": sum(row["unique_source_items"] for row in rows) / n,
                "mean_duplicate_source_items": sum(row["duplicate_source_items"] for row in rows) / n,
                "mean_correct_unique_items": sum(row["correct_unique_items"] for row in rows) / n,
                "mean_wrong_unique_items": sum(row["wrong_unique_items"] for row in rows) / n,
            }
        )

    # Hard examples expose representation loss: exact block routing succeeds while
    # ours fails. ParisKV provides a token-granularity reference at equal budget.
    hard = []
    for budget in (512, 1024, 2048):
        ours = vectors.get(("Ours", budget), {})
        exact = vectors.get(("Exact block-LSE", budget), {})
        paris = vectors.get(("ParisKV", budget), {})
        full = vectors.get(("Full attention", None), {})
        for sample in sorted(set(ours) & set(exact)):
            row = {
                "budget": budget,
                "sample": sample,
                "ours": ours[sample],
                "exact_block_lse": exact[sample],
                "proxy_gap": exact[sample] - ours[sample],
                "pariskv": paris.get(sample, ""),
                "full": full.get(sample, ""),
            }
            hard.append(row)
    hard.sort(key=lambda row: (-row["proxy_gap"], row["budget"], row["sample"]))

    detail_fields = list(detail[0]) if detail else []
    summary_fields = list(summary[0]) if summary else []
    hard_fields = list(hard[0]) if hard else []
    write_csv(args.output / "cwe_prediction_detail.csv", detail, detail_fields)
    write_csv(args.output / "cwe_prediction_summary.csv", summary, summary_fields)
    write_csv(args.output / "cwe_hard_samples.csv", hard, hard_fields)

    print(f"wrote {len(detail)} per-sample rows from {len(unique_files)} cells")
    print(f"output={args.output}")
    for row in summary:
        print(
            f"{row['method']:<18} B={row['budget']:<4} "
            f"acc={row['accuracy_pct']:5.2f} unique={row['mean_unique_source_items']:.2f} "
            f"dup={row['mean_duplicate_source_items']:.2f} correct={row['mean_correct_unique_items']:.2f}"
        )
    print("hardest exact-vs-proxy gaps:")
    for row in hard[:12]:
        print(row)


if __name__ == "__main__":
    main()
