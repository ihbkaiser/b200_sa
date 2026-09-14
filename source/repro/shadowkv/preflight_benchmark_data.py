#!/usr/bin/env python3
"""CPU-only readiness check for all ShadowKV benchmark adapters."""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
LONG_BENCH_TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
)
REQUIRED_COMMON = {"context", "question", "answer_prefix", "max_new_tokens"}


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--skip-ruler", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    shadow = Path(os.environ.get("SHADOWKV_DIR", repo / "ShadowKV"))
    sys.path.insert(0, str(shadow))
    from data.dataset import Dataset

    models = {
        "qwen": os.environ["SHADOWKV_QWEN3_PATH"],
        "llama-3": os.environ["SHADOWKV_LLAMA32_PATH"],
    }
    lengths = (4096, 8192, 16384, 32768, 65536, 131072)
    failures = []

    if not args.skip_ruler:
        ruler_root = shadow / "data/ruler/data"
        for model in models:
            for length in lengths:
                for task in RULER_TASKS:
                    path = ruler_root / model / str(length) / task / "validation.jsonl"
                    count = count_lines(path)
                    if count != args.samples:
                        failures.append(f"RULER {model}/{length}/{task}: {count}/{args.samples}")
        print(f"RULER: {2 * len(lengths) * len(RULER_TASKS) - len(failures)}/"
              f"{2 * len(lengths) * len(RULER_TASKS)} files exact")

    lb_path = os.environ["SHADOWKV_LONGBENCH_PATH"]
    for task in LONG_BENCH_TASKS:
        rows = load_dataset(lb_path, data_dir=task, split="test")
        missing = (REQUIRED_COMMON | {"answers", "all_classes", "task"}) - set(rows.column_names)
        if missing:
            failures.append(f"LongBench/{task}: missing {sorted(missing)}")
    print(f"LongBench: {len(LONG_BENCH_TASKS)} tasks readable")

    lbv2 = load_dataset(os.environ["SHADOWKV_LONGBENCH_V2_PATH"], split="test")
    missing = (
        REQUIRED_COMMON
        | {"answer", "difficulty", "length", "domain", "sub_domain", "_id"}
    ) - set(lbv2.column_names)
    if len(lbv2) != 503 or missing or set(lbv2["max_new_tokens"]) != {16}:
        failures.append(
            f"LongBench-v2: rows={len(lbv2)}, missing={sorted(missing)}, "
            f"generation={sorted(set(lbv2['max_new_tokens']))}"
        )
    print(f"LongBench-v2: {len(lbv2)} rows readable")

    aime = load_dataset(os.environ["SHADOWKV_AIME25_PATH"], split="test")
    missing = (REQUIRED_COMMON | {"answer"}) - set(aime.column_names)
    if len(aime) != 30 or missing or set(aime["max_new_tokens"]) != {32000}:
        failures.append(
            f"AIME-25: rows={len(aime)}, missing={sorted(missing)}, "
            f"generation={sorted(set(aime['max_new_tokens']))}"
        )
    print(f"AIME-25: {len(aime)} rows readable")

    math500 = load_dataset(os.environ["SHADOWKV_MATH500_PATH"], split="test")
    missing = (
        REQUIRED_COMMON | {"answer", "subject", "level", "unique_id"}
    ) - set(math500.column_names)
    if (
        len(math500) != 500
        or missing
        or set(math500["max_new_tokens"]) != {4096}
    ):
        failures.append(
            f"MATH-500: rows={len(math500)}, missing={sorted(missing)}, "
            f"generation={sorted(set(math500['max_new_tokens']))}"
        )
    print(f"MATH-500: {len(math500)} rows readable")

    gpqa = load_dataset(os.environ["SHADOWKV_GPQA_PATH"], data_dir="diamond", split="test")
    missing = (REQUIRED_COMMON | {"answer"}) - set(gpqa.column_names)
    if len(gpqa) != 198 or missing or set(gpqa["max_new_tokens"]) != {16384}:
        failures.append(
            f"GPQA/diamond: rows={len(gpqa)}, missing={sorted(missing)}, "
            f"generation={sorted(set(gpqa['max_new_tokens']))}"
        )
    print(f"GPQA/diamond: {len(gpqa)} rows readable")

    # Exercise the real adapter, tokenizer/chat template and official scorer
    # imports once for both architectures. This remains strictly CPU-only.
    smoke_names = (
        "longbench/narrativeqa", "longbench-v2", "aime25", "math500",
        "gpqa/diamond",
    )
    for model_name, model_path in models.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        for dataset_name in smoke_names:
            dataset = Dataset(dataset_name, tokenizer, 32768, 1)
            if dataset.tokenized_prompts[0].shape[1] <= 0:
                failures.append(f"adapter produced empty prompt: {model_name}/{dataset_name}")
        print(f"adapter: {model_name} tokenization/scorers OK")

    if failures:
        print(f"\nNOT READY: {len(failures)} failure(s)")
        for failure in failures[:30]:
            print(f"  FAIL {failure}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        raise SystemExit(1)
    print("\nREADY: all benchmark data and CPU adapters passed")


if __name__ == "__main__":
    main()
