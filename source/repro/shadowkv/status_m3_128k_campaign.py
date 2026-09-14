#!/usr/bin/env python3
"""Show machine-3 128K pool progress and honest partial RULER scores."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
TASK_LABELS = {
    "niah_single_1": "S1", "niah_single_2": "S2", "niah_single_3": "S3",
    "niah_multikey_1": "MK1", "niah_multikey_2": "MK2", "niah_multikey_3": "MK3",
    "niah_multivalue": "MV", "niah_multiquery": "MQ", "vt": "VT",
    "cwe": "CWE", "fwe": "FWE", "qa_1": "QA1", "qa_2": "QA2",
}
METHODS = ("Full", "Ours", "Quest", "ShadowKV", "ParisKV")


def marker_names(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir()} if path.is_dir() else set()


def identify(path: Path) -> tuple[str, str, str, int | None] | None:
    model = path.parent.name
    prefix = f"{model}_131072_"
    if not path.stem.startswith(prefix):
        return None
    remainder = path.stem[len(prefix):]
    task = next((task for task in sorted(TASKS, key=len, reverse=True)
                 if remainder.startswith(task + "_")), None)
    if task is None:
        return None
    suffix = remainder[len(task) + 1:]
    if suffix == "full":
        method = "Full"
        budget = None
    elif suffix.startswith("adaptive_lse_stream"):
        method = "Ours"
    elif suffix.startswith("quest_stream"):
        method = "Quest"
    elif suffix.startswith("shadowkv_cpu"):
        method = "ShadowKV"
    elif suffix.startswith("pariskv_author_common"):
        method = "ParisKV"
    else:
        return None
    if method != "Full":
        match = re.search(r"(?:^|_)b(\d+)(?:_|$)", suffix)
        if match is None:
            return None
        budget = int(match.group(1))
    return model, task, method, budget


def read_score(path: Path) -> tuple[int, float] | None:
    # eval_acc writes one record per completed sample, but avg_score in each
    # record is the running aggregate through that sample.  The cell score is
    # therefore the LAST valid record, never the mean of all records.
    count = 0
    last_score: float | None = None
    try:
        with path.open() as handle:
            for line in handle:
                try:
                    score = json.loads(line).get("avg_score")
                except (json.JSONDecodeError, AttributeError):
                    continue
                count += 1
                if isinstance(score, list):
                    if score:
                        last_score = sum(float(value) for value in score) / len(score)
                elif score is not None:
                    last_score = float(score)
    except OSError:
        return None
    if last_score is None:
        return None
    return count, 100.0 * last_score


def format_cell(value: tuple[int, float] | None) -> str:
    if value is None:
        return "NaN"
    count, score = value
    return f"{score:.2f}" if count >= 100 else f"{score:.2f}*{count}"


def print_table(
    model: str,
    budget: int,
    scores: dict[tuple[str, str, str, int | None], tuple[int, float]],
) -> None:
    headers = [TASK_LABELS[task] for task in TASKS]
    rows: list[list[str]] = []
    for method in METHODS:
        values = [
            scores.get((model, task, method, None if method == "Full" else budget))
            for task in TASKS
        ]
        available = [value[1] for value in values if value is not None]
        complete = all(value is not None and value[0] >= 100 for value in values)
        mean = statistics.mean(available) if available else None
        mean_text = "NaN" if mean is None else f"{mean:.2f}{'' if complete else '*'}"
        rows.append([method, *(format_cell(value) for value in values), mean_text])

    widths = [max(len(title), *(len(row[index]) for row in rows))
              for index, title in enumerate(["Method", *headers, "Mean"])]
    print(f"\n=== {model} / 128K / B{budget} ===")
    print("  ".join(title.rjust(widths[index]) for index, title in
                    enumerate(["Method", *headers, "Mean"])))
    for row in rows:
        print("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    queue = root / "queue.txt"
    state = root / ".state"

    queue_rows = []
    if queue.is_file():
        queue_rows = [line for line in queue.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
    expected = len(queue_rows) or 234
    done = marker_names(state / "done")
    running = marker_names(state / "running")
    failed = marker_names(state / "failed")
    claimed = done | running | failed
    unclaimed = max(0, expected - len(claimed))

    print(time.strftime("updated=%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"cells: done={len(done)}/{expected} running={len(running)} "
          f"failed={len(failed)} unclaimed={unclaimed}")
    lock = state / "pool.lock"
    if done and lock.exists():
        elapsed = max(1.0, time.time() - lock.stat().st_mtime)
        remaining = max(0, expected - len(done) - len(failed))
        eta_hours = remaining * elapsed / len(done) / 3600
        print(f"elapsed={elapsed / 3600:.2f}h rough_eta={eta_hours:.2f}h")

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        print("gpu processes:\n" + (gpu or "  none"))
    except (OSError, subprocess.CalledProcessError):
        print("gpu processes: unavailable")

    if running:
        print("running cells:")
        for name in sorted(running):
            print("  " + name)
    if failed:
        print("FAILED cells:")
        for name in sorted(failed):
            print("  " + name)

    scores: dict[tuple[str, str, str, int | None], tuple[int, float]] = {}
    for path in (root / "cells").glob("*/*.jsonl"):
        identity = identify(path)
        value = read_score(path)
        if identity is not None and value is not None:
            scores[identity] = value
    for model in ("llama32", "qwen3"):
        for budget in (4096, 8192):
            print_table(model, budget, scores)
    print("\n* = partial or mean over currently available cells; *N means N/100 samples.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
