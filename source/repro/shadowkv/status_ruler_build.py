#!/usr/bin/env python3
"""Report the two-model RULER data build without touching a GPU."""

import argparse
import os
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get(
            "SHADOWKV_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ShadowKV")
        ) + "/data/ruler/data",
    )
    parser.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root)
    lengths = [int(item) for item in args.lengths.split(",") if item]
    total_rows = len(TASKS) * len(lengths) * 2 * args.samples
    seen_rows = 0
    done_cells = 0
    total_cells = len(TASKS) * len(lengths) * 2

    print(f"RULER data root: {root}")
    for model_dir in ("qwen", "llama-3"):
        print(f"\n[{model_dir}]")
        for length in lengths:
            counts = [
                line_count(root / model_dir / str(length) / task / "validation.jsonl")
                for task in TASKS
            ]
            seen_rows += sum(min(count, args.samples) for count in counts)
            complete = sum(count == args.samples for count in counts)
            done_cells += complete
            partial = [(task, count) for task, count in zip(TASKS, counts)
                       if count and count != args.samples]
            state = "DONE" if complete == len(TASKS) else "BUILDING"
            print(
                f"  {length:>6}: {complete:>2}/{len(TASKS)} tasks, "
                f"rows={sum(counts):>5}/{len(TASKS) * args.samples:<5} {state}"
            )
            for task, count in partial:
                print(f"          partial {task}: {count}/{args.samples}")

    print(
        f"\nTOTAL: {done_cells}/{total_cells} task-length-model cells complete; "
        f"{seen_rows}/{total_rows} rows ({100.0 * seen_rows / total_rows:.1f}%)"
    )


if __name__ == "__main__":
    main()
