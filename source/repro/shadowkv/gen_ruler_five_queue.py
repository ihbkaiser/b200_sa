#!/usr/bin/env python
"""Generate the five-method RULER matrix at 32K / 64K / 128K.

The five methods are the ones that now run their own fastest variant on the
shared frame (PROJECT.md 2e-bis).  Every cell uses the preregistered matched
budget L/32 and, through UPSTREAM_MATCHED_EXACT_REGIONS=1, the same exact sink
and local window.

Cells are emitted length-major and round-robin over task and method, so a pool
that is stopped early still leaves a readable table rather than three finished
methods and two empty ones.

The shards are weighted by context length, because a 128K cell costs roughly
four 32K cells, and a machine's share follows its GPU count.  RetroInfer is
pinned to the machine named by --retroinfer-host: its author kernels
(retroinfer_kernels, weighted_flash_decoding) are built there and nowhere else.
"""

from __future__ import annotations

import argparse

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]
METHODS = [
    "adaptive_centroid_lse_streaming_prefix4_querymean",
    "quest_streaming",
    "pariskv_author_common",
    "retroinfer_author_common",
    "shadowkv_cpu",
]
# ShadowKV's landmark chunk is 4; every other method blocks at 8.
CHUNK = {"shadowkv_cpu": 4}


def cost(length: int) -> int:
    return max(1, length // 32768)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument(
        "--shards", default="m1:7,m2:4,m4:2",
        help="host:gpu_count pairs; the share of work follows the GPU count",
    )
    parser.add_argument("--retroinfer-host", default="m1")
    parser.add_argument("--out-prefix", default="queue")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    tasks = [x for x in args.tasks.split(",") if x]
    shards = []
    for pair in args.shards.split(","):
        host, _, gpus = pair.partition(":")
        shards.append([host, int(gpus), 0, []])  # host, gpus, weight, rows

    rows = []
    for length in lengths:
        budget = length // 32
        for task in tasks:
            for method in methods:
                chunk = CHUNK.get(method, 8)
                rows.append((
                    method, cost(length),
                    f"{args.model} {length} {task} {method} "
                    f"{budget} 160 {chunk} 8 0",
                ))

    total_gpus = sum(shard[1] for shard in shards)
    total_weight = sum(row[1] for row in rows)
    target = {shard[0]: total_weight * shard[1] / total_gpus for shard in shards}
    pinned = {shard[0]: shard for shard in shards}[args.retroinfer_host]

    for method, weight, line in rows:
        if method == "retroinfer_author_common":
            shard = pinned
        else:
            # Whichever machine is furthest below its share takes the next cell.
            shard = min(shards, key=lambda s: s[2] - target[s[0]])
        shard[2] += weight
        shard[3].append(line)

    for host, gpus, weight, lines in shards:
        # Emit each shard round-robin over the context lengths.  The assignment
        # above is weight-balanced, which happens to hand one machine every 32K
        # cell and another every 128K one; left in that order a machine spends
        # hours on a single length and no length is comparable across methods
        # until its owner finishes.  Interleaving costs nothing and makes every
        # partial table readable.
        by_length: dict[int, list[str]] = {}
        for line in lines:
            by_length.setdefault(int(line.split()[1]), []).append(line)
        lines = []
        while any(by_length.values()):
            for length in sorted(by_length):
                if by_length[length]:
                    lines.append(by_length[length].pop(0))
        path = f"{args.out_prefix}_{host}.txt"
        header = (
            f"# {len(lines)} cells | weight {weight} of {total_weight} "
            f"| {gpus} GPUs | matched budget L/32 | model {args.model}\n"
        )
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(header)
            stream.write("\n".join(lines) + "\n")
        print(f"{path}: {len(lines)} cells, weight {weight} "
              f"(share {target[host]:.0f})")


if __name__ == "__main__":
    main()
