#!/usr/bin/env python
"""Merge distributed upstream-baseline cells into strict Markdown tables.

Only the final cumulative checkpoint of a cell is read.  By default a cell is
reported only when it has exactly ``--expected-samples`` scores; partial cells
are listed separately and never enter a headline mean.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path


METHODS = (
    "pqcache_author_common",
    "magicpig_author_common",
    "infllm_author_common",
)
MODELS = ("llama32", "qwen3")
LENGTHS = (32768, 65536)
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
)
LABEL = {
    "pqcache_author_common": "PQCache",
    "magicpig_author_common": "MagicPIG",
    "infllm_author_common": "InfLLM",
    "pqcache_author_native": "PQCache native",
}
AUTHOR_FIELD = {
    "pqcache_author_common": "pqcache_author",
    "magicpig_author_common": "magicpig_author",
    "infllm_author_common": "infllm_author",
}
AUTHOR_PIN = {
    "pqcache_author_common": "0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828",
    "magicpig_author_common": "ac9aa36c866330ca6ad2ce342a7848d7df6f49bb",
    "infllm_author_common": "12b70798f56e56ebb23c53c7018091a3f540a028",
}
ADAPTER_FILE = {
    "pqcache_author_common": "ShadowKV/models/pqcache_author_cache.py",
    "magicpig_author_common": "ShadowKV/models/magicpig_author_cache.py",
    "infllm_author_common": "ShadowKV/models/infllm_author_cache.py",
}
REPO = Path(__file__).resolve().parents[2]


class _NumericAst(ast.NodeTransformer):
    """Discard provenance-neutral syntax and recorded config defaults.

    Campaign rows record and validate the effective seed in both the cell key
    and stamp.  A later commit changed MagicPIG's *fallback* from 0 to the
    released seed 43, while the already-running workers had explicitly used
    43.  Treating that unused fallback literal as a new numerical adapter would
    incorrectly split otherwise identical rows.
    """

    RECORDED_ENV_DEFAULTS = {
        "MAGICPIG_SEED": "<recorded-magicpig-seed>",
        "PQCACHE_SEED": "<recorded-pqcache-seed>",
    }

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Call(self, node):
        node = self.generic_visit(node)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in self.RECORDED_ENV_DEFAULTS
        ):
            node.args[1] = ast.Constant(
                self.RECORDED_ENV_DEFAULTS[node.args[0].value]
            )
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list) and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node


def numeric_ast_hash(revision, path):
    try:
        source = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{revision}:{path}"]
        )
        tree = _NumericAst().visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    except (OSError, subprocess.CalledProcessError, SyntaxError, TypeError):
        return None


def last_json(path: Path):
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def task_from_stamp(stamp: dict) -> str:
    cell = stamp["cell"]
    prefix = f"{cell.split('_', 1)[0]}_{int(stamp['datalen'])}_"
    marker = f"_{stamp['method']}_"
    if not cell.startswith(prefix) or marker not in cell:
        raise ValueError(f"cannot recover task from {cell}")
    return cell[len(prefix):].split(marker, 1)[0]


def load_cells(roots, expected, require_matched=False):
    complete, partial, duplicates = {}, {}, []
    for root in roots:
        for stamp_path in sorted((root / "cells").glob("*/*.stamp.json")):
            result = None
            try:
                stamp = json.loads(stamp_path.read_text())
                if stamp.get("method") not in METHODS:
                    continue
                if require_matched and stamp.get(
                    "upstream_matched_exact_regions"
                ) is not True:
                    continue
                result = stamp_path.with_name(
                    stamp_path.name.removesuffix(".stamp.json") + ".jsonl"
                )
                row = last_json(result)
                scores = row.get("correct") if row else None
                if not isinstance(scores, list) or not scores:
                    continue
                key = (
                    stamp["cell"].split("_", 1)[0], int(stamp["datalen"]),
                    task_from_stamp(stamp), stamp["method"],
                )
                traffic_path = result.with_name(result.stem + ".traffic.jsonl")
                record = {
                    "score": sum(map(float, scores)) / len(scores),
                    "count": len(scores),
                    "source": str(result),
                    "mtime": result.stat().st_mtime,
                    "stamp": stamp,
                    "traffic": (
                        last_json(traffic_path) if traffic_path.is_file() else {}
                    ),
                    "numeric_hash": numeric_ast_hash(
                        stamp.get("git_sha"), ADAPTER_FILE[stamp["method"]]
                    ),
                }
            except Exception as error:
                # A worker writes its stamp before its first result checkpoint.
                # That is an ordinary running cell, not a malformed result.
                if result is not None and not result.is_file():
                    continue
                print(f"warning: skipped {stamp_path}: {error}", file=sys.stderr)
                continue
            target = complete if len(scores) == expected else partial
            if key in target:
                duplicates.append((key, target[key]["source"], str(result)))
                # Prefer the deeper checkpoint; ties prefer the newer file.
                old = target[key]
                if (record["count"], record["mtime"]) <= (
                    old["count"], old.get("mtime", 0.0)
                ):
                    continue
            target[key] = record
    return complete, partial, duplicates


def export_summary(complete, partial, duplicates, expected_samples):
    """Emit a compact control-plane artifact; raw generations stay remote."""

    def encode(cells):
        encoded = []
        for key, record in sorted(cells.items()):
            encoded.append({
                "key": list(key),
                "score": record["score"],
                "count": record["count"],
                "source": record["source"],
                "stamp": record["stamp"],
                "traffic": record["traffic"],
                "numeric_hash": record.get("numeric_hash"),
            })
        return encoded

    return {
        "schema": "shadowkv-upstream-summary-v1",
        "expected_samples": expected_samples,
        "complete": encode(complete),
        "partial": encode(partial),
        "duplicate_inputs": len(duplicates),
    }


def load_summaries(paths, expected, require_matched=False):
    complete, partial, duplicates = {}, {}, []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "shadowkv-upstream-summary-v1":
            raise ValueError(f"unsupported summary schema in {path}")
        if int(payload.get("expected_samples", -1)) != expected:
            raise ValueError(f"sample-count mismatch in {path}")
        for kind, target in (("complete", complete), ("partial", partial)):
            for row in payload.get(kind, []):
                key = tuple(row["key"])
                key = (key[0], int(key[1]), key[2], key[3])
                stamp = row.get("stamp") or {}
                if require_matched and stamp.get(
                    "upstream_matched_exact_regions"
                ) is not True:
                    continue
                record = {
                    "score": float(row["score"]),
                    "count": int(row["count"]),
                    "source": f"{path}:{row.get('source', key)}",
                    "mtime": 0.0,
                    "stamp": stamp,
                    "traffic": row.get("traffic") or {},
                    "numeric_hash": row.get("numeric_hash"),
                }
                if key in target:
                    duplicates.append((key, target[key]["source"], record["source"]))
                    if record["count"] <= target[key]["count"]:
                        continue
                target[key] = record
        duplicates.extend(
            [(('summary', str(path), i), str(path), str(path))
             for i in range(int(payload.get("duplicate_inputs", 0)))]
        )
    return complete, partial, duplicates


def merge_cell_maps(primary, incoming, duplicates):
    for key, record in incoming.items():
        if key in primary:
            duplicates.append((key, primary[key]["source"], record["source"]))
            if record["count"] <= primary[key]["count"]:
                continue
        primary[key] = record


def accuracy_tables(cells):
    for length in sorted({key[1] for key in cells}):
        print(f"\n## RULER {length // 1024}K (complete cells only)\n")
        print("| Model | Task | " + " | ".join(LABEL[m] for m in METHODS) + " |")
        print("|---|---|" + "---:|" * len(METHODS))
        models = sorted({key[0] for key in cells if key[1] == length})
        for model in models:
            tasks = sorted({key[2] for key in cells if key[:2] == (model, length)})
            for task in tasks:
                values = []
                for method in METHODS:
                    record = cells.get((model, length, task, method))
                    values.append(
                        "—" if record is None else f"{100*record['score']:.2f}"
                    )
                print(f"| {model} | {task} | " + " | ".join(values) + " |")
            means = []
            for method in METHODS:
                vals = [cells[(model, length, task, method)]["score"] for task in tasks
                        if (model, length, task, method) in cells]
                means.append("—" if len(vals) != len(tasks) else f"{100*sum(vals)/len(vals):.2f}")
            print(f"| **{model}** | **macro ({len(tasks)} tasks present)** | "
                  + " | ".join(means) + " |")


def traffic_tables(cells):
    """Report the active-token bill attached to complete accuracy cells.

    PQCache and InfLLM expose a deterministic cap.  MagicPIG is a sampler, so
    its comparable quantity is the realized mean and its observed maximum.
    Keeping the two meanings explicit prevents a nominal L/32 label from
    silently hiding method-specific mandatory or stochastic overhead.
    """
    grouped = defaultdict(list)
    for (model, length, _task, method), record in cells.items():
        row = record.get("traffic")
        if row:
            grouped[(model, length, method)].append(row)
    if not grouped:
        return

    print("\n## Active-token accounting (complete cells only)\n")
    print("| Model | Length | Method | Cells | Nominal B | Bill type | Mean/cap active | Max observed | vs B |")
    print("|---|---:|---|---:|---:|---|---:|---:|---:|")
    for (model, length, method), rows in sorted(grouped.items()):
        nominal = length // 32
        maximum = None
        if method == "magicpig_author_common":
            active = sum(r["magicpig_mean_active_tokens"] for r in rows) / len(rows)
            maximum = max(
                r["magicpig_max_sampled_remote_tokens"]
                + r["magicpig_local_sink_tokens"] for r in rows
            )
            kind = "realized mean"
        elif method == "pqcache_author_common":
            active = sum(r["pqcache_active_token_cap"] for r in rows) / len(rows)
            maximum = active
            kind = "hard cap"
        elif method == "infllm_author_common":
            active = sum(r["infllm_active_token_cap"] for r in rows) / len(rows)
            maximum = active
            kind = "hard cap"
        else:
            continue
        delta = 100.0 * (active / nominal - 1.0)
        print(
            f"| {model} | {length // 1024}K | {LABEL[method]} | {len(rows)} "
            f"| {nominal} | {kind} | {active:.1f} | {maximum:.0f} "
            f"| {delta:+.1f}% |"
        )


def provenance_table(cells):
    grouped = defaultdict(list)
    for (_model, _length, _task, method), record in cells.items():
        row = dict(record.get("stamp") or {})
        row["_traffic"] = record.get("traffic") or {}
        row["_numeric_hash"] = record.get("numeric_hash")
        grouped[method].append(row)
    if not grouped:
        return False
    print("\n## Provenance audit (complete cells only)\n")
    print("| Method | Cells | Author commit(s) | Pin | Config | Repo revisions | Numeric variants | Dirty stamps | Backing store(s) |")
    print("|---|---:|---|---|---|---:|---:|---:|---|")
    all_provenance_ok = True
    for method in METHODS:
        rows = grouped.get(method, [])
        if not rows:
            continue
        author_commits = {
            (row.get(AUTHOR_FIELD[method]) or {}).get("commit") for row in rows
        }
        author_commits.discard(None)
        revisions = {row.get("git_sha") for row in rows if row.get("git_sha")}
        numeric_hashes = {row.get("_numeric_hash") for row in rows}
        numeric_hashes.update(
            numeric_ast_hash(revision, ADAPTER_FILE[method])
            for revision in revisions if not numeric_hashes - {None}
        )
        numeric_hashes.discard(None)
        dirty = sum(bool(row.get("git_dirty")) for row in rows)
        backings = {row.get("backing_store") for row in rows if row.get("backing_store")}
        pin_ok = author_commits == {AUTHOR_PIN[method]}
        expected_backing = {
            "pqcache_author_common": "exact_postrope_cpu_pinned",
            "magicpig_author_common": "author_native_cpu_pinned",
            "infllm_author_common": "author_native_cpu_pinned",
        }[method]

        def expected_cell_suffix(row):
            budget = int(row.get("datalen", 0)) // 32
            matched = row.get("upstream_matched_exact_regions") is True
            if method == "pqcache_author_common":
                if matched:
                    return (
                        f"pqcache_author_common_routeb{budget}_pq2x6_i10_"
                        "x32_l32_s4321_exactkv_matched"
                    )
                return (
                    f"pqcache_author_common_b{budget}_pq2x6_i10_x32_"
                    "recent0.5_s4321_exactkv"
                )
            if method == "magicpig_author_common":
                if matched:
                    return (
                        f"magicpig_author_common_targetb{budget}_k10l210_"
                        "x32_l32_d0_s43_matched"
                    )
                return (
                    f"magicpig_author_common_targetb{budget}_k10l210_"
                    "x4_l64_d0_s43"
                )
            if matched:
                return (
                    f"infllm_author_common_routeb{budget}_x32_l32_"
                    "blk128_repr4_matched"
                )
            return f"infllm_author_common_b{budget}_blk128_repr4_native21"

        config_ok = all(
            int(row.get("sparse_budget", -1)) == int(row.get("datalen", 0)) // 32
            and str(row.get("cell", "")).endswith(expected_cell_suffix(row))
            and row.get("backing_store") == expected_backing
            and (
                method != "pqcache_author_common"
                or (
                    row.get("pqcache_pq") == [2, 6]
                    and row.get("pqcache_seed") == 4321
                    and row["_traffic"].get("pqcache_seed") == 4321
                    and (
                        row.get("upstream_matched_exact_regions") is not True
                        or (
                            row["_traffic"].get("pqcache_matched_exact_regions") is True
                            and row["_traffic"].get("pqcache_sink_tokens_extra") == 32
                            and row["_traffic"].get("pqcache_recent_tokens") == 32
                            and row["_traffic"].get("pqcache_retrieved_tokens")
                            == int(row.get("sparse_budget", -1))
                            and row["_traffic"].get("pqcache_active_token_cap")
                            == int(row.get("sparse_budget", -1)) + 64
                        )
                    )
                )
            )
            and (
                method != "magicpig_author_common"
                or (
                    row.get("magicpig_k_l") == [10, 210]
                    and row.get("magicpig_seed") == 43
                    and row["_traffic"].get("magicpig_seed") == 43
                    and int(row.get("dense_layers", -1)) == 0
                    and (
                        row.get("upstream_matched_exact_regions") is not True
                        or (
                            row["_traffic"].get("magicpig_matched_exact_regions") is True
                            and row["_traffic"].get("magicpig_local_sink_tokens") == 64
                        )
                    )
                )
            )
            and (
                method != "infllm_author_common"
                or (
                    row["_traffic"].get("infllm_active_token_cap")
                    == int(row.get("sparse_budget", -1))
                    + (64 if row.get("upstream_matched_exact_regions") is True else 0)
                    and (
                        row.get("upstream_matched_exact_regions") is not True
                        or (
                            row["_traffic"].get("infllm_matched_exact_regions") is True
                            and row["_traffic"].get("infllm_init_tokens") == 32
                            and row["_traffic"].get("infllm_local_tokens") == 32
                            and row["_traffic"].get("infllm_retrieved_tokens")
                            == int(row.get("sparse_budget", -1))
                        )
                    )
                )
            )
            for row in rows
        )
        # Repository revisions may differ because campaign/reporting code was
        # committed while workers were live.  The numerical adapter itself,
        # however, must be identical across every headline cell.
        numeric_ok = len(numeric_hashes) == 1
        all_provenance_ok &= pin_ok and config_ok and numeric_ok
        commits = ", ".join(sorted(c[:12] for c in author_commits)) or "missing"
        print(
            f"| {LABEL[method]} | {len(rows)} | {commits} | "
            f"{'OK' if pin_ok else 'MISMATCH'} | "
            f"{'OK' if config_ok else 'MISMATCH'} | {len(revisions)} "
            f"| {len(numeric_hashes) or 'missing'} | {dirty} "
            f"| {', '.join(sorted(backings)) or 'missing'} |"
        )
    all_provenance_ok &= set(grouped) == set(METHODS)
    return all_provenance_ok


def limitations_table():
    """Print claim boundaries that cannot be inferred from accuracy alone."""

    print("\n## Implementation and comparison limitations\n")
    print("| Method | Author components retained | Common-eval deviations | Claim boundary |")
    print("|---|---|---|---|")
    print(
        "| PQCache | Released PQ dimensions and lookup/ranking structure "
        "| Bounded `kmeans-gpu` fit on at most 8,192 keys; one-byte codes; "
        "exact CPU-pinned KV; no author LFU GPU block cache "
        "| Author-structure common-forward transfer, not bit-exact native runtime; "
        "Qwen3 is outside the released model scope |"
    )
    print(
        "| MagicPIG | Released LSH and CPU importance-sampling kernels, K=10/L=210 "
        "| Table construction is synchronous in the common adapter; dense layers "
        "disabled; realized sample count is stochastic "
        "| Accuracy uses author kernels, but common-forward runtime is not the "
        "author overlap schedule; Qwen3 is an unsupported-family transfer |"
    )
    print(
        "| InfLLM | Released ContextManager and Triton attention output "
        "| Matched B+64 regime is much smaller than the released 6,272-token "
        "operating point; short auxiliary-score tiles use the equivalent Torch "
        "expression; model-native RoPE tables are injected "
        "| Fair-budget stress test rather than reproduction of the native budget; "
        "64K needs a 48GB card in this common forward |"
    )


def latency_table(paths):
    expected = set(product(MODELS, LENGTHS, METHODS))
    runtime_methods = set(METHODS) | {"pqcache_author_native"}
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        return expected, set(), 0
    eligible = {}
    duplicates = 0
    for row in rows:
        key = (row.get("model"), row.get("datalen"), row.get("method"))
        if key[0] not in MODELS or key[1] not in LENGTHS or key[2] not in runtime_methods:
            continue
        if key in eligible:
            duplicates += 1
        # Input order is explicit on the command line; a later clean rerun is
        # authoritative when the caller intentionally supplies both files.
        eligible[key] = row
    print("\n## Isolated runtime\n")
    print("| Model | Length | Method | Prefill s | Decode p50/mean/p95/max ms | Steps | Active mean/cap | Active max | Router/index config | Peak alloc/reserved GiB | Peak host GiB |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in sorted(eligible.values(), key=lambda r: (r["datalen"], r["model"], r["method"])):
        method = row["method"]
        if method in {"pqcache_author_common", "pqcache_author_native"}:
            active = row.get("pqcache_active_token_cap")
            active_max = active
            config = (
                f"PQ {row.get('pqcache_subvectors', '?')}x"
                f"{row.get('pqcache_bits_per_subvector', '?')}b"
            )
            if method == "pqcache_author_native":
                config += (
                    f", LFU={row.get('pqcache_lfu_resident_tokens', '?')}"
                    f"/block{row.get('pqcache_lfu_block_size', '?')}"
                )
        elif method == "magicpig_author_common":
            active = row.get("magicpig_mean_active_tokens")
            remote_max = row.get("magicpig_max_sampled_remote_tokens")
            exact = row.get("magicpig_local_sink_tokens")
            active_max = (
                remote_max + exact
                if remote_max is not None and exact is not None else None
            )
            config = (
                f"LSH K={row.get('magicpig_k_bits', '?')},"
                f"L={row.get('magicpig_tables', '?')}"
            )
        elif method == "infllm_author_common":
            active = row.get("infllm_active_token_cap")
            active_max = active
            config = (
                f"block={row.get('infllm_block_size', 128)},"
                f"repr={row.get('infllm_repr_topk', 4)}"
            )
        else:
            active = active_max = None
            config = "—"

        def scalar(value):
            return "—" if value is None else f"{float(value):.1f}"

        print(
            f"| {row['model']} | {row['datalen']//1024}K | {LABEL[row['method']]} "
            f"| {row['prefill_s']:.3f} | {row['decode_ms']:.2f}/"
            f"{row['decode_mean_ms']:.2f}/{row['decode_p95_ms']:.2f}/"
            f"{row['decode_ms_max']:.2f} | {row.get('n_decode', 0)} "
            f"| {scalar(active)} | {scalar(active_max)} | {config} "
            f"| {row['peak_gib']:.2f}/{row.get('peak_reserved_gib', float('nan')):.2f} "
            f"| {row['peak_host_rss_gib']:.2f} |"
        )
    headline = {
        key for key, row in eligible.items()
        if key in expected and row.get("real_text") is True
        and row.get("n_decode", 0) >= 128
    }
    print(
        f"\nRuntime audit: {len(headline)}/{len(expected)} clean rows have a real "
        f"RULER prompt and >=128 measured decode steps; missing "
        f"{len(expected-headline)}; duplicate rows resolved by input order "
        f"{duplicates}."
    )
    return expected, headline, duplicates


def matrix_audit(cells, show_missing=False):
    expected = set(product(MODELS, LENGTHS, TASKS, METHODS))
    present = set(cells)
    missing = expected - present
    unexpected = present - expected
    print("\n## Matrix audit\n")
    print(
        f"Expected {len(expected)} unique cells; complete {len(present & expected)}; "
        f"missing {len(missing)}; unexpected {len(unexpected)}."
    )
    print("\n| Model | Length | Method | Complete | Missing |")
    print("|---|---:|---|---:|---:|")
    for model, length, method in product(MODELS, LENGTHS, METHODS):
        have = sum((model, length, task, method) in present for task in TASKS)
        print(
            f"| {model} | {length // 1024}K | {LABEL[method]} "
            f"| {have}/13 | {13-have} |"
        )
    if show_missing and missing:
        print("\n### Missing cells\n")
        for key in sorted(missing):
            print("- " + " / ".join(map(str, key)))
    if unexpected:
        print("\n### Unexpected complete cells\n")
        for key in sorted(unexpected):
            print("- " + " / ".join(map(str, key)))
    return expected, missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--latency", nargs="*", type=Path, default=[])
    parser.add_argument("--show-missing", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--require-matched-exact-regions", action="store_true",
        help="only accept cells stamped with the B + 32 prefix + 32 recent protocol",
    )
    parser.add_argument(
        "--export-summary-json", action="store_true",
        help="print a compact JSON summary suitable for control-plane transfer",
    )
    parser.add_argument(
        "--summary-input", action="append", default=[], type=Path,
        help="merge a compact JSON summary exported on another host",
    )
    args = parser.parse_args()
    if not args.roots and not args.summary_input:
        parser.error("provide at least one root or --summary-input")
    complete, partial, duplicates = load_cells(
        args.roots, args.expected_samples, args.require_matched_exact_regions
    )
    imported_complete, imported_partial, imported_duplicates = load_summaries(
        args.summary_input, args.expected_samples,
        args.require_matched_exact_regions,
    )
    merge_cell_maps(complete, imported_complete, duplicates)
    merge_cell_maps(partial, imported_partial, duplicates)
    duplicates.extend(imported_duplicates)
    if args.export_summary_json:
        print(json.dumps(export_summary(
            complete, partial, duplicates, args.expected_samples
        ), separators=(",", ":"), sort_keys=True))
        return
    expected = len(MODELS) * len(LENGTHS) * len(TASKS) * len(METHODS)
    print(f"complete={len(complete)}/{expected} partial={len(partial)} duplicate_inputs={len(duplicates)}")
    accuracy_tables(complete)
    traffic_tables(complete)
    provenance_ok = provenance_table(complete)
    limitations_table()
    runtime_expected, runtime_present, runtime_duplicates = latency_table(args.latency)
    _, missing, unexpected = matrix_audit(complete, args.show_missing)
    if partial:
        print("\n## Partial cells excluded\n")
        for key, record in sorted(partial.items()):
            print(
                f"- {' / '.join(map(str, key))}: "
                f"{record['count']}/{args.expected_samples}"
            )
    if args.require_complete:
        unresolved_partial = set(partial) - set(complete)
        failures = []
        if missing:
            failures.append(f"{len(missing)} accuracy cells missing")
        if unexpected:
            failures.append(f"{len(unexpected)} unexpected accuracy cells")
        if duplicates:
            failures.append(f"{len(duplicates)} duplicate accuracy inputs")
        if unresolved_partial:
            failures.append(f"{len(unresolved_partial)} unresolved partial cells")
        if runtime_expected - runtime_present:
            failures.append(
                f"{len(runtime_expected-runtime_present)} clean runtime rows missing"
            )
        if runtime_duplicates:
            failures.append(f"{runtime_duplicates} duplicate runtime rows")
        if not provenance_ok:
            failures.append("author pin or numeric-adapter provenance mismatched")
        if failures:
            raise SystemExit("INCOMPLETE: " + "; ".join(failures))
        print("\nSTRICT CAMPAIGN AUDIT PASSED")


if __name__ == "__main__":
    main()
