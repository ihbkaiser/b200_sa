#!/usr/bin/env python
"""Deterministically partition a generated queue across heterogeneous hosts."""

from __future__ import annotations

import argparse
import hashlib
import sys


def parse_hosts(spec: str) -> list[tuple[str, int]]:
    hosts = []
    for item in spec.split(","):
        name, weight = item.split("=", 1)
        if not name or int(weight) <= 0:
            raise ValueError("host weights must be positive NAME=INTEGER pairs")
        hosts.append((name, int(weight)))
    if len({name for name, _ in hosts}) != len(hosts):
        raise ValueError("host names must be unique")
    return hosts


def owner(index: int, hosts: list[tuple[str, int]]) -> str:
    slot = index % sum(weight for _, weight in hosts)
    for name, weight in hosts:
        if slot < weight:
            return name
        slot -= weight
    raise AssertionError("unreachable partition slot")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--hosts", default="m1=8,m2=4,m4=2")
    args = parser.parse_args()
    hosts = parse_hosts(args.hosts)
    if args.host not in {name for name, _ in hosts}:
        parser.error(f"host {args.host!r} is absent from --hosts")

    rows = [line.strip() for line in sys.stdin if line.strip() and not line.startswith("#")]
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()[:16]
    selected = [row for index, row in enumerate(rows) if owner(index, hosts) == args.host]
    print(
        f"# host={args.host} rows={len(selected)}/{len(rows)} "
        f"partition={args.hosts} source_sha256={digest}"
    )
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
