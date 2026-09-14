#!/usr/bin/env python
"""Round-robin an existing queue by method without changing its cells.

The claim pool skips running/done markers, so this filter may also be applied
atomically to a live queue.  It is useful for overnight campaigns: every
method begins producing evidence early instead of waiting behind one complete
method block.
"""

from __future__ import annotations

import collections
import sys


def main() -> int:
    comments: list[str] = []
    groups: dict[str, collections.deque[str]] = {}
    order: list[str] = []
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"expected 9 fields, got {len(fields)}: {line}")
        method = fields[3]
        if method not in groups:
            groups[method] = collections.deque()
            order.append(method)
        groups[method].append(line)

    total = sum(map(len, groups.values()))
    print(
        f"# interleaved_by=method rows={total} methods={','.join(order)}"
    )
    for comment in comments:
        print(comment)
    while any(groups.values()):
        for method in order:
            if groups[method]:
                print(groups[method].popleft())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
