#!/usr/bin/env python
"""ShadowKV outlier-block policy used by the reproducibility harness.

The paper uses 48 outlier blocks of 8 tokens at 128K context, i.e. 3/1024
of the context tokens.  Short-context evaluations must scale the number of
blocks; carrying the literal 48 down to 8K/16K silently gives the baseline a
much larger permanently-resident exact cache.
"""

from __future__ import annotations

import os
import sys


REFERENCE_CONTEXT = 131_072
REFERENCE_CHUNK = 8
REFERENCE_OUTLIER_BLOCKS = 48


def shadow_outlier_chunks(datalen: int, chunk_size: int) -> int:
    """Nearest integer block count at the paper's 384/131072 token rate.

    ``SHADOWKV_OUTLIER_CHUNKS`` is an explicit escape hatch for controlled
    ablations.  It is intentionally part of the cell name (via cell_key.py),
    so an override cannot reuse a result produced under another policy.
    """
    override = os.environ.get("SHADOWKV_OUTLIER_CHUNKS")
    if override is not None:
        value = int(override)
        if value < 0:
            raise ValueError("SHADOWKV_OUTLIER_CHUNKS must be non-negative")
        return value

    datalen = int(datalen)
    chunk_size = int(chunk_size)
    if datalen <= 0 or chunk_size <= 0:
        raise ValueError("datalen and chunk_size must be positive")

    # Round half up, rather than Python's bankers' rounding.  This gives
    # c8: 4K->2, 8K->3, 16K->6, 32K->12, 64K->24, 128K->48.
    numerator = datalen * REFERENCE_OUTLIER_BLOCKS * REFERENCE_CHUNK
    denominator = REFERENCE_CONTEXT * chunk_size
    value = (numerator + denominator // 2) // denominator
    total_blocks = datalen // chunk_size
    return max(0, min(value, total_blocks))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: outlier_policy.py DATALEN CHUNK_SIZE")
    print(shadow_outlier_chunks(int(sys.argv[1]), int(sys.argv[2])))
