"""Shared append-only block lifecycle for streaming KV retrieval.

One frame, four regions, identical for every method::

    [ sink ][ retrievable: indexed ][ local ][ buffer 0..interval )
      exact        top-k chosen       exact      exact, not yet indexed

* a block is *sealed* as soon as ``block_size`` chronological tokens exist,
  so a router can build its metadata once;
* a sealed block becomes *active* -- eligible for retrieval -- only after it
  falls outside ``local_tokens`` plus the pending update buffer, so no KV pair
  is ever both retrieved and attended exactly;
* the buffer fills over ``update_interval`` tokens and then empties in one
  step, which is the batch flush: ``update_interval`` tokens enter the index
  at once.  Methods whose index is a batch algorithm (product quantisation,
  LSH tables, IVF assignment, landmark means) can therefore index generated
  text using the same code they use at prefill.

Setting ``local_tokens`` to the old recent window and ``update_interval`` to
``block_size`` reproduces the previous per-block lifecycle exactly; that
degenerate case is what pins this refactor.

No scoring rule lives here, so methods keep their own algorithms while sharing
identical token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockAppend:
    """State transition produced by :meth:`StreamingBlockState.append`."""

    old_total: int
    new_total: int
    sealed: tuple[int, ...]
    activated: tuple[int, ...]


class StreamingBlockState:
    """Chronological fixed-size blocks with exact prefix and recent regions."""

    def __init__(
        self,
        block_size: int,
        *,
        prefix_tokens: int = 0,
        local_tokens: int = 0,
        update_interval: int | None = None,
        update_origin: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if prefix_tokens < 0 or prefix_tokens % block_size:
            raise ValueError("prefix_tokens must be a non-negative block multiple")
        if local_tokens < 0 or local_tokens % block_size:
            raise ValueError("local_tokens must be a non-negative block multiple")
        interval = block_size if update_interval is None else int(update_interval)
        if interval <= 0 or interval % block_size:
            raise ValueError("update_interval must be a positive block multiple")
        self.block_size = int(block_size)
        self.prefix_tokens = int(prefix_tokens)
        self.local_tokens = int(local_tokens)
        self.update_interval = interval
        # Tokens already present when flushing starts counting: set to the
        # prompt length so the buffer measures generated text, or left at zero
        # for the degenerate per-block lifecycle.
        self.update_origin = int(update_origin)
        self.total_tokens = 0

    @property
    def recent_tokens(self) -> int:
        """Exact suffix at this instant: the local window plus the buffer."""
        return self.local_tokens + self.pending_buffer

    @property
    def pending_buffer(self) -> int:
        """Tokens waiting for the next batch flush."""
        grown = max(0, self.total_tokens - self.update_origin)
        return grown % self.update_interval

    @property
    def prefix_blocks(self) -> int:
        return self.prefix_tokens // self.block_size

    @property
    def sealed_blocks(self) -> int:
        return self.total_tokens // self.block_size

    @property
    def pending_tokens(self) -> int:
        return self.total_tokens % self.block_size

    @property
    def active_blocks(self) -> int:
        """Oldest complete blocks outside the local window and the buffer."""
        old_enough = max(0, self.total_tokens - self.recent_tokens)
        return min(self.sealed_blocks, old_enough // self.block_size)

    @property
    def candidate_block_range(self) -> tuple[int, int]:
        """Half-open range of blocks eligible for dynamic retrieval."""
        start = min(self.prefix_blocks, self.active_blocks)
        return start, self.active_blocks

    @property
    def exact_ranges(self) -> tuple[tuple[int, int], ...]:
        """Disjoint half-open token ranges that must be attended exactly."""
        ranges: list[tuple[int, int]] = []
        prefix_end = min(self.prefix_tokens, self.total_tokens)
        if prefix_end:
            ranges.append((0, prefix_end))

        suffix_start = min(self.active_blocks * self.block_size, self.total_tokens)
        if suffix_start < self.total_tokens:
            if ranges and suffix_start <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], self.total_tokens)
            else:
                ranges.append((suffix_start, self.total_tokens))
        return tuple(ranges)

    @property
    def exact_tokens(self) -> int:
        return sum(end - start for start, end in self.exact_ranges)

    def reset(self, total_tokens: int = 0) -> None:
        """Seat the state on a prompt of ``total_tokens``.

        The prompt also anchors the flush counter, so ``pending_buffer``
        measures generated text rather than absolute position: a prompt of any
        length leaves the buffer empty and the first flush lands exactly
        ``update_interval`` generated tokens later.
        """
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        self.total_tokens = int(total_tokens)
        if self.update_interval != self.block_size:
            self.update_origin = int(total_tokens)

    def append(self, count: int = 1) -> BlockAppend:
        if count <= 0:
            raise ValueError("append count must be positive")
        old_total = self.total_tokens
        old_sealed = self.sealed_blocks
        old_active = self.active_blocks
        self.total_tokens += int(count)
        return BlockAppend(
            old_total=old_total,
            new_total=self.total_tokens,
            sealed=tuple(range(old_sealed, self.sealed_blocks)),
            activated=tuple(range(old_active, self.active_blocks)),
        )
