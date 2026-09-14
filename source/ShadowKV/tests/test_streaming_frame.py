"""The shared four-region frame: sink, retrievable, local, buffer.

Every method is supposed to see the same token accounting, so the rules that
define it are pinned here rather than in any one cache.
"""

import pytest

from models.streaming_blocks import StreamingBlockState


def test_degenerate_case_reproduces_the_old_per_block_lifecycle():
    """local = old recent window, interval = block_size, origin 0.

    This is the configuration the refactor has to reproduce exactly; if it
    drifts, every number produced before the frame landed becomes unreadable.
    """
    for block_size, recent in ((8, 32), (16, 32), (4, 64)):
        state = StreamingBlockState(
            block_size, prefix_tokens=32, local_tokens=recent,
            update_interval=block_size,
        )
        for total in range(0, 4096, 7):
            state.total_tokens = total
            old_active = min(total // block_size,
                             max(0, total - recent) // block_size)
            assert state.active_blocks == old_active, (block_size, recent, total)


def test_buffer_fills_then_empties_in_one_step():
    state = StreamingBlockState(
        8, prefix_tokens=32, local_tokens=256, update_interval=256,
        update_origin=1024,
    )
    state.total_tokens = 1024
    assert state.pending_buffer == 0
    assert state.recent_tokens == 256
    indexed_at_prompt = state.active_blocks

    # Generating does not index anything until the buffer is full.
    for generated in range(1, 256):
        state.total_tokens = 1024 + generated
        assert state.pending_buffer == generated
        assert state.recent_tokens == 256 + generated
        assert state.active_blocks == indexed_at_prompt

    # The 256th token empties the buffer: one batch enters the index.
    state.total_tokens = 1024 + 256
    assert state.pending_buffer == 0
    assert state.active_blocks == indexed_at_prompt + 256 // 8


def test_a_token_is_never_both_retrievable_and_exact():
    state = StreamingBlockState(
        8, prefix_tokens=32, local_tokens=256, update_interval=256,
        update_origin=2048,
    )
    for total in range(2048, 2048 + 600):
        state.total_tokens = total
        first, last = state.candidate_block_range
        retrievable_end = last * state.block_size
        exact = state.exact_ranges
        for start, end in exact:
            if start >= state.prefix_tokens:          # the suffix range
                assert start >= retrievable_end, (total, start, retrievable_end)


def test_nothing_falls_between_the_index_and_the_exact_suffix():
    """Coverage, not fairness: an unindexed token outside the exact suffix
    would simply be invisible to the model."""
    state = StreamingBlockState(
        8, prefix_tokens=32, local_tokens=256, update_interval=256,
        update_origin=2048,
    )
    for total in range(2048, 2048 + 600):
        state.total_tokens = total
        covered = state.active_blocks * state.block_size
        suffix_start = min(covered, total)
        assert suffix_start + sum(
            end - start for start, end in state.exact_ranges
            if start >= state.prefix_tokens
        ) >= total - state.prefix_tokens or state.prefix_tokens >= total


@pytest.mark.parametrize("bad", [{"local_tokens": 100}, {"update_interval": 12}])
def test_regions_must_be_whole_blocks(bad):
    with pytest.raises(ValueError):
        StreamingBlockState(8, prefix_tokens=32, **bad)
