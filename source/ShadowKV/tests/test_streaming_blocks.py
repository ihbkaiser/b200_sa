import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from streaming_blocks import StreamingBlockState


def test_prompt_tail_seals_across_prefill_decode_boundary():
    state = StreamingBlockState(8, prefix_tokens=32, local_tokens=32)
    state.reset(1025)
    assert state.sealed_blocks == 128
    assert state.pending_tokens == 1
    assert state.candidate_block_range == (4, 124)
    assert state.exact_ranges == ((0, 32), (992, 1025))

    transition = state.append(7)
    assert transition.sealed == (128,)
    assert transition.activated == (124,)
    assert state.pending_tokens == 0
    assert state.exact_ranges == ((0, 32), (1000, 1032))


def test_1029_token_prompt_has_five_token_tail_and_needs_three_more():
    state = StreamingBlockState(8)
    state.reset(1029)
    assert state.pending_tokens == 5
    assert state.append(2).sealed == ()
    assert state.append(1).sealed == (128,)


def test_sealed_block_waits_until_it_leaves_recent_window():
    state = StreamingBlockState(8, local_tokens=32)
    state.reset(40)
    assert state.sealed_blocks == 5
    assert state.active_blocks == 1
    transition = state.append(8)
    assert transition.sealed == (5,)
    assert transition.activated == (1,)
    assert state.sealed_blocks == 6
    assert state.active_blocks == 2


def test_short_sequence_merges_overlapping_prefix_and_suffix():
    state = StreamingBlockState(8, prefix_tokens=32, local_tokens=32)
    state.reset(24)
    assert state.candidate_block_range == (0, 0)
    assert state.exact_ranges == ((0, 24),)


@pytest.mark.parametrize("prefix,recent", [(1, 0), (0, 7), (-8, 0)])
def test_exact_regions_must_align_to_blocks(prefix, recent):
    with pytest.raises(ValueError):
        StreamingBlockState(8, prefix_tokens=prefix, local_tokens=recent)
