import torch

from models.quill_router import KeyDiffRouterScorer, combine_mean_and_exact_logits


def test_exact_landmark_can_rescue_a_mean_diluted_block():
    # One KV head, one query head, one query, three blocks.  The mean routes to
    # block 0, but QUILL has retained a strong exact key in block 2.
    means = torch.tensor([[[[[5.0, 2.0, 1.0]]]]])
    exact = torch.tensor([[[[[0.5, 7.0]]]]])
    blocks = torch.tensor([[[1, 2]]])
    valid = torch.tensor([[[True, True]]])
    merged = combine_mean_and_exact_logits(means, exact, blocks, valid)
    assert merged.argmax(dim=-1).item() == 2
    assert merged[0, 0, 0, 0, 0].item() == 5.0
    assert merged[0, 0, 0, 0, 2].item() == 7.0


def test_padding_never_changes_a_block_score():
    means = torch.tensor([[[[[1.0, 2.0]]]]])
    exact = torch.tensor([[[[[100.0]]]]])
    blocks = torch.tensor([[[0]]])
    valid = torch.tensor([[[False]]])
    merged = combine_mean_and_exact_logits(means, exact, blocks, valid)
    torch.testing.assert_close(merged, means)


def test_no_exact_keys_is_identity():
    means = torch.randn(1, 2, 4, 1, 7)
    assert combine_mean_and_exact_logits(means, None, None, None) is means


def test_keydiff_router_matches_direct_definition_per_pool():
    torch.manual_seed(7)
    keys = torch.randn(1, 2, 12, 4)
    scorer = KeyDiffRouterScorer(score_chunk=6, exact_fraction=1 / 3)
    actual = scorer.exact_mask(keys_postrope=keys)
    expected = torch.zeros_like(actual)
    for start in (0, 6):
        part = keys[:, :, start : start + 6]
        anchor = torch.nn.functional.normalize(part, dim=-1).mean(2, keepdim=True)
        score = -torch.nn.functional.cosine_similarity(part, anchor, dim=-1)
        expected.scatter_(2, score.topk(2, dim=-1).indices + start, True)
    assert torch.equal(actual, expected)


def test_logsumexp_aggregation_matches_blockwise_reference():
    means = torch.zeros(1, 1, 1, 1, 2)
    exact = torch.tensor([[[[[1.0, 2.0, -1.0, 3.0]]]]])
    blocks = torch.tensor([[[0, 0, 1, 1]]])
    valid = torch.ones_like(blocks, dtype=torch.bool)
    actual = combine_mean_and_exact_logits(
        means, exact, blocks, valid, aggregation="logsumexp"
    )
    expected = torch.stack(
        [
            torch.logsumexp(exact[..., :2], dim=-1),
            torch.logsumexp(exact[..., 2:], dim=-1),
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
