from dataclasses import replace

import torch

from models.retroinfer_reference import (
    retroinfer_partition_attention,
    retroinfer_reference_attention,
    segmented_spherical_kmeans,
)


def _dense(queries, keys, values, steady_keys=None, steady_values=None):
    if steady_keys is not None:
        keys = torch.cat((steady_keys, keys), dim=1)
        values = torch.cat((steady_values, values), dim=1)
    logits = torch.einsum("gqd,gnd->gqn", queries, keys) / keys.shape[-1] ** 0.5
    return torch.softmax(logits, dim=-1) @ values


def test_segmented_kmeans_shapes_and_cluster_statistics_are_consistent():
    torch.manual_seed(0)
    keys = torch.randn(2, 32, 16)
    values = torch.randn(2, 32, 12)
    index = segmented_spherical_kmeans(
        keys, values, num_centroids=8, num_segments=2, num_iters=4
    )
    assert index.centroids.shape == (2, 8, 16)
    assert index.value_sum.shape == (2, 8, 12)
    assert torch.all(index.cluster_size.sum(-1) == 32)
    for group in range(2):
        for cluster in range(8):
            mask = index.assignments[group] == cluster
            if torch.any(mask):
                torch.testing.assert_close(
                    index.centroids[group, cluster], keys[group, mask].mean(0)
                )
                torch.testing.assert_close(
                    index.value_sum[group, cluster], values[group, mask].sum(0)
                )


def test_retrieving_all_clusters_is_exact_dense_attention():
    torch.manual_seed(1)
    keys = torch.randn(2, 40, 16)
    values = torch.randn(2, 40, 16)
    queries = torch.randn(2, 3, 16)
    steady_keys = torch.randn(2, 5, 16)
    steady_values = torch.randn(2, 5, 16)
    index = segmented_spherical_kmeans(
        keys, values, num_centroids=8, num_segments=2, num_iters=3
    )
    actual = retroinfer_reference_attention(
        queries,
        keys,
        values,
        index,
        retrieve_clusters=8,
        estimation_clusters=0,
        steady_keys=steady_keys,
        steady_values=steady_values,
    )
    expected = _dense(queries, keys, values, steady_keys, steady_values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_estimation_is_exact_when_keys_are_constant_inside_each_cluster():
    torch.manual_seed(2)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    keys = torch.stack((first, first, first, second, second, second))[None]
    values = torch.randn(1, 6, 4)
    queries = torch.randn(1, 2, 4)
    index = segmented_spherical_kmeans(
        keys, values, num_centroids=2, num_segments=1, num_iters=3
    )
    actual = retroinfer_partition_attention(
        queries,
        keys,
        values,
        index,
        retrieve_ids=torch.empty(0, dtype=torch.long),
        estimation_ids=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(actual, _dense(queries, keys, values), atol=1e-6, rtol=1e-5)


def test_estimation_accepts_bfloat16_stored_cluster_summaries():
    torch.manual_seed(3)
    keys = torch.randn(1, 16, 8)
    values = torch.randn(1, 16, 8)
    queries = torch.randn(1, 2, 8)
    index = segmented_spherical_kmeans(
        keys, values, num_centroids=4, num_iters=3
    )
    stored = replace(
        index,
        centroids=index.centroids.to(torch.bfloat16),
        value_sum=index.value_sum.to(torch.bfloat16),
    )
    output = retroinfer_reference_attention(
        queries,
        keys,
        values,
        stored,
        retrieve_clusters=2,
        estimation_clusters=2,
    )
    assert output.shape == (1, 2, 8)
    assert torch.isfinite(output).all()


def test_retrieval_and_estimation_clusters_cannot_overlap():
    keys = torch.randn(1, 8, 4)
    values = torch.randn(1, 8, 4)
    queries = torch.randn(1, 1, 4)
    index = segmented_spherical_kmeans(
        keys, values, num_centroids=2, num_iters=2
    )
    try:
        retroinfer_partition_attention(
            queries,
            keys,
            values,
            index,
            retrieve_ids=torch.tensor([0]),
            estimation_ids=torch.tensor([0]),
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping zones must be rejected")
