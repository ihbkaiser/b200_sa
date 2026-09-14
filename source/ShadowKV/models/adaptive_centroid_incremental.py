"""CUDA-graph replay for the exact block-8 incremental centroid fit.

The prefill allocator is unchanged.  During decode, a newly sealed block must
solve the same 127-bipartition self-LSE problem before it can later enter the
retrieval pool.  Running the faithful PyTorch expression layer by layer pays
dozens of Python/CUDA launches every eighth token.  This helper captures that
fixed-shape expression once and replays it without changing its arithmetic.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .centroid_router_cache import _restricted_growth_partitions


class IncrementalTwoCentroidGraph:
    """Exact fixed-shape decode fitter for S=8, T={1}, and up to two slots."""

    def __init__(
        self,
        *,
        batch_size: int,
        heads: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("incremental CUDA graph requires a CUDA device")
        self.device = device
        self.dtype = dtype
        self.dim = dim
        labels = torch.tensor(
            _restricted_growth_partitions(8, 2),
            device=device,
            dtype=torch.long,
        )
        self.labels = labels
        self.membership = nn.functional.one_hot(labels, num_classes=2).float()
        self.partition_counts = self.membership.sum(1)
        self.static_keys = torch.empty(
            batch_size, heads, 1, 8, dim, device=device, dtype=dtype
        )
        self.static_threshold = torch.empty(
            batch_size, heads, device=device, dtype=torch.float32
        )

        current = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream):
            # Warm allocator/library work outside capture.
            for _ in range(3):
                self.outputs = self._compute(
                    self.static_keys, self.static_threshold
                )
        capture_stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.outputs = self._compute(
                self.static_keys, self.static_threshold
            )
        current.wait_stream(capture_stream)

    def _materialize(
        self,
        keys: torch.Tensor,
        assignment_path: torch.Tensor,
        component_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = assignment_path.gather(
            -2,
            (component_counts - 1)[..., None, None].expand(
                *component_counts.shape, 1, 8
            ),
        ).squeeze(-2).long()
        membership = nn.functional.one_hot(chosen, num_classes=8).float()[..., :2]
        populations = membership.sum(-2)
        sums = torch.einsum("...nsc,...nsd->...ncd", membership, keys.float())
        centers = sums / populations.clamp_min(1)[..., None]
        energy = torch.einsum(
            "...nsc,...ns->...nc",
            membership,
            keys.float().square().sum(-1),
        )
        trace = (
            energy / populations.clamp_min(1) - centers.square().sum(-1)
        ).clamp_min(0)
        alpha = trace / (2.0 * self.dim)
        valid = (
            torch.arange(2, device=keys.device) < component_counts[..., None]
        )
        log_count = torch.where(
            valid,
            populations.clamp_min(1).log(),
            torch.full_like(populations, float("-inf")),
        )
        return (
            torch.where(valid[..., None], centers, torch.zeros_like(centers)).to(
                keys.dtype
            ),
            log_count.to(keys.dtype),
            torch.where(valid, alpha, torch.zeros_like(alpha)),
        )

    def _compute(
        self, keys: torch.Tensor, threshold: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = keys.float()
        unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
        logits = torch.einsum("...njd,...nid->...nji", unit, x).unsqueeze(-3)
        exact = torch.logsumexp(logits, dim=-1)
        one = logits.mean(dim=-1) + math.log(8.0)
        risk_one = (exact - one).clamp_min(0).amax(dim=(-2, -1))

        grouped = torch.einsum(
            "...tji,pir->...tjpr", logits, self.membership
        )
        grouped = grouped / self.partition_counts[None, None, None, None]
        lower = torch.logsumexp(
            grouped + self.partition_counts.log()[None, None, None, None],
            dim=-1,
        )
        partition_risk = (exact[..., None] - lower).clamp_min(0).amax(
            dim=(-3, -2)
        )
        best = partition_risk.argmin(-1)
        assignment_two = self.labels[best]
        assignment_path = torch.zeros(
            *keys.shape[:-2], 8, 8, device=keys.device, dtype=torch.uint8
        )
        assignment_path[..., 1, :] = assignment_two.to(torch.uint8)
        component_counts = 1 + (
            risk_one >= threshold[..., None]
        ).long()
        centers, log_counts, alpha = self._materialize(
            keys, assignment_path, component_counts
        )
        return centers, log_counts, alpha, component_counts, risk_one

    @torch.inference_mode()
    def run(
        self, keys: torch.Tensor, threshold: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if keys.shape != self.static_keys.shape:
            raise ValueError(
                f"incremental keys have shape {tuple(keys.shape)}, expected "
                f"{tuple(self.static_keys.shape)}"
            )
        if threshold.shape != self.static_threshold.shape:
            raise ValueError("incremental threshold shape changed")
        self.static_keys.copy_(keys)
        self.static_threshold.copy_(threshold)
        self.graph.replay()
        return self.outputs
