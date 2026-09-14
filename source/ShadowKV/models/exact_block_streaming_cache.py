"""Offline exact block routers on the shared streaming lifecycle.

These controls deliberately read every candidate key to compute the routing
statistic.  They isolate representation error from the downstream sparse
attention budget and are not deployable retrieval methods.
"""

from __future__ import annotations

import math
import os

import torch

from .streaming_cache import StreamingBlockCache


class StreamingExactBlockOracleCache(StreamingBlockCache):
    """Rank blocks by exact token-logit maximum or exact log-sum-exp."""

    def __init__(
        self, config: object, *, statistic: str,
        normalize_blocks: bool = False, **kwargs
    ) -> None:
        if statistic not in {"max", "logsumexp"}:
            raise ValueError("statistic must be max or logsumexp")
        self.statistic = statistic
        self.normalize_blocks = normalize_blocks
        super().__init__(config, **kwargs)

    def print_stats(self) -> None:
        super().print_stats()
        print(
            "EXACT_BLOCK_ORACLE_STREAMING | offline full-key router | "
            f"statistic {self.statistic} | block normalization "
            f"{'softmax' if self.normalize_blocks else 'raw'}"
        )

    def _reset_metadata(self) -> None:
        pass

    def _build_blocks(
        self, layer_idx: int, block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        del layer_idx, block_ids, block_keys

    def _query(self, query_states: torch.Tensor) -> torch.Tensor:
        return query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        query = self._query(query_states)
        score_batch = int(os.environ.get(
            "SHADOWKV_EXACT_SCORE_BLOCK_BATCH", "256"
        ))
        if score_batch < 1:
            raise ValueError("exact score block batch must be positive")
        pieces = []
        for start in range(first_block, last_block, score_batch):
            stop = min(start + score_batch, last_block)
            block_ids = torch.arange(
                start, stop, device=self.k_cache.device
            )
            offsets = torch.arange(
                self.block_size, device=self.k_cache.device
            )
            positions = block_ids[:, None] * self.block_size + offsets[None]
            keys = self.k_cache[layer_idx, :, :, positions, :]
            if keys.device != query.device:
                keys = keys.to(query.device, non_blocking=False)
            logits = torch.einsum(
                "bhgqd,bhnsd->bhgqns", query.float(), keys.float()
            ) / math.sqrt(self.head_dim)
            if self.statistic == "max":
                pieces.append(logits.amax(dim=-1))
            else:
                pieces.append(torch.logsumexp(logits, dim=-1))
        block_score = torch.cat(pieces, dim=-1)
        if self.normalize_blocks:
            block_score = torch.softmax(block_score, dim=-1)
        # q_len is one in normal decode. Summation preserves the established
        # multi-query routing convention if this control is probed in batches.
        score = block_score.sum(dim=-2)
        if self.num_key_value_groups > 1:
            if self.group_reduce == "max":
                score = score.amax(dim=2)
            else:
                score = score.sum(dim=2)
        else:
            score = score.squeeze(2)
        return score
