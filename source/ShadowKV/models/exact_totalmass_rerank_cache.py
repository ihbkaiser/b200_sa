"""Offline exact-total-mass controls for temporal block routing.

The parent exact-block cache supplies the one-shot oracle.  This subclass adds
an optional, deliberately non-deployable control: admit complete blocks holding
``factor * B`` tokens by exact total mass, then retain the best ``B`` individual
tokens by exact live-query mass.
"""

from __future__ import annotations

import math
import os

import torch

from .exact_block_streaming_cache import StreamingExactBlockOracleCache


class StreamingExactTotalMassRerankCache(StreamingExactBlockOracleCache):
    """Exact block-8 mass oracle with optional exact token reranking."""

    def __init__(
        self,
        config: object,
        *,
        refine_factor: float = 1.0,
        refine_tokens: bool = False,
        **kwargs,
    ) -> None:
        self.refine_factor = float(refine_factor)
        self.refine_tokens = bool(refine_tokens)
        self.query_group_mean = (
            os.environ.get("SHADOWKV_EXACT_QUERY_GROUP_MEAN", "0") == "1"
        )
        if self.refine_factor < 1.0:
            raise ValueError("refine_factor must be at least one")
        if self.refine_tokens and kwargs.get("statistic") != "logsumexp":
            raise ValueError("exact token refinement requires logsumexp block mass")
        if self.refine_tokens and not kwargs.get("normalize_blocks", False):
            raise ValueError("exact token refinement requires normalized block mass")
        super().__init__(config, **kwargs)

    def print_stats(self) -> None:
        super().print_stats()
        print(
            "EXACT_TOTAL_MASS_CONTROL | refinement "
            f"{'token' if self.refine_tokens else 'none'} "
            f"x{self.refine_factor:g} | GQA "
            f"{'query-mean-first' if self.query_group_mean else 'mass-aggregate'}"
        )

    def _query(self, query_states: torch.Tensor) -> torch.Tensor:
        query = query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )
        if self.query_group_mean:
            query = query.mean(dim=(2, 3), keepdim=True)
        return query

    def _reduce_query_groups(
        self, score: torch.Tensor, *, query_dim: int
    ) -> torch.Tensor:
        """Reduce live-query scores with the same GQA rule as block routing."""
        score = score.sum(dim=query_dim)
        if self.query_group_mean:
            return score.squeeze(2)
        if self.num_key_value_groups == 1:
            return score.squeeze(2)
        if self.group_reduce == "max":
            return score.amax(dim=2)
        return score.sum(dim=2)

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        return super()._score_blocks(
            layer_idx, query_states, first_block, last_block
        )

    def get_retrieval_position_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        if not self.refine_tokens:
            return super().get_retrieval_position_ids(layer_idx, query_states)

        self.incoming_q_len = query_states.shape[-2]
        first, last = self.block_state[layer_idx].candidate_block_range
        available = last - first
        if available < self.select_blocks:
            raise RuntimeError(
                f"only {available} active blocks for budget {self.select_blocks}"
            )

        block_ids = torch.arange(first, last, device=self.compute_device)
        offsets = torch.arange(self.block_size, device=self.compute_device)
        positions = block_ids[:, None] * self.block_size + offsets[None]
        keys = self.k_cache[layer_idx, :, :, positions, :]
        query = self._query(query_states)
        token_logits = torch.einsum(
            "bhgqd,bhnsd->bhgqns", query.float(), keys.float()
        ) / math.sqrt(self.head_dim)

        block_probability = torch.softmax(
            torch.logsumexp(token_logits, dim=-1), dim=-1
        )
        block_score = self._reduce_query_groups(
            block_probability, query_dim=-2
        )
        candidate_blocks = min(
            available,
            max(
                self.select_blocks,
                int(math.ceil(self.refine_factor * self.select_blocks)),
            ),
        )
        candidate_relative = block_score.topk(candidate_blocks, dim=-1).indices

        token_probability = torch.softmax(
            token_logits.flatten(-2), dim=-1
        ).view_as(token_logits)
        token_score = self._reduce_query_groups(
            token_probability, query_dim=-3
        )
        gather_index = candidate_relative[..., None].expand(
            self.batch_size,
            self.num_key_value_heads,
            candidate_blocks,
            self.block_size,
        )
        candidate_score = token_score.gather(-2, gather_index).flatten(-2)
        winner = candidate_score.topk(self.sparse_budget, dim=-1).indices

        candidate_positions = (
            (candidate_relative + first)[..., None] * self.block_size + offsets
        ).flatten(-2)
        self.pending_block_ids[layer_idx] = None
        return candidate_positions.gather(-1, winner)
