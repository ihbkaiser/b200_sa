"""Streaming Quest with dynamically sealed and activated pages."""

from __future__ import annotations

import torch

from .quest_triton import quest_page_scores
from .streaming_cache import StreamingBlockCache


class StreamingQuestCache(StreamingBlockCache):
    """Quest's exact min/max page bound on an append-only page index."""

    def __init__(self, config: object, *, page_size: int = 16, **kwargs) -> None:
        super().__init__(config, block_size=page_size, **kwargs)
        shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_blocks,
            self.head_dim,
        )
        self.page_min = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.page_max = torch.zeros(shape, device=self.device, dtype=self.dtype)

    @property
    def page_size(self) -> int:
        return self.block_size

    @property
    def select_pages(self) -> int:
        return self.select_blocks

    def _reset_metadata(self) -> None:
        self.page_min.zero_()
        self.page_max.zero_()

    def _build_blocks(
        self, layer_idx: int, block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        if not block_ids:
            return
        ids, blocks = self._load_block_keys(
            layer_idx, block_ids, block_keys
        )
        self.page_min[layer_idx].index_copy_(2, ids, blocks.amin(dim=-2))
        self.page_max[layer_idx].index_copy_(2, ids, blocks.amax(dim=-2))

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        query = query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )
        if query.is_cuda and self.incoming_q_len == 1:
            return quest_page_scores(
                query,
                self.page_min[layer_idx],
                self.page_max[layer_idx],
                first_page=first_block,
                last_page=last_block,
                group_reduce=self.group_reduce,
            )
        minimum = self.page_min[layer_idx, :, :, first_block:last_block]
        maximum = self.page_max[layer_idx, :, :, first_block:last_block]
        q = query.unsqueeze(-2)
        score = torch.maximum(
            q * minimum[:, :, None, None],
            q * maximum[:, :, None, None],
        ).sum(-1)
        score = score.sum(dim=-2)
        if self.num_key_value_groups > 1:
            if self.group_reduce == "max":
                score = score.amax(dim=2)
            else:
                score = score.sum(dim=2)
        else:
            score = score.squeeze(2)
        return score
