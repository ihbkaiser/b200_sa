"""ShadowKV cache whose router augments each mean with QUILL exact keys."""

from __future__ import annotations

import math

import torch
from torch import nn

from .kv_cache import ShadowKVCache
from .quill_router import combine_mean_and_exact_logits


class QuillRouterShadowKVCache(ShadowKVCache):
    """Keep ShadowKV attention intact and change only block routing logits.

    QUILL candidates are routing metadata, not additionally attended tokens.
    Hence ``sparse_budget`` retains exactly the same meaning as in ShadowKV.
    """

    def __init__(
        self, *args, exact_fraction: float = 0.125,
        score_name: str = "QUILL", aggregation: str = "max", **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if aggregation not in {"max", "logsumexp"}:
            raise ValueError("aggregation must be max or logsumexp")
        if aggregation == "logsumexp" and exact_fraction != 1.0:
            raise ValueError("logsumexp routing requires exact_fraction=1")
        self.exact_fraction = exact_fraction
        self.score_name = score_name
        self.aggregation = aggregation
        self.router_exact_keys: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_exact_blocks: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_exact_valid: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_candidate_positions: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.quill_gamma: dict[int, torch.Tensor] = {}
        self.router_exact_counts = torch.zeros(
            self.num_layers,
            self.num_key_value_heads,
            dtype=torch.int32,
            device=self.device,
        )

    def print_stats(self):
        super().print_stats()
        print(
            f"ShadowKV_{self.score_name}_ROUTER | exact-key routing | "
            f"exact_fraction {self.exact_fraction} | aggregation {self.aggregation}"
        )

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
        quill_exact_mask: torch.Tensor | None = None,
    ):
        if quill_exact_mask is None:
            raise ValueError("QUILL router requires a prefill exact-candidate mask")
        super().prefill_kv_cache(
            new_v_cache, layer_idx, key_states_roped, query
        )

        full_positions = torch.arange(
            quill_exact_mask.shape[-1], device=key_states_roped.device
        ).view(1, 1, -1).expand_as(quill_exact_mask)
        counts = quill_exact_mask.sum(dim=-1)
        if not torch.equal(counts, counts[..., :1].expand_as(counts)):
            raise RuntimeError("QUILL candidate count differs across KV heads")
        self.router_candidate_positions[layer_idx] = full_positions.masked_select(
            quill_exact_mask
        ).view(self.batch_size, self.num_key_value_heads, -1).detach().cpu()

        eligible = self.chunks * self.chunk_size
        exact_mask = quill_exact_mask[:, :, :eligible]
        positions = full_positions[:, :, :eligible]

        # Map global chunk ids to the compact landmark array, from which the
        # 48 always-resident ShadowKV outlier chunks have already been removed.
        rest_idx = self.k_landmark_idx[layer_idx]
        compact = torch.full(
            (self.batch_size, self.num_key_value_heads, self.chunks),
            -1,
            dtype=torch.long,
            device=key_states_roped.device,
        )
        compact.scatter_(
            2,
            rest_idx,
            torch.arange(rest_idx.shape[-1], device=rest_idx.device)
            .view(1, 1, -1)
            .expand_as(rest_idx),
        )

        per_head: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_count = 0
        for head in range(self.num_key_value_heads):
            selected_pos = positions[0, head][exact_mask[0, head]]
            selected_blocks = selected_pos // self.chunk_size
            selected_compact = compact[0, head, selected_blocks]
            keep = selected_compact >= 0
            selected_pos = selected_pos[keep]
            selected_compact = selected_compact[keep]
            keys = key_states_roped[0, head, selected_pos]
            per_head.append((keys, selected_compact))
            max_count = max(max_count, keys.shape[0])
            self.router_exact_counts[layer_idx, head] = keys.shape[0]

        exact_keys = torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            max_count,
            self.head_dim,
            dtype=key_states_roped.dtype,
            device=key_states_roped.device,
        )
        exact_blocks = torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            max_count,
            dtype=torch.long,
            device=key_states_roped.device,
        )
        exact_valid = torch.zeros_like(exact_blocks, dtype=torch.bool)
        for head, (keys, blocks) in enumerate(per_head):
            count = keys.shape[0]
            exact_keys[0, head, :count] = keys
            exact_blocks[0, head, :count] = blocks
            exact_valid[0, head, :count] = True
        self.router_exact_keys[layer_idx] = exact_keys
        self.router_exact_blocks[layer_idx] = exact_blocks
        self.router_exact_valid[layer_idx] = exact_valid

    def get_retrieval_position_ids(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2]
        query = query_states.view(
            -1,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )
        mean_logits = torch.einsum(
            "bhgqd,bhcd->bhgqc", query, self.k_landmark[layer_idx]
        ) / math.sqrt(self.head_dim)

        exact_keys = self.router_exact_keys[layer_idx]
        if exact_keys is None or exact_keys.shape[-2] == 0:
            exact_logits = None
        else:
            exact_logits = torch.einsum(
                "bhgqd,bhed->bhgqe", query, exact_keys
            ) / math.sqrt(self.head_dim)
        logits = combine_mean_and_exact_logits(
            mean_logits,
            exact_logits,
            self.router_exact_blocks[layer_idx],
            self.router_exact_valid[layer_idx],
            aggregation=self.aggregation,
        )
        chunk_attn = nn.functional.softmax(
            logits, dim=-1, dtype=torch.float32
        ).to(self.dtype)
        chunk_attn = chunk_attn.sum(dim=-2)
        if self.num_key_value_groups > 1:
            chunk_attn = torch.max(chunk_attn, dim=-2).values
        merged_results = torch.topk(
            chunk_attn, k=self.select_sets, dim=-1
        ).indices
        selected_chunks = self.k_landmark_idx[layer_idx].gather(
            dim=-1, index=merged_results
        )
        self.selected_chunk_idx[layer_idx].copy_(
            selected_chunks, non_blocking=True
        )
        return (
            selected_chunks.unsqueeze(-1) * self.chunk_size
            + torch.arange(self.chunk_size, device=chunk_attn.device).view(
                1, 1, 1, -1
            )
        ).view(self.batch_size, self.num_key_value_heads, -1)

    def router_metadata_fraction_of_fp16_kv(self) -> float:
        """Mean+exact router vectors divided by dense FP16 K+V scalars."""
        mean_vectors = (
            0
            if self.k_landmark is None
            else self.k_landmark.numel() // self.head_dim
        )
        exact_vectors = int(self.router_exact_counts.sum().item())
        dense_pairs = (
            self.num_layers
            * self.num_key_value_heads
            * max(self.prefill, 1)
        )
        return (mean_vectors + exact_vectors) / (2.0 * dense_pairs)

    def clear(self):
        super().clear()
        self.router_exact_keys = [None for _ in range(self.num_layers)]
        self.router_exact_blocks = [None for _ in range(self.num_layers)]
        self.router_exact_valid = [None for _ in range(self.num_layers)]
        self.router_candidate_positions = [None for _ in range(self.num_layers)]
        self.router_exact_counts.zero_()
        self.quill_gamma.clear()
