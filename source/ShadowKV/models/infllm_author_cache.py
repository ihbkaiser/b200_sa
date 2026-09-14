"""Adapter for the official InfLLM context-memory implementation.

The attention and memory-management logic is imported from a checkout of
``thunlp/InfLLM`` instead of being reimplemented here.  The adapter only maps
the repository's common Q/K/V tensors and token budget to the interface used
by our Qwen/Llama evaluator.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from .compat import head_dim_of


OFFICIAL_COMMIT = "12b7079"


def _load_official_infllm(root: str | None):
    root = root or os.environ.get("INFLLM_AUTHOR_ROOT")
    if not root:
        raise RuntimeError(
            "set INFLLM_AUTHOR_ROOT to a thunlp/InfLLM checkout "
            f"(tested commit {OFFICIAL_COMMIT})"
        )
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "inf_llm" / "attention" / "context_manager.py").is_file():
        raise RuntimeError(f"invalid INFLLM_AUTHOR_ROOT: {root_path}")
    root_text = str(root_path)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    context = importlib.import_module("inf_llm.attention.context_manager")
    rope = importlib.import_module("inf_llm.attention.rope")
    # The released Triton score kernel rounds the query length to 64 but its
    # auxiliary ``get_score`` pass reads the rounded log-normalizer buffer
    # without masking padded query rows (the kernel checks IS_EVEN_N where it
    # should check IS_EVEN_M).  Attention output is unaffected, but the local
    # scores used to choose representatives become NaN/garbage for decode
    # lengths such as one.  Keep author code intact and mask only those padded
    # rows before invoking its kernel.  +inf makes exp(qk - m) exactly zero.
    triton_attn = importlib.import_module(
        "inf_llm.attention.dot_production_attention.triton_impl"
    )
    if not getattr(triton_attn.get_score, "_shadowkv_padding_guard", False):
        author_get_score = triton_attn.get_score

        def get_score_padding_guard(q, k, m, sliding_window, complement_sliding_window):
            # The author's score-only Triton kernel also assumes a full
            # 64-row query tile. Matched x32/l32 runs violate that assumption
            # for Qwen GQA. This is only the auxiliary representative score,
            # not the attention output: evaluate the same expression with
            # Torch while retaining the released Triton attention kernel.
            if (
                os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
                and q.size(-2) < 64
            ):
                if q.size(1) != k.size(1):
                    groups = q.size(1) // k.size(1)
                    k = k[:, :, None, :, :].expand(
                        k.size(0), k.size(1), groups, k.size(2), k.size(3)
                    ).reshape(k.size(0), q.size(1), k.size(2), k.size(3))
                score = torch.matmul(q.float(), k.float().transpose(-1, -2))
                score.mul_((1.0 / math.sqrt(q.size(-1))) * math.log2(math.e))
                score.sub_(m[..., : q.size(-2)].float().unsqueeze(-1))
                score = torch.exp2(score)
                if sliding_window is not None:
                    offset, width = sliding_window
                    dist = (
                        torch.arange(q.size(-2), device=q.device)[:, None]
                        - torch.arange(k.size(-2), device=k.device)[None, :]
                        + offset
                    )
                    if complement_sliding_window:
                        mask = dist >= width
                    else:
                        mask = (dist >= 0) & (dist < width)
                    score.masked_fill_(~mask[None, None, :, :], 0)
                return score.sum(dim=-2).to(k.dtype)
            query_length = q.size(-2)
            if m.size(-1) > query_length:
                m = m.clone()
                m[..., query_length:] = float("inf")
            return author_get_score(
                q, k, m, sliding_window, complement_sliding_window
            )

        get_score_padding_guard._shadowkv_padding_guard = True
        triton_attn.get_score = get_score_padding_guard
    return context.ContextManager, rope.RotaryEmbeddingESM, root_path


class InfLLMAuthorCache:
    """Run the official InfLLM ContextManager in the common model forward.

    InfLLM's released operating point spends roughly two thirds of its active
    tokens on the local window and one third on retrieved 128-token memory
    units (4096 local + 16 * 128 retrieved).  The matched-budget configuration
    preserves that ratio, includes the exact initial tokens in the budget,
    and rounds the number of memory units down.  Any remainder is assigned to
    the local window, so the maximum attended-token count is exactly the
    requested ``sparse_budget``.  The optional fair-comparison lane instead
    interprets ``sparse_budget`` as B retrieved tokens and adds the same 32
    initial and 32 recent exact tokens used by ours.
    """

    def __init__(
        self,
        config: object,
        *,
        sparse_budget: int,
        batch_size: int = 1,
        device: str = "cuda:0",
        block_size: int = 128,
        n_init: int = 32,
        repr_topk: int = 4,
        author_root: str | None = None,
    ) -> None:
        if batch_size != 1:
            raise ValueError("InfLLM official adapter currently requires batch_size=1")
        if sparse_budget <= n_init + block_size:
            raise ValueError("InfLLM budget is too small for init + one memory unit")
        self.ContextManager, self.RotaryEmbeddingESM, self.author_root = (
            _load_official_infllm(author_root)
        )
        self.config = config
        self.device = torch.device(device)
        self.num_layers = int(config.num_hidden_layers)
        self.head_dim = head_dim_of(config)
        self.sparse_budget = int(sparse_budget)
        self.block_size = int(block_size)
        self.matched_exact_regions = (
            os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        )
        self.n_init = (
            int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
            if self.matched_exact_regions else int(n_init)
        )
        self.repr_topk = int(repr_topk)

        if self.matched_exact_regions:
            if self.sparse_budget % self.block_size:
                raise ValueError(
                    "matched InfLLM route budget must be divisible by block_size"
                )
            self.topk = self.sparse_budget // self.block_size
            self.n_local = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
            self.active_token_cap = (
                self.n_init + self.n_local + self.topk * self.block_size
            )
        else:
            global_budget = max(
                self.block_size, (self.sparse_budget - self.n_init) // 3
            )
            self.topk = max(1, global_budget // self.block_size)
            self.n_local = (
                self.sparse_budget - self.n_init - self.topk * self.block_size
            )
            self.active_token_cap = self.sparse_budget
        if self.n_local <= 0:
            raise ValueError("InfLLM matched budget leaves no local window")
        self.exc_block_size = min(512, self.n_local)
        self.contexts: list[object | None] = [None] * self.num_layers
        self.kv_offset = 0

    def _new_context(self):
        rope_base = float(getattr(self.config, "rope_theta", 10000.0))
        position_embedding = self.RotaryEmbeddingESM(
            self.head_dim, rope_base, 1.0
        ).to(self.device)
        # The released InfLLM predates Llama-3.2 and constructs plain RoPE
        # from theta.  Llama-3.2 uses the model-defined ``llama3`` frequency
        # scaling; ignoring it changes the base model at long context.  Reuse
        # InfLLM's position-embedding interface but inject the exact frequency
        # table selected by the installed Transformers config.  Qwen3's
        # default RoPE resolves to the same table as the author constructor.
        rope_scaling = getattr(self.config, "rope_scaling", None) or {}
        rope_type = rope_scaling.get(
            "rope_type", rope_scaling.get("type", "default")
        )
        rope_init = ROPE_INIT_FUNCTIONS.get(rope_type)
        if rope_init is None:
            raise ValueError(f"unsupported model RoPE type for InfLLM: {rope_type}")
        inv_freq, attention_scaling = rope_init(
            self.config, device=self.device, seq_len=None
        )
        if float(attention_scaling) != 1.0:
            raise ValueError(
                "InfLLM author position interface cannot represent RoPE "
                f"attention scaling {attention_scaling:g}"
            )
        position_embedding.inv_freq = inv_freq
        return self.ContextManager(
            position_embedding=position_embedding,
            n_init=self.n_init,
            n_local=self.n_local,
            block_size=self.block_size,
            max_cached_block=max(32, self.topk),
            topk=self.topk,
            exc_block_size=self.exc_block_size,
            score_decay=None,
            fattn=True,
            repr_topk=self.repr_topk,
            cache_strategy="lru",
            chunk_topk_calc=None,
            async_global_stream=True,
            pin_memory=True,
            faiss=False,
            perhead=False,
        )

    def attend(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        # The common Llama loader keeps fused Q/K projections flattened until
        # its RoPE helper, whereas Qwen already returns [B,H,T,D].  InfLLM's
        # author ContextManager expects the latter before applying its own
        # local/global positional views.
        if query_states.dim() == 3:
            batch, length, _ = query_states.shape
            query_states = query_states.view(
                batch, length, self.config.num_attention_heads, self.head_dim
            ).transpose(1, 2)
        if key_states.dim() == 3:
            batch, length, _ = key_states.shape
            key_states = key_states.view(
                batch, length, self.config.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
        context = self.contexts[layer_idx]
        if context is None:
            context = self._new_context()
            self.contexts[layer_idx] = context
        output = context.append(
            query_states,
            key_states,
            value_states,
            query_states,
            key_states,
            value_states,
        )
        if layer_idx == self.num_layers - 1:
            self.kv_offset += int(query_states.shape[-2])
        return output.transpose(1, 2).contiguous()

    def get_kv_len(self) -> int:
        return self.kv_offset

    def traffic(self) -> dict[str, int]:
        return {
            "infllm_init_tokens": self.n_init,
            "infllm_local_tokens": self.n_local,
            "infllm_retrieved_tokens": self.topk * self.block_size,
            "infllm_active_token_cap": self.active_token_cap,
            "infllm_block_size": self.block_size,
            "infllm_repr_topk": self.repr_topk,
            "infllm_matched_exact_regions": self.matched_exact_regions,
        }

    def H2D(self) -> None:
        # ContextManager performs and synchronizes its own block transfers.
        return None

    def clear(self) -> None:
        self.contexts = [None] * self.num_layers
        self.kv_offset = 0

    def print_stats(self) -> None:
        print(
            "InfLLMAuthorCache | official ContextManager "
            f"commit {OFFICIAL_COMMIT} | route budget {self.sparse_budget}; "
            f"active cap {self.active_token_cap} "
            f"= init {self.n_init} + local {self.n_local} + "
            f"{self.topk}x{self.block_size} retrieved | repr_topk "
            f"{self.repr_topk} | execution block {self.exc_block_size}"
        )
