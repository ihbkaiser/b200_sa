"""Common-model adapter for MagicPIG's official LSH/sparse-attention kernels.

The LSH index and CPU importance-sampling attention are imported from the
Apache-2.0 author repository.  This file only bridges their attention server
to the Q/K/V tensors produced by ShadowKV's common Qwen/Llama forward.
"""

from __future__ import annotations

import importlib.util
import os
import copy
from pathlib import Path

import torch

from .compat import head_dim_of


OFFICIAL_COMMIT = "ac9aa36"


def _load_server_class(root: str | None):
    root = root or os.environ.get("MAGICPIG_AUTHOR_ROOT")
    if not root:
        raise RuntimeError(
            "set MAGICPIG_AUTHOR_ROOT to an Infini-AI-Lab/MagicPIG checkout "
            f"(tested commit {OFFICIAL_COMMIT})"
        )
    path = Path(root).expanduser().resolve() / "models" / "attnserver.py"
    if not path.is_file():
        raise RuntimeError(f"invalid MAGICPIG_AUTHOR_ROOT: {path.parent.parent}")
    spec = importlib.util.spec_from_file_location(
        "shadowkv_magicpig_author_attnserver", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MagicPIG attention server from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # MagicPIG pins an older FlashInfer API in which the third argument is an
    # append indptr.  FlashInfer 0.2.4 split that into per-entry batch indices
    # and positions.  Keep the author call sites untouched and adapt only the
    # ABI; the cache contents and attention algorithm are unchanged.
    original_append = module.flashinfer.append_paged_kv_cache

    def append_paged_kv_cache_compat(*args, **kwargs):
        if len(args) == 7:
            (append_key, append_value, append_indptr, paged_kv_cache,
             kv_indices, kv_indptr, kv_last_page_len) = args
            seq_lens = module.flashinfer.get_seq_lens(
                kv_indptr, kv_last_page_len, paged_kv_cache.shape[-2]
            )
            batch_indices, positions = (
                module.flashinfer.get_batch_indices_positions(
                    append_indptr, seq_lens, append_key.shape[0]
                )
            )
            return original_append(
                append_key, append_value, batch_indices, positions,
                paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len,
                **kwargs,
            )
        return original_append(*args, **kwargs)

    module.flashinfer.append_paged_kv_cache = append_paged_kv_cache_compat
    return module.LSHSparseAttnServer, path.parent.parent


class MagicPIGAuthorCache:
    """MagicPIG K=10 L=210 using the released native CPU kernels.

    K=10,L=150 and K=10,L=170 are the authors' reported ~2% and ~2.5%
    operating points.  L=210 is the preregistered matched-compute point for
    this campaign (approximately 1/32 sampled remote KV); actual sampled nnz
    is recorded because LSH has no fixed token count.
    """

    def __init__(
        self,
        config: object,
        *,
        max_length: int,
        batch_size: int = 1,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        k_bits: int | None = None,
        tables: int | None = None,
        sink_tokens: int | None = None,
        local_tokens: int | None = None,
        seed: int | None = None,
        author_root: str | None = None,
    ) -> None:
        if batch_size != 1:
            raise ValueError("MagicPIG official adapter currently requires batch_size=1")
        if dtype != torch.bfloat16:
            raise ValueError("MagicPIG author CPU kernel requires bfloat16")
        k_bits = int(os.environ.get("MAGICPIG_K", "10")) if k_bits is None else int(k_bits)
        tables = int(os.environ.get("MAGICPIG_L", "210")) if tables is None else int(tables)
        self.matched_exact_regions = (
            os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        )
        sink_default = (
            os.environ.get("QUEST_PREFIX_TOKENS", "32")
            if self.matched_exact_regions
            else os.environ.get("MAGICPIG_SINK_TOKENS", "4")
        )
        local_default = (
            os.environ.get("STREAMING_RECENT_TOKENS", "32")
            if self.matched_exact_regions
            else os.environ.get("MAGICPIG_LOCAL_TOKENS", "64")
        )
        sink_tokens = int(sink_default) if sink_tokens is None else int(sink_tokens)
        local_tokens = int(local_default) if local_tokens is None else int(local_tokens)
        seed = int(os.environ.get("MAGICPIG_SEED", "43")) if seed is None else int(seed)
        Server, self.author_root = _load_server_class(author_root)
        actual_head_dim = head_dim_of(config)
        # The released server predates Qwen3 and derives head_dim from
        # hidden_size/num_heads.  Qwen3 decouples the residual width from its
        # Q-projection width, so present the server with the attention width
        # while leaving the model config itself untouched.
        server_config = copy.copy(config)
        server_config.hidden_size = (
            int(config.num_attention_heads) * actual_head_dim
        )
        # The released server samples its LSH hyperplanes with torch.randn.
        # Its released HF RULER entry point hard-codes seed 43 before model
        # construction. Seed only construction and restore the caller RNG
        # afterwards so generation remains unchanged.
        cuda_index = torch.device(device).index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[cuda_index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.server = Server(
                config=server_config,
                K=int(k_bits),
                L=int(tables),
                batch_size=batch_size,
                num_sink_tokens=int(sink_tokens),
                num_local_tokens=int(local_tokens),
                generation_buffer=2048,
                max_length=int(max_length),
                dense_layers=[],
                device=device,
                dtype=dtype,
            )
        self.config = config
        self.num_layers = int(config.num_hidden_layers)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = actual_head_dim
        self.attention_width = self.num_attention_heads * self.head_dim
        # The released Llama path names this quantity ``hidden_size``.  Qwen3
        # has a wider concatenated attention projection, so point the author's
        # final reshape at the actual Q-head width.
        self.server.hidden_size = self.attention_width
        self.k_bits = int(k_bits)
        self.tables = int(tables)
        self.sink_tokens = int(sink_tokens)
        self.local_tokens = int(local_tokens)
        self.seed = seed
        self.kv_offset = 0
        self._allocated_length = 0
        self.sampled_sum = 0
        self.sampled_square_sum = 0
        self.sampled_count = 0
        self.sampled_max = 0
        self._dense_warmup = False
        self._dense_keys: list[torch.Tensor | None] = [None] * self.num_layers
        self._dense_values: list[torch.Tensor | None] = [None] * self.num_layers

    def prefill_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        length = int(key_states.shape[-2])
        exact_capacity = self.sink_tokens + self.local_tokens
        if layer_idx == 0:
            self._dense_warmup = length <= exact_capacity
        if self._dense_warmup:
            self._dense_keys[layer_idx] = key_states.contiguous()
            self._dense_values[layer_idx] = value_states.contiguous()
            if layer_idx == self.num_layers - 1:
                self.kv_offset = length
            return
        if self._allocated_length != length:
            if layer_idx != 0:
                raise RuntimeError("MagicPIG allocation must start at layer zero")
            self.server.alloc_buffer(length)
            self._allocated_length = length
        keys = key_states[0].transpose(0, 1).contiguous()
        values = value_states[0].transpose(0, 1).contiguous()
        self.server.fill(layer_idx, 0, keys, values, length)
        # The author model overlaps table construction for layer l-1 with
        # layer l.  The common adapter builds synchronously for correctness;
        # native-runtime measurements use the author overlap schedule.
        self.server.build_table(layer_idx, 0, length)
        if layer_idx == self.num_layers - 1:
            self.kv_offset = length

    def begin_decode_step(self) -> None:
        if not self._dense_warmup:
            self.server.plan()

    def decode_attend(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._dense_warmup:
            from flash_attn import flash_attn_with_kvcache

            old_keys = self._dense_keys[layer_idx]
            old_values = self._dense_values[layer_idx]
            if old_keys is None or old_values is None:
                raise RuntimeError("MagicPIG dense warm-up state is incomplete")
            all_keys = torch.cat((old_keys, key_states), dim=-2)
            all_values = torch.cat((old_values, value_states), dim=-2)
            self._dense_keys[layer_idx] = all_keys
            self._dense_values[layer_idx] = all_values

            # Once exact prefix+local storage no longer covers the context,
            # build the released LSH index from the exact warm-up state.  The
            # crossing token is still attended exactly; the next decode step
            # starts the normal planned sparse path.
            if all_keys.shape[-2] > self.sink_tokens + self.local_tokens:
                length = int(all_keys.shape[-2])
                if self._allocated_length != length:
                    if layer_idx != 0:
                        raise RuntimeError(
                            "MagicPIG warm-up transition must start at layer zero"
                        )
                    self.server.alloc_buffer(length)
                    self._allocated_length = length
                keys = all_keys[0].transpose(0, 1).contiguous()
                values = all_values[0].transpose(0, 1).contiguous()
                self.server.fill(layer_idx, 0, keys, values, length)
                self.server.build_table(layer_idx, 0, length)

            output = flash_attn_with_kvcache(
                q=query_states.transpose(1, 2),
                k_cache=all_keys.transpose(1, 2),
                v_cache=all_values.transpose(1, 2),
                causal=True,
            ).reshape(1, 1, self.attention_width)
            if layer_idx == self.num_layers - 1:
                self.kv_offset += int(query_states.shape[-2])
                if all_keys.shape[-2] > self.sink_tokens + self.local_tokens:
                    self._dense_warmup = False
                    self._dense_keys[:] = [None] * self.num_layers
                    self._dense_values[:] = [None] * self.num_layers
            return output

        output = self.server.decode(
            query_states, key_states, value_states, layer_idx
        )
        # Author code reshapes with config.hidden_size, which is valid for
        # Llama but not Qwen3 (its attention width differs from hidden size).
        output = output.reshape(1, 1, self.attention_width)
        nnz = self.server.nnz.long()
        self.sampled_sum += int(nnz.sum().item())
        self.sampled_square_sum += int(nnz.square().sum().item())
        self.sampled_count += int(self.server.nnz.numel())
        self.sampled_max = max(self.sampled_max, int(nnz.max().item()))
        if layer_idx == self.num_layers - 1:
            self.kv_offset += int(query_states.shape[-2])
        return output

    def get_kv_len(self) -> int:
        return self.kv_offset

    def traffic(self) -> dict[str, float]:
        remote = (
            self.sampled_sum / self.sampled_count
            if self.sampled_count else float("nan")
        )
        variance = (
            self.sampled_square_sum / self.sampled_count - remote * remote
            if self.sampled_count else float("nan")
        )
        return {
            "magicpig_mean_sampled_remote_tokens": remote,
            "magicpig_std_sampled_remote_tokens": variance ** 0.5,
            "magicpig_max_sampled_remote_tokens": self.sampled_max,
            "magicpig_local_sink_tokens": self.local_tokens + self.sink_tokens,
            "magicpig_mean_active_tokens": remote + self.local_tokens + self.sink_tokens,
            "magicpig_k_bits": self.k_bits,
            "magicpig_tables": self.tables,
            "magicpig_seed": self.seed,
            "magicpig_matched_exact_regions": self.matched_exact_regions,
        }

    def H2D(self) -> None:
        return None

    def clear(self) -> None:
        self.server.clear()
        self.kv_offset = 0
        self._allocated_length = 0
        self._dense_warmup = False
        self._dense_keys[:] = [None] * self.num_layers
        self._dense_values[:] = [None] * self.num_layers

    def print_stats(self) -> None:
        sampled = (
            self.sampled_sum / self.sampled_count
            if self.sampled_count else float("nan")
        )
        print(
            "MagicPIGAuthorCache | official LSH + CPU sparse attention "
            f"commit {OFFICIAL_COMMIT} | K {self.k_bits} L {self.tables} | "
            f"sink {self.sink_tokens} local {self.local_tokens} | "
            f"dense layers 0 | mean sampled remote tokens {sampled:g}"
        )
