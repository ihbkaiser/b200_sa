"""Thin bridge to PQCache's released native cache/runtime.

Unlike :mod:`pqcache_author_cache`, this class does not reimplement the
algorithm in the common cache abstraction.  It imports the commit-pinned
author checkout at runtime and delegates indexing, CPU KV storage, LFU GPU
cache management, asynchronous transfers, and sparse attention to the
released ``PqBasedSearchCompressor``.

The bridge only adapts the already-computed post-RoPE Q/K/V tensors and maps
our explicit token accounting (B retrieved + prefix + recent) onto the ratio
parameters expected by the author runtime.  The upstream repository has no
top-level license, so none of its source is copied into this repository.
"""

from __future__ import annotations

import atexit
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

from .compat import head_dim_of


OFFICIAL_COMMIT = "0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828"


class PQCacheAuthorNativeCache:
    """Use PQCache's complete released compressor/cache-manager path.

    This is intentionally a runtime/provenance lane.  Accuracy tables continue
    to use ``pqcache_author_common`` so every method shares the same model
    forward and BF16 arithmetic.  PQCache's released native cache is FP16, so
    this lane requires an FP16 common forward as well.
    """

    def __init__(
        self,
        config: object,
        *,
        max_length: int,
        sparse_budget: int,
        batch_size: int = 1,
        device: str = "cuda:0",
        dtype=torch.float16,
        author_root: str | None = None,
    ) -> None:
        if batch_size != 1:
            raise ValueError("PQCache native runtime requires batch_size=1")
        if dtype != torch.float16:
            raise ValueError(
                "PQCache's released native GPU cache is FP16; construct the "
                "model with dtype=torch.float16"
            )
        visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
        if len(visible) != 1:
            raise ValueError(
                "PQCache native common-forward bridge requires exactly one "
                "CUDA_VISIBLE_DEVICES entry"
            )

        author_root = author_root or os.environ.get("PQCACHE_AUTHOR_ROOT")
        if not author_root:
            raise RuntimeError("set PQCACHE_AUTHOR_ROOT to the official PQCache checkout")
        self.author_root = Path(author_root).expanduser().resolve()
        entry = self.author_root / "vq_method" / "retrieval_based" / "pq_search.py"
        lfu_dir = self.author_root / "vq_method" / "retrieval_based" / "lfu" / "build"
        if not entry.is_file():
            raise RuntimeError(f"invalid PQCACHE_AUTHOR_ROOT: missing {entry}")
        if not any(lfu_dir.glob("lfucache*.so")):
            raise RuntimeError(
                f"PQCache native LFU extension is missing under {lfu_dir}; "
                "run repro/shadowkv/setup_upstream_kv_baselines.sh"
            )

        self.config = config
        self.max_length = int(max_length)
        self.sparse_budget = int(sparse_budget)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = int(config.num_hidden_layers)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = head_dim_of(config)

        self.sink_tokens = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        self.recent_tokens = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        self.retrieved_tokens = self.sparse_budget
        self.subvec = int(os.environ.get("PQCACHE_SUBVECTORS", "2"))
        self.subbits = int(os.environ.get("PQCACHE_SUBBITS", "6"))
        self.max_iter = int(os.environ.get("PQCACHE_NATIVE_MAX_ITER", "0"))
        self.cache_block_size = int(os.environ.get("PQCACHE_CACHE_BLOCK_SIZE", "128"))
        self.global_cache_size = int(os.environ.get("PQCACHE_GLOBAL_CACHE_SIZE", "4096"))
        self.cache_topk = int(
            os.environ.get(
                "PQCACHE_CACHE_TOPK",
                str(self.global_cache_size // self.cache_block_size),
            )
        )
        if self.head_dim % self.subvec:
            raise ValueError("head_dim must be divisible by PQ subvector count")
        if self.global_cache_size % self.cache_block_size:
            raise ValueError("PQCache global cache must contain whole LFU blocks")

        # These are the knobs read directly by the released worker processes.
        os.environ["SUBVEC"] = str(self.subvec)
        os.environ["SUBBITS"] = str(self.subbits)
        os.environ.setdefault("METRIC", "euc")
        os.environ.setdefault("RANDOM_SEED", "4321")

        module = self._import_author_runtime()
        self._author = module
        initial_ratio, local_ratio = self._ratios_for_length(self.max_length)
        native_config = SimpleNamespace(
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            # Qwen3's declared hidden_size/head count is not its Q/K head_dim.
            # The author runtime only uses this quotient, so expose the tensor
            # ABI it actually receives rather than the residual-stream width.
            hidden_size=self.num_attention_heads * self.head_dim,
            max_seq_len=self.max_length,
            compress_ratio=initial_ratio,
            recent_ratio=local_ratio,
            sink_size=self.sink_tokens,
            global_cache_size=self.global_cache_size,
            cache_block_size=self.cache_block_size,
            cache_topk=self.cache_topk,
        )

        # The coefficient selector only recognizes Llama/Mistral.  It predicts
        # how many CPU Lloyd iterations fit under prefill; tensor computation
        # and all cache logic are otherwise model agnostic.  max_iter>0 bypasses
        # this selector, but the class still asks for a known model label.
        module.PqBasedSearchCompressor.all_pq_compressors.clear()
        self.state_dir = Path(
            os.environ.get(
                "PQCACHE_NATIVE_STATE_DIR",
                str(
                    Path(os.environ.get("SHADOWKV_RESULTS_ROOT", "."))
                    / "_pqcache_native_state"
                ),
            )
        ).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # The released compressor calibrates a machine-specific KMeans timing
        # model into ./cluster_config.json on first use. Keep that legitimate
        # state outside the code checkout and reuse it across isolated rows.
        # Initialization alone is not timed; every prefill still includes its
        # actual full-key KMeans fit and readiness barrier.
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.state_dir)
            module.initialize_objects(native_config, model="llama")
        finally:
            os.chdir(previous_cwd)
        self.compressors = [
            module.PqBasedSearchCompressor(
                initial_ratio,
                local_ratio,
                self.subvec,
                self.subbits,
                True,
                self.sink_tokens,
                layer_idx=layer_idx,
                cur_device=self.device,
                max_iter=self.max_iter,
                kv_head=self.num_key_value_heads,
                dim=self.head_dim,
                num_layer_cnt=self.num_layers,
            )
            for layer_idx in range(self.num_layers)
        ]
        self.kv_offset = 0
        self._configured_length = None
        self._closed = False
        atexit.register(self.close)

    def _import_author_runtime(self):
        root = str(self.author_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # pq_search.py imports two SparQ names but never references them in the
        # PQ path.  Importing the real parent package eagerly imports WandB,
        # torchaudio and experiment tasks.  Supply only those two unused names
        # so the native PQ runtime does not acquire unrelated benchmark/UI
        # dependencies; no PQCache execution path is replaced.
        unused_sparq = (
            "vq_method.retrieval_based.sparq_official.methods.ann_attention"
        )
        if unused_sparq not in sys.modules:
            stub = ModuleType(unused_sparq)
            stub.MistralAttentionWithANN = object
            stub.Settings = object
            sys.modules[unused_sparq] = stub
        module = importlib.import_module("vq_method.retrieval_based.pq_search")
        source = Path(module.__file__).resolve()
        if self.author_root not in source.parents:
            raise RuntimeError(
                "vq_method was imported from a different checkout: " f"{source}"
            )
        return module

    def _ratios_for_length(self, length: int) -> tuple[float, float]:
        available = length - self.sink_tokens
        requested = self.retrieved_tokens + self.recent_tokens
        if available <= requested:
            raise ValueError(
                f"PQCache native budget {requested}+sink exceeds context {length}"
            )
        # A quarter-token interior point avoids floating-point truncation at an
        # exact integer boundary in both author expressions:
        # int(n*c*l) == recent and int(n*c*(1-l)) == retrieved.
        total = requested + 0.5
        local = self.recent_tokens + 0.25
        ratio = total / available
        local_ratio = local / total
        if int(available * ratio * local_ratio) != self.recent_tokens:
            raise AssertionError("failed to encode exact PQCache recent budget")
        if int(available * ratio * (1.0 - local_ratio)) != self.retrieved_tokens:
            raise AssertionError("failed to encode exact PQCache retrieval budget")
        return ratio, local_ratio

    def _configure_length(self, length: int) -> None:
        ratio, local_ratio = self._ratios_for_length(length)
        for compressor in self.compressors:
            compressor.compress_ratio = ratio
            compressor.recent_ratio = local_ratio
            compressor.topk_ratio = 1.0 - local_ratio
        for manager in self._author.cache_managers:
            manager.compress_ratio = ratio
            manager.local_ratio = local_ratio
        self._configured_length = length

    def prefill_attend(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        length = int(key_states.shape[-2])
        if layer_idx == 0:
            self._configure_length(length)
        elif self._configured_length != length:
            raise RuntimeError("PQCache layers received inconsistent prefill lengths")
        output, _ = self.compressors[layer_idx].prefill_attn(
            query_states, (key_states, value_states)
        )
        if layer_idx == self.num_layers - 1:
            self.kv_offset = length
        return output

    def decode_attend(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        repeat_k = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        repeat_v = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        output = self.compressors[layer_idx].decoding_attn(
            self.num_key_value_groups, query_states, repeat_k, repeat_v
        )
        if layer_idx == self.num_layers - 1:
            self.kv_offset += int(key_states.shape[-2])
        return output.to(query_states.dtype)

    def get_kv_len(self) -> int:
        return self.kv_offset

    def H2D(self) -> None:
        # The last layer's completion implies every earlier layer's clustering
        # task has completed because each worker traverses layers in order.
        self._author.wait()
        torch.cuda.current_stream(self.device).synchronize()

    def clear(self) -> None:
        # ``prefill_attn`` refreshes every author buffer and layer 0 refreshes
        # the shared-memory pool.  Do not destroy worker processes between
        # benchmark repetitions.
        self.kv_offset = 0
        self._configured_length = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._author.del_objects()
        except Exception:
            # Interpreter shutdown may already have torn down CUDA/modules.
            pass

    def traffic(self) -> dict[str, int | float | bool | str]:
        return {
            "pqcache_native_author_runtime": True,
            "pqcache_native_author_commit": OFFICIAL_COMMIT,
            "pqcache_native_author_root": str(self.author_root),
            "pqcache_native_compat_model": "llama-coefficients",
            "pqcache_sink_tokens_extra": self.sink_tokens,
            "pqcache_retrieved_tokens": self.retrieved_tokens,
            "pqcache_recent_tokens": self.recent_tokens,
            "pqcache_active_token_cap": (
                self.sink_tokens + self.retrieved_tokens + self.recent_tokens
            ),
            "pqcache_subvectors": self.subvec,
            "pqcache_bits_per_subvector": self.subbits,
            "pqcache_lfu_block_size": self.cache_block_size,
            "pqcache_lfu_resident_tokens": self.global_cache_size,
            "pqcache_lfu_cache_topk_blocks": self.cache_topk,
            "pqcache_native_state_dir": str(self.state_dir),
        }

    def print_stats(self) -> None:
        print(
            "PQCacheAuthorNativeCache | complete upstream runtime "
            f"commit {OFFICIAL_COMMIT[:12]} | PQ {self.subvec}x{self.subbits} bit | "
            f"retrieve {self.retrieved_tokens} + recent {self.recent_tokens} "
            f"+ sink {self.sink_tokens} | LFU {self.global_cache_size} tokens/"
            f"block {self.cache_block_size} | FP16"
        )
