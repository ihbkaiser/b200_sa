"""Common exact-backing cache for streaming block retrieval prototypes.

This is intentionally separate from released ShadowKV.  It provides token
accounting, full post-RoPE KV storage, exact prefix/recent handling, and block
activation.  Subclasses only implement block metadata and scoring.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .compat import head_dim_of
from .offload_gather import append_kv_uva, gather_blocks_reuse_uva, gather_kv
from .streaming_blocks import StreamingBlockState


class StreamingBlockCache(ABC):
    """Append-only retrieval cache with one shared chronological block grid."""

    block_selection = True

    def __init__(
        self,
        config: object,
        *,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        sparse_budget: int = 2048,
        block_size: int = 8,
        dense_layers: int = 0,
        group_reduce: str = "max",
        prefix_tokens: int = 0,
        recent_tokens: int = 32,
        update_interval: int | None = None,
        offload: bool = False,
        offload_backend: str = "auto",
    ) -> None:
        if batch_size != 1:
            raise ValueError("streaming block caches currently require batch_size=1")
        if group_reduce not in {"max", "sum"}:
            raise ValueError("group_reduce must be max or sum")
        if sparse_budget <= 0 or sparse_budget % block_size:
            raise ValueError("sparse_budget must be a positive block multiple")
        if prefix_tokens < 0 or prefix_tokens % block_size:
            raise ValueError("prefix_tokens must be a non-negative block multiple")
        if recent_tokens < 0 or recent_tokens % block_size:
            raise ValueError("recent_tokens must be a non-negative block multiple")
        if update_interval is not None and (
            update_interval <= 0 or update_interval % block_size
        ):
            raise ValueError("update_interval must be a positive block multiple")

        self.config = config
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self.dtype = dtype
        self.sparse_budget = int(sparse_budget)
        self.block_size = int(block_size)
        self.dense_layers = int(dense_layers)
        self.group_reduce = group_reduce
        self.prefix_tokens = int(prefix_tokens)
        # recent_tokens is the LOCAL window: exact, already indexed, excluded
        # from retrieval. update_interval is the batch-flush cadence; leaving it
        # at block_size is the per-block lifecycle this class shipped with.
        self.recent_tokens = int(recent_tokens)
        self.update_interval = (
            self.block_size if update_interval is None else int(update_interval)
        )
        self.offload = bool(offload)
        self.offload_backend = str(offload_backend)
        # Cross-step reuse keeps blocks that the previous step already fetched.
        # It is an implementation optimization, not part of any method, so a
        # runtime comparison must be able to switch it off for everybody.
        self.gather_reuse = os.environ.get("STREAMING_GATHER_REUSE", "1") == "1"
        self.reuse_stats = os.environ.get("STREAMING_REUSE_STATS", "0") == "1"
        self.reused_blocks = 0
        self.fetched_blocks = 0
        self._previous_token_ids = [None] * int(config.num_hidden_layers)
        # Sealed blocks waiting for the next batch flush: (ids, key chunks).
        self._pending_block_builds = [
            ([], []) for _ in range(int(config.num_hidden_layers))
        ]
        if self.reuse_stats:
            import atexit

            atexit.register(self._report_reuse)
        self.compute_device = torch.device(device)
        if self.offload and self.compute_device.type != "cuda":
            raise ValueError("streaming CPU offload requires a CUDA compute device")
        if self.offload_backend not in {"auto", "uva", "torch"}:
            raise ValueError("offload_backend must be auto, uva, or torch")

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.num_layers = config.num_hidden_layers
        self.head_dim = head_dim_of(config)
        self.select_blocks = self.sparse_budget // self.block_size
        # A token-level router (ParisKV, or ours under token refinement) has a
        # retrieval unit of one token.  The reuse kernel matches ids inside the
        # launch, so unit=1 is simply block_size=1 and needs no new kernel --
        # only an id bank sized by the unit rather than by the block grid.
        self.retrieval_unit = self.block_size if self.block_selection else 1
        self.retrieval_slots = self.sparse_budget // self.retrieval_unit
        self.max_blocks = self.max_length // self.block_size

        cache_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
        )
        if self.offload:
            # Avoid eagerly zeroing tens of GiB at 128K.  State bookkeeping
            # ensures that only initialized positions are ever read.
            self.k_cache = torch.empty(
                cache_shape, device="cpu", dtype=dtype, pin_memory=True
            )
            self.v_cache = torch.empty(
                cache_shape, device="cpu", dtype=dtype, pin_memory=True
            )
        else:
            self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
            self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

        # The exact suffix is the local window plus the flush buffer, and the
        # buffer grows to update_interval-1 tokens before a flush hands it to
        # the index.  Sizing for block_size alone was correct only in the
        # degenerate configuration where the interval *is* the block size; at
        # interval 256 the region grows by 256 tokens between flushes and the
        # gather overruns its buffer partway through the first generation.
        self.gather_capacity = (
            self.sparse_budget
            + self.prefix_tokens
            + self.recent_tokens
            + self.update_interval
            + self.block_size
        )
        gather_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.gather_capacity,
            self.head_dim,
        )
        self.k_gather = torch.zeros(gather_shape, device=device, dtype=dtype)
        self.v_gather = torch.zeros(gather_shape, device=device, dtype=dtype)
        if self.offload:
            # Ping-pong attention buffers allow unchanged retrieval blocks to
            # remain on GPU.  The next step reads them from the previous bank
            # and fetches only replacements from pinned CPU memory.
            self.k_gather_alt = torch.zeros(
                gather_shape, device=device, dtype=dtype
            )
            self.v_gather_alt = torch.zeros(
                gather_shape, device=device, dtype=dtype
            )
            self.gather_block_ids = torch.full(
                (
                    2,
                    self.num_layers,
                    self.batch_size,
                    self.num_key_value_heads,
                    self.retrieval_slots,
                ),
                -1,
                device=device,
                dtype=torch.long,
            )
            self.gather_bank = [0] * self.num_layers
        else:
            self.k_gather_alt = None
            self.v_gather_alt = None
            self.gather_block_ids = None
            self.gather_bank = []
        self.pending_block_ids: list[torch.Tensor | None] = [
            None
        ] * self.num_layers
        staging_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.block_size,
            self.head_dim,
        )
        # Metadata for a newly sealed decode block is built from this GPU tail,
        # so the hot path never round-trips through CPU merely to index it.
        self.k_block_staging = torch.empty(
            staging_shape, device=device, dtype=dtype
        )

        self.block_state = [
            StreamingBlockState(
                self.block_size,
                prefix_tokens=self.prefix_tokens,
                local_tokens=self.recent_tokens,
                update_interval=self.update_interval,
            )
            for _ in range(self.num_layers)
        ]
        self.kv_offset = 0
        self.incoming_q_len = 1
        self.last_attention_tokens = 0
        self.copy_stream = (
            torch.cuda.Stream(device=device)
            if torch.device(device).type == "cuda"
            else None
        )

    @abstractmethod
    def _reset_metadata(self) -> None:
        """Clear method-specific metadata."""

    @abstractmethod
    def _build_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        """Build metadata for newly sealed chronological blocks."""

    def _before_streaming_append(self, layer_idx: int) -> None:
        """Allow a router to finish deferred metadata before activation."""

    def _build_or_defer_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor,
    ) -> None:
        """Build sealed-block metadata, or queue it while the block is recent.

        A block's metadata is needed only once the block may be retrieved, and
        the frame guarantees that cannot happen before the next flush: a block
        sealed while the buffer is filling still lies inside the exact suffix,
        so ``active_blocks`` excludes it.  Queuing the build until the flush is
        therefore invisible to selection -- the same blocks are chosen and the
        same metadata describes them -- while turning ``update_interval /
        block_size`` separate fits into one batched fit, which is how ParisKV's
        ``dynamic_update_interval`` works and why the interval exists at all.

        The degenerate configuration (interval == block_size) flushes on every
        seal, so it keeps the original immediate build.
        """
        if self.update_interval == self.block_size:
            self._build_blocks(layer_idx, block_ids, block_keys)
            return
        ids, keys = self._pending_block_builds[layer_idx]
        ids.extend(block_ids)
        # The staging buffer is overwritten by the next eight tokens, so the
        # queue must own its copy.
        keys.append(block_keys.clone())

    def _flush_block_builds(self, layer_idx: int) -> None:
        ids, keys = self._pending_block_builds[layer_idx]
        if not ids:
            return
        self._pending_block_builds[layer_idx] = ([], [])
        batched = keys[0] if len(keys) == 1 else torch.cat(keys, dim=2)
        self._build_blocks(layer_idx, tuple(ids), batched)

    def _load_block_keys(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ids and [B,H,block,S,D] keys on the compute device."""
        ids = torch.tensor(
            block_ids, device=self.compute_device, dtype=torch.long
        )
        if block_keys is not None:
            expected = (
                self.batch_size,
                self.num_key_value_heads,
                len(block_ids),
                self.block_size,
                self.head_dim,
            )
            if block_keys.shape != expected:
                raise ValueError(
                    f"block_keys has shape {tuple(block_keys.shape)}, "
                    f"expected {expected}"
                )
            return ids, block_keys.to(self.compute_device)
        if self.offload:
            # This fallback is used only for a batched update that seals more
            # than the staged tail. Make its preceding async D2H copy visible
            # before CPU index_select reads the completed blocks.
            torch.cuda.current_stream(self.compute_device).synchronize()
        cpu_ids = ids.cpu() if self.k_cache.device.type == "cpu" else ids
        offsets = torch.arange(self.block_size, device=cpu_ids.device)
        positions = (cpu_ids[:, None] * self.block_size + offsets[None]).reshape(-1)
        keys = self.k_cache[layer_idx].index_select(2, positions)
        keys = keys.reshape(
            self.batch_size,
            self.num_key_value_heads,
            len(block_ids),
            self.block_size,
            self.head_dim,
        )
        return ids, keys.to(self.compute_device, non_blocking=self.offload)

    @abstractmethod
    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        """Return one score per eligible block, shaped [batch, kv_head, block]."""

    def print_stats(self) -> None:
        print(
            f"{type(self).__name__} | token budget {self.sparse_budget} | "
            f"block {self.block_size} | prefix {self.prefix_tokens} | "
            f"recent {self.recent_tokens} | dense layers {self.dense_layers} | "
            f"cached {self.kv_offset} | backing "
            f"{'CPU-pinned/' + self.offload_backend if self.offload else 'GPU'}"
        )

    def _report_reuse(self) -> None:
        if not self.fetched_blocks:
            return
        share = self.reused_blocks / self.fetched_blocks
        print(
            f"REUSE {type(self).__name__} block {self.block_size} "
            f"reused {self.reused_blocks} of {self.fetched_blocks} "
            f"({share:.1%}) enabled={self.gather_reuse}"
        )

    def get_kv_len(self) -> int:
        return self.kv_offset

    def H2D(self) -> None:
        # Prefill copies the full exact K/V asynchronously to pinned memory.
        # Generation calls H2D once after prefill; synchronizing here makes the
        # backing store visible before the first UVA gather.
        if self.offload:
            torch.cuda.current_stream(self.compute_device).synchronize()

    def is_dense_layer(self, layer_idx: int) -> bool:
        return layer_idx < self.dense_layers

    def clear(self) -> None:
        if not self.offload:
            self.k_cache.zero_()
            self.v_cache.zero_()
        self.k_gather.zero_()
        self.v_gather.zero_()
        if self.offload:
            self.k_gather_alt.zero_()
            self.v_gather_alt.zero_()
            self.gather_block_ids.fill_(-1)
            self.gather_bank[:] = [0] * self.num_layers
        self.pending_block_ids[:] = [None] * self.num_layers
        self._pending_block_builds = [
            ([], []) for _ in range(self.num_layers)
        ]
        for state in self.block_state:
            state.reset()
        self.kv_offset = 0
        self.incoming_q_len = 1
        self.last_attention_tokens = 0
        self._reset_metadata()

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> None:
        del query
        incoming = new_v_cache.shape[-2]
        if incoming > self.max_length:
            raise ValueError("prefill exceeds max_length")
        state = self.block_state[layer_idx]
        if state.total_tokens:
            raise RuntimeError("prefill_kv_cache called twice without clear")
        self.k_cache[layer_idx, :, :, :incoming].copy_(
            key_states_roped, non_blocking=self.offload
        )
        self.v_cache[layer_idx, :, :, :incoming].copy_(
            new_v_cache, non_blocking=self.offload
        )
        state.reset(incoming)
        sealed_ids = tuple(range(state.sealed_blocks))
        if sealed_ids:
            sealed_keys = key_states_roped[...,
                : state.sealed_blocks * self.block_size, :
            ].reshape(
                self.batch_size,
                self.num_key_value_heads,
                state.sealed_blocks,
                self.block_size,
                self.head_dim,
            )
            self._build_blocks(layer_idx, sealed_ids, sealed_keys)
        tail = incoming % self.block_size
        if tail:
            self.k_block_staging[layer_idx, :, :, :tail].copy_(
                key_states_roped[..., -tail:, :]
            )
        if layer_idx == self.num_layers - 1:
            self.kv_offset = incoming

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ) -> None:
        self._before_streaming_append(layer_idx)
        incoming = new_k_cache.shape[-2]
        state = self.block_state[layer_idx]
        start = state.total_tokens
        end = start + incoming
        if end > self.max_length:
            raise ValueError("streaming KV cache exceeds max_length")
        first_offset = start % self.block_size
        fits_tail = incoming <= self.block_size - first_offset
        fused_append = (
            self.offload
            and self.offload_backend in {"auto", "uva"}
            and fits_tail
            and new_k_cache.is_contiguous()
            and new_v_cache.is_contiguous()
        )
        if fused_append:
            append_kv_uva(
                new_k_cache,
                new_v_cache,
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                self.k_block_staging[layer_idx],
                destination_start=start,
                staging_start=first_offset,
            )
        else:
            self.k_cache[layer_idx, :, :, start:end].copy_(
                new_k_cache, non_blocking=self.offload
            )
            self.v_cache[layer_idx, :, :, start:end].copy_(
                new_v_cache, non_blocking=self.offload
            )
        if fits_tail:
            if not fused_append:
                self.k_block_staging[
                    layer_idx, :, :, first_offset : first_offset + incoming
                ].copy_(new_k_cache)
            combined_keys = None
        else:
            combined_keys = torch.cat(
                (
                    self.k_block_staging[layer_idx, :, :, :first_offset],
                    new_k_cache,
                ),
                dim=-2,
            )
        before_flush = state.pending_buffer
        transition = state.append(incoming)
        if transition.sealed:
            if fits_tail:
                sealed_keys = self.k_block_staging[layer_idx].unsqueeze(2)
                self._build_or_defer_blocks(
                    layer_idx, transition.sealed, sealed_keys
                )
            else:
                sealed_tokens = len(transition.sealed) * self.block_size
                sealed_keys = combined_keys[..., :sealed_tokens, :].reshape(
                    self.batch_size,
                    self.num_key_value_heads,
                    len(transition.sealed),
                    self.block_size,
                    self.head_dim,
                )
                self._build_or_defer_blocks(
                    layer_idx, transition.sealed, sealed_keys
                )
                remainder = combined_keys[..., sealed_tokens:, :]
                if remainder.shape[-2]:
                    self.k_block_staging[
                        layer_idx, :, :, : remainder.shape[-2]
                    ].copy_(remainder)
        # A flush has happened when the buffer wrapped back to (or through)
        # zero.  Everything queued since the last flush becomes retrievable on
        # this step, so its metadata must exist before selection runs -- which
        # it does, because layer_compute calls update before select.
        if state.pending_buffer <= before_flush and self._pending_block_builds[
            layer_idx
        ][0]:
            self._flush_block_builds(layer_idx)
        if layer_idx == self.num_layers - 1:
            self.kv_offset = end

    def _select_block_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        self.incoming_q_len = query_states.shape[-2]
        state = self.block_state[layer_idx]
        first, last = state.candidate_block_range
        available = last - first
        selected = min(available, self.select_blocks)
        if selected == 0:
            return torch.empty(
                self.batch_size,
                self.num_key_value_heads,
                0,
                device=self.compute_device,
                dtype=torch.long,
            )
        scores = self._score_blocks(
            layer_idx, query_states, first, last
        )
        relative = torch.topk(scores, k=selected, dim=-1).indices
        return relative + first

    def get_retrieval_position_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        block_ids = self._select_block_ids(layer_idx, query_states)
        self.pending_block_ids[layer_idx] = block_ids
        offsets = torch.arange(self.block_size, device=block_ids.device)
        return (
            block_ids.unsqueeze(-1) * self.block_size + offsets
        ).reshape(self.batch_size, self.num_key_value_heads, -1)

    def _can_reuse_selected_blocks(self) -> bool:
        # The CUDA reuse kernel assigns one thread block to one retrieval
        # unit.  Larger scientific block sizes remain supported by the
        # ordinary fused UVA gather instead of failing its launch constraint.
        vectors_per_token = self.head_dim * self.k_gather.element_size() // 16
        return (
            self.offload
            and self.offload_backend in {"auto", "uva"}
            and self.retrieval_unit * vectors_per_token <= 1024
        )

    def select_key_value_cache(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and gather without materializing unused token ids.

        The temporal UVA kernel consumes block ids directly.  Other backends
        and token-level routers retain the public position-id path.
        """
        if self._can_reuse_selected_blocks():
            unit_ids = (
                self._select_block_ids(layer_idx, query_states)
                if self.block_selection
                else self.get_retrieval_position_ids(layer_idx, query_states)
            )
            if unit_ids.shape[-1] == self.retrieval_slots:
                self.pending_block_ids[layer_idx] = unit_ids
                return self.get_key_value_cache(layer_idx, unit_ids)
            # During the short-context warm-up there may be fewer eligible
            # units than the configured budget.  Attend every eligible unit
            # exactly; the fixed-width temporal-reuse kernel starts only once
            # the candidate pool reaches its normal width.
            self.pending_block_ids[layer_idx] = None
            if self.block_selection:
                offsets = torch.arange(
                    self.block_size, device=unit_ids.device
                )
                unit_ids = (
                    unit_ids.unsqueeze(-1) * self.block_size + offsets
                ).reshape(self.batch_size, self.num_key_value_heads, -1)
            return self.get_key_value_cache(layer_idx, unit_ids)
        dynamic_ids = self.get_retrieval_position_ids(layer_idx, query_states)
        return self.get_key_value_cache(layer_idx, dynamic_ids)

    def _exact_position_ids(self, layer_idx: int) -> torch.Tensor:
        state = self.block_state[layer_idx]
        pieces = [
            torch.arange(start, end, device=self.compute_device)
            for start, end in state.exact_ranges
        ]
        if not pieces:
            return torch.empty(0, device=self.compute_device, dtype=torch.long)
        return torch.cat(pieces)

    def _gather(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        layer_idx: int,
        dynamic_ids: torch.Tensor,
    ) -> torch.Tensor:
        exact = self._exact_position_ids(layer_idx)
        if exact.numel():
            exact = exact.view(1, 1, -1).expand(
                self.batch_size, self.num_key_value_heads, -1
            )
            position_ids = torch.cat((dynamic_ids, exact), dim=-1)
        else:
            position_ids = dynamic_ids
        count = position_ids.shape[-1]
        if count > self.gather_capacity:
            raise RuntimeError("gather buffer capacity was underestimated")
        gathered = source[layer_idx].gather(
            -2,
            position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        destination[layer_idx, :, :, :count].copy_(gathered, non_blocking=True)
        self.last_attention_tokens = count
        return destination[layer_idx, :, :, :count]

    def get_value_cache(
        self, layer_idx: int, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._gather(
            self.v_cache, self.v_gather, layer_idx, position_ids
        )

    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        rope_func=None,
        cos_sin_cache=None,
    ) -> torch.Tensor:
        del rope_func, cos_sin_cache
        return self._gather(
            self.k_cache, self.k_gather, layer_idx, position_ids
        )

    def get_key_value_cache(
        self, layer_idx: int, dynamic_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K and V together, using one UVA launch when offloaded."""
        pending_blocks = self.pending_block_ids[layer_idx]
        if self._can_reuse_selected_blocks() and pending_blocks is not None:
            # Check capacity here rather than letting the kernel's own bound
            # check raise: the fallback below swallows kernel exceptions so a
            # machine without a compiler still runs, and an undersized buffer
            # would otherwise surface as an unrelated error one branch later.
            planned = (
                pending_blocks.shape[-1] * self.retrieval_unit
                + self.block_state[layer_idx].exact_tokens
            )
            if planned > self.gather_capacity:
                raise RuntimeError(
                    f"gather buffer holds {self.gather_capacity} tokens but "
                    f"this step needs {planned}: budget "
                    f"{self.sparse_budget} + prefix {self.prefix_tokens} + "
                    f"local {self.recent_tokens} + buffer up to "
                    f"{self.update_interval}"
                )
            current_bank = self.gather_bank[layer_idx]
            previous_bank = 1 - current_bank
            key_banks = (self.k_gather, self.k_gather_alt)
            value_banks = (self.v_gather, self.v_gather_alt)
            previous_ids = self.gather_block_ids[previous_bank, layer_idx]
            if not self.gather_reuse:
                previous_ids = torch.full_like(previous_ids, -1)
            if self.reuse_stats:
                matched = (
                    pending_blocks.unsqueeze(-1) == previous_ids.unsqueeze(-2)
                ).any(-1)
                self.reused_blocks += int(matched.sum())
                self.fetched_blocks += matched.numel()
            try:
                keys, values = gather_blocks_reuse_uva(
                    self.k_cache[layer_idx],
                    self.v_cache[layer_idx],
                    key_banks[previous_bank][layer_idx],
                    value_banks[previous_bank][layer_idx],
                    previous_ids,
                    pending_blocks.contiguous(),
                    key_banks[current_bank][layer_idx],
                    value_banks[current_bank][layer_idx],
                    self.gather_block_ids[current_bank, layer_idx],
                    block_size=self.retrieval_unit,
                    exact_ranges=self.block_state[layer_idx].exact_ranges,
                )
                self.gather_bank[layer_idx] = previous_bank
                self.pending_block_ids[layer_idx] = None
                self.last_attention_tokens = keys.shape[-2]
                return keys, values
            except Exception:
                if self.offload_backend == "uva":
                    raise
                # select_key_value_cache hands this method BLOCK ids when the
                # reuse kernel is eligible, because that kernel consumes them
                # directly.  The fallback below reads its argument as TOKEN
                # positions, so it must be expanded first.  Without this the
                # gather silently reads token `b` for every block `b` -- a
                # plausible-looking cache of wrong rows, not an error.
                if self.block_selection:
                    offsets = torch.arange(
                        self.block_size, device=pending_blocks.device
                    )
                    dynamic_ids = (
                        pending_blocks.unsqueeze(-1) * self.block_size
                        + offsets
                    ).reshape(self.batch_size, self.num_key_value_heads, -1)
                else:
                    dynamic_ids = pending_blocks
                self.pending_block_ids[layer_idx] = None

        if self.reuse_stats and dynamic_ids.numel():
            # Token-level routers (ParisKV, and ours under token refinement)
            # never reach the block kernel, so count what a reuse scheme would
            # have saved them: how much of this step's selection the previous
            # step already fetched.
            previous = self._previous_token_ids[layer_idx]
            if previous is not None and previous.shape == dynamic_ids.shape:
                matched = (
                    dynamic_ids.unsqueeze(-1) == previous.unsqueeze(-2)
                ).any(-1)
                self.reused_blocks += int(matched.sum())
                self.fetched_blocks += matched.numel()
            self._previous_token_ids[layer_idx] = dynamic_ids.clone()

        exact = self._exact_position_ids(layer_idx)
        if exact.numel():
            exact = exact.view(1, 1, -1).expand(
                self.batch_size, self.num_key_value_heads, -1
            )
            position_ids = torch.cat((dynamic_ids, exact), dim=-1)
        else:
            position_ids = dynamic_ids
        position_ids = position_ids.contiguous()
        count = position_ids.shape[-1]
        if count > self.gather_capacity:
            raise RuntimeError("gather buffer capacity was underestimated")
        if self.offload:
            keys, values = gather_kv(
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                position_ids,
                self.k_gather[layer_idx],
                self.v_gather[layer_idx],
                backend=self.offload_backend,
            )
        else:
            expanded = position_ids.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            )
            keys = self.k_cache[layer_idx].gather(-2, expanded)
            values = self.v_cache[layer_idx].gather(-2, expanded)
            self.k_gather[layer_idx, :, :, :count].copy_(keys)
            self.v_gather[layer_idx, :, :, :count].copy_(values)
            keys = self.k_gather[layer_idx, :, :, :count]
            values = self.v_gather[layer_idx, :, :, :count]
        self.last_attention_tokens = count
        self.pending_block_ids[layer_idx] = None
        return keys, values

    def get_dense_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.offload:
            raise RuntimeError("dense layers are incompatible with CPU-offloaded KV")
        end = self.block_state[layer_idx].total_tokens
        return (
            self.k_cache[layer_idx, :, :, :end],
            self.v_cache[layer_idx, :, :, :end],
        )
