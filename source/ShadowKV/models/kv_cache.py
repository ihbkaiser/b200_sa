################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import torch

from .compat import head_dim_of
import math
import os
import gc
from torch import nn
from models.tensor_op import batch_gather_gemm_rotary_pos_emb_cuda
from kernels import shadowkv
from .offload_gather import gather_blocks_reuse_values_uva

class KV_Cache:
    """Full Attention"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            head_dim_of(config),
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            head_dim_of(config),
            device='cpu',
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

        # batch prefill record
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        if incoming > 1: # prefill
            key = key.to(self.device)
            value = value.to(self.device)

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        print(f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset

class ShadowKVCache:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        outlier_chunk: int = 48,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = head_dim_of(config)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        # Streaming extension of the authors' cache.  Keep a rolling
        # exact suffix and append aged decode tokens to the existing ShadowKV
        # landmark/SVD index once per update wave.  For the fair-metadata
        # chunk=4 setup this means two new landmarks every eight tokens.
        # The shared frame: an exact local window plus a batch flush. Both are
        # read from the environment so a campaign sets them once for every
        # method rather than per class.
        self.recent_tokens = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        self.update_tokens = int(
            os.environ.get("STREAMING_UPDATE_INTERVAL", "0")
        ) or self.chunk_size
        if self.recent_tokens % self.chunk_size:
            raise ValueError("ShadowKV recent window must use whole chunks")
        if self.update_tokens % self.chunk_size:
            raise ValueError("ShadowKV update wave must use whole chunks")
        # The authors reserve a fixed 128-token generation tail.  The flush
        # buffer now holds up to update_tokens-1 tokens before it drains, so
        # the tail has to be at least that wide or the append runs off the end.
        # Prompt-suffix normalisation also leaves up to chunk_size-1 extra
        # local tokens (target_local = recent_tokens + prompt % chunk_size).
        self.generation_headroom = (
            max(128, self.update_tokens) + self.chunk_size
        )
        self.local_chunk = self.recent_tokens // self.chunk_size
        if outlier_chunk < 0:
            raise ValueError("outlier_chunk must be non-negative")
        self.outlier_chunk = int(outlier_chunk)

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            head_dim_of(self.config),
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            head_dim_of(self.config),
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            head_dim_of(self.config),
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.copy_stream = torch.cuda.Stream()
        pending_shape = (
            self.num_layers, self.batch_size, self.num_key_value_heads,
            self.update_tokens, self.head_dim,
        )
        self._pending_k = torch.empty(
            pending_shape, device=self.device, dtype=self.dtype
        )
        self._pending_v = torch.empty_like(self._pending_k)
        self._pending_count = [0] * self.num_layers
        self._ring_write = [0] * self.num_layers
        self._ring_size = [self.recent_tokens] * self.num_layers
        self._next_evict = [0] * self.num_layers
        self._total_tokens = [0] * self.num_layers
        self._stream_landmarks = [None] * self.num_layers
        self._stream_landmark_ids = [None] * self.num_layers
        self._stream_landmark_count = [0] * self.num_layers

    def print_stats(self):
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, self.max_length, self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx, :, :k_cache.shape[1]].copy_(
            u[:, :, :self.rank].to(self.dtype)
        ) # [bsz, prefill, rank]
        self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def _allocate_stream_pool(self, layer_idx, incoming):
        capacity = math.ceil(max(0, self.max_length - incoming) / self.chunk_size)
        self._stream_landmarks[layer_idx] = torch.empty(
            self.batch_size, self.num_key_value_heads, capacity, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self._stream_landmark_ids[layer_idx] = torch.empty(
            self.batch_size, self.num_key_value_heads, capacity,
            device=self.device, dtype=torch.long,
        )

    def _append_stream_landmarks(self, layer_idx, keys, first_position):
        if first_position % self.chunk_size or keys.shape[-2] % self.chunk_size:
            raise ValueError("ShadowKV stream update is not chunk aligned")
        count = keys.shape[-2] // self.chunk_size
        landmarks = keys.reshape(
            self.batch_size, self.num_key_value_heads, count,
            self.chunk_size, self.head_dim,
        ).mean(dim=-2)
        ids = torch.arange(
            first_position // self.chunk_size,
            first_position // self.chunk_size + count,
            device=self.device, dtype=torch.long,
        ).view(1, 1, count).expand(
            self.batch_size, self.num_key_value_heads, count
        )
        pool = self._stream_landmarks[layer_idx]
        pool_ids = self._stream_landmark_ids[layer_idx]
        start = self._stream_landmark_count[layer_idx]
        end = start + count
        if pool is None or pool_ids is None or end > pool.shape[-2]:
            raise RuntimeError("ShadowKV stream landmark capacity exceeded")
        pool[:, :, start:end].copy_(landmarks)
        pool_ids[:, :, start:end].copy_(ids)
        self._stream_landmark_count[layer_idx] = end

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        self._allocate_stream_pool(layer_idx, incoming)
        self._pending_count[layer_idx] = 0
        self._ring_write[layer_idx] = 0
        self._ring_size[layer_idx] = self.prefill_local
        self._next_evict[layer_idx] = incoming - self.prefill_local
        self._total_tokens[layer_idx] = incoming

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        count = self._stream_landmark_count[layer_idx]
        landmarks = self.k_landmark[layer_idx]
        landmark_ids = self.k_landmark_idx[layer_idx]
        if count:
            landmarks = torch.cat(
                (landmarks, self._stream_landmarks[layer_idx][:, :, :count]),
                dim=2,
            )
            landmark_ids = torch.cat(
                (landmark_ids,
                 self._stream_landmark_ids[layer_idx][:, :, :count]),
                dim=2,
            )
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), landmarks.transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = landmark_ids.gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        # this is chunk idx, which can be used to offload value cache and decide if the cache hits
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        # gather value cache
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        end = self.sparse_end + self._pending_count[layer_idx]
        return self.v_cache_buffer[layer_idx][:, :, :end]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        # gather key cache and rope them
        u = self.U[layer_idx] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, rank, 128]

        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids)

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        end = self.sparse_end + self._pending_count[layer_idx]
        return self.k_cache_buffer[layer_idx][:, :, :end]

    def _project_decode_keys(self, layer_idx, key_states_prerope, start):
        sv = self.SV[layer_idx].float().permute(0, 2, 1, 3).flatten(2)
        key = key_states_prerope.transpose(1, 2).reshape(
            self.batch_size, key_states_prerope.shape[-2], -1
        ).float()
        denominator = sv.square().sum(-1).clamp_min(1e-12)
        coefficient = torch.einsum("btd,brd->btr", key, sv)
        coefficient = coefficient / denominator[:, None, :]
        end = start + key_states_prerope.shape[-2]
        self.U[layer_idx, :, start:end].copy_(coefficient.to(self.dtype))

    def update_kv_cache(
            self, new_k_cache, new_v_cache, layer_idx,
            key_states_prerope=None):
        if key_states_prerope is None:
            raise ValueError("ShadowKV streaming needs pre-RoPE decode K")
        incoming = new_k_cache.shape[-2]
        start = self._total_tokens[layer_idx]
        end = start + incoming
        if end > self.max_length:
            raise ValueError("ShadowKV cache exceeds max_length")
        self._project_decode_keys(layer_idx, key_states_prerope, start)
        self.v_cache_cpu[layer_idx, :, :, start:end].copy_(new_v_cache)

        for offset in range(incoming):
            slot = self._ring_write[layer_idx]
            pending = self._pending_count[layer_idx]
            aged_k = self.k_cache_buffer[layer_idx, :, :, slot].clone()
            aged_v = self.v_cache_buffer[layer_idx, :, :, slot].clone()
            self._pending_k[layer_idx, :, :, pending].copy_(aged_k)
            self._pending_v[layer_idx, :, :, pending].copy_(aged_v)
            self.k_cache_buffer[
                layer_idx, :, :, self.sparse_end + pending
            ].copy_(aged_k)
            self.v_cache_buffer[
                layer_idx, :, :, self.sparse_end + pending
            ].copy_(aged_v)
            self.k_cache_buffer[layer_idx, :, :, slot].copy_(
                new_k_cache[:, :, offset]
            )
            self.v_cache_buffer[layer_idx, :, :, slot].copy_(
                new_v_cache[:, :, offset]
            )
            self._ring_write[layer_idx] = (slot + 1) % self._ring_size[layer_idx]
            pending += 1
            self._pending_count[layer_idx] = pending
            if pending == self.update_tokens:
                self._append_stream_landmarks(
                    layer_idx, self._pending_k[layer_idx],
                    self._next_evict[layer_idx],
                )
                self._next_evict[layer_idx] += self.update_tokens
                self._pending_count[layer_idx] = 0

        self._total_tokens[layer_idx] = end
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
        self._pending_count[:] = [0] * self.num_layers
        self._ring_write[:] = [0] * self.num_layers
        self._ring_size[:] = [self.recent_tokens] * self.num_layers
        self._next_evict[:] = [0] * self.num_layers
        self._total_tokens[:] = [0] * self.num_layers
        self._stream_landmark_count[:] = [0] * self.num_layers
        self._stream_landmarks[:] = [None] * self.num_layers
        self._stream_landmark_ids[:] = [None] * self.num_layers
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_CPU:
    """ShadowKV, can be used for Llama-3-8B, Llama-3.1-8B, GLM-4-9B, Yi-200K"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        outlier_chunk: int = 48,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = head_dim_of(config)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        # The authors' cached H2D/D2D gather kernels move one fixed 2 KiB
        # record per offset (8 BF16 tokens at head_dim=128).  Passing a
        # smaller chunk silently changes the logical record width while the
        # kernel continues to copy 2 KiB, corrupting both K and V.  Keep the
        # fast cached path for its native width and use the exact generic
        # pinned-CPU path for other widths.
        self._native_cached_gather = (
            self.chunk_size * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()
            == 2048
        )
        self.rank = rank
        # Extend the official CPU lifecycle to generated tokens: keep a
        # rolling exact suffix and seal aged tokens into the ShadowKV index in
        # fixed waves.  These are the shared frame's local window and batch
        # flush, so they must come from the campaign environment like every
        # other method's -- hard-coding them left this cache alone on the old
        # per-eight-token lifecycle while everything else ran the 256 frame.
        self.recent_tokens = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        self.update_tokens = int(
            os.environ.get("STREAMING_UPDATE_INTERVAL", "0")
        ) or self.chunk_size
        if self.recent_tokens % self.chunk_size:
            raise ValueError("ShadowKV recent window must use whole chunks")
        if self.update_tokens % self.chunk_size:
            raise ValueError("ShadowKV update wave must use whole chunks")
        # The authors reserve a fixed 128-token generation tail.  The flush
        # buffer now holds up to update_tokens-1 tokens before it drains, so
        # the tail has to be at least that wide or the append runs off the end.
        # Prompt-suffix normalisation also leaves up to chunk_size-1 extra
        # local tokens (target_local = recent_tokens + prompt % chunk_size).
        self.generation_headroom = (
            max(128, self.update_tokens) + self.chunk_size
        )
        self.local_chunk = self.recent_tokens // self.chunk_size
        if outlier_chunk < 0:
            raise ValueError("outlier_chunk must be non-negative")
        self.outlier_chunk = int(outlier_chunk)

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length // self.chunk_size,
            head_dim_of(self.config) * self.chunk_size,
            device='cpu',
            dtype=self.dtype,
            pin_memory=True
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + self.generation_headroom + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            head_dim_of(self.config),
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + self.generation_headroom + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            head_dim_of(self.config),
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()
        # Two GPU-resident value banks let the block kernel keep whatever the
        # previous step already fetched and pull only the replacements.
        stage_shape = (
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            self.sparse_budget,
            self.head_dim,
        )
        self.v_stage = torch.zeros(
            stage_shape, device=self.device, dtype=self.dtype
        )
        self.v_stage_alt = torch.zeros_like(self.v_stage)
        self.stage_ids = torch.full(
            (2, config.num_hidden_layers, batch_size,
             self.num_key_value_heads, self.sparse_budget // self.chunk_size),
            -1, device=self.device, dtype=torch.long,
        )
        self.gather_bank = [0] * config.num_hidden_layers
        self.reuse_stats = os.environ.get("STREAMING_REUSE_STATS", "0") == "1"
        self.reused_blocks = 0
        self.fetched_blocks = 0
        if self.reuse_stats:
            import atexit

            atexit.register(self._report_reuse)

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(self.sparse_budget // self.chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # k compute kernels
        self.output = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.sparse_budget,
            self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

        pending_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.update_tokens,
            self.head_dim,
        )
        self._pending_k = torch.empty(
            pending_shape, device=self.device, dtype=self.dtype
        )
        self._pending_v = torch.empty_like(self._pending_k)
        self._pending_count = [0] * self.num_layers
        self._ring_write = [0] * self.num_layers
        self._ring_size = [self.recent_tokens] * self.num_layers
        self._next_evict = [0] * self.num_layers
        self._total_tokens = [0] * self.num_layers
        self._stream_landmarks = [None] * self.num_layers
        self._stream_landmark_ids = [None] * self.num_layers
        self._stream_landmark_count = [0] * self.num_layers
        self._landmark_view = [None] * self.num_layers
        self._projection_cache = [None] * self.num_layers
        self.dense_warmup = [False] * self.num_layers

    def print_stats(self):
        print(
            f"ShadowKV_CPU | sparse budget {self.sparse_budget} | "
            f"chunk size {self.chunk_size} | rank {self.rank} | "
            f"cached {self.kv_offset} | rolling recent {self.recent_tokens} | "
            f"stream update {self.update_tokens} | outlier_chunk {self.outlier_chunk}"
        )

    ##### Encoding #####
    def prefill_dense_warmup(
            self, new_v_cache, layer_idx, key_states_roped):
        """Keep a short prompt exact until it grows to the sparse budget."""
        incoming = new_v_cache.shape[-2]
        if incoming > self.sparse_budget:
            raise ValueError("dense warm-up exceeds sparse budget")
        self.k_cache_buffer[layer_idx, :, :, :incoming].copy_(
            key_states_roped
        )
        self.v_cache_buffer[layer_idx, :, :, :incoming].copy_(new_v_cache)
        self._total_tokens[layer_idx] = incoming
        self.dense_warmup[layer_idx] = True
        if layer_idx == self.num_layers - 1:
            self.kv_offset = incoming

    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init U, SV
            # Reserve the generation suffix as well.  The original cache never
            # indexed decode keys, so prompt length was sufficient; the
            # streaming extension appends their coefficients in the frozen
            # prompt basis after they leave its rolling recent window.
            self.U = torch.zeros(self.num_layers, self.batch_size, self.max_length, self.rank, device='cpu', dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank, device='cpu', dtype=self.dtype)
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        
        bsz = k_cache.shape[0]
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx][
            self.prefilled_batch:self.prefilled_batch + bsz,
            :k_cache.shape[1],
        ].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, prefill, rank]
        
        temp_sv = torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [bsz, 8, 160, 128]

        # used for kernel
        temp_sv = temp_sv.transpose(-1, -2) # [bsz, 8, 128, 160]
        
        self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(temp_sv) # [bsz, 8, 128, 160]

        del u, s, v

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def _allocate_stream_pool(self, layer_idx, incoming):
        capacity = math.ceil(
            max(0, self.max_length - incoming + self.prefill_local)
            / self.chunk_size
        )
        self._stream_landmarks[layer_idx] = torch.empty(
            self.batch_size, self.num_key_value_heads, capacity, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self._stream_landmark_ids[layer_idx] = torch.empty(
            self.batch_size, self.num_key_value_heads, capacity,
            device=self.device, dtype=torch.long,
        )

    def _append_stream_landmarks(self, layer_idx, keys, first_position):
        if first_position % self.chunk_size or keys.shape[-2] % self.chunk_size:
            raise ValueError("ShadowKV stream update is not chunk aligned")
        pool = self._stream_landmarks[layer_idx]
        pool_ids = self._stream_landmark_ids[layer_idx]
        if pool is None or pool_ids is None:
            raise RuntimeError("ShadowKV stream pool is not initialized")
        count = keys.shape[-2] // self.chunk_size
        start = self._stream_landmark_count[layer_idx]
        end = start + count
        if end > pool.shape[-2]:
            raise RuntimeError("ShadowKV stream landmark capacity exceeded")
        landmarks = keys.reshape(
            self.batch_size, self.num_key_value_heads, count,
            self.chunk_size, self.head_dim,
        ).mean(dim=-2)
        ids = torch.arange(
            first_position // self.chunk_size,
            first_position // self.chunk_size + count,
            device=self.device, dtype=torch.long,
        ).view(1, 1, count).expand(
            self.batch_size, self.num_key_value_heads, count
        )
        pool[:, :, start:end].copy_(landmarks)
        pool_ids[:, :, start:end].copy_(ids)
        self._stream_landmark_count[layer_idx] = end

    def _initialize_streaming_layer(
            self, layer_idx, incoming, key_states_roped, new_v_cache):
        """Normalize the prompt suffix to recent_tokens and start the ring."""
        old_local = self.prefill_local
        target_local = self.recent_tokens + incoming % self.chunk_size
        if old_local < target_local:
            raise RuntimeError("ShadowKV prompt suffix is shorter than recent window")
        self._allocate_stream_pool(layer_idx, incoming)

        # Alignment to the authors' eight-landmark kernel can leave a few
        # extra prompt chunks local.  They are ordinary indexed chunks in the
        # streaming lifecycle, so append their landmarks and compact the
        # resident [recent, outlier, sparse] regions.
        excess = old_local - target_local
        if excess:
            self._append_stream_landmarks(
                layer_idx,
                key_states_roped[
                    ..., incoming - old_local:incoming - target_local, :
                ],
                incoming - old_local,
            )
            resident_k = self.k_cache_buffer[
                layer_idx, :, :, old_local:self.sparse_end
            ].clone()
            resident_v = self.v_cache_buffer[
                layer_idx, :, :, old_local:self.sparse_end
            ].clone()
            self.k_cache_buffer[
                layer_idx, :, :, :target_local
            ].copy_(key_states_roped[..., -target_local:, :])
            self.v_cache_buffer[
                layer_idx, :, :, :target_local
            ].copy_(new_v_cache[..., -target_local:, :])
            resident_end = target_local + resident_k.shape[-2]
            self.k_cache_buffer[
                layer_idx, :, :, target_local:resident_end
            ].copy_(resident_k)
            self.v_cache_buffer[
                layer_idx, :, :, target_local:resident_end
            ].copy_(resident_v)
            self.prefill_local = target_local
            self.sparse_start = (
                self.prefill_local + self.outlier_chunk * self.chunk_size
            )
            self.sparse_end = self.sparse_start + self.sparse_budget
            self.kernel_offset = self.sparse_start * self.head_dim
            self.kernel_stride = (
                self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim
            )

        self._pending_count[layer_idx] = 0
        self._ring_write[layer_idx] = 0
        self._ring_size[layer_idx] = self.prefill_local
        self._next_evict[layer_idx] = incoming - self.prefill_local
        self._total_tokens[layer_idx] = incoming

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):
        
        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        max_ctx_chunks = incoming // self.chunk_size
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size*self.head_dim), non_blocking=True) # [bsz, num_kv_heads, max_ctx_chunks, chunk_size*head_dim]

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        # ensure self.chunks is even
        self.chunks = self.chunks - self.chunks % 8
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget

        self.kernel_offset = self.sparse_start * self.head_dim
        self.kernel_stride = self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(bsz, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        # fill cache for the first time
        chunk_attn = torch.einsum('bhgd,bhcd->bhgc', last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim), self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)) / math.sqrt(self.head_dim) # [bsz, 8, 4, chunks]
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device).gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, self.num_key_value_heads, -1)
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        key_ = key_states_roped.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(key_, non_blocking=True)

        self._initialize_streaming_layer(
            layer_idx, incoming, key_states_roped, new_v_cache
        )

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            # self.kv_offset += incoming
            self.prefilled_batch += bsz

            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

                assert torch.any(self.position_ids == -1) == False, f"The cache for offloading is not built correctly, {self.position_ids}"

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2]
        count = self._stream_landmark_count[layer_idx]
        # Streamed landmarks live in their own pool, so scoring has to see the
        # prompt landmarks and the decode landmarks as one tensor.  Rebuilding
        # that concatenation every step copied the whole prompt index -- 16k
        # landmarks per layer at 64K/chunk 4 -- and measured 30% of decode GPU
        # time, twice what the PCIe gather costs.  The pool only grows at a
        # flush, so build it when the count changes and reuse it in between.
        cached = self._landmark_view[layer_idx]
        if cached is None or cached[0] != count:
            if count:
                landmarks = torch.cat(
                    (self.k_landmark[layer_idx],
                     self._stream_landmarks[layer_idx][:, :, :count]),
                    dim=2,
                )
                landmark_ids = torch.cat(
                    (self.k_landmark_idx[layer_idx],
                     self._stream_landmark_ids[layer_idx][:, :, :count]),
                    dim=2,
                )
            else:
                landmarks = self.k_landmark[layer_idx]
                landmark_ids = self.k_landmark_idx[layer_idx]
            self._landmark_view[layer_idx] = (count, landmarks, landmark_ids)
        _, landmarks, landmark_ids = self._landmark_view[layer_idx]
        query = query_states.reshape(
            self.batch_size, self.num_key_value_heads,
            self.num_key_value_groups, self.incoming_q_len, self.head_dim,
        )
        chunk_attn = torch.einsum("bhgtd,bhnd->bhgtn", query, landmarks)
        chunk_attn = torch.softmax(
            chunk_attn.float() / math.sqrt(self.head_dim), dim=-1
        ).amax(dim=(2, 3))
        merged_results = chunk_attn.topk(self.select_sets, dim=-1).indices
        selected_chunks = landmark_ids.gather(dim=-1, index=merged_results)
        self.position_ids[layer_idx].copy_(selected_chunks)
        # Streaming ids are not compatible with the authors' fixed prompt
        # cache-reuse offsets.  Reconstruct every selected chunk exactly as an
        # SVD record; zero counts marks every output slot as a miss.
        self.cnts.zero_()
        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):
        # The pinned store is [B,H,chunks,chunk*D], which is the same bytes in
        # the same order as [B,H,tokens,D]: chunk records hold their tokens
        # consecutively.  So the shared block kernel reads it directly with
        # block_size = chunk_size, and the authors' chunk ids are its block
        # ids.  That kernel matches ids on the GPU inside the launch, so it
        # reuses across steps even though our streaming ids no longer fit the
        # authors' prefill-anchored cache bookkeeping.
        source = self.v_cache_cpu[layer_idx].view(
            self.batch_size, self.num_key_value_heads,
            (self.max_length // self.chunk_size) * self.chunk_size,
            self.head_dim,
        )
        current = self.gather_bank[layer_idx]
        previous = 1 - current
        banks = (self.v_stage, self.v_stage_alt)
        if self.reuse_stats:
            matched = (
                position_ids.unsqueeze(-1)
                == self.stage_ids[previous, layer_idx].unsqueeze(-2)
            ).any(-1)
            self.reused_blocks += int(matched.sum())
            self.fetched_blocks += matched.numel()
        selected = gather_blocks_reuse_values_uva(
            source,
            banks[previous][layer_idx],
            self.stage_ids[previous, layer_idx],
            position_ids.contiguous(),
            banks[current][layer_idx],
            self.stage_ids[current, layer_idx],
            block_size=self.chunk_size,
        )
        self.gather_bank[layer_idx] = previous
        self.v_cache_buffer[layer_idx][
            :, :, self.sparse_start:self.sparse_end
        ].copy_(selected, non_blocking=True)
        end = self.sparse_end + self._pending_count[layer_idx]
        return self.v_cache_buffer[layer_idx][:, :, :end]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        batch_gather_gemm_rotary_pos_emb_cuda(
            self.U[layer_idx], self.SV[layer_idx], cos_sin_cache,
            position_ids, self.output, self.chunk_size,
            self.k_cache_buffer[layer_idx], self.sparse_start,
            self.sparse_end, self.cnts,
        )
        end = self.sparse_end + self._pending_count[layer_idx]
        return self.k_cache_buffer[layer_idx][:, :, :end]

    def H2D(self):
        if all(self.dense_warmup):
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.SV = self.SV.to(self.device)
        self.U = self.U.to(self.device)
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _projection_basis(self, layer_idx):
        # SV is diag(s)V^T, stored [B,H,D,R].  Its rank rows are orthogonal,
        # hence x A^T / ||A_r||^2 is the least-squares coefficient in the
        # frozen prompt basis.  SV does not change after prefill, so the
        # float copy and its row norms are constants -- rebuilding them per
        # generated token cast 164k elements 36 times a step for nothing.
        cached = self._projection_cache[layer_idx]
        if cached is None:
            sv = self.SV[layer_idx].float().permute(0, 3, 1, 2).flatten(2)
            denominator = sv.square().sum(-1).clamp_min(1e-12)
            cached = (sv, denominator)
            self._projection_cache[layer_idx] = cached
        return cached

    def _project_decode_keys(
            self, layer_idx, key_states_prerope, start):
        sv, denominator = self._projection_basis(layer_idx)
        key = key_states_prerope.transpose(1, 2).reshape(
            self.batch_size, key_states_prerope.shape[-2], -1
        ).float()
        coefficient = torch.einsum("btd,brd->btr", key, sv)
        coefficient = coefficient / denominator[:, None, :]
        end = start + key_states_prerope.shape[-2]
        self.U[layer_idx, :, start:end].copy_(coefficient.to(self.dtype))

    def update_kv_cache(
            self, new_k_cache, new_v_cache, layer_idx,
            key_states_prerope=None):
        if self.dense_warmup[layer_idx]:
            start = self._total_tokens[layer_idx]
            end = start + new_k_cache.shape[-2]
            if end > self.sparse_budget:
                raise RuntimeError(
                    "ShadowKV exact warm-up reached sparse_budget; dynamic "
                    "SVD transition is not implemented"
                )
            self.k_cache_buffer[layer_idx, :, :, start:end].copy_(new_k_cache)
            self.v_cache_buffer[layer_idx, :, :, start:end].copy_(new_v_cache)
            self._total_tokens[layer_idx] = end
            if layer_idx == self.num_layers - 1:
                self.kv_offset = end
            return
        if key_states_prerope is None:
            raise ValueError("ShadowKV CPU streaming needs pre-RoPE decode K")
        incoming = new_k_cache.shape[-2]
        start = self._total_tokens[layer_idx]
        end = start + incoming
        if end > self.max_length:
            raise ValueError("ShadowKV CPU cache exceeds max_length")
        # U is read only when a chunk is already in the index, and decode
        # chunks enter the index at a flush, so the projection can wait for
        # one and run over the whole wave -- the same batching the landmark
        # seal gets.  Stage the pre-RoPE keys until then.
        pending_before = self._pending_count[layer_idx]
        self._pending_k[
            layer_idx, :, :, pending_before:pending_before + incoming
        ].copy_(key_states_prerope)

        # Future sparse V retrieval reads the newly generated values from the
        # same pinned backing store as prompt values.
        value_tokens = self.v_cache_cpu[layer_idx].view(
            self.batch_size, self.num_key_value_heads, -1, self.head_dim
        )
        value_tokens[:, :, start:end].copy_(new_v_cache, non_blocking=True)

        # Append-only decode tail instead of a per-token ring.
        #
        # The ring rotated one token at a time: read the aged slot, stage it,
        # copy it to the exact tail, write the new token into the freed slot.
        # Eight indexed tensor expressions per token per layer, each costing
        # several dispatches -- measured ~12k dispatches per decode step with
        # the GPU only 29% busy, i.e. the step was bound by Python, not by the
        # card.
        #
        # Attention over already-RoPEd keys does not depend on their order in
        # the buffer, so the exact region can be: oldest `prefill_local` tokens
        # at the front, newest appended to the tail.  One slice copy per step,
        # and the eviction bookkeeping runs once per flush over a whole wave --
        # the same batching the shared frame gives every other method.
        pending = self._pending_count[layer_idx]
        if incoming > self.update_tokens - pending:
            raise ValueError(
                "ShadowKV decode append crosses more than one flush boundary"
            )
        tail = self.sparse_end + pending
        self.k_cache_buffer[layer_idx, :, :, tail:tail + incoming].copy_(
            new_k_cache
        )
        self.v_cache_buffer[layer_idx, :, :, tail:tail + incoming].copy_(
            new_v_cache
        )
        pending += incoming
        if pending == self.update_tokens:
            wave = self.update_tokens
            if wave > self.prefill_local:
                raise ValueError(
                    "ShadowKV flush wave is wider than the local window"
                )
            self._project_decode_keys(
                layer_idx,
                self._pending_k[layer_idx, :, :, :wave],
                end - wave,
            )
            self._append_stream_landmarks(
                layer_idx,
                self.k_cache_buffer[layer_idx, :, :, :wave],
                self._next_evict[layer_idx],
            )
            self._next_evict[layer_idx] += wave
            keep = self.prefill_local - wave
            if keep:
                self.k_cache_buffer[layer_idx, :, :, :keep].copy_(
                    self.k_cache_buffer[
                        layer_idx, :, :, wave:self.prefill_local
                    ].clone()
                )
                self.v_cache_buffer[layer_idx, :, :, :keep].copy_(
                    self.v_cache_buffer[
                        layer_idx, :, :, wave:self.prefill_local
                    ].clone()
                )
            self.k_cache_buffer[layer_idx, :, :, keep:self.prefill_local].copy_(
                self.k_cache_buffer[
                    layer_idx, :, :, self.sparse_end:self.sparse_end + wave
                ].clone()
            )
            self.v_cache_buffer[layer_idx, :, :, keep:self.prefill_local].copy_(
                self.v_cache_buffer[
                    layer_idx, :, :, self.sparse_end:self.sparse_end + wave
                ].clone()
            )
            pending = 0
        self._pending_count[layer_idx] = pending

        self._total_tokens[layer_idx] = end
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0
        self._pending_count[:] = [0] * self.num_layers
        self._ring_write[:] = [0] * self.num_layers
        self._ring_size[:] = [self.recent_tokens] * self.num_layers
        self._next_evict[:] = [0] * self.num_layers
        self._total_tokens[:] = [0] * self.num_layers
        self._stream_landmark_count[:] = [0] * self.num_layers
        self._stream_landmarks[:] = [None] * self.num_layers
        self._stream_landmark_ids[:] = [None] * self.num_layers
        self._landmark_view[:] = [None] * self.num_layers
        self._projection_cache[:] = [None] * self.num_layers
        self.dense_warmup[:] = [False] * self.num_layers
        # Stale ids would let the reuse kernel copy a previous sample's rows
        # for a block id that happens to repeat.  Invalidate both banks.
        self.stage_ids.fill_(-1)
        self.gather_bank[:] = [0] * self.num_layers

    def _report_reuse(self):
        if not self.fetched_blocks:
            return
        share = self.reused_blocks / self.fetched_blocks
        print(
            f"REUSE ShadowKVCache_CPU chunk {self.chunk_size} "
            f"reused {self.reused_blocks} of {self.fetched_blocks} "
            f"({share:.1%})"
        )

    def get_dense_cache(self, layer_idx):
        end = self._total_tokens[layer_idx]
        return (
            self.k_cache_buffer[layer_idx, :, :, :end],
            self.v_cache_buffer[layer_idx, :, :, :end],
        )

    def get_kv_len(self):
        return self.kv_offset
