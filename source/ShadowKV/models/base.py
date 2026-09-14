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

# Base LLM class

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from flash_attn import flash_attn_with_kvcache

from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU
from .streaming_cache import StreamingBlockCache
from .quest_streaming_cache import StreamingQuestCache
from .retroinfer_streaming_cache import StreamingRetroInferReferenceCache
from .adaptive_centroid_streaming_cache import StreamingAdaptiveCentroidLSECache
from .exact_block_streaming_cache import StreamingExactBlockOracleCache
from .exact_totalmass_rerank_cache import StreamingExactTotalMassRerankCache
from .pariskv_author_streaming_cache import StreamingParisKVAuthorCache
from .retroinfer_author_streaming_cache import StreamingRetroInferAuthorCache
from .routed_pq_cache import RoutedPQCache
from .quill_router_cache import QuillRouterShadowKVCache
from .quill_router import CanonicalQuillRouterScorer, KeyDiffRouterScorer
from .centroid_router_cache import CentroidLSERouterShadowKVCache
from .adaptive_centroid_cache import AdaptiveCentroidLSECache
from .infllm_author_cache import InfLLMAuthorCache
from .magicpig_author_cache import MagicPIGAuthorCache
from .pqcache_author_cache import PQCacheAuthorCache
from .pqcache_author_native_cache import PQCacheAuthorNativeCache

class LLM:

    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    def init_kv_cache(self, sparse_budget: int, rank: int, chunk_size: int, config,
                      page_size: int = 16, dense_layers: int = 2, group_reduce: str = 'max',
                      quest_prefix_tokens: int = 0, quest_recent_tokens: int = 0,
                      streaming_update_interval: int | None = None,
                      m51_anchors: str = None, m51_leaf: int = 8, m51_anchors_n: int = 8,
                      m51_coverage: float = 0.90, m51_eta_q: float = 0.999,
                      m51_pq_mode: str = 'offline', m51_pq_warm_iters: int = 2, m51_pq_codes: int = 64, m51_group_select: str = 'per_query',
                      m51_decode_mode: str = 'dense_mask', m51_load_batch: int = 128,
                      m51_pq_subdim: int = 8,
                      m51_select_mode: str = 'adaptive', m51_budget: int = 0,
                      m51_rank_anchors: int = 1, m51_rank_reduce: str = 'max',
                      quill_router_exact_fraction: float = 0.125,
                      quill_router_score_chunk: int = 1024,
                      router_aggregation: str = "max",
                      router_centroids: int = 2,
                      router_centroid_method: str = "kmeans",
                      router_split_fraction: float = 1.0,
                      router_calibration_tokens: int = 8,
                      self_lse_temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
                      streaming_offload: bool = False,
                      streaming_gather_backend: str = "auto",
                      streaming_router_backend: str = "torch",
                      streaming_refine_factor: float = 1.0,
                      streaming_refine_candidate_ratio: float = None,
                      streaming_refine_tokens: bool = False,
                      streaming_max_components: int = None,
                      streaming_compact_metadata: bool = False,
                      streaming_center_bits: int = 16,
                      retroinfer_prefix_tokens: int = 4,
                      retroinfer_recent_tokens: int = 64,
                      retroinfer_update_segment: int = 1024,
                      retroinfer_average_cluster_size: int = 16,
                      retroinfer_n_centroids: int = 0,
                      retroinfer_n_segment: int = 16,
                      retroinfer_estimation_ratio: float = 0.232,
                      retroinfer_kmeans_iters: int = 10,
                      pariskv_author_root: str = None,
                      retroinfer_author_root: str = None,
                      shadow_outlier_chunks: int = 48):
        if self.attn_mode == 'full':
            self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'm51':
            self.kv_cache = RoutedPQCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, anchors_path=m51_anchors, leaf_size=m51_leaf, n_anchors=m51_anchors_n, coverage=m51_coverage, eta_quantile=m51_eta_q, dense_layers=dense_layers, pq_mode=m51_pq_mode, pq_warm_iters=m51_pq_warm_iters, pq_codes=m51_pq_codes, group_select=m51_group_select, decode_mode=m51_decode_mode, load_batch=m51_load_batch, pq_subdim=m51_pq_subdim, select_mode=m51_select_mode, sparse_budget=m51_budget, rank_anchors=m51_rank_anchors, rank_reduce=m51_rank_reduce)
        elif self.attn_mode.lower() == 'quest_streaming':
            self.kv_cache = StreamingQuestCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, page_size=page_size,
                dense_layers=dense_layers, group_reduce=group_reduce,
                prefix_tokens=quest_prefix_tokens,
                recent_tokens=quest_recent_tokens,
                update_interval=streaming_update_interval,
                offload=streaming_offload,
                offload_backend=streaming_gather_backend,
            )
        elif self.attn_mode.lower() in {
            'exact_block_lse_streaming', 'exact_block_max_streaming',
            'exact_block_lse_softmax_streaming',
            'exact_block_max_softmax_streaming',
        }:
            self.kv_cache = StreamingExactTotalMassRerankCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, block_size=chunk_size,
                dense_layers=0, group_reduce=group_reduce,
                prefix_tokens=quest_prefix_tokens,
                recent_tokens=quest_recent_tokens,
                update_interval=streaming_update_interval,
                statistic=(
                    'logsumexp'
                    if '_lse_' in self.attn_mode.lower()
                    else 'max'
                ),
                normalize_blocks='softmax' in self.attn_mode.lower(),
                refine_factor=streaming_refine_factor,
                refine_tokens=streaming_refine_tokens,
                offload=streaming_offload,
                offload_backend=streaming_gather_backend,
            )
        elif self.attn_mode.lower() == 'retroinfer_reference_streaming':
            self.kv_cache = StreamingRetroInferReferenceCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget,
                prefix_tokens=retroinfer_prefix_tokens,
                recent_tokens=retroinfer_recent_tokens,
                update_segment=retroinfer_update_segment,
                average_cluster_size=retroinfer_average_cluster_size,
                estimation_ratio=retroinfer_estimation_ratio,
                kmeans_iters=retroinfer_kmeans_iters,
            )
        elif self.attn_mode.lower() == 'pariskv_author_common':
            self.kv_cache = StreamingParisKVAuthorCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, block_size=chunk_size,
                dense_layers=0, group_reduce='max',
                prefix_tokens=quest_prefix_tokens,
                recent_tokens=quest_recent_tokens,
                update_interval=streaming_update_interval,
                offload=streaming_offload,
                offload_backend=streaming_gather_backend,
                author_root=pariskv_author_root,
            )
        elif self.attn_mode.lower() == 'retroinfer_author_common':
            self.kv_cache = StreamingRetroInferAuthorCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, block_size=chunk_size,
                dense_layers=0, group_reduce='max',
                prefix_tokens=quest_prefix_tokens,
                recent_tokens=quest_recent_tokens,
                update_interval=streaming_update_interval,
                offload=streaming_offload,
                offload_backend=streaming_gather_backend,
                author_root=retroinfer_author_root,
                average_cluster_size=retroinfer_average_cluster_size,
                n_centroids=retroinfer_n_centroids,
                n_segment=retroinfer_n_segment,
                estimation_ratio=retroinfer_estimation_ratio,
                kmeans_iters=retroinfer_kmeans_iters,
            )
        elif self.attn_mode.lower() == 'infllm_author_common':
            self.kv_cache = InfLLMAuthorCache(
                config,
                sparse_budget=sparse_budget,
                batch_size=self.batch_size,
                device=self.device,
            )
        elif self.attn_mode.lower() == 'magicpig_author_common':
            self.kv_cache = MagicPIGAuthorCache(
                config,
                max_length=self.max_length,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.attn_mode.lower() == 'pqcache_author_common':
            self.kv_cache = PQCacheAuthorCache(
                config,
                max_length=self.max_length,
                sparse_budget=sparse_budget,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
                offload=streaming_offload,
            )
        elif self.attn_mode.lower() == 'pqcache_author_native':
            self.kv_cache = PQCacheAuthorNativeCache(
                config,
                max_length=self.max_length,
                sparse_budget=sparse_budget,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, outlier_chunk=shadow_outlier_chunks)
        elif self.attn_mode.lower() == 'shadowkv_quill':
            self.kv_cache = QuillRouterShadowKVCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
                exact_fraction=quill_router_exact_fraction,
                outlier_chunk=shadow_outlier_chunks,
                aggregation=router_aggregation,
            )
            self.router_scorer = CanonicalQuillRouterScorer(
                score_chunk=quill_router_score_chunk,
                exact_fraction=quill_router_exact_fraction,
            )
        elif self.attn_mode.lower() == 'shadowkv_keydiff':
            self.kv_cache = QuillRouterShadowKVCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
                exact_fraction=quill_router_exact_fraction,
                outlier_chunk=shadow_outlier_chunks,
                score_name="KEYDIFF",
                aggregation=router_aggregation,
            )
            self.router_scorer = KeyDiffRouterScorer(
                score_chunk=quill_router_score_chunk,
                exact_fraction=quill_router_exact_fraction,
            )
        elif self.attn_mode.lower() == 'shadowkv_centroid_lse':
            self.kv_cache = CentroidLSERouterShadowKVCache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
                outlier_chunk=shadow_outlier_chunks,
                n_centroids=router_centroids,
                centroid_method=router_centroid_method,
                split_fraction=router_split_fraction,
                calibration_tokens=router_calibration_tokens,
                self_lse_temperatures=self_lse_temperatures,
            )
        elif self.attn_mode.lower() in {
            'adaptive_centroid_lse', 'adaptive_centroid_lse_prefix4'
        }:
            self.kv_cache = AdaptiveCentroidLSECache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
                n_centroids=router_centroids,
                centroid_method=router_centroid_method,
                split_fraction=router_split_fraction,
                calibration_tokens=router_calibration_tokens,
                self_lse_temperatures=self_lse_temperatures,
                prefix_chunks=(
                    4 if self.attn_mode.lower() == 'adaptive_centroid_lse_prefix4'
                    else 0
                ),
            )
        elif self.attn_mode.lower() in {
            'adaptive_centroid_lse_streaming',
            'adaptive_centroid_lse_streaming_prefix4',
            'adaptive_centroid_lse_streaming_prefix4_querymean',
        }:
            self.kv_cache = StreamingAdaptiveCentroidLSECache(
                config, max_length=self.max_length, device=self.device,
                dtype=self.dtype, batch_size=self.batch_size,
                sparse_budget=sparse_budget, block_size=chunk_size,
                dense_layers=0, group_reduce=group_reduce,
                prefix_tokens=(
                    quest_prefix_tokens
                    if quest_prefix_tokens > 0 else
                    (4 * chunk_size
                     if '_prefix4' in self.attn_mode.lower() else 0)
                ),
                recent_tokens=quest_recent_tokens,
                update_interval=streaming_update_interval,
                extra_fraction=router_split_fraction,
                self_lse_temperatures=self_lse_temperatures,
                query_group_mean=self.attn_mode.lower().endswith('_querymean'),
                router_backend=streaming_router_backend,
                refine_factor=streaming_refine_factor,
                refine_candidate_ratio=streaming_refine_candidate_ratio,
                refine_tokens=streaming_refine_tokens,
                max_components=streaming_max_components,
                compact_metadata=streaming_compact_metadata,
                center_bits=streaming_center_bits,
                offload=streaming_offload,
                offload_backend=streaming_gather_backend,
            )
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, outlier_chunk=shadow_outlier_chunks)
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        begin_decode_step = getattr(self.kv_cache, "begin_decode_step", None)
        if input_ids.shape[-1] == 1 and begin_decode_step is not None:
            begin_decode_step()

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, InfLLMAuthorCache):
            # InfLLM owns both its local/global RoPE views and its multi-stage
            # attention.  Passing pre-RoPE Q/K is required by the author code.
            hidden_states = self.kv_cache.attend(
                layer_idx, query_states, key_states, value_states
            )

        elif isinstance(self.kv_cache, MagicPIGAuthorCache):
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )
            if q_len > 1:
                self.kv_cache.prefill_layer(
                    layer_idx, key_states, value_states
                )
                hidden_states = flash_attn_with_kvcache(
                    q=query_states.transpose(1, 2),
                    k_cache=key_states.transpose(1, 2),
                    v_cache=value_states.transpose(1, 2),
                    causal=True,
                )
            else:
                hidden_states = self.kv_cache.decode_attend(
                    layer_idx, query_states, key_states, value_states
                )

        elif isinstance(self.kv_cache, PQCacheAuthorCache):
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )
            if q_len > 1:
                self.kv_cache.prefill_layer(
                    layer_idx, key_states, value_states
                )
                hidden_states = flash_attn_with_kvcache(
                    q=query_states.transpose(1, 2),
                    k_cache=key_states.transpose(1, 2),
                    v_cache=value_states.transpose(1, 2),
                    causal=True,
                )
            else:
                self.kv_cache.update_kv_cache(
                    key_states, value_states, layer_idx
                )
                hidden_states = self.kv_cache.decode_attend(
                    layer_idx, query_states
                )

        elif isinstance(self.kv_cache, PQCacheAuthorNativeCache):
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )
            if q_len > 1:
                hidden_states = self.kv_cache.prefill_attend(
                    layer_idx, query_states, key_states, value_states
                )
            else:
                hidden_states = self.kv_cache.decode_attend(
                    layer_idx, query_states, key_states, value_states
                )

        elif isinstance(self.kv_cache, RoutedPQCache):

            # Generation enters this method with exactly one token.  RULER's
            # nominal 4K examples reserve room for the answer and are therefore
            # slightly shorter than 4096 tokens; a 4K threshold misclassified
            # their first full-context pass as decode.
            if q_len > 1: # prefill
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
                # selection and attention are one pass: the stopping rule needs
                # the exact mass of what it has already loaded
                hidden_states = self.kv_cache.decode_attend(layer_idx, query_states)

        elif isinstance(self.kv_cache, StreamingRetroInferReferenceCache):

            if q_len > 1: # prefill
                query_states, key_states = self.apply_rotary_pos_emb(
                    query_states, key_states, position_ids
                )
                self.kv_cache.prefill_kv_cache(
                    value_states, layer_idx, key_states, query_states[:, :, -1:]
                )
                hidden_states = flash_attn_with_kvcache(
                    q=query_states.transpose(1, 2),
                    k_cache=key_states.transpose(1, 2),
                    v_cache=value_states.transpose(1, 2), causal=True,
                )
            else: # decode
                query_states, key_states = self.apply_rotary_pos_emb(
                    query_states, key_states, position_ids
                )
                self.kv_cache.update_kv_cache(
                    key_states, value_states, layer_idx
                )
                hidden_states = self.kv_cache.decode_attend(
                    layer_idx, query_states
                )

        elif isinstance(self.kv_cache, StreamingBlockCache):

            if q_len > 1: # prefill
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                prepare_query = getattr(
                    self.kv_cache, "prepare_prefill_query", None
                )
                router_query = (
                    prepare_query(query_states)
                    if prepare_query is not None
                    else query_states[:, :, -1:]
                )
                self.kv_cache.prefill_kv_cache(
                    value_states, layer_idx, key_states, router_query
                )
                hidden_states = flash_attn_with_kvcache(
                    q=query_states.transpose(1, 2),
                    k_cache=key_states.transpose(1, 2),
                    v_cache=value_states.transpose(1, 2), causal=True,
                )

            else: # decode
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
                if self.kv_cache.is_dense_layer(layer_idx):
                    key_states, value_states = self.kv_cache.get_dense_cache(layer_idx)
                elif getattr(self.kv_cache, "head_alloc_tau", None) is not None:
                    self.kv_cache.set_head_output_projection(
                        layer_idx, buffer.wo
                    )
                    hidden_states = self.kv_cache.decode_attend_ragged(
                        layer_idx, query_states
                    )
                    key_states = value_states = None
                else:
                    key_states, value_states = (
                        self.kv_cache.select_key_value_cache(
                            layer_idx, query_states
                        )
                    )
                if key_states is not None:
                    # A method whose attention is not a single softmax over the
                    # gathered rows -- RetroInfer, which carries an estimation
                    # zone weighted by cluster size -- owns this call.  Every
                    # other cache leaves the hook undefined and pays nothing.
                    attend = getattr(
                        self.kv_cache, "decode_attend_gathered", None
                    )
                    if attend is None:
                        hidden_states = flash_attn_with_kvcache(
                            q=query_states.transpose(1, 2),
                            k_cache=key_states.transpose(1, 2),
                            v_cache=value_states.transpose(1, 2), causal=True,
                        )
                    else:
                        hidden_states = attend(
                            layer_idx, query_states, key_states, value_states
                        )

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU):

            if q_len > 1: # prefill
                query_states_prerope = query_states
                key_states_prerope = key_states
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                if (
                    isinstance(self.kv_cache, ShadowKVCache_CPU)
                    and q_len <= self.kv_cache.sparse_budget
                ):
                    # Before a generation-dominant benchmark grows beyond the
                    # sparse budget, the exact answer is simply every token.
                    # Avoid constructing an impossible rank-r SVD of a prompt
                    # shorter than r; the CPU cache keeps an exact warm-up.
                    self.kv_cache.prefill_dense_warmup(
                        value_states, layer_idx, key_states
                    )
                elif isinstance(self.kv_cache, QuillRouterShadowKVCache):
                    # SVD consumes pre-RoPE keys.
                    self.kv_cache.get_svd(
                        key_states_prerope, layer_idx=layer_idx
                    )
                    quill_exact_mask = self.router_scorer.exact_mask(
                        layer_idx=layer_idx,
                        queries_prerope=query_states_prerope,
                        keys_postrope=key_states,
                        values=value_states,
                        position_ids=position_ids,
                        cos_cache=self.cos_cache,
                        sin_cache=self.sin_cache,
                        gamma_store=self.kv_cache.quill_gamma,
                    )
                    self.kv_cache.prefill_kv_cache(
                        value_states, layer_idx, key_states,
                        query_states[:, :, -1:],
                        quill_exact_mask=quill_exact_mask,
                    )
                else:
                    # SVD consumes pre-RoPE keys.
                    self.kv_cache.get_svd(
                        key_states_prerope, layer_idx=layer_idx
                    )
                    cache_query = (
                        query_states
                        if isinstance(self.kv_cache, CentroidLSERouterShadowKVCache)
                        and self.kv_cache.centroid_method == "direct_lse2"
                        else query_states[:, :, -1:]
                    )
                    self.kv_cache.prefill_kv_cache(
                        value_states, layer_idx, key_states, cache_query
                    )
                
                if self.minference == True:
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode
                # rope query and key
                key_states_prerope = key_states
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # update kv cache to buffer
                if type(self.kv_cache) in (ShadowKVCache, ShadowKVCache_CPU):
                    self.kv_cache.update_kv_cache(
                        key_states, value_states, layer_idx,
                        key_states_prerope=key_states_prerope,
                    )
                else:
                    self.kv_cache.update_kv_cache(
                        key_states, value_states, layer_idx
                    )

                if (
                    isinstance(self.kv_cache, ShadowKVCache_CPU)
                    and self.kv_cache.dense_warmup[layer_idx]
                ):
                    key_states, value_states = self.kv_cache.get_dense_cache(
                        layer_idx
                    )
                    hidden_states = flash_attn_with_kvcache(
                        q=query_states.transpose(1, 2),
                        k_cache=key_states.transpose(1, 2),
                        v_cache=value_states.transpose(1, 2),
                        causal=True,
                    )
                    position_ids = None
                else:
                    # get retrieval idx
                    position_ids = self.kv_cache.get_retrieval_position_ids(
                        layer_idx=layer_idx, query_states=query_states
                    )

                if position_ids is not None:
                    # multi-stream
                    curr_stream = torch.cuda.current_stream()
                    get_value_stream = self.kv_cache.copy_stream

                    with torch.cuda.stream(get_value_stream):
                        get_value_stream.wait_stream(curr_stream)
                        value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                    # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
                    key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)

                    curr_stream.wait_stream(get_value_stream)

                    # flash attention
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        # Q/K/V are no longer needed once attention has produced its output.
        # Releasing the Python references before the output projection avoids
        # overlapping the long-prefill attention buffers with Qwen's large MLP
        # temporaries.  This is especially important for exact 64K attention
        # on a 24 GB card and does not alter any computation.
        del query_states, key_states, value_states

        hidden_states = hidden_states.reshape(bsz, q_len, getattr(self, 'attn_output_size', self.hidden_size))
        
        # Qwen3-4B's MLP widens 2560 -> 9728, so one untiled prefill temporary
        # is 1.18 GiB at 64K -- and there are several live at once.  Dense
        # attention survives that on a 24 GB card; a sparse cache does not,
        # because it holds the same full KV *plus* its index and gather
        # buffers.  The threshold used to be 64K, which meant a 64902-token
        # prompt ran untiled and OOMed every sparse method while full
        # attention passed.  Tiling is row-independent, so this changes peak
        # memory only.
        if bsz*q_len > 8*1024: # [bsz, seq, 128]
            # Qwen3's concatenated attention width can differ from hidden_size
            # (4096 vs 2560 for the 4B model).  post_attention_compute applies
            # o_proj and therefore returns the residual/hidden width.
            output = torch.empty_like(residual)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """accuracy eval usage, not for throughput eval"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """throughput eval usage"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())

        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)
