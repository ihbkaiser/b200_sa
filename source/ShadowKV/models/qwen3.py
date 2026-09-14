################################################################################
#
# Qwen3 support for ShadowKV.
#
# Modelled directly on models/qwen.py (Qwen2). Three things differ in Qwen3 and
# each one is fatal if carried over from the Qwen2 path:
#
#   1. head_dim is stated explicitly in the config and is NOT
#      hidden_size // num_attention_heads. Qwen3-4B: 2560 // 32 = 80, but the
#      real head_dim is 128, so num_heads * head_dim (4096) also differs from
#      hidden_size (2560) -- o_proj is [2560, 4096], not square.
#   2. q and k are RMSNorm'd per head (q_norm / k_norm over head_dim) after the
#      projection and before RoPE.
#   3. q/k/v projections carry no bias (attention_bias=false).
#
################################################################################

import torch
import torch.nn.functional as F
import gc

import transformers
from transformers import Qwen3ForCausalLM, Qwen3Config, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single
from .compat import get_inv_freq, head_dim_of
from .prompt_template import Templates, Chat_Templates
from .base import LLM


class Qwen3Layer:
    def __init__(self, layer_idx) -> None:

        self.wq: torch.Tensor = None
        self.wk: torch.Tensor = None
        self.wv: torch.Tensor = None
        self.wo: torch.Tensor = None

        # per-head RMSNorm on q and k (Qwen3 only)
        self.q_norm_weight: torch.Tensor = None
        self.q_norm_variance_epsilon: float = 0.0
        self.k_norm_weight: torch.Tensor = None
        self.k_norm_variance_epsilon: float = 0.0

        self.gate_proj: torch.Tensor = None
        self.up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: Qwen3DecoderLayer):

        self.wq = hf_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_layer.self_attn.v_proj.weight.detach()
        self.wo = hf_layer.self_attn.o_proj.weight.detach()

        assert hf_layer.self_attn.q_proj.bias is None, "Qwen3 is expected to have no attention bias"

        self.q_norm_weight = hf_layer.self_attn.q_norm.weight.detach()
        self.q_norm_variance_epsilon = hf_layer.self_attn.q_norm.variance_epsilon
        self.k_norm_weight = hf_layer.self_attn.k_norm.weight.detach()
        self.k_norm_variance_epsilon = hf_layer.self_attn.k_norm.variance_epsilon

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def init_gpu(self, device: str = 'cuda:0'):
        for name in ('input_layernorm_weight', 'post_attention_layernorm_weight',
                     'q_norm_weight', 'k_norm_weight',
                     'wq', 'wk', 'wv', 'wo', 'gate_proj', 'up_proj', 'down_proj'):
            setattr(self, name, getattr(self, name).to(device, non_blocking=True))


class Qwen3(LLM):
    def __init__(self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        batch_size: int = 1,
        max_length: int = 64*1024,
        device: str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        sparse_budget: int = 2048,
        rank=160,
        chunk_size=8,
        minference=False,
        page_size=16,
        dense_layers=2,
        group_reduce='max',
        quest_prefix_tokens=0,
        quest_recent_tokens=0,
        streaming_update_interval=None,
        m51_anchors=None,
        m51_leaf=8,
        m51_anchors_n=8,
        m51_coverage=0.90,
        m51_eta_q=0.999,
        m51_pq_mode='offline',
        m51_pq_warm_iters=2,
        m51_pq_codes=64,
        m51_group_select='per_query',
        m51_decode_mode='dense_mask',
        m51_load_batch=128,
        m51_pq_subdim=8,
        m51_select_mode='adaptive',
        m51_budget=0,
        m51_rank_anchors=1,
        m51_rank_reduce='max',
        quill_router_exact_fraction=0.125,
        quill_router_score_chunk=1024,
        router_aggregation="max",
        router_centroids=2,
        router_centroid_method="kmeans",
        router_split_fraction=1.0,
        router_calibration_tokens=8,
        self_lse_temperatures=(1.0, 1.5, 2.0),
        streaming_offload=False,
        streaming_gather_backend="auto",
        streaming_router_backend="torch",
        streaming_refine_factor=1.0,
        streaming_refine_candidate_ratio=None,
        streaming_refine_tokens=False,
        streaming_max_components=None,
        streaming_compact_metadata=False,
        streaming_center_bits=16,
        retroinfer_prefix_tokens=4,
        retroinfer_recent_tokens=64,
        retroinfer_update_segment=1024,
        retroinfer_average_cluster_size=16,
        retroinfer_estimation_ratio=0.232,
        retroinfer_kmeans_iters=10,
        pariskv_author_root=None,
        retroinfer_author_root=None,
        retroinfer_n_centroids=0,
        retroinfer_n_segment=16,
        shadow_outlier_chunks=48) -> None:

        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen3Config.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = head_dim_of(self.config)
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.vocab_size = self.config.vocab_size
        # o_proj input width; differs from hidden_size on Qwen3
        self.attn_output_size = self.num_heads * self.head_dim

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference
        assert not minference, "MInference patterns are not published for Qwen3"

        self.ctx_template = Templates['qwen']
        self.chat_template = Chat_Templates['qwen']

        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config,
                           page_size=page_size, dense_layers=dense_layers, group_reduce=group_reduce,
                           quest_prefix_tokens=quest_prefix_tokens, quest_recent_tokens=quest_recent_tokens,
                           streaming_update_interval=streaming_update_interval,
                           m51_anchors=m51_anchors, m51_leaf=m51_leaf, m51_anchors_n=m51_anchors_n,
                           m51_coverage=m51_coverage, m51_eta_q=m51_eta_q,
                           m51_pq_mode=m51_pq_mode, m51_pq_warm_iters=m51_pq_warm_iters, m51_pq_codes=m51_pq_codes, m51_group_select=m51_group_select,
                           m51_decode_mode=m51_decode_mode, m51_load_batch=m51_load_batch,
                           m51_pq_subdim=m51_pq_subdim,
                           m51_select_mode=m51_select_mode, m51_budget=m51_budget,
                           m51_rank_anchors=m51_rank_anchors, m51_rank_reduce=m51_rank_reduce,
                           quill_router_exact_fraction=quill_router_exact_fraction,
                           quill_router_score_chunk=quill_router_score_chunk,
                           router_aggregation=router_aggregation,
                           router_centroids=router_centroids,
                           router_centroid_method=router_centroid_method,
                           router_split_fraction=router_split_fraction,
                           router_calibration_tokens=router_calibration_tokens,
                           self_lse_temperatures=self_lse_temperatures,
                           streaming_offload=streaming_offload,
                           streaming_gather_backend=streaming_gather_backend,
                           streaming_router_backend=streaming_router_backend,
                           streaming_refine_factor=streaming_refine_factor,
                           streaming_refine_candidate_ratio=streaming_refine_candidate_ratio,
                           streaming_refine_tokens=streaming_refine_tokens,
                           streaming_max_components=streaming_max_components,
                           streaming_compact_metadata=streaming_compact_metadata,
                           streaming_center_bits=streaming_center_bits,
                           retroinfer_prefix_tokens=retroinfer_prefix_tokens,
                           retroinfer_recent_tokens=retroinfer_recent_tokens,
                           retroinfer_update_segment=retroinfer_update_segment,
                           retroinfer_average_cluster_size=retroinfer_average_cluster_size,
                           retroinfer_estimation_ratio=retroinfer_estimation_ratio,
                           retroinfer_kmeans_iters=retroinfer_kmeans_iters,
                           pariskv_author_root=pariskv_author_root,
                           retroinfer_author_root=retroinfer_author_root,
                           retroinfer_n_centroids=retroinfer_n_centroids,
                           retroinfer_n_segment=retroinfer_n_segment,
                           shadow_outlier_chunks=shadow_outlier_chunks)

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):
        hf_model = Qwen3ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(get_inv_freq(hf_model).to(self.device))
        # packed layout the ShadowKV CUDA rope kernel expects: [cos[:half] | sin[:half]]
        half_dim = self.head_dim // 2
        self.cos_sin_cache = torch.cat((self.cos_cache[:, :half_dim], self.sin_cache[:, :half_dim]), dim=-1)

        self.layers: list[Qwen3Layer] = []
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen3Layer,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int
    ):
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()

        query_states = F.linear(hidden_states, buffer.wq).view(bsz, q_len, num_heads, head_dim)
        key_states = F.linear(hidden_states, buffer.wk).view(bsz, q_len, num_key_value_heads, head_dim)
        value_states = F.linear(hidden_states, buffer.wv).view(bsz, q_len, num_key_value_heads, head_dim)

        # Qwen3: RMSNorm each head vector before RoPE
        query_states = layer_norm(query_states, buffer.q_norm_variance_epsilon, buffer.q_norm_weight)
        key_states = layer_norm(key_states, buffer.k_norm_variance_epsilon, buffer.k_norm_weight)

        return (query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2))

    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen3Layer
    ):
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.silu(F.linear(hidden_states, buffer.gate_proj))
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(q, k, self.cos_cache, self.sin_cache, position_ids)
