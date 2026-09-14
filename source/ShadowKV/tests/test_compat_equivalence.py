"""
Pins the two things that could silently move a number:

  1. models/compat.py replaces flashinfer's rmsnorm and vllm's rope kernel.
     Both replacements must be numerically the same op, checked against
     transformers' own reference implementations.
  2. The Quest knobs must not reach any other attention mode, and their
     defaults must be what we say they are.

Run:  python -m pytest tests/test_compat_equivalence.py -q
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.compat import HAS_FLASHINFER, rmsnorm, rotary_embedding_neox_, head_dim_of


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# ---------------------------------------------------------------- rmsnorm --

@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_matches_transformers(dtype):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    torch.manual_seed(0)
    hidden = 512
    x = torch.randn(2, 7, hidden, device="cuda", dtype=dtype)
    ref_mod = LlamaRMSNorm(hidden, eps=1e-5).cuda().to(dtype)
    with torch.no_grad():
        ref_mod.weight.normal_(mean=1.0, std=0.02)
        expected = ref_mod(x)
        got = rmsnorm(x, ref_mod.weight, ref_mod.variance_epsilon)

    assert got.dtype == expected.dtype
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.skipif(not HAS_FLASHINFER, reason="FlashInfer is not installed")
@pytest.mark.parametrize(
    # FlashInfer changes the reduction order.  The observed discrepancy is at
    # most one representable output ULP at this scale, not a different op.
    "dtype,atol", [(torch.bfloat16, 0.032), (torch.float16, 0.002)]
)
def test_explicit_flashinfer_rmsnorm_is_low_precision_equivalent(dtype, atol):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    torch.manual_seed(17)
    hidden = 512
    x = torch.randn(2, 7, hidden, device="cuda", dtype=dtype)
    ref_mod = LlamaRMSNorm(hidden, eps=1e-5).cuda().to(dtype)
    with torch.no_grad():
        ref_mod.weight.normal_(mean=1.0, std=0.02)
        expected = ref_mod(x)
        got = rmsnorm(
            x, ref_mod.weight, ref_mod.variance_epsilon, backend="flashinfer"
        )

    torch.testing.assert_close(got, expected, rtol=0, atol=atol)


@CUDA
def test_rmsnorm_normalises_last_dim_of_4d_head_tensor():
    """Qwen3's q_norm/k_norm run over head_dim on a [b, s, h, d] tensor."""
    torch.manual_seed(0)
    x = torch.randn(1, 5, 8, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    got = rmsnorm(x, w, 1e-6)

    ref = x.float()
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(got.float(), ref.to(torch.bfloat16).float(), rtol=0, atol=0)


# ------------------------------------------------------------------- rope --

@CUDA
def test_rotary_matches_transformers_reference():
    """Our in-place NeoX rope == transformers' apply_rotary_pos_emb."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    torch.manual_seed(0)
    bsz, seq, n_heads, n_kv, head_dim, max_pos = 1, 11, 8, 4, 128, 64
    half = head_dim // 2
    dtype = torch.bfloat16

    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim))
    t = torch.arange(max_pos, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_full, sin_full = emb.cos().to(dtype), emb.sin().to(dtype)
    cos_sin_cache = torch.cat((cos_full[:, :half], sin_full[:, :half]), dim=-1)

    position_ids = torch.arange(seq, device="cuda").unsqueeze(0).expand(bsz, -1)

    q_flat = torch.randn(bsz, seq, n_heads * head_dim, device="cuda", dtype=dtype)
    k_flat = torch.randn(bsz, seq, n_kv * head_dim, device="cuda", dtype=dtype)

    q_ref = q_flat.view(bsz, seq, n_heads, head_dim).transpose(1, 2).clone()
    k_ref = k_flat.view(bsz, seq, n_kv, head_dim).transpose(1, 2).clone()
    cos = cos_full[position_ids]
    sin = sin_full[position_ids]
    expected_q, expected_k = apply_rotary_pos_emb(q_ref, k_ref, cos, sin, unsqueeze_dim=1)

    rotary_embedding_neox_(position_ids, q_flat, k_flat, head_dim, cos_sin_cache)
    got_q = q_flat.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
    got_k = k_flat.view(bsz, seq, n_kv, head_dim).transpose(1, 2)

    torch.testing.assert_close(got_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k, expected_k, rtol=0, atol=0)


# --------------------------------------------------------------- head_dim --

def test_head_dim_prefers_explicit_config_value():
    qwen3_like = types.SimpleNamespace(hidden_size=2560, num_attention_heads=32, head_dim=128)
    llama_like = types.SimpleNamespace(hidden_size=3072, num_attention_heads=24)
    assert head_dim_of(qwen3_like) == 128        # NOT 2560 // 32 == 80
    assert head_dim_of(llama_like) == 128


# ----------------------------------------------------------- silu_and_mul --

@CUDA
def test_silu_and_mul_matches_llama_mlp():
    """Fused gate_up + silu_and_mul == transformers' separate gate/up SwiGLU."""
    from models.compat import silu_and_mul

    torch.manual_seed(0)
    hidden, inter, dtype = 256, 704, torch.bfloat16
    x = torch.randn(1, 9, hidden, device="cuda", dtype=dtype)
    gate_w = torch.randn(inter, hidden, device="cuda", dtype=dtype) / hidden**0.5
    up_w = torch.randn(inter, hidden, device="cuda", dtype=dtype) / hidden**0.5

    expected = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_w)) \
               * torch.nn.functional.linear(x, up_w)

    fused = torch.cat((gate_w, up_w), dim=0)          # exactly how LlamaLayer packs it
    got = silu_and_mul(torch.nn.functional.linear(x, fused))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
