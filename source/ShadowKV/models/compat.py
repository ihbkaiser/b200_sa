"""
Portability shims so the released ShadowKV code runs on a modern stack
(torch 2.6 / transformers 4.55) and on models the original release predates.

Every helper here is a *numerical no-op* with respect to the released code:
each one reproduces the op the upstream dependency performed. The equivalence
is pinned by tests/test_compat_equivalence.py -- do not change a helper
without re-running that suite.

Deviations from the upstream environment, and why:
  * vllm._custom_ops.rotary_embedding -> pure-torch NeoX rope.
    vllm 0.5.3.post1 pins torch 2.3.1, which cannot load Qwen3. The torch
    version applies the identical NeoX rotation and is checked bit-for-bit
    against transformers' own apply_rotary_pos_emb.
  * flashinfer.norm.rmsnorm -> pure-torch RMSNorm.
    This is the fallback the ShadowKV authors themselves left commented out
    in tensor_op.py; it is reinstated verbatim.
  * minference -> lazy. Only reachable behind --minference, which we do not use.
"""

import os

import torch

# --------------------------------------------------------------------------
# optional upstream accelerators
# --------------------------------------------------------------------------

try:  # pragma: no cover - depends on the machine's env
    from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm
    HAS_FLASHINFER = True
except Exception:
    _flashinfer_rmsnorm = None
    HAS_FLASHINFER = False

try:  # pragma: no cover
    import vllm as _vllm
    HAS_VLLM = True
except Exception:
    _vllm = None
    HAS_VLLM = False


def rmsnorm(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim with an explicit implementation backend."""
    backend = backend or os.environ.get("SHADOWKV_RMSNORM_BACKEND", "torch")
    if backend not in {"torch", "flashinfer"}:
        raise ValueError("SHADOWKV_RMSNORM_BACKEND must be torch or flashinfer")
    if backend == "flashinfer":
        if not HAS_FLASHINFER:
            raise RuntimeError("FlashInfer RMSNorm requested but flashinfer is unavailable")
        flat = hidden_states.view(-1, hidden_states.size(-1))
        return _flashinfer_rmsnorm(flat, w, eps).view_as(hidden_states)
    # Keep this path independent of optional runtime packages.  In particular,
    # installing FlashInfer for the authors' ParisKV runtime must not silently
    # change the numerics of our quality-reference model.
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return w * x.to(input_dtype)


# --------------------------------------------------------------------------
# rope
# --------------------------------------------------------------------------

def rotary_embedding_neox_(position_ids, query, key, head_size, cos_sin_cache):
    """Pure-torch stand-in for vllm._custom_ops.rotary_embedding(..., is_neox=True).

    Operates on the flat layouts the released ShadowKV code uses:
        query [bsz, seq, num_heads     * head_size]
        key   [bsz, seq, num_kv_heads  * head_size]
        position_ids  [bsz, seq]
        cos_sin_cache [max_pos, head_size]  with cos in [:, :head_size//2]
                                            and sin in [:, head_size//2:]
    Writes back in place, exactly like the vllm kernel, and returns nothing.
    """
    half = head_size // 2
    cos = cos_sin_cache[position_ids, :half]   # [bsz, seq, half]
    sin = cos_sin_cache[position_ids, half:]   # [bsz, seq, half]
    cos = cos.unsqueeze(2)                     # [bsz, seq, 1, half]
    sin = sin.unsqueeze(2)

    for tensor in (query, key):
        bsz, seq, flat = tensor.shape
        view = tensor.view(bsz, seq, flat // head_size, head_size)
        x1 = view[..., :half]
        x2 = view[..., half:]
        # NeoX / HF rotate_half, computed out-of-place then written back so the
        # second half does not read already-rotated values.
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        view[..., :half] = o1
        view[..., half:] = o2


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Pure-torch stand-in for vllm._custom_ops.silu_and_mul.

    Splits the fused gate_up projection down the middle and returns
    silu(gate) * up -- the SwiGLU the released code computes.
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def get_inv_freq(hf_model) -> torch.Tensor:
    """Fetch rope inv_freq across transformers layouts.

    transformers <= 4.47 kept a rotary_emb on every attention module;
    >= 4.48 moved a single one onto the model. Both carry any rope scaling
    (e.g. Llama-3.2's 'llama3' type) already applied.
    """
    attn = hf_model.model.layers[0].self_attn
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(hf_model.model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            "Could not locate a rotary embedding on this model; "
            "checked layers[0].self_attn.rotary_emb and model.rotary_emb"
        )
    return rotary.inv_freq


def head_dim_of(config) -> int:
    """Head dim, honouring models that state it explicitly.

    Qwen3 sets head_dim=128 with hidden_size=2560 and 32 heads, so the
    hidden_size // num_heads shortcut used throughout the release is wrong
    there. Falling back to that shortcut keeps every other model unchanged.
    """
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return int(head_dim)
