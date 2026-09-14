"""Pure-PyTorch Query-Robust page summaries and routing.

This module is a correctness/reference port of the QR path in
``ihb-sparse``.  It deliberately keeps the expensive calibration/summary
math in FP32 while storing the runtime landmark in BF16, matching the
certificate that the serving path actually uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class QueryRobustAsset:
    """Validated QR vertices for one runtime tensor-parallel rank."""

    vertices: torch.Tensor
    num_valid_vertices: torch.Tensor
    meta: dict[str, Any]


@dataclass(frozen=True)
class QueryRobustSummary:
    """Per-page QR summary and certificate diagnostics."""

    landmark: torch.Tensor
    bias: torch.Tensor
    epsilon: torch.Tensor
    dual: torch.Tensor
    gap: torch.Tensor
    errors: torch.Tensor


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if name == "vertices" and tensor.dtype == torch.uint16:
            # NumPy has no BF16 dtype; NPZ assets store raw BF16 bits.
            return tensor.view(torch.bfloat16)
        return tensor
    raise TypeError(f"Query-Robust asset field {name!r} must be a tensor or ndarray.")


def _load_asset_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Query-Robust asset not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload: dict[str, Any] = {
                "vertices": data["vertices"],
                "num_valid_vertices": data["num_valid_vertices"],
            }
            if "meta" in data:
                raw_meta = data["meta"].item() if data["meta"].ndim == 0 else data["meta"]
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                payload["meta"] = (
                    json.loads(str(raw_meta)) if isinstance(raw_meta, str) else raw_meta
                )
            return payload
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before weights_only
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Query-Robust asset must contain a mapping payload.")
    return payload


def _canonical_rope_config(value: Any) -> dict[str, Any]:
    """Normalize equivalent Transformers RoPE metadata before comparison."""

    if isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(value, (tuple, list)):
        try:
            raw = dict(value)
        except (TypeError, ValueError) as error:
            raise TypeError("RoPE metadata must be a mapping or key-value sequence.") from error
    else:
        raise TypeError(f"RoPE metadata must be a mapping, got {type(value).__name__}.")

    nested = raw.get("rope_parameters")
    if nested is None:
        nested = raw.get("rope_scaling")
    if nested is not None:
        normalized = _canonical_rope_config(nested)
        for name, nested_value in raw.items():
            if name not in {"rope_parameters", "rope_scaling"}:
                normalized.setdefault(name, nested_value)
        return _canonical_rope_config(normalized)

    if "type" in raw and "rope_type" not in raw:
        raw["rope_type"] = raw["type"]
    raw.pop("type", None)
    raw["rope_type"] = str(raw.get("rope_type", "default")).lower()
    for name in (
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "beta_fast",
        "beta_slow",
        "attention_factor",
        "mscale",
        "mscale_all_dim",
        "rope_theta",
    ):
        if name in raw and raw[name] is not None:
            raw[name] = float(raw[name])
    if "original_max_position_embeddings" in raw:
        raw["original_max_position_embeddings"] = int(raw["original_max_position_embeddings"])
    return raw


def load_query_robust_asset(
    path: str | Path,
    *,
    num_layers: int,
    global_num_kv_heads: int,
    head_dim: int,
    num_vertices: int,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    expected_model_id: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_rope_config: Any | None = None,
    expected_sha256: str | None = None,
    device: torch.device | str | None = None,
) -> QueryRobustAsset:
    """Load, validate, checksum, and tensor-parallel-slice QR vertices."""

    path = Path(path)
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256):
            raise ValueError(
                "Query-Robust asset SHA-256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256}."
            )

    payload = _load_asset_payload(path)
    vertices = _as_tensor(payload.get("vertices"), name="vertices")
    valid = _as_tensor(payload.get("num_valid_vertices"), name="num_valid_vertices")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("Query-Robust asset meta must be a mapping.")
    if vertices.ndim != 4:
        raise ValueError(
            "Query-Robust vertices must have shape [layers, global_kv_heads, M, D], "
            f"got {tuple(vertices.shape)}."
        )
    if vertices.dtype != torch.bfloat16:
        raise TypeError(
            "Query-Robust vertex assets must store vertices as BF16; "
            f"got {vertices.dtype}."
        )
    if valid.ndim != 2:
        raise ValueError(
            "Query-Robust num_valid_vertices must have shape [layers, global_kv_heads], "
            f"got {tuple(valid.shape)}."
        )

    actual_layers, actual_heads, actual_vertices, actual_dim = map(int, vertices.shape)
    if tuple(valid.shape) != (actual_layers, actual_heads):
        raise ValueError("Query-Robust validity shape does not match vertices.")
    expected = {
        "num_layers": int(num_layers),
        "global_num_kv_heads": int(global_num_kv_heads),
        "head_dim": int(head_dim),
        "num_vertices": int(num_vertices),
    }
    actual = {
        "num_layers": actual_layers,
        "global_num_kv_heads": actual_heads,
        "head_dim": actual_dim,
        "num_vertices": actual_vertices,
    }
    for name, wanted in expected.items():
        if actual[name] != wanted:
            raise ValueError(
                f"Query-Robust asset {name} mismatch: expected={wanted} got={actual[name]}."
            )

    if "tp_world_size" not in meta:
        raise ValueError("Query-Robust asset metadata must declare tp_world_size.")
    rank = int(tensor_parallel_rank)
    tp_size = int(tensor_parallel_size)
    if int(meta["tp_world_size"]) != tp_size:
        raise ValueError(
            "Query-Robust asset TP partition does not match the runtime: "
            f"asset_tp_world_size={meta['tp_world_size']} runtime_tp_world_size={tp_size}."
        )
    if tp_size <= 0 or not 0 <= rank < tp_size:
        raise ValueError(f"Invalid Query-Robust TP rank={rank} for size={tp_size}.")
    if actual_heads % tp_size:
        raise ValueError(
            "Query-Robust global KV heads must be divisible by asset TP size: "
            f"heads={actual_heads} tp={tp_size}."
        )

    model_id = meta.get("model_id")
    if expected_model_id is not None and model_id is None:
        raise ValueError("Query-Robust asset metadata must declare model_id.")
    if expected_model_id is not None and model_id is not None:
        expected_name = str(expected_model_id).rstrip("/").split("/")[-1]
        actual_name = str(model_id).rstrip("/").split("/")[-1]
        if expected_name != actual_name:
            raise ValueError(
                f"Query-Robust asset model_id mismatch: expected={expected_model_id!r} "
                f"asset={model_id!r}."
            )

    asset_fingerprint = meta.get("model_fingerprint")
    if asset_fingerprint is not None:
        if expected_model_fingerprint is None:
            raise ValueError(
                "Query-Robust asset declares model_fingerprint but the runtime "
                "did not provide one."
            )
        if str(asset_fingerprint) != str(expected_model_fingerprint):
            raise ValueError(
                "Query-Robust asset model_fingerprint mismatch: "
                f"expected={expected_model_fingerprint!r} asset={asset_fingerprint!r}."
            )
    elif expected_model_fingerprint is not None:
        raise ValueError("Query-Robust asset lacks the expected model_fingerprint.")

    asset_rope = meta.get("rope_config")
    if expected_rope_config is not None and asset_rope is None:
        # The frozen Qwen3 asset records only rope_theta.  Treat the missing
        # rope type as the standard/default Transformers representation.
        asset_rope = {"rope_theta": meta.get("rope_theta", 5_000_000.0)}
    if asset_rope is not None and expected_rope_config is not None:
        if _canonical_rope_config(asset_rope) != _canonical_rope_config(expected_rope_config):
            raise ValueError(
                "Query-Robust asset RoPE configuration does not match runtime: "
                f"asset={_canonical_rope_config(asset_rope)!r} "
                f"runtime={_canonical_rope_config(expected_rope_config)!r}."
            )

    for meta_name, actual_value in (
        ("num_layers", actual_layers),
        ("num_kv_heads", actual_heads),
        ("head_dim", actual_dim),
        ("M", actual_vertices),
    ):
        if meta_name in meta and int(meta[meta_name]) != actual_value:
            raise ValueError(
                f"Query-Robust metadata {meta_name} disagrees with tensor shape."
            )

    valid = valid.to(torch.int32)
    if bool((valid < 2).any()) or bool((valid > actual_vertices).any()):
        raise ValueError("Query-Robust num_valid_vertices must be in [2, M].")
    if not torch.isfinite(vertices.float()).all():
        raise ValueError("Query-Robust vertices contain non-finite values.")
    for layer in range(actual_layers):
        for kv_head in range(actual_heads):
            count = int(valid[layer, kv_head])
            if count == actual_vertices:
                continue
            real_vertices = vertices[layer, kv_head, :count]
            padded_vertices = vertices[layer, kv_head, count:]
            repeated = (
                padded_vertices[:, None, :] == real_vertices[None, :, :]
            ).all(dim=-1).any(dim=-1)
            if not bool(repeated.all()):
                raise ValueError(
                    "Query-Robust padded vertices must repeat a valid support vertex "
                    f"at layer={layer}, kv_head={kv_head}."
                )

    local_heads = actual_heads // tp_size
    head_start = rank * local_heads
    head_end = head_start + local_heads
    sliced_vertices = vertices[:, head_start:head_end].to(torch.float32).contiguous()
    sliced_valid = valid[:, head_start:head_end].contiguous()
    if device is not None:
        sliced_vertices = sliced_vertices.to(device=device)
        sliced_valid = sliced_valid.to(device=device)
    return QueryRobustAsset(sliced_vertices, sliced_valid, dict(meta))


def _validate_solver_inputs(
    keys: torch.Tensor, vertices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 2 or vertices.ndim != 2:
        raise ValueError("Query-Robust solver expects keys [N, D] and vertices [M, D].")
    if int(keys.shape[0]) <= 0 or int(vertices.shape[0]) < 2:
        raise ValueError("Query-Robust pages need N > 0 and at least two vertices.")
    if keys.shape[1] != vertices.shape[1]:
        raise ValueError("Query-Robust keys and vertices must share head_dim.")
    return keys.float(), vertices.float()


@torch.no_grad()
def solve_query_robust_page(
    keys: torch.Tensor,
    vertices: torch.Tensor,
    *,
    scale: float,
    solver_iters: int = 24,
    solver_lr: float = 0.25,
    uniform_p: bool = False,
) -> QueryRobustSummary:
    """Solve one page's minimax dual in stable FP32 arithmetic."""

    keys, vertices = _validate_solver_inputs(keys, vertices)
    if not math.isfinite(float(scale)):
        raise ValueError("Query-Robust attention scale must be finite.")
    solver_iters = int(solver_iters)
    solver_lr = float(solver_lr)
    if solver_iters <= 0 or not math.isfinite(solver_lr) or solver_lr <= 0:
        raise ValueError("Query-Robust solver_iters and solver_lr must be positive.")

    logits = float(scale) * (vertices @ keys.transpose(0, 1))
    page_lse = torch.logsumexp(logits, dim=-1)
    lambda_logits = torch.zeros(vertices.shape[0], dtype=torch.float32, device=keys.device)
    if uniform_p:
        p = torch.full(
            (keys.shape[0],), 1.0 / float(keys.shape[0]), dtype=torch.float32, device=keys.device
        )
        lam = torch.full_like(lambda_logits, 1.0 / float(vertices.shape[0]))
        bar_s = lam @ logits
    else:
        for step_index in range(solver_iters):
            lam = torch.softmax(lambda_logits, dim=0)
            bar_s = lam @ logits
            p = torch.softmax(bar_s, dim=0)
            grad = page_lse - logits @ p
            grad = grad - grad.mean()
            grad_scale = grad.abs().max().clamp_min(1e-6)
            step = solver_lr / grad_scale / math.sqrt(1.0 + float(step_index))
            lambda_logits = lambda_logits + step * grad
        lam = torch.softmax(lambda_logits, dim=0)
        bar_s = lam @ logits
        p = torch.softmax(bar_s, dim=0)

    landmark = p @ keys
    entropy = (
        torch.full((), math.log(float(keys.shape[0])), dtype=torch.float32, device=keys.device)
        if uniform_p
        else torch.logsumexp(bar_s, dim=0) - torch.sum(p * bar_s)
    )
    stored_landmark = landmark.to(torch.bfloat16).float()
    errors = page_lse - (float(scale) * (vertices @ stored_landmark) + entropy)
    epsilon = errors.max().clamp_min(0.0)
    dual = errors.mean() if uniform_p else torch.sum(lam * errors)
    return QueryRobustSummary(
        landmark=landmark.to(torch.bfloat16),
        bias=entropy.float(),
        epsilon=epsilon.float(),
        dual=dual.float(),
        gap=(epsilon - dual).float(),
        errors=errors.float(),
    )


def _validate_batched_summary_inputs(
    keys: torch.Tensor, vertices: torch.Tensor
) -> tuple[int, int, int, int, int]:
    if keys.ndim != 4 or vertices.ndim != 3:
        raise ValueError(
            "Query-Robust builder expects keys [P, N, H, D] and vertices [H, M, D]."
        )
    pages, page_size, num_heads, head_dim = map(int, keys.shape)
    if pages <= 0 or page_size <= 0:
        raise ValueError("Query-Robust builder needs non-empty pages.")
    if tuple(vertices.shape[::2]) != (num_heads, head_dim):
        raise ValueError("Query-Robust vertex head dimensions do not match keys.")
    if int(vertices.shape[1]) < 2:
        raise ValueError("Query-Robust builder needs at least two vertices.")
    return pages, page_size, num_heads, head_dim, int(vertices.shape[1])


def _build_flat_query_robust_summary(
    flat_keys: torch.Tensor,
    flat_vertices: torch.Tensor,
    *,
    page_size: int,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
) -> tuple[torch.Tensor, ...]:
    """Tensor-only summary body shared by eager and compiled backends."""
    pages_heads, _, head_dim = flat_keys.shape
    vertices = int(flat_vertices.shape[1])
    logits = float(scale) * torch.bmm(flat_vertices, flat_keys.transpose(1, 2))
    page_lse = torch.logsumexp(logits, dim=-1)
    lambda_logits = torch.zeros(
        pages_heads,
        vertices,
        dtype=torch.float32,
        device=flat_keys.device,
    )
    if uniform_p:
        p = torch.full(
            (pages_heads, page_size),
            1.0 / float(page_size),
            dtype=torch.float32,
            device=flat_keys.device,
        )
        lam = torch.full_like(lambda_logits, 1.0 / float(vertices))
        bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
    else:
        for step_index in range(int(solver_iters)):
            lam = torch.softmax(lambda_logits, dim=-1)
            bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
            p = torch.softmax(bar_s, dim=-1)
            grad = page_lse - torch.bmm(logits, p.unsqueeze(-1)).squeeze(-1)
            grad = grad - grad.mean(dim=-1, keepdim=True)
            grad_scale = grad.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            step = float(solver_lr) / grad_scale / math.sqrt(1.0 + step_index)
            lambda_logits = lambda_logits + step * grad
        lam = torch.softmax(lambda_logits, dim=-1)
        bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
        p = torch.softmax(bar_s, dim=-1)

    flat_landmark = torch.bmm(p.unsqueeze(1), flat_keys).squeeze(1)
    entropy = (
        torch.full(
            (pages_heads,),
            math.log(float(page_size)),
            dtype=torch.float32,
            device=flat_keys.device,
        )
        if uniform_p
        else torch.logsumexp(bar_s, dim=-1) - torch.sum(p * bar_s, dim=-1)
    )
    stored_landmark = flat_landmark.to(torch.bfloat16).float()
    errors = page_lse - (
        float(scale)
        * torch.bmm(flat_vertices, stored_landmark.unsqueeze(-1)).squeeze(-1)
        + entropy[:, None]
    )
    epsilon = errors.amax(dim=-1).clamp_min(0.0)
    dual = errors.mean(dim=-1) if uniform_p else torch.sum(lam * errors, dim=-1)
    gap = epsilon - dual
    return (
        flat_landmark.to(torch.bfloat16),
        entropy.float(),
        epsilon.float(),
        dual.float(),
        gap.float(),
        errors.float(),
    )


def _summary_from_flat_outputs(
    outputs: tuple[torch.Tensor, ...],
    *,
    pages: int,
    num_heads: int,
    head_dim: int,
    vertices: int,
) -> QueryRobustSummary:
    flat_landmark, entropy, epsilon, dual, gap, errors = outputs
    return QueryRobustSummary(
        landmark=flat_landmark.view(pages, num_heads, head_dim),
        bias=entropy.view(pages, num_heads),
        epsilon=epsilon.view(pages, num_heads),
        dual=dual.view(pages, num_heads),
        gap=gap.view(pages, num_heads),
        errors=errors.view(pages, num_heads, vertices),
    )


@lru_cache(maxsize=16)
def _compiled_summary_builder(
    pages: int,
    page_size: int,
    num_heads: int,
    head_dim: int,
    vertices: int,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
):
    del pages, num_heads, head_dim, vertices
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch version has no torch.compile")

    def compiled(flat_keys: torch.Tensor, flat_vertices: torch.Tensor):
        return _build_flat_query_robust_summary(
            flat_keys,
            flat_vertices,
            page_size=page_size,
            scale=scale,
            solver_iters=solver_iters,
            solver_lr=solver_lr,
            uniform_p=uniform_p,
        )

    return torch.compile(compiled, backend="inductor", dynamic=False)


class QueryRobustSummaryWorkspace:
    """Reusable flattened buffers and optional compiled QR summary body."""

    def __init__(self, device: torch.device | str, *, backend: str = "eager") -> None:
        self.device = torch.device(device)
        self.backend = str(backend).lower()
        if self.backend not in {"eager", "compile"}:
            raise ValueError("Query-Robust summary backend must be eager or compile")
        self.backend_used = "eager"
        self._flat_keys: torch.Tensor | None = None
        self._flat_vertices: torch.Tensor | None = None
        self.allocation_count = 0
        self.compile_fallback: str | None = None

    def _ensure_buffers(
        self, pages: int, page_size: int, num_heads: int, head_dim: int, vertices: int
    ) -> None:
        key_shape = (pages * num_heads, page_size, head_dim)
        vertex_shape = (pages * num_heads, vertices, head_dim)
        if self._flat_keys is None or self._flat_keys.shape != key_shape:
            self._flat_keys = torch.empty(
                key_shape, device=self.device, dtype=torch.float32
            )
            self.allocation_count += 1
        if self._flat_vertices is None or self._flat_vertices.shape != vertex_shape:
            self._flat_vertices = torch.empty(
                vertex_shape, device=self.device, dtype=torch.float32
            )
            self.allocation_count += 1

    @torch.no_grad()
    def build(
        self,
        keys: torch.Tensor,
        vertices: torch.Tensor,
        *,
        scale: float,
        solver_iters: int,
        solver_lr: float,
        uniform_p: bool,
    ) -> QueryRobustSummary:
        pages, page_size, num_heads, head_dim, num_vertices = (
            _validate_batched_summary_inputs(keys, vertices)
        )
        if keys.device != self.device or vertices.device != self.device:
            raise ValueError("Query-Robust workspace and inputs must share a device")
        if not math.isfinite(float(scale)):
            raise ValueError("Query-Robust attention scale must be finite.")
        solver_iters = int(solver_iters)
        solver_lr = float(solver_lr)
        if solver_iters <= 0 or not math.isfinite(solver_lr) or solver_lr <= 0:
            raise ValueError("Query-Robust solver_iters and solver_lr must be positive.")

        self._ensure_buffers(pages, page_size, num_heads, head_dim, num_vertices)
        assert self._flat_keys is not None and self._flat_vertices is not None
        self._flat_keys.copy_(keys.permute(0, 2, 1, 3).reshape(-1, page_size, head_dim))
        expanded_vertices = vertices.unsqueeze(0).expand(
            pages, -1, -1, -1
        ).reshape(-1, num_vertices, head_dim)
        self._flat_vertices.copy_(expanded_vertices)

        if self.backend == "compile":
            try:
                compiled = _compiled_summary_builder(
                    pages,
                    page_size,
                    num_heads,
                    head_dim,
                    num_vertices,
                    float(scale),
                    solver_iters,
                    solver_lr,
                    bool(uniform_p),
                )
                outputs = compiled(self._flat_keys, self._flat_vertices)
                self.compile_fallback = None
                self.backend_used = "compile"
            except Exception as error:
                self.compile_fallback = type(error).__name__
                self.backend_used = "eager"
                outputs = _build_flat_query_robust_summary(
                    self._flat_keys,
                    self._flat_vertices,
                    page_size=page_size,
                    scale=float(scale),
                    solver_iters=solver_iters,
                    solver_lr=solver_lr,
                    uniform_p=bool(uniform_p),
                )
        else:
            self.backend_used = "eager"
            outputs = _build_flat_query_robust_summary(
                self._flat_keys,
                self._flat_vertices,
                page_size=page_size,
                scale=float(scale),
                solver_iters=solver_iters,
                solver_lr=solver_lr,
                uniform_p=bool(uniform_p),
            )
        return _summary_from_flat_outputs(
            outputs,
            pages=pages,
            num_heads=num_heads,
            head_dim=head_dim,
            vertices=num_vertices,
        )


@torch.no_grad()
def build_query_robust_page_summaries(
    keys: torch.Tensor,
    vertices: torch.Tensor,
    *,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
    workspace: QueryRobustSummaryWorkspace | None = None,
) -> QueryRobustSummary:
    """Build summaries for keys ``[pages, page_size, kv_heads, dim]``."""
    if workspace is not None:
        return workspace.build(
            keys,
            vertices,
            scale=scale,
            solver_iters=solver_iters,
            solver_lr=solver_lr,
            uniform_p=uniform_p,
        )
    pages, page_size, num_heads, head_dim, num_vertices = (
        _validate_batched_summary_inputs(keys, vertices)
    )
    flat_keys = keys.float().permute(0, 2, 1, 3).reshape(
        pages * num_heads, page_size, head_dim
    )
    flat_vertices = vertices.float().unsqueeze(0).expand(
        pages, -1, -1, -1
    ).reshape(
        pages * num_heads, num_vertices, head_dim
    )
    return _summary_from_flat_outputs(
        _build_flat_query_robust_summary(
            flat_keys,
            flat_vertices,
            page_size=page_size,
            scale=float(scale),
            solver_iters=int(solver_iters),
            solver_lr=float(solver_lr),
            uniform_p=bool(uniform_p),
        ),
        pages=pages,
        num_heads=num_heads,
        head_dim=head_dim,
        vertices=num_vertices,
    )


@torch.no_grad()
def score_query_robust_pages_reference(
    query: torch.Tensor,
    landmark: torch.Tensor,
    bias: torch.Tensor,
    epsilon: torch.Tensor,
    metadata_valid: torch.Tensor,
    row_page_slots: torch.Tensor,
    *,
    scale: float,
    alpha: float,
) -> torch.Tensor:
    """Score pages with one shared route per request using GQA max reduction."""

    if query.ndim != 3 or landmark.ndim != 3:
        raise ValueError("QR scorer expects query [B, QH, D] and landmark [P, H, D].")
    if bias.ndim != 2 or bias.shape != epsilon.shape or bias.shape != metadata_valid.shape:
        raise ValueError("QR scalar metadata must all have shape [P, H].")
    if row_page_slots.ndim != 2 or row_page_slots.dtype != torch.int32:
        raise ValueError("QR row_page_slots must be int32 [batch, pages].")
    batch, query_heads, dim = map(int, query.shape)
    pages, kv_heads, metadata_dim = map(int, landmark.shape)
    if pages <= 0 or metadata_dim != dim or query_heads % kv_heads:
        raise ValueError("QR query heads must be divisible by local KV heads.")
    if row_page_slots.shape[0] != batch:
        raise ValueError("QR row_page_slots batch dimension does not match query.")
    if query.device != landmark.device or row_page_slots.device != query.device:
        raise ValueError("QR scorer inputs must share one device.")

    safe_pages = row_page_slots.to(torch.long).clamp(0, pages - 1)
    width = int(row_page_slots.shape[1])
    page_landmark = landmark.index_select(0, safe_pages.reshape(-1)).view(
        batch, width, kv_heads, dim
    )
    page_bias = bias.index_select(0, safe_pages.reshape(-1)).view(batch, width, kv_heads)
    page_epsilon = epsilon.index_select(0, safe_pages.reshape(-1)).view(batch, width, kv_heads)
    page_valid = metadata_valid.index_select(0, safe_pages.reshape(-1)).view(
        batch, width, kv_heads
    )
    page_valid = page_valid & row_page_slots.ge(0).unsqueeze(-1)
    group_size = query_heads // kv_heads
    grouped_query = query.float().view(batch, kv_heads, group_size, dim)
    dots = torch.einsum("bhgd,bwhd->bwhg", grouped_query, page_landmark.float())
    dot_max = dots.amax(dim=-1)
    scores = dot_max * float(scale) + page_bias.float() + float(alpha) * page_epsilon.float()
    scores = scores.amax(dim=-1)
    return torch.where(
        ~page_valid.all(dim=-1), torch.full_like(scores, float("inf")), scores
    ).float().contiguous()


__all__ = [
    "QueryRobustAsset",
    "QueryRobustSummary",
    "QueryRobustSummaryWorkspace",
    "build_query_robust_page_summaries",
    "load_query_robust_asset",
    "score_query_robust_pages_reference",
    "solve_query_robust_page",
]
