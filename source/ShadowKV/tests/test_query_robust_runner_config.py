from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_runner_exposes_qr_backend_selectors() -> None:
    script = (ROOT / "run_b200_ruler.sh").read_text()
    assert "QUERY_ROBUST_ROUTER_BACKEND" in script
    assert "QUERY_ROBUST_SUMMARY_BACKEND" in script
    assert "SHADOWKV_RUNTIME_TIMINGS" in script


def test_runner_keeps_quest_dense_layers_compatible_with_offload() -> None:
    script = (ROOT / "run_b200_ruler.sh").read_text()
    assert "QUEST_OFFLOAD" in script
    assert 'QUEST_DENSE_LAYERS=${QUEST_DENSE_LAYERS:-2}' in script
    assert 'QUEST_DENSE_LAYERS=0' in script
    assert 'QUEST_OFFLOAD_ARGS=(--streaming_offload' in script


def test_runner_uses_compile_qr_on_b200_by_default() -> None:
    script = (ROOT / "run_b200_ruler.sh").read_text()
    assert 'QUERY_ROBUST_SUMMARY_BACKEND=${QUERY_ROBUST_SUMMARY_BACKEND:-compile}' in script
    assert 'QUERY_ROBUST_ROUTER_BACKEND=${QUERY_ROBUST_ROUTER_BACKEND:-auto}' in script

