"""MVP-A: paper-aligned single-layer compile + block size + reference e2e."""

import json

import pytest

from llmir.compiler.block_size import analyze_block_size, optimize_block_size_attr
from llmir.compiler.kv_emit import KVMicroPipelineConfig
from llmir.compiler.mvp_pipeline import mvp_result_to_json, run_mvp_single_layer_e2e
from llmir.compiler.opt_driver import find_mlir_opt


def test_block_size_algorithm_seq_len_1():
    result = analyze_block_size([1])
    assert result.optimal_block_size == 64


def test_block_size_algorithm_seq_len_16():
    result = analyze_block_size([16])
    assert result.optimal_block_size == 32


def test_block_size_reduces_oversized_block():
    assert optimize_block_size_attr(1024, [4]) == 32


def test_mvp_python_block_size_patch():
    result = run_mvp_single_layer_e2e(
        KVMicroPipelineConfig(seq_len=4, block_size=16),
        oversized_block_size=1024,
        run_mlir_passes=False,
        run_reference=True,
        compare_torch=False,
        seed=1,
    )
    assert result.block_size_before == 1024
    assert result.block_size_after == 32
    assert "llm.append_kv" in result.mlir_after_block_size
    assert result.reference_backend in ("numpy", "native")


@pytest.mark.skipif(not find_mlir_opt(), reason="mlir-opt / llmir-opt not on PATH")
def test_mvp_mlir_opt_lowers_single_layer():
    result = run_mvp_single_layer_e2e(
        KVMicroPipelineConfig(),
        oversized_block_size=1024,
        run_mlir_passes=True,
        run_reference=False,
        compare_torch=False,
    )
    assert result.opt_optimize is not None and result.opt_optimize.success
    assert result.opt_lower is not None and result.opt_lower.success
    assert "mlir_llm_append_kv" in result.lowered_mlir
    assert result.metadata.get("mlir_opt_block_size") == 32


def test_mvp_reference_matches_torch():
    pytest.importorskip("torch")
    result = run_mvp_single_layer_e2e(
        KVMicroPipelineConfig(seq_len=8, num_heads=2, head_dim=16),
        oversized_block_size=512,
        run_mlir_passes=False,
        run_reference=True,
        compare_torch=True,
        seed=42,
    )
    assert result.torch_max_abs_diff is not None
    assert result.torch_max_abs_diff < 1e-5


def test_mvp_json_metadata():
    result = run_mvp_single_layer_e2e(run_mlir_passes=False, run_reference=False)
    payload = json.loads(mvp_result_to_json(result))
    assert payload["block_size_after"] == 32
    assert payload["metadata"]["pipeline"] == "mvp-a-single-layer"
