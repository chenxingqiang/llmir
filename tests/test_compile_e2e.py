"""P2 compile pipeline tests (MLIR emit + reference interpreter)."""

import pytest

from llmir.compiler import (
    compile_kv_micro_pipeline,
    emit_kv_micro_pipeline_mlir,
    find_mlir_opt,
    run_mlir_opt,
)
from llmir.compiler.kv_emit import KVMicroPipelineConfig


def test_emit_kv_micro_pipeline_mlir_contains_ops():
    mlir = emit_kv_micro_pipeline_mlir(KVMicroPipelineConfig())
    assert "llm.append_kv" in mlir
    assert "llm.lookup_kv" in mlir
    assert "llm.paged_attention" in mlir
    assert "func.func @kv_micro_pipeline" in mlir


def test_reference_matches_torch_sdpa():
    pytest.importorskip("torch")
    cfg = KVMicroPipelineConfig(batch_size=1, seq_len=8, num_heads=2, head_dim=16)
    result = compile_kv_micro_pipeline(
        cfg, run_opt=False, run_reference=True, compare_torch=True, seed=42
    )
    assert result.reference_output is not None
    assert result.torch_max_abs_diff is not None
    assert result.torch_max_abs_diff < 1e-5


def test_import_toy_attention_emits_llm_ops():
    pytest.importorskip("torch")
    from llmir.importers.toy_attention import import_toy_attention_to_mlir

    mlir = import_toy_attention_to_mlir()
    assert "module" in mlir
    assert "llm.paged_attention" in mlir or "llm.attention" in mlir


@pytest.mark.skipif(not find_mlir_opt(), reason="mlir-opt / llmir-opt not on PATH")
def test_mlir_opt_lowers_kv_ops():
    mlir = emit_kv_micro_pipeline_mlir(KVMicroPipelineConfig())
    opt = run_mlir_opt(mlir, passes=("-llm-lower-kv-cache-ops",))
    assert opt.success, opt.stderr
    lowered = opt.stdout
    assert "llm.append_kv" not in lowered or "call" in lowered
