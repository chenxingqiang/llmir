"""M5: lowered mlir_llm hot path verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llmir.compiler.kv_emit import KVMicroPipelineConfig
from llmir.compiler.lowered_hot_path import (
    LOWERED_RUNTIME_SYMBOLS,
    execute_semantic_lowered_hot_path,
    run_lowered_hot_path_verification,
    verify_lowered_mlir,
)
from llmir.compiler.opt_driver import find_mlir_opt


def test_verify_lowered_symbols_detects_calls():
    sample = "call @mlir_llm_append_kv\n call @mlir_llm_lookup_kv\n call @mlir_llm_paged_attention"
    present = verify_lowered_mlir(sample)
    assert all(present[s] for s in LOWERED_RUNTIME_SYMBOLS)


def test_semantic_hot_path_matches_reference_shapes():
    cfg = KVMicroPipelineConfig(seq_len=4, num_heads=2, head_dim=8)
    rng = np.random.default_rng(1)
    keys = rng.standard_normal((1, 4, 2, 8), dtype=np.float32)
    values = rng.standard_normal((1, 4, 2, 8), dtype=np.float32)
    query = keys[:, -1:, :, :].copy()
    out, path = execute_semantic_lowered_hot_path(keys, values, query, cfg=cfg)
    assert out.shape == (1, 1, 2, 8)
    assert path.startswith("semantic_lowered::")


def test_m5_full_verification_passes():
    result = run_lowered_hot_path_verification(
        KVMicroPipelineConfig(seq_len=8, num_heads=2, head_dim=16),
        seed=7,
    )
    assert result.matches_reference is True
    assert result.max_abs_diff_vs_reference is not None
    assert result.max_abs_diff_vs_reference < 1e-5


@pytest.mark.skipif(not find_mlir_opt(), reason="mlir-opt not on PATH")
def test_m5_lowering_produces_runtime_symbols():
    result = run_lowered_hot_path_verification(KVMicroPipelineConfig())
    assert result.mlir_lowered is True
    assert all(result.lowered_symbols_present.values())


def test_m5_json_schema(tmp_path):
    out = tmp_path / "m5.json"
    result = run_lowered_hot_path_verification(KVMicroPipelineConfig(seq_len=4))
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert data["experiment"] == "M5"
    assert "lowered_symbols_present" in data
