"""E6 multi-backend correctness parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmir.benchmark.e6_backend_parity import (
    E6ParityConfig,
    compare_decode_parity,
    run_e6_backend_parity,
    run_kv_micro_parity,
)
from llmir.runtime.config import KVCacheConfig


@pytest.fixture
def kv_config_small() -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        block_size=4,
        max_seq_len=32,
        dtype="float32",
        enable_gpu=False,
    )


def test_kv_micro_parity_numpy_vs_torch(kv_config_small):
    pytest.importorskip("torch")
    result = run_kv_micro_parity(kv_config_small, backends=("numpy", "torch_cuda"))
    assert result["all_match"] is True
    for row in result["rows"]:
        assert row["matches_reference"] is True
        assert "error" not in row


def test_decode_parity_numpy_vs_torch(tiny_llama):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    rows = compare_decode_parity(
        model,
        tokenizer,
        E6ParityConfig(
            backends=("numpy", "torch_cuda"),
            prompts=("hello world", "a b c"),
            max_new_tokens=3,
            seed=7,
        ),
    )
    assert len(rows) == 2
    assert all(row["all_match"] for row in rows)
    for row in rows:
        comps = {c["backend"]: c for c in row["comparisons"]}
        assert comps["numpy"]["matches_reference"] is True
        assert comps["torch_cuda"]["matches_reference"] is True
        assert comps["numpy"]["generated_token_ids"] == comps["torch_cuda"]["generated_token_ids"]


def test_e6_full_run(tiny_llama, kv_config_small):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    result = run_e6_backend_parity(
        model,
        tokenizer,
        kv_config=kv_config_small,
        cfg=E6ParityConfig(max_new_tokens=2, seed=1),
    )
    payload = result.to_dict()
    assert payload["experiment"] == "E6"
    assert payload["summary"]["overall_pass"] is True


def test_e6_json_schema(tmp_path, tiny_llama, kv_config_small):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    out = tmp_path / "e6.json"
    result = run_e6_backend_parity(model, tokenizer, kv_config=kv_config_small)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert "decode_parity" in data
    assert "kv_micro_parity" in data
    assert data["summary"]["backends_tested"] == ["numpy", "torch_cuda"]
