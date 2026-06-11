"""E8 optional empirical GPU bench (B-class)."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.e8_empirical_gpu import run_e8_empirical_gpu_bench


def test_e8_skips_without_cuda():
    result = run_e8_empirical_gpu_bench()
    payload = result.to_dict()
    assert payload["experiment"] == "E8"
    assert payload["evidence_class"] == "B"
    if not payload["cuda_stack"].get("torch_cuda"):
        assert payload["status"] == "skipped"
        assert payload.get("reason") == "no_cuda_available"


def test_e8_json_schema(tmp_path):
    out = tmp_path / "e8.json"
    result = run_e8_empirical_gpu_bench()
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert "claim_scope" in data
    assert "cuda_stack" in data
