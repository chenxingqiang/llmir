"""MLIR lit suite catalog and optional mlir-opt execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmir.compiler.mlir_lit_suite import (
    LIT_SUITE_FILES,
    default_lit_dir,
    parse_run_lines,
    run_lit_suite,
)
from llmir.compiler.opt_driver import find_mlir_opt

ROOT = Path(__file__).resolve().parents[1]


def test_lit_catalog_matches_files_on_disk():
    lit_dir = default_lit_dir(ROOT)
    on_disk = sorted(p.name for p in lit_dir.glob("*.mlir"))
    catalog = sorted(LIT_SUITE_FILES)
    assert "decoder_workload_buckets.mlir" in catalog
    for name in catalog:
        assert (lit_dir / name).is_file(), name
    # optimization_passes.mlir is aspirational (llmir-opt-only); not in Tier-A catalog.
    assert "optimization_passes.mlir" in on_disk


def test_lit_files_have_run_lines():
    lit_dir = default_lit_dir(ROOT)
    for name in LIT_SUITE_FILES:
        text = (lit_dir / name).read_text(encoding="utf-8")
        runs = parse_run_lines(text)
        assert runs, f"{name} missing RUN lines"


def test_decoder_workload_buckets_checks_all_shapes():
    text = (default_lit_dir(ROOT) / "decoder_workload_buckets.mlir").read_text(encoding="utf-8")
    assert "bucket_s1_short_multitenant" in text
    assert "bucket_s2_rag_shared_system" in text
    assert "bucket_s3_long_document" in text
    assert "block_size = 1024" in text
    assert "block_size = 32" in text


def test_run_lit_suite_skips_without_opt():
    summary = run_lit_suite(default_lit_dir(ROOT))
    assert summary["passed"] + summary["failed"] + summary["skipped"] == len(LIT_SUITE_FILES)
    if not find_mlir_opt():
        assert summary["status"] == "skipped"
        assert summary["skipped"] == len(LIT_SUITE_FILES)


@pytest.mark.skipif(not find_mlir_opt(), reason="mlir-opt / llmir-opt not on PATH")
def test_run_lit_suite_passes_with_opt():
    summary = run_lit_suite(default_lit_dir(ROOT))
    assert summary["status"] == "passed"
    assert summary["failed"] == 0
    assert summary["passed"] == len(LIT_SUITE_FILES)


def test_lit_status_json_roundtrip(tmp_path):
    out = tmp_path / "status.json"
    summary = run_lit_suite(default_lit_dir(ROOT))
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert "files" in data
    assert len(data["files"]) == len(LIT_SUITE_FILES)
