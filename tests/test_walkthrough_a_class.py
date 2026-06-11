"""A-class walkthrough script smoke checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/walkthrough_a_class.sh"


def test_walkthrough_script_exists_and_executable():
    assert SCRIPT.is_file()
    assert oct(SCRIPT.stat().st_mode)[-3:] in ("755", "775", "777")


def test_walkthrough_script_lists_core_steps():
    text = SCRIPT.read_text(encoding="utf-8")
    for needle in (
        "test_mvp_a_e2e",
        "test_e4_e5_multi_bucket",
        "verify_artifact_bundle",
        "test_e8_empirical_gpu",
    ):
        assert needle in text


def test_optional_artifacts_in_manifest():
    from llmir.benchmark.artifact_bundle import load_manifest, verify_artifact_bundle

    manifest = load_manifest(ROOT)
    ids = {a["id"] for a in manifest["artifacts"]}
    assert "e8_empirical_gpu" in ids
    report = verify_artifact_bundle(ROOT, check_figures=False)
    by_id = {row["id"]: row for row in report.artifacts}
    assert by_id["e8_empirical_gpu"]["ok"] is True
    assert by_id["e4_compositional_buckets"]["ok"] is True
    assert by_id["e5_ablation_buckets"]["ok"] is True
