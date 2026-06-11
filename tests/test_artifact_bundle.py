"""M6 artifact bundle manifest checks."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.artifact_bundle import load_manifest, verify_artifact_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_loads():
    manifest = load_manifest(ROOT)
    assert manifest["version"] == "1"
    assert len(manifest["artifacts"]) >= 6


def test_artifact_bundle_required_json_present():
    report = verify_artifact_bundle(ROOT, check_figures=False)
    assert report.all_pass is True
    failed = [a for a in report.artifacts if not a["ok"]]
    assert not failed, failed


def test_e6_and_m5_assertions_in_bundle():
    report = verify_artifact_bundle(ROOT, check_figures=False)
    by_id = {row["id"]: row for row in report.artifacts}
    assert by_id["e6_backend_parity"]["ok"] is True
    assert by_id["m5_lowered_hot_path"]["ok"] is True


def test_artifact_bundle_status_roundtrip(tmp_path):
    out = tmp_path / "status.json"
    report = verify_artifact_bundle(ROOT, check_figures=False)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert data["all_pass"] is True
