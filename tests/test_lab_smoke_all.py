"""Unified lab smoke scripts and summary JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_scripts_exist():
    for name in (
        "lab_smoke_all.sh",
        "check_native_build_prereqs.sh",
        "lab_status_summary.py",
    ):
        assert (ROOT / "scripts" / name).is_file()


def test_check_native_build_prereqs_runs():
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/check_native_build_prereqs.sh")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1)
    assert "Native / MLIR build prerequisites" in proc.stdout


def test_lab_status_summary_writes_json(tmp_path):
    out = tmp_path / "lab_status_summary.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lab_status_summary.py"), "--json-out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "lab_status_summary"
    assert data["package_version"] == "0.2.2"
    assert data["mlir_lit_status"] in ("skipped", "passed", "failed", "missing")
