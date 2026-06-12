"""Walkthrough summary JSON builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llmir.benchmark.walkthrough_summary import build_walkthrough_summary

ROOT = Path(__file__).resolve().parents[1]


def test_build_walkthrough_summary_in_process():
    payload = build_walkthrough_summary(ROOT)
    assert payload["m6_all_pass"] is True
    assert payload["artifact_count"] >= 9
    assert payload["package_version"] == "0.2.2"
    assert payload["pypi_release_status"] in ("published", "pending", "unavailable")


def test_walkthrough_summary_script():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/walkthrough_summary.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = ROOT / "IEEE-conference/benchmarks/walkthrough_summary.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "a_class_walkthrough_summary"
    assert data["m6_all_pass"] is True
    assert data["e8_status"] in ("skipped", "completed")
    assert "walkthrough_command" in data
