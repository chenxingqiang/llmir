"""Evidence dashboard markdown generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llmir.benchmark.evidence_dashboard import build_evidence_dashboard_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_contains_core_sections():
    md = build_evidence_dashboard_markdown(ROOT)
    assert "# LLMIR Evidence Dashboard" in md
    assert "M6 artifact bundle" in md
    assert "`e6_backend_parity`" in md
    assert "a-class-walkthrough.yml/badge.svg" in md
    assert "python-package.yml/badge.svg" in md
    assert "Package (local)" in md
    assert "PyPI latest" in md
    assert "Lab snapshot" in md
    assert "lab_smoke_all.sh" in md


def test_generate_script_writes_file():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_evidence_dashboard.py")],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = ROOT / "docs/EVIDENCE_DASHBOARD.md"
    assert out.is_file()
    assert "walkthrough_a_class.sh" in out.read_text(encoding="utf-8")
