"""LAB_RUNBOOK and lab-smoke workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_runbook_exists_and_links_workflows():
    text = (ROOT / "docs/LAB_RUNBOOK.md").read_text(encoding="utf-8")
    assert "lab_smoke_all.sh" in text
    assert "PYPI_TRUSTED_PUBLISHER.md" in text
    assert "MLIR_LIT_RUNBOOK.md" in text


def test_walkthrough_doc_lists_nine_steps():
    text = (ROOT / "docs/WALKTHROUGH.md").read_text(encoding="utf-8")
    assert "lab_status_summary.json" in text
    assert "LAB_RUNBOOK.md" in text
    assert "| 9 |" in text
