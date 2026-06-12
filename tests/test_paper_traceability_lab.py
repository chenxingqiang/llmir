"""Paper traceability includes lab commands."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_traceability_lists_lab_and_release_commands():
    text = (ROOT / "docs/PAPER_REVISION_TRACEABILITY.md").read_text(encoding="utf-8")
    assert "lab_smoke_all.sh" in text
    assert "verify_lab_gates.py" in text
    assert "mlir_lit_smoke.sh" in text
    assert "CI_WORKFLOW_INDEX.md" in text
