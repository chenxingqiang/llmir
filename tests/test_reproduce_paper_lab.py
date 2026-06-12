"""reproduce_paper.sh lab tail alignment."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_paper.sh"


def test_reproduce_paper_uses_lab_smoke_scripts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "mlir_lit_smoke.sh" in text
    assert "e8_lab_smoke.sh" in text
    assert "lab_status_summary.py" in text
    assert "verify_lab_gates.py" in text
