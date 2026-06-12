"""Lab closure matrix documentation and hub scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_closure_matrix_doc_exists():
    text = (ROOT / "docs/LAB_CLOSURE_MATRIX.md").read_text(encoding="utf-8")
    for needle in (
        "mlir_lit_preflight.sh",
        "e8_lab_preflight.sh",
        "pypi_republish_preflight.sh",
        "lab_smoke_all.sh",
        "--require-passed",
        "--require-completed",
        "--require-published",
    ):
        assert needle in text


def test_lab_smoke_all_runs_preflights_and_gates():
    text = (ROOT / "scripts/lab_smoke_all.sh").read_text(encoding="utf-8")
    assert "mlir_lit_preflight.sh" in text
    assert "e8_lab_preflight.sh" in text
    assert "verify_lab_gates.py" in text


def test_walkthrough_script_runs_gates():
    text = (ROOT / "scripts/walkthrough_a_class.sh").read_text(encoding="utf-8")
    assert "verify_lab_gates.py" in text
    assert "verify_walkthrough_gates.py" in text


def test_lab_status_summary_lists_closure_commands():
    from llmir.benchmark.lab_status_summary import build_lab_status_summary

    data = build_lab_status_summary(ROOT)
    commands = data["commands"]
    for key in (
        "mlir_lit_preflight",
        "mlir_lit_strict",
        "e8_preflight",
        "e8_strict",
        "pypi_strict",
        "lab_gates",
    ):
        assert key in commands
