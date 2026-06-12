"""CI workflow files for walkthrough and E8 lab."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_a_class_walkthrough_workflow_exists():
    path = ROOT / ".github/workflows/a-class-walkthrough.yml"
    text = path.read_text(encoding="utf-8")
    assert "walkthrough_a_class.sh" in text
    assert "workflow_dispatch" in text


def test_release_prep_workflow_exists():
    path = ROOT / ".github/workflows/release-prep.yml"
    text = path.read_text(encoding="utf-8")
    assert "prepare_release.sh" in text
    assert "workflow_dispatch" in text


def test_mlir_lit_lab_workflow_exists():
    path = ROOT / ".github/workflows/mlir-lit-lab.yml"
    text = path.read_text(encoding="utf-8")
    assert "mlir_lit_smoke.sh" in text
    assert "workflow_dispatch" in text


def test_release_tag_workflow_exists():
    path = ROOT / ".github/workflows/release-tag.yml"
    text = path.read_text(encoding="utf-8")
    assert "tag_release.sh" in text
    assert "workflow_dispatch" in text


def test_e8_workflow_and_lab_script_exist():
    wf = ROOT / ".github/workflows/e8-empirical-gpu.yml"
    lab = ROOT / "scripts/e8_lab_run.sh"
    assert wf.is_file()
    assert lab.is_file()
    assert "e8_empirical_gpu_bench.py" in wf.read_text(encoding="utf-8")
    assert "status=completed" in lab.read_text(encoding="utf-8")
