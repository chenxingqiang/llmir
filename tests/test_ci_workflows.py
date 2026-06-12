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


def test_pypi_republish_workflow_exists():
    path = ROOT / ".github/workflows/pypi-republish.yml"
    text = path.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "pypa/gh-action-pypi-publish" in text


def test_release_tag_workflow_exists():
    path = ROOT / ".github/workflows/release-tag.yml"
    text = path.read_text(encoding="utf-8")
    assert "tag_release.sh" in text
    assert "workflow_dispatch" in text


def test_e8_workflow_and_lab_script_exist():
    wf = ROOT / ".github/workflows/e8-empirical-gpu.yml"
    lab_wf = ROOT / ".github/workflows/e8-gpu-lab.yml"
    smoke = ROOT / "scripts/e8_lab_smoke.sh"
    lab = ROOT / "scripts/e8_lab_run.sh"
    assert wf.is_file()
    assert lab_wf.is_file()
    assert smoke.is_file()
    assert lab.is_file()
    assert "e8_lab_smoke.sh" in wf.read_text(encoding="utf-8")
    assert "e8_lab_run.sh" in lab_wf.read_text(encoding="utf-8")
    assert "require_completed" in lab_wf.read_text(encoding="utf-8")
