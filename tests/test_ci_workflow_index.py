"""CI workflow index documentation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_index_lists_core_workflows():
    text = (ROOT / "docs/CI_WORKFLOW_INDEX.md").read_text(encoding="utf-8")
    for wf in (
        "a-class-walkthrough.yml",
        "python-package.yml",
        "lab-smoke.yml",
        "native-runtime.yml",
        "pypi-republish.yml",
        "mlir-lit-lab.yml",
        "e8-gpu-lab.yml",
    ):
        assert wf in text
    assert "workflow_dispatch" in text or "Manual" in text
