"""Release tag script and workflow contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tag_release.sh"
WORKFLOW = ROOT / ".github/workflows/release-tag.yml"


def test_tag_release_script_exists():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "test_prepare_release.py" in text
    assert "--dry-run" in text
    assert "--push" in text


def test_release_tag_workflow_dispatch():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "tag_release.sh" in text
    assert "contents: write" in text


def test_tag_release_dry_run_passes():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "DRY RUN" in proc.stdout
    assert "0.2.2" in proc.stdout or "v0.2.2" in proc.stdout or "already exists" in proc.stdout
