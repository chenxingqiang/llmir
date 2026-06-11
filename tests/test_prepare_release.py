"""Release prep script and version/changelog alignment."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_release.sh"
CHECKLIST = ROOT / "docs/PYPI_RELEASE_CHECKLIST.md"


def test_prepare_release_script_exists():
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ci_lint_gate.sh" in text
    assert "python3 -m build" in text


def test_release_checklist_documents_tag_publish():
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "prepare_release.sh" in text
    assert "python-package.yml" in text


def test_pyproject_version_in_changelog():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"\[{re.escape(version)}\]", changelog), (
        f"version {version} not found as [X.Y.Z] section in CHANGELOG.md"
    )


def test_prepare_release_dry_imports():
    """Ensure tomllib version read works (used by prepare_release.sh banner)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
