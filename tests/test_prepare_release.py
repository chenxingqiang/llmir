"""Release prep script and version/changelog alignment."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_release.sh"
CHECKLIST = ROOT / "docs/PYPI_RELEASE_CHECKLIST.md"
TOOLS = ROOT / "scripts/pyproject_tools.py"


def _load_pyproject() -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "from pyproject_tools import load_pyproject; "
            "import json; print(json.dumps(load_pyproject()))",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return json.loads(proc.stdout)


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
    data = _load_pyproject()
    version = data["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"\[{re.escape(version)}\]", changelog), (
        f"version {version} not found as [X.Y.Z] section in CHANGELOG.md"
    )


def test_package_version_matches_pyproject():
    data = _load_pyproject()
    expected = data["project"]["version"]
    init = (ROOT / "src/llmir/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    assert match, "__version__ not found in src/llmir/__init__.py"
    assert match.group(1) == expected


def test_pyproject_tools_reads_version():
    proc = subprocess.run(
        [sys.executable, str(TOOLS), "version"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    version = proc.stdout.strip()
    assert version == _load_pyproject()["project"]["version"]
