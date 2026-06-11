"""pyproject_tools works on all supported Python versions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts/pyproject_tools.py"


def test_pyproject_tools_version_matches_package():
    proc = subprocess.run(
        [sys.executable, str(TOOLS), "version"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    version = proc.stdout.strip()
    init = (ROOT / "src/llmir/__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version}"' in init or f"__version__ = '{version}'" in init


def test_pyproject_tools_check_alignment():
    proc = subprocess.run(
        [sys.executable, str(TOOLS), "check-alignment", "0.2.2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "version alignment OK" in proc.stdout
