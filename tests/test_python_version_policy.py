"""Python version policy matches packaging and CI."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requires_python_is_39_plus():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"(>=3\.(\d+))"', text)
    assert match is not None
    minor = int(match.group(2))
    assert minor >= 9, match.group(1)


def test_ci_matrix_excludes_python38():
    workflow = (ROOT / ".github/workflows/python-package.yml").read_text(encoding="utf-8")
    assert '"3.8"' not in workflow
    for version in ("3.9", "3.10", "3.11", "3.12"):
        assert version in workflow


def test_no_py38_classifier():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "Python :: 3.8" not in text
