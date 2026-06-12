"""PyPI release status helper and verify script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from llmir.benchmark.pypi_release_status import (
    build_pypi_release_status,
    read_local_version,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pypi_release.py"


def test_read_local_version_matches_pyproject():
    version = read_local_version(ROOT)
    assert version == "0.2.2"


def test_build_pypi_release_status_published_when_versions_match():
    with patch(
        "llmir.benchmark.pypi_release_status.fetch_pypi_version",
        return_value=read_local_version(ROOT),
    ):
        status = build_pypi_release_status(ROOT, fetch_remote=True)
    assert status["status"] == "published"
    assert status["published"] is True


def test_build_pypi_release_status_pending_when_behind():
    with patch(
        "llmir.benchmark.pypi_release_status.fetch_pypi_version",
        return_value="0.0.19",
    ):
        status = build_pypi_release_status(ROOT, fetch_remote=True)
    assert status["status"] == "pending"
    assert status["published"] is False
    assert status["local_version"] == "0.2.2"


def test_verify_pypi_release_offline_writes_json(tmp_path):
    out = tmp_path / "pypi_release_status.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--offline", "--json-out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["local_version"] == "0.2.2"
    assert data["status"] == "unavailable"
