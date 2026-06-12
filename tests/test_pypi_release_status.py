"""PyPI release status helper and verify script."""

from __future__ import annotations

import importlib.util
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


def _load_verify_main():
    spec = importlib.util.spec_from_file_location("verify_pypi_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


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


def test_verify_pypi_release_require_published_fails_when_pending(tmp_path):
    out = tmp_path / "pypi_release_status.json"
    main = _load_verify_main()
    with patch(
        "llmir.benchmark.pypi_release_status.fetch_pypi_version",
        return_value="0.0.19",
    ), patch(
        "sys.argv",
        [
            "verify_pypi_release.py",
            "--require-published",
            "--retries",
            "1",
            "--json-out",
            str(out),
        ],
    ):
        code = main()
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "pending"


def test_verify_pypi_release_require_published_ok_when_match(tmp_path):
    out = tmp_path / "pypi_release_status.json"
    main = _load_verify_main()
    with patch(
        "llmir.benchmark.pypi_release_status.fetch_pypi_version",
        return_value=read_local_version(ROOT),
    ), patch(
        "sys.argv",
        ["verify_pypi_release.py", "--require-published", "--json-out", str(out)],
    ):
        code = main()
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "published"


def test_pypi_republish_preflight_script():
    script = ROOT / "scripts/pypi_republish_preflight.sh"
    assert script.is_file()
    proc = subprocess.run(
        ["bash", str(script), "v0.2.2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Preflight OK" in proc.stdout
    assert "version alignment OK" in proc.stdout
