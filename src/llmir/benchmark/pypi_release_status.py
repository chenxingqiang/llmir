"""Compare local package version with PyPI (optional network)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

PYPI_JSON_URL = "https://pypi.org/pypi/llmir/json"


def read_local_version(root: Path) -> str:
    init = root / "src/llmir/__init__.py"
    text = init.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise ValueError(f"__version__ not found in {init}")
    return match.group(1)


def fetch_pypi_version(
    package: str = "llmir",
    *,
    timeout: float = 10.0,
) -> Optional[str]:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    info = data.get("info") or {}
    version = info.get("version")
    return str(version) if version else None


def build_pypi_release_status(
    root: Path,
    *,
    fetch_remote: bool = True,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    local = read_local_version(root)
    remote: Optional[str] = None
    if fetch_remote:
        remote = fetch_pypi_version(timeout=timeout)

    if remote is None:
        status = "unavailable"
        published = False
        note = "Could not reach PyPI or parse response"
    elif remote == local:
        status = "published"
        published = True
        note = "PyPI latest matches local __version__"
    else:
        status = "pending"
        published = False
        note = (
            f"PyPI latest ({remote}) differs from local ({local}); "
            "check trusted publisher or manual upload"
        )

    return {
        "experiment": "pypi_release",
        "evidence_class": "A",
        "package": "llmir",
        "local_version": local,
        "pypi_version": remote,
        "published": published,
        "status": status,
        "pypi_json_url": PYPI_JSON_URL,
        "note": note,
    }
