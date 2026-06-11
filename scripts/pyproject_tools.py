#!/usr/bin/env python3
"""Read and validate pyproject.toml on Python 3.9+ (tomllib or tomli)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pyproject(path: Path | None = None) -> dict:
    path = path or ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef,import-not-found]
    return tomllib.loads(text)


def project_version(path: Path | None = None) -> str:
    return str(load_pyproject(path)["project"]["version"])


def check_version_alignment(version: str) -> None:
    data = load_pyproject()
    if data["project"]["version"] != version:
        raise SystemExit(
            f"pyproject.toml version {data['project']['version']!r} != {version!r}"
        )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"\[{re.escape(version)}\]", changelog):
        raise SystemExit(f"CHANGELOG.md missing [{version}] section")
    init = (ROOT / "src/llmir/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    if not match or match.group(1) != version:
        raise SystemExit(f"src/llmir/__init__.py __version__ must be {version}")
    print("version alignment OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pyproject.toml helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print project.version")
    check = sub.add_parser("check-alignment", help="validate version files")
    check.add_argument("version")
    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(project_version())
    elif args.cmd == "check-alignment":
        check_version_alignment(args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
