#!/usr/bin/env python3
"""Post-walkthrough gate checks for CI and local runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.walkthrough_summary import build_walkthrough_summary  # noqa: E402

SUMMARY = ROOT / "IEEE-conference/benchmarks/walkthrough_summary.json"
DASHBOARD = ROOT / "docs/EVIDENCE_DASHBOARD.md"


def main() -> int:
    errors: list[str] = []

    if not SUMMARY.is_file():
        errors.append(f"missing {SUMMARY}")
    else:
        on_disk = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if not on_disk.get("m6_all_pass"):
            errors.append("walkthrough_summary.json: m6_all_pass is not true")
        fresh = build_walkthrough_summary(ROOT)
        if fresh["m6_all_pass"] is not True:
            errors.append("live walkthrough summary: m6_all_pass is not true")
        if not fresh.get("package_version"):
            errors.append("walkthrough summary: package_version missing")
        pypi_status = fresh.get("pypi_release_status")
        if pypi_status == "pending":
            print(
                "NOTE: PyPI release pending "
                f"(local={fresh.get('package_version')}, "
                f"pypi={fresh.get('pypi_version')})",
                file=sys.stderr,
            )

    if not DASHBOARD.is_file():
        errors.append(f"missing {DASHBOARD}")
    else:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/generate_evidence_dashboard.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            errors.append(f"generate_evidence_dashboard failed: {proc.stderr}")
        elif "M6 artifact bundle" not in DASHBOARD.read_text(encoding="utf-8"):
            errors.append("EVIDENCE_DASHBOARD.md missing M6 section")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("Walkthrough gates OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
