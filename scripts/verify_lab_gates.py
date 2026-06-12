#!/usr/bin/env python3
"""Validate lab_status_summary.json (optional lab; honest skip OK)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.lab_status_summary import build_lab_status_summary  # noqa: E402

LAB_JSON = ROOT / "IEEE-conference/benchmarks/lab_status_summary.json"
ALLOWED = {
    "mlir_lit_status": ("skipped", "passed", "failed", "missing"),
    "e8_status": ("skipped", "completed", "missing"),
    "pypi_release_status": ("published", "pending", "unavailable", "missing"),
}


def verify_lab_gates(root: Path | None = None) -> tuple[list[str], list[str]]:
    root = root or ROOT
    errors: list[str] = []
    notes: list[str] = []

    if LAB_JSON.is_file():
        summary = json.loads(LAB_JSON.read_text(encoding="utf-8"))
    else:
        summary = build_lab_status_summary(root)
        notes.append("lab_status_summary.json missing; used live summary")

    if summary.get("mode") != "lab_status_summary":
        errors.append("lab summary: mode must be lab_status_summary")

    for key in ("package_version", "mlir_lit_status", "e8_status", "pypi_release_status"):
        if key not in summary:
            errors.append(f"lab summary: missing {key}")

    for key, allowed in ALLOWED.items():
        value = summary.get(key)
        if value is not None and value not in allowed:
            errors.append(f"lab summary: {key}={value!r} not in {allowed}")

    if summary.get("mlir_lit_status") == "skipped":
        notes.append("mlir lit skipped (expected without in-tree mlir-opt)")
    if summary.get("e8_status") == "skipped":
        notes.append("E8 skipped (expected on CPU CI)")
    if summary.get("pypi_release_status") == "pending":
        notes.append("PyPI publish pending (configure trusted publisher)")
    if summary.get("native_build_prereqs_ok") is False:
        notes.append("native build prereqs incomplete (see MLIR_NATIVE_BUILD.md)")

    return errors, notes


def main() -> int:
    errors, notes = verify_lab_gates(ROOT)
    for note in notes:
        print(f"NOTE: {note}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("Lab gates OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
