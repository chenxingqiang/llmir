#!/usr/bin/env python3
"""Validate E8 empirical GPU JSON (optional B-class lab closure)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "IEEE-conference/benchmarks/e8_empirical_gpu.json"


def verify_e8_lab(
    json_path: Path,
    *,
    require_completed: bool = False,
) -> tuple[list[str], dict]:
    errors: list[str] = []
    if not json_path.is_file():
        errors.append(f"missing {json_path}")
        return errors, {}

    data = json.loads(json_path.read_text(encoding="utf-8"))

    if data.get("experiment") != "E8":
        errors.append("experiment must be E8")
    if data.get("evidence_class") != "B":
        errors.append("evidence_class must be B")

    status = data.get("status")
    if status not in ("skipped", "completed"):
        errors.append(f"unexpected status: {status!r}")

    results = data.get("results", [])
    if status == "completed" and len(results) < 1:
        errors.append("status=completed but results is empty")

    if require_completed:
        if status != "completed":
            errors.append(f"require_completed: status={status!r}")
        elif len(results) < 1:
            errors.append("require_completed: no result rows")

    return errors, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify E8 empirical GPU JSON")
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to e8_empirical_gpu.json",
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Exit 1 unless status=completed with result rows",
    )
    args = parser.parse_args()

    errors, data = verify_e8_lab(args.json, require_completed=args.require_completed)

    print("E8 lab verify")
    print("=" * 50)
    if data:
        print(f"status: {data.get('status')}")
        print(f"rows:   {len(data.get('results', []))}")
        if data.get("reason"):
            print(f"reason: {data['reason']}")
    else:
        print(f"file:   {args.json} (missing)")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("E8 lab verify OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
