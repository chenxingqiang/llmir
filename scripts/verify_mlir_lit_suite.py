#!/usr/bin/env python3
"""Verify MLIR lit files under test/Dialect/LLM/ (runs when mlir-opt is available)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.compiler.mlir_lit_suite import (  # noqa: E402
    LIT_SUITE_FILES,
    default_lit_dir,
    run_lit_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLIR lit suite verify")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "IEEE-conference/benchmarks/mlir_lit_suite_status.json",
    )
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="Exit 1 unless lit suite status is passed (4/4 green)",
    )
    args = parser.parse_args()

    summary = run_lit_suite(default_lit_dir(ROOT))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MLIR lit suite")
    print("=" * 50)
    print(f"mlir-opt: {summary.get('mlir_opt') or 'not found'}")
    print(f"catalog: {len(LIT_SUITE_FILES)} files")
    for row in summary["files"]:
        print(f"  {Path(row['path']).name}: {row['status']}")
        if row.get("reason"):
            print(f"    reason: {row['reason']}")
    print(f"Wrote {args.json_out}")

    if summary["status"] == "failed":
        return 1
    if args.require_passed and summary["status"] != "passed":
        print(
            f"ERROR: MLIR lit suite status={summary['status']!r}; expected passed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
