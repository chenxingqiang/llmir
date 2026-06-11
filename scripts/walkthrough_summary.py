#!/usr/bin/env python3
"""Build a reviewer-facing JSON summary after walkthrough_a_class.sh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.walkthrough_summary import build_walkthrough_summary  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/walkthrough_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Walkthrough summary JSON")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = build_walkthrough_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Walkthrough summary")
    print("=" * 40)
    print(f"m6_all_pass: {payload['m6_all_pass']}")
    print(f"e8_status: {payload['e8_status']}")
    print(f"mlir_lit: {payload['mlir_lit_suite_status']}")
    print(f"Wrote {args.output}")
    return 0 if payload["m6_all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
