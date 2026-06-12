#!/usr/bin/env python3
"""Write lab status summary JSON from smoke artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.lab_status_summary import build_lab_status_summary  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/lab_status_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write lab status summary JSON")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = build_lab_status_summary(ROOT)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
