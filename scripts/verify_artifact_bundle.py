#!/usr/bin/env python3
"""M6: verify CPU artifact bundle against artifact_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.artifact_bundle import verify_artifact_bundle  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/artifact_bundle_status.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify A-class artifact bundle")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not require figure PDFs (CI without matplotlib regen)",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = verify_artifact_bundle(ROOT, check_figures=not args.skip_figures)
    payload = report.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("M6 artifact bundle verification")
    print("=" * 50)
    for row in payload["artifacts"]:
        status = "OK" if row["ok"] else "FAIL"
        print(f"  [{status}] {row['id']:22s} {row.get('message', '')}")
    if payload["figures"]:
        print("Figures:")
        for row in payload["figures"]:
            status = "OK" if row["ok"] else "MISSING"
            print(f"  [{status}] {row['path']}")
    print(f"all_pass: {payload['all_pass']}")
    print(f"Wrote {args.output}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
