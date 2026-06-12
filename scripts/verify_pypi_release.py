#!/usr/bin/env python3
"""Write PyPI vs local version status JSON (optional network)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.pypi_release_status import build_pypi_release_status  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/pypi_release_status.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PyPI release alignment")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip PyPI fetch; local version only",
    )
    args = parser.parse_args()

    status = build_pypi_release_status(ROOT, fetch_remote=not args.offline)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("PyPI release status")
    print("=" * 50)
    print(f"local:  {status['local_version']}")
    print(f"pypi:   {status['pypi_version']}")
    print(f"status: {status['status']}")
    print(f"note:   {status['note']}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
