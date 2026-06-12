#!/usr/bin/env python3
"""Write PyPI vs local version status JSON (optional network)."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument(
        "--require-published",
        action="store_true",
        help="Exit 1 unless PyPI latest matches local __version__",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry PyPI fetch when --require-published (default: 1)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds between retries (default: 10)",
    )
    args = parser.parse_args()

    if args.offline and args.require_published:
        print("ERROR: --require-published needs network fetch", file=sys.stderr)
        return 2

    attempts = max(1, args.retries)
    status: dict = {}
    for attempt in range(attempts):
        status = build_pypi_release_status(ROOT, fetch_remote=not args.offline)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(status, indent=2), encoding="utf-8")

        if not args.require_published or status["status"] == "published":
            break
        if attempt < attempts - 1:
            print(f"PyPI not yet published; retry {attempt + 2}/{attempts} in {args.retry_delay}s")
            time.sleep(args.retry_delay)

    print("PyPI release status")
    print("=" * 50)
    print(f"local:  {status['local_version']}")
    print(f"pypi:   {status['pypi_version']}")
    print(f"status: {status['status']}")
    print(f"note:   {status['note']}")
    print(f"Wrote {args.json_out}")

    if args.require_published and status["status"] != "published":
        print("ERROR: PyPI latest does not match local version", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
