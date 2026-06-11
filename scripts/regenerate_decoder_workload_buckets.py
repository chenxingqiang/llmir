#!/usr/bin/env python3
"""Regenerate or verify S1/S2/S3 shared-prefix decoder sim JSON artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.decoder_workload_buckets import (  # noqa: E402
    DECODER_WORKLOAD_BUCKETS,
    verify_bucket_artifact,
    write_bucket_artifact,
)

BENCH_DIR = ROOT / "IEEE-conference/benchmarks"


def main() -> int:
    parser = argparse.ArgumentParser(description="S1/S2/S3 decoder workload bucket artifacts")
    parser.add_argument(
        "--bucket",
        choices=sorted(DECODER_WORKLOAD_BUCKETS),
        help="Single bucket (default: all)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check committed JSON configs without rewriting",
    )
    args = parser.parse_args()

    buckets = (
        [DECODER_WORKLOAD_BUCKETS[args.bucket]]
        if args.bucket
        else list(DECODER_WORKLOAD_BUCKETS.values())
    )

    errors: list[str] = []
    for bucket in buckets:
        if args.verify_only:
            err = verify_bucket_artifact(bucket, BENCH_DIR)
            if err:
                errors.append(err)
            else:
                print(f"OK {bucket.bucket_id} {bucket.artifact_name}")
        else:
            path = write_bucket_artifact(bucket, BENCH_DIR)
            print(f"Wrote {path}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
