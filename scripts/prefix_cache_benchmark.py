#!/usr/bin/env python3
"""Prefix cache benchmark with JSON output."""

from __future__ import annotations

import argparse
import json
import sys

from llmir.benchmark.prefix_cache_bench import (
    bench_prefix_kv_reuse,
    bench_prefix_lookup_throughput,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLMIR prefix cache benchmark")
    parser.add_argument("-o", "--output", default="prefix_cache_bench.json")
    parser.add_argument("--num-queries", type=int, default=10_000)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=200)
    args = parser.parse_args()

    lookup = bench_prefix_lookup_throughput(num_queries=args.num_queries)
    kv_rows = bench_prefix_kv_reuse(
        num_requests=args.num_requests,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
    )

    print(f"lookup throughput: {lookup.throughput_ops_s:,.0f} ops/s")
    print(f"lookup hit ratio:  {lookup.hit_ratio:.1%}")
    for row in kv_rows:
        print(
            f"{row.scenario}: {row.elapsed_s * 1000:.2f} ms "
            f"(speedup {row.speedup_vs_baseline:.2f}x vs baseline)"
        )

    payload = {
        "mode": "prefix_cache",
        "lookup": lookup.to_dict(),
        "kv_reuse": [r.to_dict() for r in kv_rows],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
