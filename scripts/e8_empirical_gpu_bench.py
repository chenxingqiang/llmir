#!/usr/bin/env python3
"""E8 optional empirical GPU benchmark (B-class). Skips cleanly without CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.e8_empirical_gpu import (  # noqa: E402
    E8EmpiricalConfig,
    e8_result_to_json,
    run_e8_empirical_gpu_bench,
)

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/e8_empirical_gpu.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="E8 empirical GPU bench (optional B-class)")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--backends", default="hf,llmir-paged")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = E8EmpiricalConfig(
        model=args.model,
        backends=tuple(b.strip() for b in args.backends.split(",") if b.strip()),
    )
    result = run_e8_empirical_gpu_bench(cfg)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(e8_result_to_json(result), encoding="utf-8")

    print("E8 empirical GPU benchmark (B-class)")
    print("=" * 50)
    print(f"status: {payload['status']}")
    print(f"cuda: {payload['cuda_stack']}")
    if payload.get("reason"):
        print(f"reason: {payload['reason']}")
    print(f"rows: {len(payload.get('results', []))}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
