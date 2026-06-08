#!/usr/bin/env python3
"""MVP-C: NumPy vs torch-CUDA KV backend benchmark for llmir_paged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmir.benchmark.mvp_c_cuda_kv_bench import (  # noqa: E402
    MVPCudaKVBenchConfig,
    print_mvp_c_results,
    run_mvp_c_cuda_kv_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MVP-C CUDA KV backend benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect)")
    parser.add_argument(
        "--backends",
        default="numpy,torch_cuda",
        help="Comma-separated LLMIR_KV_BACKEND values",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("-o", "--output", default="mvp_c_cuda_kv.json")
    args = parser.parse_args()

    cfg = MVPCudaKVBenchConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        backends=[b.strip() for b in args.backends.split(",") if b.strip()],
        device=args.device,
    )
    print("LLMIR MVP-C CUDA KV benchmark")
    print("=" * 50)
    payload = run_mvp_c_cuda_kv_benchmark(cfg)
    print_mvp_c_results(payload)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
