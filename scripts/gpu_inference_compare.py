#!/usr/bin/env python3
"""GPU-oriented E2E inference compare (HF / vLLM / llmir-paged)."""

from __future__ import annotations

import argparse
import json
import sys

from llmir.benchmark.device import cuda_available, resolve_inference_device
from llmir.benchmark.inference_compare import (
    print_inference_results,
    results_to_json,
    run_inference_compare,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare inference backends on GPU when available"
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--backends",
        default="hf,vllm,llmir-paged",
        help="Comma-separated backends",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto prefers CUDA",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("-o", "--output", default="gpu_inference_compare.json")
    args = parser.parse_args()

    resolved = resolve_inference_device(args.device)
    print(f"CUDA available: {cuda_available()}")
    print(f"Resolved device: {resolved.device} ({resolved.note})")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = run_inference_compare(
        args.model,
        backends,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        device=args.device,
        dtype=args.dtype,
    )
    print_inference_results(results)
    payload = {
        "mode": "gpu_inference_compare",
        "model": args.model,
        "cuda_available": cuda_available(),
        "device": args.device,
        "dtype": args.dtype,
        "results": results_to_json(results),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
