#!/usr/bin/env python3
"""Compare CPU inference throughput between LLMIR serving and vLLM."""

from __future__ import annotations

import argparse
import json
import sys

from llmir.benchmark.inference_compare import (
    print_inference_results,
    results_to_json,
    run_inference_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU inference benchmark comparing LLMIR with vLLM"
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-llmir-vllm-backend", action="store_true")
    parser.add_argument("--skip-llmir-paged", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = ["llmir"]
    if not args.skip_vllm:
        backends.append("vllm")
    if not args.skip_llmir_vllm_backend:
        backends.append("llmir+vllm")
    if not args.skip_llmir_paged:
        backends.append("llmir-paged")

    results = run_inference_compare(
        args.model,
        backends,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
    )
    print_inference_results(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_to_json(results), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
