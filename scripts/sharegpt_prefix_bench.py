#!/usr/bin/env python3
"""MVP-B: ShareGPT-style prefix reuse benchmark (KV sim + llmir_paged E2E)."""

from __future__ import annotations

import argparse
import json
import sys

from llmir.benchmark.sharegpt_prefix_bench import (
    ShareGPTPrefixBenchConfig,
    print_sharegpt_results,
    run_sharegpt_prefix_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ShareGPT-style shared system prompt + N user variants (MVP-B)"
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--system-prompt-tokens",
        type=int,
        default=128,
        help="Approximate shared system prompt length (use 2048 on GPU demos)",
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--user-suffix-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Skip HuggingFace llmir_paged E2E",
    )
    parser.add_argument(
        "--llmir-only",
        action="store_true",
        help="Skip KV-layer simulation",
    )
    parser.add_argument("-o", "--output", default="sharegpt_prefix_bench.json")
    args = parser.parse_args()

    cfg = ShareGPTPrefixBenchConfig(
        system_prompt_tokens=args.system_prompt_tokens,
        num_requests=args.num_requests,
        user_suffix_tokens=args.user_suffix_tokens,
        max_new_tokens=args.max_new_tokens,
        model=args.model,
        device=args.device,
    )
    payload = run_sharegpt_prefix_benchmark(
        cfg,
        run_simulation=not args.llmir_only,
        run_llmir_paged=not args.simulation_only,
    )
    print_sharegpt_results(payload)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")
    if payload.get("llmir_paged_error"):
        print(f"llmir_paged skipped: {payload['llmir_paged_error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
