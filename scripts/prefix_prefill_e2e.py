#!/usr/bin/env python3
"""E2E prefix prefill savings on llmir_paged (requires llmir[full])."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefix prefill e2e benchmark")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--system-prompt", default="hello " * 32)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("-o", "--output", default="prefix_prefill_e2e.json")
    args = parser.parse_args()

    try:
        from llmir import LLMEngine, SamplingParams
    except ImportError:
        print("llmir not installed", file=sys.stderr)
        return 1

    engine = LLMEngine.from_pretrained(args.model, backend="llmir_paged", dtype="float32")
    assert engine._paged_decoder is not None
    engine._paged_decoder.warm_prefix(args.system_prompt)

    rows = []
    for i in range(args.requests):
        prompt = args.system_prompt + f" suffix{i}"
        start = time.perf_counter()
        out = engine.generate(
            [prompt],
            SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - start
        metrics = out.metrics or {}
        rows.append(
            {
                "request": i,
                "elapsed_s": elapsed,
                "prefix_hit_tokens": metrics.get("prefix_hit_tokens", 0),
                "prefill_tokens_computed": metrics.get("prefill_tokens_computed", 0),
            }
        )

    engine.shutdown()
    payload = {"model": args.model, "rows": rows}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"{'req':>4} {'hit':>6} {'prefill':>8} {'time_ms':>10}")
    for row in rows:
        print(
            f"{row['request']:>4} {row['prefix_hit_tokens']:>6} "
            f"{row['prefill_tokens_computed']:>8} {row['elapsed_s'] * 1000:>10.1f}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
