#!/usr/bin/env python3
"""E5 ablation at verifiable layers — switch matrix + JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.e4_compositional import E4WorkloadTrace, trace_from_sim_json  # noqa: E402
from llmir.benchmark.e5_ablation import run_e5_ablation  # noqa: E402

DEFAULT_SIM = ROOT / "IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json"
DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/e5_ablation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="E5 ablation at verifiable layers")
    parser.add_argument(
        "--from-sim",
        type=Path,
        default=None,
        help="Load trace from shared_prefix_decoder sim JSON",
    )
    parser.add_argument("--shared-prefix-tokens", type=int, default=None)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--suffix-tokens", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--model-preset", default="qwen3-8b")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.from_sim:
        trace = trace_from_sim_json(args.from_sim)
    else:
        ls = args.shared_prefix_tokens if args.shared_prefix_tokens is not None else 2048
        n = args.num_requests if args.num_requests is not None else 32
        lu = args.suffix_tokens if args.suffix_tokens is not None else 8
        trace = E4WorkloadTrace(
            shared_prefix_tokens=ls,
            num_requests=n,
            suffix_tokens=lu,
            decode_steps=args.decode_steps,
            model_preset=args.model_preset,
        )

    result = run_e5_ablation(trace)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("E5 ablation at verifiable layers")
    print("=" * 50)
    print(f"trace: L_s={trace.shared_prefix_tokens} N={trace.num_requests} L_u={trace.suffix_tokens}")
    for row in payload["configurations"]:
        p = row["proxies"]
        print(
            f"  {row['name']:22s}  "
            f"E1={p['block_size_reduction_ratio']:.3f}  "
            f"E2={p['prefill_reduction_ratio']:.3f}  "
            f"E3={p['host_copy_reduction_ratio']:.3f}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
