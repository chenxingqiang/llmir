#!/usr/bin/env python3
"""E4 compositional verification — trace-driven E1+E2+E3 analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.e4_compositional import (  # noqa: E402
    E4WorkloadTrace,
    run_e4_compositional_verification,
    trace_from_sim_json,
)

DEFAULT_SIM = ROOT / "IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json"
DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/e4_compositional.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="E4 compositional verification (E1+E2+E3)")
    parser.add_argument(
        "--from-sim",
        type=Path,
        default=None,
        help="Load L_s/N/L_u from shared_prefix_decoder sim JSON",
    )
    parser.add_argument("--shared-prefix-tokens", type=int, default=None)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--suffix-tokens", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--block-size-before", type=int, default=1024)
    parser.add_argument("--block-size-after", type=int, default=32)
    parser.add_argument("--model-preset", default="qwen3-8b")
    parser.add_argument(
        "--compare-sim",
        type=Path,
        default=DEFAULT_SIM,
        help="E2 sim JSON for measured_comparison (pass '' to skip)",
    )
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
            block_size_before=args.block_size_before,
            block_size_after=args.block_size_after,
            model_preset=args.model_preset,
        )

    sim_path = args.compare_sim if str(args.compare_sim) else None
    result = run_e4_compositional_verification(
        trace,
        measured_sim_json=sim_path,
    )
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    comp = payload["composite"]["compile_time_levers"]
    print("E4 compositional verification")
    print("=" * 50)
    print(f"trace: L_s={trace.shared_prefix_tokens} N={trace.num_requests} L_u={trace.suffix_tokens}")
    print(f"E1 block_size reduction ratio: {comp['e1_block_size_reduction']:.3f}")
    print(f"E2 prefill reduction ratio: {comp['e2_prefill_token_reduction']:.3f}")
    print(f"E3 host copy elimination: {comp['e3_host_copy_elimination']:.3f}")
    if payload.get("measured_comparison"):
        mc = payload["measured_comparison"]
        print(
            f"E2 sim speedup {mc.get('measured_kv_sim_speedup'):.2f}x "
            f"(ideal bound {mc.get('ideal_kv_speedup_model'):.2f}x)"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
