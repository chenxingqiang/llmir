#!/usr/bin/env python3
"""E4/E5 multi-bucket verification across S1/S2/S3 decoder workload traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.e4_compositional import run_e4_multi_bucket_verification  # noqa: E402
from llmir.benchmark.e5_ablation import run_e5_multi_bucket_ablation  # noqa: E402

BENCH_DIR = ROOT / "IEEE-conference/benchmarks"
DEFAULT_E4_OUT = BENCH_DIR / "e4_compositional_buckets.json"
DEFAULT_E5_OUT = BENCH_DIR / "e5_ablation_buckets.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="E4/E5 S1/S2/S3 multi-bucket verify")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=BENCH_DIR,
        help="Directory with shared_prefix_decoder_*_sim.json",
    )
    parser.add_argument(
        "--e4-output",
        type=Path,
        default=DEFAULT_E4_OUT,
        help="Aggregated E4 multi-bucket JSON",
    )
    parser.add_argument(
        "--e5-output",
        type=Path,
        default=DEFAULT_E5_OUT,
        help="Aggregated E5 multi-bucket JSON",
    )
    parser.add_argument("--e4-only", action="store_true")
    parser.add_argument("--e5-only", action="store_true")
    args = parser.parse_args()

    run_e4 = not args.e5_only
    run_e5 = not args.e4_only

    if run_e4:
        e4_payload = run_e4_multi_bucket_verification(args.benchmarks_dir)
        args.e4_output.parent.mkdir(parents=True, exist_ok=True)
        args.e4_output.write_text(json.dumps(e4_payload, indent=2), encoding="utf-8")
        print("E4 multi-bucket compositional verification")
        print("=" * 50)
        for row in e4_payload["buckets"]:
            analysis = row["analysis"]
            comp = analysis["composite"]["compile_time_levers"]
            mc = analysis.get("measured_comparison") or {}
            print(
                f"  {row['bucket_id']}: "
                f"E1={comp['e1_block_size_reduction']:.3f} "
                f"E2={comp['e2_prefill_token_reduction']:.3f} "
                f"E3={comp['e3_host_copy_elimination']:.3f} "
                f"sim={mc.get('measured_kv_sim_speedup', 'n/a')}"
            )
        print(f"Wrote {args.e4_output}")

    if run_e5:
        e5_payload = run_e5_multi_bucket_ablation(args.benchmarks_dir)
        args.e5_output.parent.mkdir(parents=True, exist_ok=True)
        args.e5_output.write_text(json.dumps(e5_payload, indent=2), encoding="utf-8")
        print("E5 multi-bucket ablation")
        print("=" * 50)
        for row in e5_payload["buckets"]:
            full = next(
                c for c in row["ablation"]["configurations"] if c["name"] == "full"
            )
            p = full["proxies"]
            print(
                f"  {row['bucket_id']}: "
                f"E1={p['block_size_reduction_ratio']:.3f} "
                f"E2={p['prefill_reduction_ratio']:.3f} "
                f"E3={p['host_copy_reduction_ratio']:.3f}"
            )
        print(f"Wrote {args.e5_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
