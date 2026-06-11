#!/usr/bin/env python3
"""M5 lowered hot path verification — mlir_llm runtime call sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.compiler.kv_emit import KVMicroPipelineConfig  # noqa: E402
from llmir.compiler.lowered_hot_path import run_lowered_hot_path_verification  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/m5_lowered_hot_path.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 lowered hot path verification")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = KVMicroPipelineConfig(
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )
    result = run_lowered_hot_path_verification(cfg, seed=args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("M5 lowered hot path verification")
    print("=" * 50)
    print(f"mlir_lowered: {payload['mlir_lowered']}")
    print(f"execution_path: {payload['execution_path']}")
    print(f"matches_reference: {payload['matches_reference']}")
    print(f"max_abs_diff: {payload['max_abs_diff_vs_reference']}")
    print(f"Wrote {args.output}")
    return 0 if payload["matches_reference"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
