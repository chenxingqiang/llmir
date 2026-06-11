#!/usr/bin/env python3
"""
Collect reproducible benchmark JSON for the ICCD revised paper.

Outputs:
  IEEE-conference/benchmarks/paper_results.json
  IEEE-conference/benchmarks/gpt2_e1_snippet.mlir
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "IEEE-conference" / "benchmarks"


def _run_inference(model: str, prompt_tokens: int, max_tokens: int, warmup: int) -> list:
    from llmir.benchmark.inference_compare import results_to_json, run_inference_compare

    backends = ["hf", "llmir-paged", "vllm"]
    rows = run_inference_compare(
        model,
        backends,
        batch_size=1,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        warmup=warmup,
    )
    return results_to_json(rows)


def _run_shared_prefix_decoder(
    model: str,
    system_tokens: int,
    num_requests: int,
    suffix_tokens: int,
    *,
    simulation_only: bool = False,
) -> dict:
    from llmir.benchmark.sharegpt_prefix_bench import (
        ShareGPTPrefixBenchConfig,
        run_sharegpt_prefix_benchmark,
    )

    cfg = ShareGPTPrefixBenchConfig(
        model=model,
        system_prompt_tokens=system_tokens,
        num_requests=num_requests,
        user_suffix_tokens=suffix_tokens,
        max_new_tokens=4,
        device="auto",
    )
    return run_sharegpt_prefix_benchmark(
        cfg,
        run_simulation=True,
        run_llmir_paged=not simulation_only,
    )


def _run_e1_snippet(out_mlir: Path) -> dict:
    from llmir.compiler.kv_emit import KVMicroPipelineConfig
    from llmir.compiler.mvp_pipeline import run_mvp_single_layer_e2e

    result = run_mvp_single_layer_e2e(
        KVMicroPipelineConfig(seq_len=8, num_heads=4, head_dim=32, block_size=16),
        oversized_block_size=1024,
        run_mlir_passes=False,
        run_reference=True,
        compare_torch=False,
        seed=0,
    )
    snippet = result.mlir_after_block_size
    out_mlir.write_text(snippet, encoding="utf-8")
    return {
        "block_size_before": result.block_size_before,
        "block_size_after": result.block_size_after,
        "reference_backend": result.reference_backend,
        "mlir_snippet_path": str(out_mlir.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper benchmark JSON")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--sharegpt-system-tokens",
        "--shared-prefix-tokens",
        dest="shared_prefix_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--sharegpt-requests",
        "--shared-prefix-requests",
        dest="shared_prefix_requests",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sharegpt-suffix-tokens",
        "--shared-prefix-suffix-tokens",
        dest="shared_prefix_suffix_tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sharegpt-simulation-only",
        "--shared-prefix-simulation-only",
        dest="shared_prefix_simulation_only",
        action="store_true",
    )
    parser.add_argument("-o", "--output", default=str(OUT_DIR / "paper_results.json"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mlir_path = OUT_DIR / "gpt2_e1_snippet.mlir"

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/paper_benchmark_collect.py",
        "model": args.model,
        "inference_compare": _run_inference(
            args.model, args.prompt_tokens, args.max_tokens, args.warmup
        ),
        "shared_prefix_decoder": _run_shared_prefix_decoder(
            args.model,
            args.shared_prefix_tokens,
            args.shared_prefix_requests,
            args.shared_prefix_suffix_tokens,
            simulation_only=args.shared_prefix_simulation_only,
        ),
        "e1": _run_e1_snippet(mlir_path),
        "notes": {
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "shared_prefix_tokens": args.shared_prefix_tokens,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
