#!/usr/bin/env python3
"""One-command P2 MVP: emit MLIR + run reference + compare to PyTorch SDPA."""

from __future__ import annotations

import sys

from llmir.compiler import KVMicroPipelineConfig, compile_kv_micro_pipeline


def main() -> int:
    cfg = KVMicroPipelineConfig(batch_size=1, seq_len=8, num_heads=2, head_dim=16)
    result = compile_kv_micro_pipeline(
        cfg,
        run_opt=True,
        run_reference=True,
        compare_torch=True,
        seed=0,
    )
    print("=== Emitted MLIR (first 15 lines) ===")
    for line in result.mlir.splitlines()[:15]:
        print(line)
    print("...")
    if result.opt:
        print(f"\nmlir-opt: success={result.opt.success} exe={result.opt.executable!r}")
    print(f"reference backend: {result.reference_backend}")
    if result.torch_max_abs_diff is not None:
        print(f"max |ref - torch| = {result.torch_max_abs_diff:.6e}")
        if result.torch_max_abs_diff >= 1e-5:
            print("FAIL: numerical error too large", file=sys.stderr)
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
