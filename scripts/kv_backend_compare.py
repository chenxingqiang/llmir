#!/usr/bin/env python3
"""Compare NumPy vs native KV backend append/lookup throughput."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache, kv_cache_backend_name


def bench_backend(prefer_native: bool, batch: int, seq: int, iters: int) -> dict:
    config = KVCacheConfig(
        num_layers=1, num_heads=4, head_dim=32, block_size=16, dtype="float32"
    )
    cache = create_paged_kv_cache(config, prefer_native=prefer_native)
    backend = kv_cache_backend_name(cache)
    keys = np.random.randn(batch, seq, 4, 32).astype(np.float32)
    values = np.random.randn(batch, seq, 4, 32).astype(np.float32)
    seq_ids = np.arange(batch, dtype=np.int32)
    for _ in range(3):
        cache.append(keys, values, seq_ids)
        cache.reset()
    start = time.perf_counter()
    for _ in range(iters):
        cache.append(keys, values, seq_ids)
        cache.lookup(
            np.zeros((batch, config.num_layers), dtype=np.int32),
            np.full(batch, seq, dtype=np.int32),
        )
        cache.reset()
    elapsed = time.perf_counter() - start
    tokens = batch * seq * iters
    return {
        "backend": backend,
        "batch": batch,
        "seq_len": seq,
        "iterations": iters,
        "throughput_tokens_s": tokens / elapsed if elapsed else 0.0,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="KV backend microbenchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("-o", "--output", default="kv_backend_compare.json")
    args = parser.parse_args()

    rows = [
        bench_backend(False, args.batch, args.seq_len, args.iterations),
    ]
    if os.environ.get("LLMIR_LIB_PATH") or os.environ.get("LLMIR_KV_BACKEND") != "numpy":
        try:
            rows.append(
                bench_backend(True, args.batch, args.seq_len, args.iterations)
            )
        except RuntimeError as exc:
            print(f"native backend skipped: {exc}")

    print(f"{'backend':<8} {'tok/s':>12}")
    for r in rows:
        print(f"{r['backend']:<8} {r['throughput_tokens_s']:>12.0f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
