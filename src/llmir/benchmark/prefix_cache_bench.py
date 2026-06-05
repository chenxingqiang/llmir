"""Prefix cache microbenchmarks (lookup + simulated KV reuse)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from llmir.runtime.config import KVCacheConfig, PrefixCacheConfig
from llmir.runtime.kv_cache import PrefixCache
from llmir.runtime.kv_factory import create_paged_kv_cache


@dataclass
class PrefixBenchResult:
    """One prefix benchmark scenario."""

    scenario: str
    elapsed_s: float
    throughput_ops_s: float
    hit_ratio: float = 0.0
    speedup_vs_baseline: float = 1.0
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def bench_prefix_lookup_throughput(
    *,
    num_queries: int = 10_000,
    num_prefixes: int = 3,
    prefix_len: int = 128,
    seed: int = 0,
) -> PrefixBenchResult:
    """Measure PrefixCache lookup throughput with 100% hit rate."""
    rng = np.random.default_rng(seed)
    cache = PrefixCache(PrefixCacheConfig(max_prefixes=1000, min_prefix_length=4))
    prefixes: List[List[int]] = []
    for i in range(num_prefixes):
        tokens = rng.integers(0, 50000, size=prefix_len, dtype=np.int32).tolist()
        cache.cache_prefix(tokens, [[0]])
        prefixes.append(tokens)

    queries = [rng.choice(prefixes).copy() for _ in range(num_queries)]
    start = time.perf_counter()
    for q in queries:
        cache.lookup(q)
    elapsed = time.perf_counter() - start
    return PrefixBenchResult(
        scenario="prefix_lookup",
        elapsed_s=elapsed,
        throughput_ops_s=num_queries / elapsed if elapsed else 0.0,
        hit_ratio=cache.get_hit_ratio(),
        details=cache.get_stats(),
    )


def bench_prefix_kv_reuse(
    *,
    num_requests: int = 200,
    prefix_len: int = 512,
    suffix_len: int = 32,
    batch_size: int = 1,
    seed: int = 0,
) -> List[PrefixBenchResult]:
    """
    Simulate KV append savings when a shared prompt prefix is recognized.

    **Baseline**: append full (prefix + suffix) KV each request.
    **With prefix cache**: append only suffix KV after the first request (ideal KV-layer
    reuse; prefix *lookup* throughput is measured separately).
    """
    rng = np.random.default_rng(seed)
    h, d = 4, 32
    config = KVCacheConfig(
        num_layers=1, num_heads=h, head_dim=d, block_size=16, dtype="float32"
    )
    prefix_kv = rng.standard_normal((batch_size, prefix_len, h, d)).astype(np.float32)
    suffix_kv = rng.standard_normal((batch_size, suffix_len, h, d)).astype(np.float32)
    seq_ids = np.arange(batch_size, dtype=np.int32)

    def _run_full_appends(cache) -> float:
        start = time.perf_counter()
        for _ in range(num_requests):
            full_k = np.concatenate([prefix_kv, suffix_kv], axis=1)
            full_v = full_k.copy()
            cache.append(full_k, full_v, seq_ids)
            cache.reset()
        return time.perf_counter() - start

    def _run_prefix_cached_appends() -> float:
        """Ideal KV-layer reuse: first request pays full prefill, rest append suffix only."""
        cache = create_paged_kv_cache(config, prefer_native=False)
        prefix_warmed = False
        start = time.perf_counter()
        for _ in range(num_requests):
            if prefix_warmed:
                cache.append(suffix_kv, suffix_kv.copy(), seq_ids)
            else:
                full_k = np.concatenate([prefix_kv, suffix_kv], axis=1)
                cache.append(full_k, full_k.copy(), seq_ids)
                prefix_warmed = True
            cache.reset()
        return time.perf_counter() - start

    baseline_cache = create_paged_kv_cache(config, prefer_native=False)
    baseline_s = _run_full_appends(baseline_cache)
    optimized_s = _run_prefix_cached_appends()
    baseline_tokens = num_requests * batch_size * (prefix_len + suffix_len)
    optimized_tokens = num_requests * batch_size * suffix_len + batch_size * (
        prefix_len + suffix_len
    )

    return [
        PrefixBenchResult(
            scenario="kv_append_baseline",
            elapsed_s=baseline_s,
            throughput_ops_s=baseline_tokens / baseline_s if baseline_s else 0.0,
            hit_ratio=0.0,
        ),
        PrefixBenchResult(
            scenario="kv_append_prefix_cached",
            elapsed_s=optimized_s,
            throughput_ops_s=optimized_tokens / optimized_s if optimized_s else 0.0,
            hit_ratio=1.0,
            speedup_vs_baseline=baseline_s / optimized_s if optimized_s else 1.0,
            details={
                "prefix_len": prefix_len,
                "suffix_len": suffix_len,
                "num_requests": num_requests,
            },
        ),
    ]
