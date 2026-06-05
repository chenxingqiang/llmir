"""Tests for prefix cache benchmarks."""

from llmir.benchmark.prefix_cache_bench import (
    bench_prefix_kv_reuse,
    bench_prefix_lookup_throughput,
)


def test_prefix_lookup_hit_ratio():
    row = bench_prefix_lookup_throughput(num_queries=1000, num_prefixes=3, prefix_len=32)
    assert row.hit_ratio == 1.0
    assert row.throughput_ops_s > 0


def test_prefix_kv_reuse_speedup():
    rows = bench_prefix_kv_reuse(
        num_requests=50, prefix_len=128, suffix_len=16, batch_size=1
    )
    assert len(rows) == 2
    cached = [r for r in rows if r.scenario == "kv_append_prefix_cached"][0]
    assert cached.speedup_vs_baseline >= 1.0
