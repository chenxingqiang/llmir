"""Reproducible benchmarks (inference compare, prefix cache)."""

from llmir.benchmark.inference_compare import (
    BenchmarkResult,
    print_inference_results,
    run_inference_compare,
)
from llmir.benchmark.prefix_cache_bench import (
    PrefixBenchResult,
    bench_prefix_kv_reuse,
    bench_prefix_lookup_throughput,
)

__all__ = [
    "BenchmarkResult",
    "run_inference_compare",
    "print_inference_results",
    "PrefixBenchResult",
    "bench_prefix_lookup_throughput",
    "bench_prefix_kv_reuse",
]
