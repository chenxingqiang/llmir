"""E2 shared-prefix decoder workload tests (legacy module names)."""

from __future__ import annotations

import pytest

from llmir.benchmark.sharegpt_prefix_bench import (
    ShareGPTPrefixBenchConfig,
    bench_sharegpt_kv_simulation,
    build_sharegpt_prompts,
    run_sharegpt_prefix_benchmark,
)


def test_build_sharegpt_prompts_count():
    cfg = ShareGPTPrefixBenchConfig(num_requests=5, system_prompt_tokens=10)
    system, prompts = build_sharegpt_prompts(cfg)
    assert len(system.split()) == 10
    assert len(prompts) == 5
    assert all(system in p for p in prompts)


def test_sharegpt_kv_simulation_speedup():
    cfg = ShareGPTPrefixBenchConfig(
        num_requests=50,
        system_prompt_tokens=256,
        user_suffix_tokens=16,
    )
    rows = bench_sharegpt_kv_simulation(cfg)
    assert len(rows) == 2
    baseline, optimized = rows
    assert baseline.scenario == "sharegpt_kv_baseline"
    assert optimized.scenario == "sharegpt_kv_prefix_cached"
    assert optimized.speedup_vs_baseline > 1.5


def test_run_sharegpt_simulation_only():
    payload = run_sharegpt_prefix_benchmark(
        ShareGPTPrefixBenchConfig(num_requests=20, system_prompt_tokens=64),
        run_simulation=True,
        run_llmir_paged=False,
    )
    assert payload["mode"] == "shared_prefix_decoder"
    assert len(payload["results"]) == 2


@pytest.mark.network
def test_run_sharegpt_llmir_paged_e2e():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    payload = run_sharegpt_prefix_benchmark(
        ShareGPTPrefixBenchConfig(
            num_requests=3,
            system_prompt_tokens=16,
            user_suffix_tokens=4,
            max_new_tokens=2,
            model="gpt2",
            device="cpu",
        ),
        run_simulation=False,
        run_llmir_paged=True,
    )
    assert "per_request" in payload
    warmed = next(
        r for r in payload["results"] if r["scenario"] == "sharegpt_llmir_warm_prefix"
    )
    assert warmed["prefix_hit_tokens_total"] >= 0
    if payload["config"]["num_requests"] > 1:
        assert warmed["speedup_vs_baseline"] >= 0.5
