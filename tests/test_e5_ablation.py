"""E5 ablation at verifiable layers."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.e4_compositional import (
    E4WorkloadTrace,
    run_e4_compositional_verification,
)
from llmir.benchmark.e5_ablation import (
    E5AblationSwitches,
    analyze_layer_proxies,
    run_e5_ablation,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_JSON = ROOT / "IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json"


def _s2_trace() -> E4WorkloadTrace:
    return E4WorkloadTrace(
        shared_prefix_tokens=2048,
        num_requests=32,
        suffix_tokens=8,
        decode_steps=4,
    )


def test_baseline_all_proxies_zero_except_host_trips():
    trace = _s2_trace()
    proxies = analyze_layer_proxies(trace, E5AblationSwitches(False, False, False))
    assert proxies["block_size_reduction_ratio"] == 0.0
    assert proxies["prefill_reduction_ratio"] == 0.0
    assert proxies["host_copy_reduction_ratio"] == 0.0
    assert proxies["host_round_trips_total"] > 0


def test_full_stack_matches_e4_levers():
    trace = _s2_trace()
    full = analyze_layer_proxies(trace, E5AblationSwitches(True, True, True))
    e4 = run_e4_compositional_verification(trace).to_dict()
    levers = e4["composite"]["compile_time_levers"]
    assert full["block_size_reduction_ratio"] == levers["e1_block_size_reduction"]
    assert full["prefill_reduction_ratio"] == levers["e2_prefill_token_reduction"]
    assert full["host_copy_reduction_ratio"] == levers["e3_host_copy_elimination"]


def test_isolated_e2_only_moves_prefill_proxy():
    trace = _s2_trace()
    baseline = analyze_layer_proxies(trace, E5AblationSwitches(False, False, False))
    e2_only = analyze_layer_proxies(trace, E5AblationSwitches(False, True, False))
    assert e2_only["block_size_reduction_ratio"] == baseline["block_size_reduction_ratio"]
    assert e2_only["host_copy_reduction_ratio"] == baseline["host_copy_reduction_ratio"]
    assert e2_only["prefill_reduction_ratio"] > 0.9
    assert e2_only["saved_prefill_tokens"] > baseline["saved_prefill_tokens"]


def test_isolated_e1_only_moves_block_proxy():
    trace = _s2_trace()
    baseline = analyze_layer_proxies(trace, E5AblationSwitches(False, False, False))
    e1_only = analyze_layer_proxies(trace, E5AblationSwitches(True, False, False))
    assert e1_only["prefill_reduction_ratio"] == baseline["prefill_reduction_ratio"]
    assert e1_only["host_copy_reduction_ratio"] == baseline["host_copy_reduction_ratio"]
    assert e1_only["block_size_reduction_ratio"] > 0.9


def test_cumulative_stack_monotonic_prefill_or_host():
    result = run_e5_ablation(_s2_trace())
    stack = {row["name"]: row["proxies"] for row in result.cumulative_stack}
    assert stack["baseline"]["prefill_reduction_ratio"] == 0.0
    assert (
        stack["cumulative_e1_e2"]["prefill_reduction_ratio"]
        >= stack["cumulative_e1"]["prefill_reduction_ratio"]
    )
    assert stack["full"]["host_copy_reduction_ratio"] >= stack["cumulative_e1_e2"][
        "host_copy_reduction_ratio"
    ]


def test_e5_json_schema(tmp_path):
    out = tmp_path / "e5.json"
    result = run_e5_ablation(_s2_trace())
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert data["experiment"] == "E5"
    assert len(data["configurations"]) >= 7
    assert "isolated_contributions" in data
    assert "e1_block_opt" in data["isolated_contributions"]
    names = {row["name"] for row in data["configurations"]}
    assert "baseline" in names and "full" in names
