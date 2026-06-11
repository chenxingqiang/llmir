"""E4 compositional verification tests."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.e4_compositional import (
    E4WorkloadTrace,
    analyze_e1_block_sizing,
    analyze_e2_prefix_prefill,
    run_e4_compositional_verification,
    trace_from_sim_json,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_JSON = ROOT / "IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json"


def test_e1_block_reduction_s2_trace():
    trace = E4WorkloadTrace(
        shared_prefix_tokens=2048,
        num_requests=32,
        suffix_tokens=8,
        block_size_before=1024,
        block_size_after=32,
    )
    e1 = analyze_e1_block_sizing(trace)
    assert e1["block_size_attr_before"] == 1024
    assert e1["block_size_attr_after_e1_pass"] <= 64
    assert e1["block_size_reduction_ratio"] > 0.9
    assert e1["blocks_per_request_before_attr"] <= e1["blocks_per_request_after_e1_pass"]


def test_e2_prefill_savings_multi_request():
    trace = E4WorkloadTrace(shared_prefix_tokens=2048, num_requests=32, suffix_tokens=8)
    e2 = analyze_e2_prefix_prefill(trace)
    assert e2["saved_prefill_tokens"] == (32 - 1) * 2048
    assert e2["prefill_reduction_ratio"] > 0.9
    assert e2["ideal_kv_layer_speedup_upper_bound"] > 1.0


def test_e4_end_to_end_with_sim_json():
    trace = trace_from_sim_json(SIM_JSON)
    result = run_e4_compositional_verification(trace, measured_sim_json=SIM_JSON)
    payload = result.to_dict()
    assert payload["experiment"] == "E4"
    assert payload["measured_comparison"]["measured_kv_sim_speedup"] is not None
    assert payload["measured_comparison"]["measured_within_ideal_bound"] is True


def test_e4_script_output_schema(tmp_path):
    out = tmp_path / "e4.json"
    trace = E4WorkloadTrace(shared_prefix_tokens=128, num_requests=8, suffix_tokens=16)
    result = run_e4_compositional_verification(trace)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text())
    assert "e1_block_sizing" in data
    assert "e2_prefix_prefill" in data
    assert "e3_kv_host_copies" in data
