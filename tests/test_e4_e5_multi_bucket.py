"""E4/E5 multi-bucket S1/S2/S3 trace verification."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.e4_compositional import run_e4_multi_bucket_verification
from llmir.benchmark.e5_ablation import run_e5_multi_bucket_ablation

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "IEEE-conference/benchmarks"


def test_e4_multi_bucket_all_within_ideal_bound():
    payload = run_e4_multi_bucket_verification(BENCH)
    assert payload["mode"] == "multi_bucket_compositional"
    assert len(payload["buckets"]) == 3
    ids = {row["bucket_id"] for row in payload["buckets"]}
    assert ids == {"S1", "S2", "S3"}
    for row in payload["buckets"]:
        mc = row["analysis"]["measured_comparison"]
        assert mc["measured_kv_sim_speedup"] is not None
        assert mc["measured_within_ideal_bound"] is True


def test_e4_prefill_savings_scale_with_bucket():
    payload = run_e4_multi_bucket_verification(BENCH)
    saved = {
        row["bucket_id"]: row["analysis"]["e2_prefix_prefill"]["saved_prefill_tokens"]
        for row in payload["buckets"]
    }
    assert saved["S1"] < saved["S2"] < saved["S3"]


def test_e5_multi_bucket_full_stack_monotonic():
    payload = run_e5_multi_bucket_ablation(BENCH)
    assert payload["mode"] == "multi_bucket_ablation"
    assert len(payload["buckets"]) == 3
    for row in payload["buckets"]:
        stack = {c["name"]: c["proxies"] for c in row["ablation"]["cumulative_stack"]}
        assert stack["baseline"]["prefill_reduction_ratio"] == 0.0
        assert (
            stack["cumulative_e1_e2"]["prefill_reduction_ratio"]
            >= stack["cumulative_e1"]["prefill_reduction_ratio"]
        )
        assert stack["full"]["host_copy_reduction_ratio"] >= stack["cumulative_e1_e2"][
            "host_copy_reduction_ratio"
        ]


def test_multi_bucket_json_roundtrip(tmp_path):
    e4 = run_e4_multi_bucket_verification(BENCH)
    e5 = run_e5_multi_bucket_ablation(BENCH)
    e4_path = tmp_path / "e4_buckets.json"
    e5_path = tmp_path / "e5_buckets.json"
    e4_path.write_text(json.dumps(e4, indent=2), encoding="utf-8")
    e5_path.write_text(json.dumps(e5, indent=2), encoding="utf-8")
    e4_loaded = json.loads(e4_path.read_text())
    e5_loaded = json.loads(e5_path.read_text())
    assert e4_loaded["experiment"] == "E4"
    assert e5_loaded["experiment"] == "E5"
    assert all("bucket_id" in row for row in e4_loaded["buckets"])
    assert all(len(row["ablation"]["configurations"]) >= 7 for row in e5_loaded["buckets"])
