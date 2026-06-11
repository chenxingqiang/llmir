"""S1/S2/S3 decoder workload bucket artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from llmir.benchmark.decoder_workload_buckets import (
    DECODER_WORKLOAD_BUCKETS,
    generate_bucket_sim_payload,
    verify_bucket_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "IEEE-conference/benchmarks"


def test_bucket_presets_match_doc_shapes():
    s1, s2, s3 = (
        DECODER_WORKLOAD_BUCKETS["S1"],
        DECODER_WORKLOAD_BUCKETS["S2"],
        DECODER_WORKLOAD_BUCKETS["S3"],
    )
    assert s1.system_prompt_tokens == 128 and s1.num_requests == 32
    assert s2.system_prompt_tokens == 2048 and s2.num_requests == 32
    assert s3.system_prompt_tokens == 8192 and s3.num_requests == 16


def test_generate_bucket_sim_payload():
    payload = generate_bucket_sim_payload(DECODER_WORKLOAD_BUCKETS["S1"])
    assert payload["bucket_id"] == "S1"
    assert payload["mode"] == "shared_prefix_decoder"
    assert len(payload["results"]) == 2
    assert payload["results"][1]["speedup_vs_baseline"] > 1.0


def test_committed_bucket_artifacts_verify():
    for bucket in DECODER_WORKLOAD_BUCKETS.values():
        err = verify_bucket_artifact(bucket, BENCH)
        assert err is None, err
        data = json.loads(bucket.artifact_path(BENCH).read_text(encoding="utf-8"))
        assert data["config"]["system_prompt_tokens"] == bucket.system_prompt_tokens
