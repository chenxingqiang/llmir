"""Tests for inference compare registry."""

from llmir.benchmark.inference_compare import (
    build_prompts,
    make_result,
    run_inference_compare,
)


def test_build_prompts_length():
    prompts = build_prompts(2, 8)
    assert len(prompts) == 2
    assert len(prompts[0].split()) == 8


def test_make_result_metrics():
    r = make_result("hf", "m", 1, 16, 32, 1.0)
    assert r.throughput_tokens_s == 32.0
    assert r.latency_ms_per_token == 1000.0 / 32


def test_unknown_backend_skipped(capsys):
    results = run_inference_compare(
        "test-model", ["not-a-real-backend"], max_tokens=1, warmup=0
    )
    assert results == []
    assert "Unknown backend" in capsys.readouterr().out
