"""Tests for CPU inference comparison (llmir.benchmark.inference_compare)."""

import sys
import types

from llmir.benchmark.inference_compare import (
    build_prompts,
    make_result,
    run_inference_compare,
    run_llmir_paged,
    run_llmir_smoke,
    run_llmir_vllm_backend,
)


def test_build_prompts():
    prompts = build_prompts(batch_size=3, prompt_tokens=4)
    assert prompts == ["hello hello hello hello"] * 3


def test_make_result_metrics():
    result = make_result(
        "engine",
        "model",
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=20,
        elapsed_s=2.0,
    )
    assert result.throughput_tokens_s == 10.0
    assert result.latency_ms_per_token == 100.0


def test_llmir_cpu_benchmark_runs_without_vllm():
    result = run_llmir_smoke(
        model="test-model",
        batch_size=2,
        prompt_tokens=4,
        max_tokens=3,
        warmup=0,
    )
    assert result.engine == "llmir"
    assert result.generated_tokens == 6
    assert result.throughput_tokens_s > 0


def test_llmir_paged_cpu_returns_none_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = run_llmir_paged(
        model="test-model",
        batch_size=1,
        prompt_tokens=4,
        max_tokens=3,
        warmup=0,
    )
    assert result is None


def test_llmir_vllm_backend_returns_none_without_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    result = run_llmir_vllm_backend(
        model="test-model",
        batch_size=1,
        prompt_tokens=4,
        max_tokens=3,
        warmup=0,
    )
    assert result is None


def test_llmir_vllm_backend_runs_with_fake_vllm(monkeypatch):
    class FakeVLLMSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCompletion:
        text = "fake"
        token_ids = [1, 2, 3]
        finish_reason = "length"
        logprobs = None
        cumulative_logprob = 0.0

    class FakeRequestOutput:
        prompt = "hello hello hello hello"
        prompt_token_ids = [1, 2, 3, 4]
        outputs = [FakeCompletion()]

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params):
            return [FakeRequestOutput() for _ in prompts]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeVLLMSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    result = run_llmir_vllm_backend(
        model="test-model",
        batch_size=2,
        prompt_tokens=4,
        max_tokens=3,
        warmup=1,
    )
    assert result is not None
    assert result.engine == "llmir+vllm"
    assert result.generated_tokens == 6
    assert result.throughput_tokens_s > 0


def test_run_inference_compare_integration_smoke():
    results = run_inference_compare(
        "test-model", ["llmir"], max_tokens=2, warmup=0, batch_size=1
    )
    assert len(results) == 1
    assert results[0].engine == "llmir"
