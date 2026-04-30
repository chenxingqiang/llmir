"""Tests for the CPU inference comparison benchmark helpers."""

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cpu_inference_compare.py"


def load_module():
    """Load the benchmark script as a module."""
    spec = importlib.util.spec_from_file_location("cpu_inference_compare", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_prompts():
    """Prompts are deterministic and sized by batch."""
    module = load_module()
    prompts = module.build_prompts(batch_size=3, prompt_tokens=4)
    assert prompts == ["hello hello hello hello"] * 3


def test_make_result_metrics():
    """Derived throughput and latency metrics are computed correctly."""
    module = load_module()
    result = module.make_result(
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
    """The LLMIR benchmark path runs without vLLM or model downloads."""
    module = load_module()
    result = module.run_llmir_cpu(
        model="test-model",
        batch_size=2,
        prompt_tokens=4,
        max_tokens=3,
        warmup=0,
    )
    assert result.engine == "llmir"
    assert result.generated_tokens == 6
    assert result.throughput_tokens_s > 0


def test_llmir_vllm_backend_returns_none_without_vllm(monkeypatch):
    """The LLMIR+vLLM backend path is skipped when vLLM is not installed."""
    module = load_module()
    monkeypatch.setitem(sys.modules, "vllm", None)
    result = module.run_llmir_vllm_backend(
        model="test-model",
        batch_size=1,
        prompt_tokens=4,
        max_tokens=3,
        warmup=0,
    )
    assert result is None


def test_llmir_vllm_backend_runs_with_fake_vllm(monkeypatch):
    """The LLMIR+vLLM backend benchmark path runs against a fake vLLM module."""
    module = load_module()

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

    result = module.run_llmir_vllm_backend(
        model="test-model",
        batch_size=2,
        prompt_tokens=4,
        max_tokens=3,
        warmup=1,
    )
    assert result is not None
    assert result.engine == "llmir+vllm"
    # 2 prompts * 3 generated token_ids each = 6
    assert result.generated_tokens == 6
    assert result.throughput_tokens_s > 0
