"""Tests for the CPU inference comparison benchmark helpers."""

import importlib.util
import sys
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
