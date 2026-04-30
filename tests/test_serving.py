"""Tests for LLMIR serving components."""

import sys
import types

import pytest

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.serving.config import (
    BackendType,
    SamplingParams,
    SchedulerConfig,
    SchedulingPolicy,
)
from llmir.serving.engine import (
    ContinuousBatchingEngine,
    LLMEngine,
)


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_defaults(self):
        """Test default values."""
        params = SamplingParams()

        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.max_tokens == 256
        assert params.n == 1

    def test_custom_params(self):
        """Test custom parameter values."""
        params = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=100, presence_penalty=0.1
        )

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 100

    def test_to_dict(self):
        """Test dictionary conversion."""
        params = SamplingParams(temperature=0.5, max_tokens=50)
        d = params.to_dict()

        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 50

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"temperature": 0.8, "top_k": 50, "max_tokens": 128}
        params = SamplingParams.from_dict(d)

        assert params.temperature == 0.8
        assert params.top_k == 50
        assert params.max_tokens == 128


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SchedulerConfig()

        assert config.policy == SchedulingPolicy.FCFS
        assert config.max_batch_size == 256
        assert config.enable_preemption is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SchedulerConfig(
            policy=SchedulingPolicy.ADAPTIVE,
            max_batch_size=128,
            preemption_threshold=0.8,
        )

        assert config.policy == SchedulingPolicy.ADAPTIVE
        assert config.max_batch_size == 128


class TestContinuousBatchingEngine:
    """Tests for ContinuousBatchingEngine."""

    def test_create_engine(self):
        """Test engine creation."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)

        engine = ContinuousBatchingEngine(cache)

        assert not engine.is_running()

    def test_start_stop(self):
        """Test starting and stopping the engine."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)

        engine.start()
        assert engine.is_running()

        engine.stop()
        assert not engine.is_running()

    def test_submit_request(self):
        """Test submitting a request."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)
        engine.start()

        request_id = engine.submit(
            prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10)
        )

        assert request_id.startswith("req-")
        assert engine.has_pending_requests()

    def test_custom_request_id(self):
        """Test custom request ID."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)
        engine.start()

        request_id = engine.submit(prompt_tokens=[1, 2, 3], request_id="custom-123")

        assert request_id == "custom-123"

    def test_abort_request(self):
        """Test aborting a request."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)
        engine.start()

        request_id = engine.submit([1, 2, 3])
        result = engine.abort(request_id)

        assert result is True

    def test_iterate(self):
        """Test running iterations."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)
        engine.start()

        engine.submit([1, 2, 3], SamplingParams(max_tokens=5))

        outputs = []
        while engine.has_pending_requests():
            outputs.extend(engine.iterate())

        assert len(outputs) > 0
        assert outputs[-1].finished

    def test_get_stats(self):
        """Test getting engine statistics."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        engine = ContinuousBatchingEngine(cache)
        engine.start()

        engine.submit([1, 2, 3], SamplingParams(max_tokens=5))
        while engine.has_pending_requests():
            engine.iterate()

        stats = engine.get_stats()

        assert "total_requests" in stats
        assert "completed_requests" in stats
        assert stats["completed_requests"] == 1


class TestLLMEngine:
    """Tests for LLMEngine."""

    def test_create_engine(self):
        """Test engine creation."""
        engine = LLMEngine(model_path="test-model")

        assert engine.model_path == "test-model"

    def test_from_pretrained(self):
        """Test from_pretrained factory method."""
        engine = LLMEngine.from_pretrained(
            "meta-llama/Llama-3.1-8B", tensor_parallel_size=2, dtype="bfloat16"
        )

        assert engine.engine_config.tensor_parallel_size == 2
        assert engine.engine_config.dtype == "bfloat16"

    def test_from_pretrained_vllm_backend(self):
        """Test creating an engine that targets the optional vLLM backend."""
        engine = LLMEngine.from_pretrained("test-model", backend=BackendType.VLLM)

        assert engine.backend == "vllm"
        assert engine.engine_config.backend == "vllm"

    def test_generate_vllm_backend(self, monkeypatch):
        """Test vLLM backend generation with a fake vLLM module."""

        class FakeVLLMSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeCompletion:
            text = "fake completion"
            token_ids = [10, 11]
            finish_reason = "length"
            logprobs = None
            cumulative_logprob = 0.0

        class FakeRequestOutput:
            prompt = "hello"
            prompt_token_ids = [1]
            outputs = [FakeCompletion()]

        class FakeLLM:
            init_kwargs = None
            sampling_kwargs = None

            def __init__(self, **kwargs):
                FakeLLM.init_kwargs = kwargs

            def generate(self, prompts, sampling_params):
                FakeLLM.sampling_kwargs = sampling_params.kwargs
                assert prompts == ["hello"]
                return [FakeRequestOutput()]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeVLLMSamplingParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        engine = LLMEngine.from_pretrained(
            "test-model",
            backend="vllm",
            dtype="float32",
            tensor_parallel_size=2,
        )
        outputs = engine.generate("hello", SamplingParams(max_tokens=2, stop=[]))

        assert FakeLLM.init_kwargs["model"] == "test-model"
        assert FakeLLM.init_kwargs["tensor_parallel_size"] == 2
        assert FakeLLM.init_kwargs["dtype"] == "float32"
        assert FakeLLM.sampling_kwargs["max_tokens"] == 2
        assert "stop" not in FakeLLM.sampling_kwargs
        assert len(outputs) == 1
        assert outputs[0].prompt == "hello"
        assert outputs[0].outputs[0].text == "fake completion"
        assert outputs[0].outputs[0].token_ids == [10, 11]
        assert outputs[0].finished

    def test_generate_single(self):
        """Test generating from single prompt."""
        engine = LLMEngine(model_path="test-model")

        outputs = engine.generate("Hello world", SamplingParams(max_tokens=5))

        assert len(outputs) == 1
        assert outputs[0].finished

    def test_generate_batch(self):
        """Test generating from multiple prompts."""
        engine = LLMEngine(model_path="test-model")

        outputs = engine.generate(["Hello", "World"], SamplingParams(max_tokens=5))

        assert len(outputs) == 2

    def test_shutdown(self):
        """Test engine shutdown."""
        engine = LLMEngine(model_path="test-model")
        engine.generate("Test", SamplingParams(max_tokens=1))
        engine.shutdown()

        assert not engine._engine.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
