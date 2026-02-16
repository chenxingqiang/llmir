"""Integration tests for config round-trip and memory estimates.

These tests run without network - they use the built-in model registry
and verify config consistency, memory estimation, and KV cache setup.
"""

import pytest

from llmir import (
    ModelRegistry,
    LlamaOptimizer,
    MistralOptimizer,
    PhiOptimizer,
    PagedKVCache,
    KVCacheConfig,
)
from llmir.models import ModelMemoryEstimator


class TestConfigRoundTrip:
    """Test config consistency between registry and optimizers."""

    def test_llama_config_consistency(self):
        """Llama optimizer config matches KV cache output."""
        opt = LlamaOptimizer.for_llama3_8b()
        kv = opt.get_optimized_kv_cache_config()

        assert kv.num_layers == opt.config.num_layers
        assert kv.num_heads == opt.config.num_key_value_heads
        assert kv.head_dim == opt.config.get_head_dim()

    def test_mistral_config_consistency(self):
        """Mistral optimizer config matches KV cache output."""
        opt = MistralOptimizer.for_mistral_7b()
        kv = opt.get_optimized_kv_cache_config()

        assert kv.num_layers == opt.config.num_layers
        assert kv.num_heads == opt.config.num_key_value_heads

    def test_registry_to_optimizer_roundtrip(self):
        """ModelRegistry returns optimizer with correct config."""
        registry = ModelRegistry()
        for name in ["llama3-8b", "mistral-7b", "phi-3-mini", "qwen2-7b", "gemma-2b", "falcon-40b"]:
            opt = registry.get_optimizer(name)
            assert opt is not None, f"Missing optimizer for {name}"
            kv = opt.get_optimized_kv_cache_config()
            assert kv.num_layers > 0
            assert kv.num_heads > 0
            assert kv.head_dim > 0


class TestMemoryEstimate:
    """Test memory estimation sanity."""

    def test_memory_estimate_positive(self):
        """Memory estimate is positive for valid config."""
        opt = LlamaOptimizer.for_llama3_8b()
        mem = opt.estimate_memory(batch_size=1, seq_len=128)
        assert mem > 0

    def test_memory_estimate_increases_with_batch(self):
        """Memory increases with batch size."""
        opt = LlamaOptimizer.for_llama3_8b()
        m1 = opt.estimate_memory(batch_size=1, seq_len=128)
        m4 = opt.estimate_memory(batch_size=4, seq_len=128)
        assert m4 > m1

    def test_memory_estimate_increases_with_seq_len(self):
        """Memory increases with sequence length."""
        opt = LlamaOptimizer.for_llama3_8b()
        m128 = opt.estimate_memory(batch_size=1, seq_len=128)
        m512 = opt.estimate_memory(batch_size=1, seq_len=512)
        assert m512 > m128

    def test_memory_breakdown_no_crash(self):
        """ModelMemoryEstimator.print_breakdown runs without error."""
        opt = LlamaOptimizer.for_llama3_8b()
        estimator = ModelMemoryEstimator(opt.config)
        estimator.print_breakdown(batch_size=1, seq_len=64)


class TestKVCacheFromOptimizer:
    """Test PagedKVCache creation from optimizer config."""

    def test_create_cache_from_llama_optimizer(self):
        """Create PagedKVCache from Llama optimizer config."""
        opt = LlamaOptimizer.for_llama3_8b()
        kv_config = opt.get_optimized_kv_cache_config()
        cache = PagedKVCache(kv_config)

        assert cache.num_layers == kv_config.num_layers
        assert cache.block_size == kv_config.block_size

    def test_create_cache_from_registry_optimizer(self):
        """Create PagedKVCache from registry optimizer."""
        opt = ModelRegistry().get_optimizer("llama3.1-8b")
        assert opt is not None
        kv_config = opt.get_optimized_kv_cache_config()
        cache = PagedKVCache(kv_config)
        assert cache.get_num_sequences() == 0
