"""Tests for LLMIR model optimizations."""

import pytest

from llmir.models import (
    ModelArchitecture,
    ModelConfig,
    ModelOptimizer,
    LlamaOptimizer,
    MistralOptimizer,
    PhiOptimizer,
    ModelRegistry,
    ModelMemoryEstimator,
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        config = ModelConfig()
        
        assert config.num_layers == 32
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
    
    def test_get_head_dim(self):
        """Test head dimension calculation."""
        config = ModelConfig(hidden_size=4096, num_attention_heads=32)
        
        assert config.get_head_dim() == 128
    
    def test_is_gqa(self):
        """Test GQA detection."""
        gqa_config = ModelConfig(
            num_attention_heads=32,
            num_key_value_heads=8
        )
        mha_config = ModelConfig(
            num_attention_heads=32,
            num_key_value_heads=32
        )
        
        assert gqa_config.is_gqa() is True
        assert mha_config.is_gqa() is False
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test ModelConfig serialization round-trip."""
        config = ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            num_layers=32,
            hidden_size=4096,
            num_key_value_heads=8,
        )
        d = config.to_dict()
        config2 = ModelConfig.from_dict(d)
        assert config2.num_layers == config.num_layers
        assert config2.architecture == config.architecture

    def test_queries_per_kv(self):
        """Test queries per KV head calculation."""
        config = ModelConfig(
            num_attention_heads=32,
            num_key_value_heads=8
        )
        
        assert config.get_num_queries_per_kv() == 4


class TestLlamaOptimizer:
    """Tests for LlamaOptimizer."""
    
    def test_llama_7b(self):
        """Test Llama 7B configuration."""
        optimizer = LlamaOptimizer.for_llama_7b()
        
        assert optimizer.config.num_layers == 32
        assert optimizer.config.hidden_size == 4096
        assert optimizer.config.architecture == ModelArchitecture.LLAMA
    
    def test_llama3_8b(self):
        """Test Llama 3 8B configuration."""
        optimizer = LlamaOptimizer.for_llama3_8b()
        
        assert optimizer.config.num_layers == 32
        assert optimizer.config.num_key_value_heads == 8  # GQA
        assert optimizer.config.architecture == ModelArchitecture.LLAMA3
    
    def test_llama2_70b_gqa(self):
        """Test Llama 2 70B GQA configuration."""
        optimizer = LlamaOptimizer.for_llama2_70b()
        
        assert optimizer.config.num_layers == 80
        assert optimizer.config.num_attention_heads == 64
        assert optimizer.config.num_key_value_heads == 8
        assert optimizer.config.is_gqa()
    
    def test_llama31_8b(self):
        """Test Llama 3.1 8B with 128K context."""
        optimizer = LlamaOptimizer.for_llama31_8b()
        
        assert optimizer.config.max_position_embeddings == 131072
        assert optimizer.config.rope_scaling_factor == 8.0
    
    def test_kv_cache_config(self):
        """Test optimized KV cache configuration."""
        optimizer = LlamaOptimizer.for_llama3_8b()
        kv_config = optimizer.get_optimized_kv_cache_config()
        
        assert kv_config.num_layers == 32
        assert kv_config.num_heads == 8  # KV heads for GQA
        assert kv_config.head_dim == 128


class TestMistralOptimizer:
    """Tests for MistralOptimizer."""
    
    def test_mistral_7b(self):
        """Test Mistral 7B configuration."""
        optimizer = MistralOptimizer.for_mistral_7b()
        
        assert optimizer.config.num_layers == 32
        assert optimizer.config.sliding_window_size == 4096
        assert optimizer.uses_sliding_window()
    
    def test_mixtral_8x7b(self):
        """Test Mixtral 8x7B configuration."""
        optimizer = MistralOptimizer.for_mixtral_8x7b()
        
        assert optimizer.config.architecture == ModelArchitecture.MIXTRAL
        assert optimizer.config.rope_theta == 1000000.0
    
    def test_mixtral_8x22b(self):
        """Test Mixtral 8x22B configuration."""
        optimizer = MistralOptimizer.for_mixtral_8x22b()
        
        assert optimizer.config.num_layers == 56
        assert optimizer.config.hidden_size == 6144
    
    def test_block_size(self):
        """Test optimized block size for sliding window."""
        optimizer = MistralOptimizer.for_mistral_7b()
        
        # Mistral uses smaller block size for sliding window
        assert optimizer.get_optimized_block_size() == 8


class TestPhiOptimizer:
    """Tests for PhiOptimizer."""
    
    def test_phi2(self):
        """Test Phi-2 configuration."""
        optimizer = PhiOptimizer.for_phi2()
        
        assert optimizer.config.num_layers == 32
        assert optimizer.config.hidden_size == 2560
        assert optimizer.config.head_dim == 80
    
    def test_phi3_mini(self):
        """Test Phi-3 Mini configuration."""
        optimizer = PhiOptimizer.for_phi3_mini()
        
        assert optimizer.config.architecture == ModelArchitecture.PHI3
        assert optimizer.config.num_key_value_heads == 8  # GQA
    
    def test_phi3_block_size(self):
        """Test block size for smaller head dimensions."""
        phi2 = PhiOptimizer.for_phi2()
        phi3 = PhiOptimizer.for_phi3_medium()
        
        # Phi-2 has smaller head_dim, should use larger block
        assert phi2.get_optimized_block_size() == 32
        assert phi3.get_optimized_block_size() == 16


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_singleton(self):
        """Test singleton pattern."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        
        assert registry1 is registry2
    
    def test_list_models(self):
        """Test listing registered models."""
        registry = ModelRegistry()
        models = registry.list_models()
        
        assert 'llama-7b' in models
        assert 'llama3-8b' in models
        assert 'mistral-7b' in models
        assert 'phi-2' in models
    
    def test_get_config(self):
        """Test getting model configuration."""
        registry = ModelRegistry()
        config = registry.get('llama3-8b')
        
        assert config is not None
        assert config.num_layers == 32
    
    def test_get_optimizer(self):
        """Test getting model optimizer."""
        registry = ModelRegistry()
        optimizer = registry.get_optimizer('llama3-8b')
        
        assert optimizer is not None
        assert isinstance(optimizer, LlamaOptimizer)
    
    def test_has_model(self):
        """Test checking model existence."""
        registry = ModelRegistry()
        
        assert registry.has_model('llama3-8b')
        assert not registry.has_model('nonexistent-model')
    
    def test_register_custom(self):
        """Test registering custom model."""
        registry = ModelRegistry()
        
        custom_config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM,
            num_layers=24,
            hidden_size=2048
        )
        registry.register('my-custom-model', custom_config)
        
        assert registry.has_model('my-custom-model')


class TestModelMemoryEstimator:
    """Tests for ModelMemoryEstimator."""
    
    def test_weight_memory(self):
        """Test weight memory estimation."""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000
        )
        estimator = ModelMemoryEstimator(config)
        
        memory = estimator.estimate_weight_memory()
        
        # Llama 7B is ~14GB in float16
        assert 12e9 < memory < 16e9
    
    def test_kv_cache_memory(self):
        """Test KV cache memory estimation."""
        config = ModelConfig(
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128
        )
        estimator = ModelMemoryEstimator(config)
        
        memory = estimator.estimate_kv_cache_memory(
            batch_size=1,
            seq_len=2048
        )
        
        # Should be reasonable size
        assert memory > 0
    
    def test_gqa_kv_memory(self):
        """Test that GQA reduces KV cache memory."""
        mha_config = ModelConfig(
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128
        )
        gqa_config = ModelConfig(
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            head_dim=128
        )
        
        mha_est = ModelMemoryEstimator(mha_config)
        gqa_est = ModelMemoryEstimator(gqa_config)
        
        mha_mem = mha_est.estimate_kv_cache_memory(1, 2048)
        gqa_mem = gqa_est.estimate_kv_cache_memory(1, 2048)
        
        # GQA should use 1/4 of MHA memory
        assert gqa_mem == mha_mem // 4
    
    def test_find_max_batch_size(self):
        """Test finding maximum batch size."""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128
        )
        estimator = ModelMemoryEstimator(config)
        
        # 80GB GPU
        max_batch = estimator.find_max_batch_size(
            memory_budget=80 * 1024**3,
            seq_len=2048
        )
        
        assert max_batch > 0
    
    def test_total_memory(self):
        """Test total memory estimation."""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000
        )
        estimator = ModelMemoryEstimator(config)
        
        total = estimator.estimate_total_memory(
            batch_size=8,
            seq_len=2048
        )
        
        weight = estimator.estimate_weight_memory()
        kv = estimator.estimate_kv_cache_memory(8, 2048)
        act = estimator.estimate_activation_memory(8, 2048)
        
        assert total == weight + kv + act


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
