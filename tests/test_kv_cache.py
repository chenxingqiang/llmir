"""Tests for KV Cache implementations."""

import numpy as np
import pytest

from llmir.runtime.kv_cache import (
    PagedKVCache,
    QuantizedKVCache,
    DistributedKVCache,
    SpeculativeKVCache,
    PrefixCache,
)
from llmir.runtime.config import (
    KVCacheConfig,
    QuantizationConfig,
    QuantizationType,
    SpeculativeConfig,
    ShardingConfig,
    ShardingStrategy,
)


class TestPagedKVCache:
    """Tests for PagedKVCache."""
    
    def test_create_cache(self):
        """Test cache creation."""
        config = KVCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            max_seq_len=4096
        )
        cache = PagedKVCache(config)
        
        assert cache.num_layers == 32
        assert cache.block_size == 16
        assert cache.get_num_sequences() == 0
    
    def test_append_and_lookup(self):
        """Test append and lookup operations."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        batch_size = 2
        seq_len = 10
        
        keys = np.random.randn(batch_size, seq_len, 8, 64).astype(np.float16)
        values = np.random.randn(batch_size, seq_len, 8, 64).astype(np.float16)
        seq_ids = np.array([0, 1], dtype=np.int32)
        
        block_indices = cache.append(keys, values, seq_ids)
        
        assert block_indices.shape == (batch_size, config.num_layers)
        assert cache.get_num_sequences() == 2
    
    def test_clear_sequence(self):
        """Test clearing a sequence."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        keys = np.random.randn(1, 10, 8, 64).astype(np.float16)
        values = np.random.randn(1, 10, 8, 64).astype(np.float16)
        seq_ids = np.array([0], dtype=np.int32)
        
        cache.append(keys, values, seq_ids)
        assert cache.get_num_sequences() == 1
        
        cache.clear_sequence(0)
        assert cache.get_num_sequences() == 0
    
    def test_reset(self):
        """Test resetting the cache."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        keys = np.random.randn(2, 10, 8, 64).astype(np.float16)
        values = np.random.randn(2, 10, 8, 64).astype(np.float16)
        seq_ids = np.array([0, 1], dtype=np.int32)
        
        cache.append(keys, values, seq_ids)
        cache.reset()
        
        assert cache.get_num_sequences() == 0


class TestQuantizedKVCache:
    """Tests for QuantizedKVCache."""
    
    def test_create_quantized_cache(self):
        """Test quantized cache creation."""
        config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        quant_config = QuantizationConfig(quant_type=QuantizationType.INT8)
        
        cache = QuantizedKVCache(config, quant_config)
        
        assert cache.get_compression_ratio() == 4.0
    
    def test_int4_compression(self):
        """Test INT4 compression ratio."""
        config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        quant_config = QuantizationConfig(quant_type=QuantizationType.INT4)
        
        cache = QuantizedKVCache(config, quant_config)
        
        assert cache.get_compression_ratio() == 8.0
    
    def test_accuracy_loss(self):
        """Test accuracy loss estimation."""
        config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        
        int8_cache = QuantizedKVCache(
            config, QuantizationConfig(quant_type=QuantizationType.INT8))
        int4_cache = QuantizedKVCache(
            config, QuantizationConfig(quant_type=QuantizationType.INT4))
        
        assert int8_cache.get_accuracy_loss() < int4_cache.get_accuracy_loss()


class TestDistributedKVCache:
    """Tests for DistributedKVCache."""
    
    def test_create_distributed_cache(self):
        """Test distributed cache creation."""
        config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        shard_config = ShardingConfig(
            strategy=ShardingStrategy.LAYER_WISE,
            num_devices=4
        )
        
        cache = DistributedKVCache(config, shard_config)
        
        assert len(cache._shards) == 4
    
    def test_per_device_memory(self):
        """Test per-device memory reporting."""
        config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        shard_config = ShardingConfig(num_devices=2)
        
        cache = DistributedKVCache(config, shard_config)
        memory = cache.get_per_device_memory()
        
        assert len(memory) == 2


class TestSpeculativeKVCache:
    """Tests for SpeculativeKVCache."""
    
    def test_create_branch(self):
        """Test branch creation."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        spec_config = SpeculativeConfig(max_draft_tokens=8)
        
        cache = SpeculativeKVCache(config, spec_config)
        
        branch_id = cache.create_branch(seq_id=0)
        assert branch_id == 0
        
        branch_id2 = cache.create_branch(seq_id=0)
        assert branch_id2 == 1
    
    def test_append_speculative(self):
        """Test appending speculative tokens."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        spec_config = SpeculativeConfig(max_draft_tokens=8)
        
        cache = SpeculativeKVCache(config, spec_config)
        branch_id = cache.create_branch(seq_id=0)
        
        keys = np.random.randn(1, 4, 8, 64).astype(np.float16)
        values = np.random.randn(1, 4, 8, 64).astype(np.float16)
        
        cache.append_speculative(keys, values, seq_id=0, branch_id=branch_id)
    
    def test_rollback(self):
        """Test rollback of speculative tokens."""
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
        spec_config = SpeculativeConfig(max_draft_tokens=8)
        
        cache = SpeculativeKVCache(config, spec_config)
        branch_id = cache.create_branch(seq_id=0)
        
        keys = np.random.randn(1, 4, 8, 64).astype(np.float16)
        values = np.random.randn(1, 4, 8, 64).astype(np.float16)
        
        cache.append_speculative(keys, values, seq_id=0, branch_id=branch_id)
        cache.rollback(seq_id=0, branch_id=branch_id)


class TestPrefixCache:
    """Tests for PrefixCache."""
    
    def test_cache_and_lookup(self):
        """Test caching and looking up prefixes."""
        cache = PrefixCache()
        
        tokens = list(range(100))
        block_indices = [[i] for i in range(10)]
        
        cache.cache_prefix(tokens, block_indices)
        
        match_len, cached = cache.lookup(tokens)
        assert match_len == 100
        assert cached == block_indices
    
    def test_partial_match(self):
        """Test partial prefix matching."""
        cache = PrefixCache()
        
        tokens = list(range(100))
        block_indices = [[i] for i in range(10)]
        cache.cache_prefix(tokens, block_indices)
        
        # Lookup with extended tokens
        extended = list(range(150))
        match_len, cached = cache.lookup(extended)
        
        assert match_len == 100
    
    def test_no_match(self):
        """Test no match case."""
        cache = PrefixCache()
        
        tokens = list(range(100))
        block_indices = [[i] for i in range(10)]
        cache.cache_prefix(tokens, block_indices)
        
        # Different tokens
        different = list(range(200, 300))
        match_len, cached = cache.lookup(different)
        
        assert match_len == 0
        assert cached is None
    
    def test_hit_ratio(self):
        """Test hit ratio tracking."""
        cache = PrefixCache()
        
        tokens = list(range(100))
        cache.cache_prefix(tokens, [[0]])
        
        # Hit
        cache.lookup(tokens)
        # Miss
        cache.lookup([999] * 100)
        
        assert cache.get_hit_ratio() == 0.5
    
    def test_clear(self):
        """Test clearing the cache."""
        cache = PrefixCache()
        
        cache.cache_prefix(list(range(100)), [[0]])
        cache.clear()
        
        stats = cache.get_stats()
        assert stats['num_prefixes'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
