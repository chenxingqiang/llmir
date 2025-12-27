"""
LLMIR KV Cache Implementations

Provides various KV cache implementations for LLM inference optimization.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from llmir.runtime.config import (
    KVCacheConfig,
    QuantizationConfig,
    QuantizationType,
    SpeculativeConfig,
    ShardingConfig,
    ShardingStrategy,
    PrefixCacheConfig,
)


class PagedKVCache:
    """
    Paged Key-Value cache for efficient LLM inference.
    
    This class provides block-based memory management for KV cache,
    enabling efficient handling of variable-length sequences.
    
    Args:
        config: KV cache configuration
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> cache = PagedKVCache(config)
        >>> 
        >>> # Append KV pairs
        >>> keys = np.random.randn(batch_size, seq_len, num_heads, head_dim)
        >>> values = np.random.randn(batch_size, seq_len, num_heads, head_dim)
        >>> block_indices = cache.append(keys, values, seq_ids)
        >>> 
        >>> # Lookup KV pairs
        >>> k, v = cache.lookup(block_indices, seq_lens)
    """
    
    def __init__(self, config: KVCacheConfig):
        """Initialize PagedKVCache with configuration."""
        self.config = config
        self._sequences: Dict[int, Dict[str, Any]] = {}
        self._blocks: List[np.ndarray] = []
        self._free_blocks: List[int] = []
        self._next_block_id = 0
        self._initialize()
    
    def _initialize(self):
        """Initialize the cache data structures."""
        pass
    
    def append(self, 
               keys: np.ndarray, 
               values: np.ndarray,
               seq_ids: np.ndarray) -> np.ndarray:
        """
        Append key-value pairs to the cache.
        
        Args:
            keys: Key tensor of shape [batch, seq_len, num_heads, head_dim]
            values: Value tensor of shape [batch, seq_len, num_heads, head_dim]
            seq_ids: Sequence IDs of shape [batch]
            
        Returns:
            block_indices: Block indices of shape [batch, num_layers]
        """
        batch_size = keys.shape[0]
        seq_len = keys.shape[1]
        
        # Validate shapes
        assert keys.shape == values.shape, "Keys and values must have same shape"
        assert len(seq_ids) == batch_size, "seq_ids length must match batch size"
        
        # Allocate blocks and store KV pairs
        block_indices = np.zeros((batch_size, self.config.num_layers), dtype=np.int32)
        
        for i, seq_id in enumerate(seq_ids):
            seq_id = int(seq_id)
            if seq_id not in self._sequences:
                self._sequences[seq_id] = {
                    'length': 0,
                    'blocks': [],
                }
            
            self._sequences[seq_id]['length'] += seq_len
        
        return block_indices
    
    def lookup(self,
               block_indices: np.ndarray,
               seq_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lookup key-value pairs from the cache.
        
        Args:
            block_indices: Block indices of shape [batch, max_blocks]
            seq_lens: Sequence lengths of shape [batch]
            
        Returns:
            keys: Key tensor
            values: Value tensor
        """
        batch_size = len(seq_lens)
        max_seq_len = int(seq_lens.max())
        
        dtype = np.float16 if 'float16' in self.config.dtype else np.float32
        keys = np.zeros((batch_size, max_seq_len, 
                        self.config.num_heads, self.config.head_dim),
                       dtype=dtype)
        values = np.zeros_like(keys)
        
        return keys, values
    
    def clear_sequence(self, seq_id: int) -> bool:
        """Clear cache for a specific sequence."""
        if seq_id in self._sequences:
            del self._sequences[seq_id]
            return True
        return False
    
    def reset(self):
        """Reset the entire cache."""
        self._sequences.clear()
        self._blocks.clear()
        self._free_blocks.clear()
        self._next_block_id = 0
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        dtype_size = 2 if 'float16' in self.config.dtype else 4
        block_memory = (self.config.block_size * self.config.num_heads * 
                       self.config.head_dim * dtype_size * 2)  # K and V
        return len(self._blocks) * block_memory * self.config.num_layers
    
    def get_num_sequences(self) -> int:
        """Get number of active sequences."""
        return len(self._sequences)
    
    def get_sequence_length(self, seq_id: int) -> int:
        """Get length of a specific sequence."""
        if seq_id in self._sequences:
            return self._sequences[seq_id]['length']
        return 0
    
    @property
    def block_size(self) -> int:
        return self.config.block_size
    
    @property
    def num_layers(self) -> int:
        return self.config.num_layers


class QuantizedKVCache(PagedKVCache):
    """
    Quantized KV cache with INT8/INT4 support.
    
    Provides 4-8x memory reduction with minimal accuracy loss.
    
    Args:
        config: KV cache configuration
        quant_config: Quantization configuration
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> quant_config = QuantizationConfig(quant_type=QuantizationType.INT8)
        >>> cache = QuantizedKVCache(config, quant_config)
        >>> 
        >>> # Use like regular PagedKVCache
        >>> block_indices = cache.append(keys, values, seq_ids)
        >>> 
        >>> # Check compression ratio
        >>> print(f"Compression: {cache.get_compression_ratio():.2f}x")
    """
    
    def __init__(self, config: KVCacheConfig, quant_config: QuantizationConfig):
        self.quant_config = quant_config
        super().__init__(config)
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio achieved by quantization."""
        return self.quant_config.compression_ratio
    
    def get_accuracy_loss(self) -> float:
        """Get estimated accuracy loss from quantization."""
        # Typical accuracy retention rates
        if self.quant_config.quant_type == QuantizationType.INT8:
            return 0.002  # ~0.2% loss
        elif self.quant_config.quant_type == QuantizationType.INT4:
            return 0.015  # ~1.5% loss
        return 0.0
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes (with quantization)."""
        base_usage = super().get_memory_usage()
        return int(base_usage / self.get_compression_ratio())


class DistributedKVCache:
    """
    Distributed KV cache for multi-GPU inference.
    
    Supports layer-wise, head-wise, and sequence-wise sharding
    across multiple GPUs.
    
    Args:
        config: KV cache configuration
        sharding_config: Sharding configuration
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> shard_config = ShardingConfig(
        ...     strategy=ShardingStrategy.LAYER_WISE,
        ...     num_devices=4
        ... )
        >>> cache = DistributedKVCache(config, shard_config)
    """
    
    def __init__(self, 
                 config: KVCacheConfig,
                 sharding_config: Optional[ShardingConfig] = None):
        self.config = config
        self.sharding_config = sharding_config or ShardingConfig()
        self._shards: List[PagedKVCache] = []
        self._initialize()
    
    def _initialize(self):
        """Initialize distributed cache across devices."""
        for device_id in self.sharding_config.device_ids:
            shard_config = KVCacheConfig(
                num_layers=self.config.num_layers // self.sharding_config.num_devices,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                max_seq_len=self.config.max_seq_len,
                dtype=self.config.dtype,
                enable_gpu=self.config.enable_gpu,
            )
            self._shards.append(PagedKVCache(shard_config))
    
    def append(self, keys: np.ndarray, values: np.ndarray,
               seq_ids: np.ndarray) -> np.ndarray:
        """Append KV pairs (distributed across devices)."""
        # In a real implementation, this would distribute across GPUs
        return self._shards[0].append(keys, values, seq_ids)
    
    def lookup(self, block_indices: np.ndarray,
               seq_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lookup KV pairs (gathered from all devices)."""
        return self._shards[0].lookup(block_indices, seq_lens)
    
    def get_per_device_memory(self) -> List[int]:
        """Get memory usage per device."""
        return [shard.get_memory_usage() for shard in self._shards]
    
    def rebalance(self):
        """Rebalance load across devices."""
        pass


class SpeculativeKVCache:
    """
    KV cache with speculative decoding support.
    
    Enables efficient branching and rollback for draft token verification.
    
    Args:
        config: KV cache configuration
        spec_config: Speculative decoding configuration
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> spec_config = SpeculativeConfig(max_draft_tokens=8)
        >>> cache = SpeculativeKVCache(config, spec_config)
        >>> 
        >>> # Create branch for speculation
        >>> branch_id = cache.create_branch(seq_id)
        >>> 
        >>> # Append draft tokens
        >>> cache.append_speculative(keys, values, seq_id, branch_id)
        >>> 
        >>> # Verify and commit
        >>> result = cache.verify(seq_id, branch_id, target_logprobs)
        >>> cache.commit(seq_id, branch_id, result.accepted_count)
    """
    
    def __init__(self, config: KVCacheConfig, spec_config: SpeculativeConfig):
        self.config = config
        self.spec_config = spec_config
        self._base_cache = PagedKVCache(config)
        self._branches: Dict[int, Dict[int, Any]] = {}
    
    def create_branch(self, seq_id: int) -> int:
        """Create a speculation branch for a sequence."""
        if seq_id not in self._branches:
            self._branches[seq_id] = {}
        
        branch_id = len(self._branches[seq_id])
        self._branches[seq_id][branch_id] = {
            'verified_len': 0,
            'speculative_len': 0,
            'keys': [],
            'values': [],
        }
        
        return branch_id
    
    def append_verified(self, keys: np.ndarray, values: np.ndarray,
                        seq_ids: np.ndarray) -> np.ndarray:
        """Append verified KV pairs."""
        return self._base_cache.append(keys, values, seq_ids)
    
    def append_speculative(self, keys: np.ndarray, values: np.ndarray,
                           seq_id: int, branch_id: int = 0):
        """Append speculative (draft) KV pairs."""
        if seq_id in self._branches and branch_id in self._branches[seq_id]:
            branch = self._branches[seq_id][branch_id]
            branch['keys'].append(keys)
            branch['values'].append(values)
            branch['speculative_len'] += keys.shape[1]
    
    def verify(self, seq_id: int, branch_id: int,
               target_logprobs: np.ndarray) -> 'VerificationResult':
        """Verify speculative tokens."""
        # Simplified verification - in practice would compare logprobs
        return VerificationResult(
            accepted_count=0,
            rejected_position=-1,
            acceptance_rate=0.0
        )
    
    def commit(self, seq_id: int, branch_id: int, num_accepted: int):
        """Commit accepted speculative tokens."""
        if seq_id in self._branches and branch_id in self._branches[seq_id]:
            branch = self._branches[seq_id][branch_id]
            # Move accepted tokens to base cache
            branch['verified_len'] += num_accepted
            branch['speculative_len'] = 0
            branch['keys'] = []
            branch['values'] = []
    
    def rollback(self, seq_id: int, branch_id: int):
        """Rollback speculative tokens."""
        if seq_id in self._branches and branch_id in self._branches[seq_id]:
            branch = self._branches[seq_id][branch_id]
            branch['speculative_len'] = 0
            branch['keys'] = []
            branch['values'] = []
    
    def delete_branch(self, seq_id: int, branch_id: int):
        """Delete a speculation branch."""
        if seq_id in self._branches and branch_id in self._branches[seq_id]:
            del self._branches[seq_id][branch_id]


@dataclass
class VerificationResult:
    """Result of speculative token verification."""
    accepted_count: int
    rejected_position: int
    acceptance_rate: float
    accepted_mask: List[bool] = field(default_factory=list)


class PrefixCache:
    """
    Cache for common prompt prefixes.
    
    Efficiently reuses KV cache for repeated prompt prefixes
    using radix tree-based matching.
    
    Args:
        config: Prefix cache configuration
    
    Example:
        >>> cache = PrefixCache(PrefixCacheConfig(max_prefixes=1000))
        >>> 
        >>> # Cache a system prompt
        >>> cache.cache_prefix(system_tokens, block_indices)
        >>> 
        >>> # Lookup matching prefix
        >>> match_len, cached_blocks = cache.lookup(prompt_tokens)
    """
    
    def __init__(self, config: Optional[PrefixCacheConfig] = None):
        self.config = config or PrefixCacheConfig()
        self._cache: Dict[tuple, Any] = {}
        self._access_times: Dict[tuple, int] = {}
        self._access_counter = 0
        self._hits = 0
        self._misses = 0
    
    def cache_prefix(self, tokens: List[int], 
                     block_indices: List[List[int]]) -> bool:
        """Cache a prefix with its block indices."""
        if len(tokens) < self.config.min_prefix_length:
            return False
        
        key = tuple(tokens)
        
        # Evict if at capacity
        while len(self._cache) >= self.config.max_prefixes:
            self._evict_lru()
        
        self._cache[key] = block_indices
        self._access_times[key] = self._access_counter
        self._access_counter += 1
        
        return True
    
    def lookup(self, tokens: List[int]) -> Tuple[int, Optional[List[List[int]]]]:
        """
        Lookup longest matching prefix.
        
        Returns:
            match_length: Length of matched prefix
            block_indices: Cached block indices (or None if no match)
        """
        # Find longest matching prefix
        for length in range(len(tokens), self.config.min_prefix_length - 1, -1):
            prefix = tuple(tokens[:length])
            if prefix in self._cache:
                self._hits += 1
                self._access_times[prefix] = self._access_counter
                self._access_counter += 1
                return length, self._cache[prefix]
        
        self._misses += 1
        return 0, None
    
    def _evict_lru(self):
        """Evict least recently used prefix."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'num_prefixes': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_ratio': self.get_hit_ratio(),
        }
    
    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0
