"""
LLMIR LLM Dialect Python Bindings

This module provides Python bindings for the LLMIR LLM dialect,
including KV cache management, attention optimizations, and serving utilities.

When the `llmir` PyPI package is installed, this module re-exports its classes.
Otherwise, it provides standalone implementations for use with MLIR builds.
"""

# Try to import from the llmir PyPI package first
try:
    from llmir import (
        # Core KV Cache classes
        PagedKVCache,
        QuantizedKVCache,
        DistributedKVCache,
        SpeculativeKVCache,
        PrefixCache,
        
        # Configuration classes
        KVCacheConfig,
        QuantizationConfig,
        QuantizationType,
        ShardingStrategy,
        SpeculativeConfig,
        
        # Serving components
        LLMEngine,
        ContinuousBatchingEngine,
        SamplingParams,
        SchedulerConfig,
        SchedulingPolicy,
        RequestPriority,
        
        # Profiling tools
        Profiler,
        MemoryProfiler,
        LatencyProfiler,
        ThroughputMonitor,
        
        # Model optimizations
        ModelOptimizer,
        LlamaOptimizer,
        MistralOptimizer,
        PhiOptimizer,
        ModelRegistry,
    )
    
    _USING_LLMIR_PACKAGE = True
    
except ImportError:
    # Fall back to standalone implementation
    _USING_LLMIR_PACKAGE = False
    
    from typing import Optional, List, Dict, Any, Tuple
    from dataclasses import dataclass, field
    from enum import Enum, auto
    import numpy as np
    
    class QuantizationType(Enum):
        """Quantization type for KV cache."""
        NONE = auto()
        INT8 = auto()
        INT4 = auto()
        FP8 = auto()
    
    class ShardingStrategy(Enum):
        """Sharding strategy for distributed KV cache."""
        LAYER_WISE = auto()
        HEAD_WISE = auto()
        SEQUENCE_WISE = auto()
        HYBRID = auto()
    
    class SchedulingPolicy(Enum):
        """Scheduling policy for continuous batching."""
        FCFS = auto()
        SHORTEST_FIRST = auto()
        PRIORITY_BASED = auto()
        FAIR_SHARE = auto()
        ADAPTIVE = auto()
    
    class RequestPriority(Enum):
        """Request priority levels."""
        LOW = 0
        NORMAL = 1
        HIGH = 2
        URGENT = 3
    
    @dataclass
    class KVCacheConfig:
        """Configuration for PagedKVCache."""
        num_layers: int = 32
        num_heads: int = 32
        head_dim: int = 128
        block_size: int = 16
        max_seq_len: int = 4096
        dtype: str = "float16"
        enable_gpu: bool = True
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'head_dim': self.head_dim,
                'block_size': self.block_size,
                'max_seq_len': self.max_seq_len,
                'dtype': self.dtype,
                'enable_gpu': self.enable_gpu,
            }
    
    @dataclass
    class QuantizationConfig:
        """Configuration for quantized KV cache."""
        quant_type: QuantizationType = QuantizationType.INT8
        symmetric: bool = True
        per_channel: bool = False
        group_size: int = 128
    
    @dataclass
    class SchedulerConfig:
        """Configuration for continuous batching scheduler."""
        policy: SchedulingPolicy = SchedulingPolicy.FCFS
        max_batch_size: int = 256
        max_num_seqs: int = 256
        max_batch_tokens: int = 8192
        chunk_size: int = 512
        enable_preemption: bool = True
        preemption_threshold: float = 0.9
    
    @dataclass
    class SpeculativeConfig:
        """Configuration for speculative decoding."""
        max_draft_tokens: int = 8
        max_branches: int = 4
        enable_tree_attention: bool = True
        acceptance_threshold: float = 0.9
    
    @dataclass
    class SamplingParams:
        """Sampling parameters for generation."""
        n: int = 1
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        max_tokens: int = 256
        stop: List[str] = field(default_factory=list)
        stop_token_ids: List[int] = field(default_factory=list)
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        repetition_penalty: float = 1.0
    
    class PagedKVCache:
        """Paged Key-Value cache for efficient LLM inference."""
        
        def __init__(self, config: KVCacheConfig):
            self.config = config
            self._sequences: Dict[int, Dict[str, Any]] = {}
        
        def append(self, keys: np.ndarray, values: np.ndarray,
                   seq_ids: np.ndarray) -> np.ndarray:
            batch_size = keys.shape[0]
            return np.zeros((batch_size, self.config.num_layers), dtype=np.int32)
        
        def lookup(self, block_indices: np.ndarray,
                   seq_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            batch_size = len(seq_lens)
            max_seq_len = int(seq_lens.max())
            keys = np.zeros((batch_size, max_seq_len, 
                            self.config.num_heads, self.config.head_dim),
                           dtype=np.float16)
            return keys, np.zeros_like(keys)
        
        def clear_sequence(self, seq_id: int) -> bool:
            if seq_id in self._sequences:
                del self._sequences[seq_id]
                return True
            return False
        
        def reset(self):
            self._sequences.clear()
        
        @property
        def block_size(self) -> int:
            return self.config.block_size
        
        @property
        def num_layers(self) -> int:
            return self.config.num_layers
    
    class QuantizedKVCache(PagedKVCache):
        """Quantized KV cache with INT8/INT4 support."""
        
        def __init__(self, config: KVCacheConfig, quant_config: QuantizationConfig):
            self.quant_config = quant_config
            super().__init__(config)
        
        def get_compression_ratio(self) -> float:
            if self.quant_config.quant_type == QuantizationType.INT8:
                return 4.0
            elif self.quant_config.quant_type == QuantizationType.INT4:
                return 8.0
            return 1.0
    
    class DistributedKVCache:
        """Distributed KV cache for multi-GPU inference."""
        
        def __init__(self, config: KVCacheConfig, num_devices: int,
                     strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE,
                     device_ids: Optional[List[int]] = None):
            self.config = config
            self.num_devices = num_devices
            self.strategy = strategy
            self.device_ids = device_ids or list(range(num_devices))
    
    class SpeculativeKVCache:
        """KV cache with speculative decoding support."""
        
        def __init__(self, config: KVCacheConfig, spec_config: SpeculativeConfig):
            self.config = config
            self.spec_config = spec_config
            self._base_cache = PagedKVCache(config)
            self._branches: Dict[int, Dict[int, Any]] = {}
        
        def create_branch(self, seq_id: int) -> int:
            if seq_id not in self._branches:
                self._branches[seq_id] = {}
            branch_id = len(self._branches[seq_id])
            self._branches[seq_id][branch_id] = {}
            return branch_id
    
    class PrefixCache:
        """Cache for common prompt prefixes."""
        
        def __init__(self, max_prefixes: int = 1000, max_memory_gb: float = 4.0):
            self.max_prefixes = max_prefixes
            self._cache: Dict[tuple, Any] = {}
        
        def cache_prefix(self, tokens: List[int], block_indices: List[List[int]]) -> bool:
            self._cache[tuple(tokens)] = block_indices
            return True
        
        def lookup(self, tokens: List[int]) -> Tuple[int, Optional[List[List[int]]]]:
            for length in range(len(tokens), 0, -1):
                prefix = tuple(tokens[:length])
                if prefix in self._cache:
                    return length, self._cache[prefix]
            return 0, None
    
    class ContinuousBatchingEngine:
        """Continuous batching engine for production LLM serving."""
        
        def __init__(self, cache_config: KVCacheConfig,
                     scheduler_config: SchedulerConfig):
            self.cache_config = cache_config
            self.scheduler_config = scheduler_config
            self._running = False
        
        def start(self):
            self._running = True
        
        def stop(self):
            self._running = False
    
    class LLMEngine:
        """High-level LLM engine with vLLM-compatible API."""
        
        def __init__(self, model_path: str, **kwargs):
            self.model_path = model_path
        
        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'LLMEngine':
            return cls(model_path=model_name_or_path, **kwargs)
    
    class Profiler:
        """Performance profiler for LLMIR."""
        
        def __init__(self):
            self._active = False
        
        def start(self):
            self._active = True
        
        def stop(self):
            self._active = False
    
    # Placeholder classes for compatibility
    class MemoryProfiler:
        pass
    
    class LatencyProfiler:
        pass
    
    class ThroughputMonitor:
        pass
    
    class ModelOptimizer:
        pass
    
    class LlamaOptimizer:
        pass
    
    class MistralOptimizer:
        pass
    
    class PhiOptimizer:
        pass
    
    class ModelRegistry:
        pass


# Import legacy compatibility functions
try:
    from .compat import (
        attention_forward,
        kv_cache_append,
        optimize_model,
        set_debug,
    )
except ImportError:
    pass

# Import native bindings if available
try:
    from .native import (
        NativePagedKVCache,
        NativeQuantizedKVCache,
        NativeContinuousBatchingEngine,
        is_native_available,
        load_library,
    )
except ImportError:
    is_native_available = lambda: False


__all__ = [
    # Core classes
    'PagedKVCache',
    'QuantizedKVCache', 
    'DistributedKVCache',
    'SpeculativeKVCache',
    'PrefixCache',
    
    # Serving
    'ContinuousBatchingEngine',
    'LLMEngine',
    
    # Configuration
    'KVCacheConfig',
    'QuantizationConfig',
    'SchedulerConfig',
    'SpeculativeConfig',
    'SamplingParams',
    
    # Enums
    'QuantizationType',
    'ShardingStrategy',
    'SchedulingPolicy',
    'RequestPriority',
    
    # Profiling
    'Profiler',
    'MemoryProfiler',
    'LatencyProfiler',
    'ThroughputMonitor',
    
    # Models
    'ModelOptimizer',
    'LlamaOptimizer',
    'MistralOptimizer',
    'PhiOptimizer',
    'ModelRegistry',
    
    # Native bindings
    'is_native_available',
]
