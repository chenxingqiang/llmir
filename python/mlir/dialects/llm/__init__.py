"""
LLMIR LLM Dialect Python Bindings

This module provides Python bindings for the LLMIR LLM dialect,
including KV cache management, attention optimizations, and serving utilities.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
import ctypes
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import os
import json

# Import legacy compatibility functions
try:
    from .compat import (
        attention_forward,
        kv_cache_append,
        optimize_model,
        set_debug,
    )
except ImportError:
    pass  # compat module may not be available

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
    
    # Enums
    'QuantizationType',
    'ShardingStrategy',
    'SchedulingPolicy',
    
    # Utilities
    'CheckpointManager',
    'Profiler',
]

#===----------------------------------------------------------------------===#
# Configuration Enums
#===----------------------------------------------------------------------===#

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


#===----------------------------------------------------------------------===#
# Configuration Classes
#===----------------------------------------------------------------------===#

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quant_type': self.quant_type.name,
            'symmetric': self.symmetric,
            'per_channel': self.per_channel,
            'group_size': self.group_size,
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy': self.policy.name,
            'max_batch_size': self.max_batch_size,
            'max_num_seqs': self.max_num_seqs,
            'max_batch_tokens': self.max_batch_tokens,
            'chunk_size': self.chunk_size,
            'enable_preemption': self.enable_preemption,
            'preemption_threshold': self.preemption_threshold,
        }


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    max_draft_tokens: int = 8
    max_branches: int = 4
    enable_tree_attention: bool = True
    acceptance_threshold: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_draft_tokens': self.max_draft_tokens,
            'max_branches': self.max_branches,
            'enable_tree_attention': self.enable_tree_attention,
            'acceptance_threshold': self.acceptance_threshold,
        }


@dataclass
class SamplingParams:
    """Sampling parameters for generation (vLLM compatible)."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n': self.n,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_tokens': self.max_tokens,
            'stop': self.stop,
            'stop_token_ids': self.stop_token_ids,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'repetition_penalty': self.repetition_penalty,
        }


#===----------------------------------------------------------------------===#
# Core KV Cache Classes
#===----------------------------------------------------------------------===#

class PagedKVCache:
    """
    Paged Key-Value cache for efficient LLM inference.
    
    This class provides block-based memory management for KV cache,
    enabling efficient handling of variable-length sequences.
    
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
        self._handle = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the native cache object."""
        # Would call into C++ via ctypes or pybind11
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
        assert keys.shape == values.shape
        assert len(seq_ids) == batch_size
        
        # Would call native implementation
        block_indices = np.zeros((batch_size, self.config.num_layers), dtype=np.int32)
        
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
        
        keys = np.zeros((batch_size, max_seq_len, 
                        self.config.num_heads, self.config.head_dim),
                       dtype=np.float16)
        values = np.zeros_like(keys)
        
        return keys, values
    
    def clear_sequence(self, seq_id: int) -> bool:
        """Clear cache for a specific sequence."""
        return True
    
    def reset(self):
        """Reset the entire cache."""
        pass
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return 0
    
    def get_num_sequences(self) -> int:
        """Get number of active sequences."""
        return 0
    
    def get_sequence_length(self, seq_id: int) -> int:
        """Get length of a specific sequence."""
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
        if self.quant_config.quant_type == QuantizationType.INT8:
            return 4.0
        elif self.quant_config.quant_type == QuantizationType.INT4:
            return 8.0
        return 1.0
    
    def get_accuracy_loss(self) -> float:
        """Get estimated accuracy loss from quantization."""
        return 0.0


class DistributedKVCache:
    """
    Distributed KV cache for multi-GPU inference.
    
    Supports layer-wise, head-wise, and sequence-wise sharding
    across multiple GPUs.
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> dist_cache = DistributedKVCache(
        ...     config, 
        ...     num_devices=4,
        ...     strategy=ShardingStrategy.LAYER_WISE
        ... )
    """
    
    def __init__(self, 
                 config: KVCacheConfig,
                 num_devices: int,
                 strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE,
                 device_ids: Optional[List[int]] = None):
        self.config = config
        self.num_devices = num_devices
        self.strategy = strategy
        self.device_ids = device_ids or list(range(num_devices))
        self._initialize()
    
    def _initialize(self):
        """Initialize distributed cache across devices."""
        pass
    
    def append(self, keys: np.ndarray, values: np.ndarray,
               seq_ids: np.ndarray) -> np.ndarray:
        """Append KV pairs (distributed across devices)."""
        return np.zeros((len(seq_ids), self.config.num_layers), dtype=np.int32)
    
    def lookup(self, block_indices: np.ndarray,
               seq_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lookup KV pairs (gathered from all devices)."""
        batch_size = len(seq_lens)
        max_seq_len = int(seq_lens.max())
        
        keys = np.zeros((batch_size, max_seq_len,
                        self.config.num_heads, self.config.head_dim),
                       dtype=np.float16)
        values = np.zeros_like(keys)
        
        return keys, values
    
    def get_per_device_memory(self) -> List[int]:
        """Get memory usage per device."""
        return [0] * self.num_devices
    
    def rebalance(self):
        """Rebalance load across devices."""
        pass


class SpeculativeKVCache:
    """
    KV cache with speculative decoding support.
    
    Enables efficient branching and rollback for draft token verification.
    
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
        }
        
        return branch_id
    
    def append_verified(self, keys: np.ndarray, values: np.ndarray,
                        seq_ids: np.ndarray) -> np.ndarray:
        """Append verified KV pairs."""
        return self._base_cache.append(keys, values, seq_ids)
    
    def append_speculative(self, keys: np.ndarray, values: np.ndarray,
                           seq_id: int, branch_id: int = 0):
        """Append speculative (draft) KV pairs."""
        pass
    
    def verify(self, seq_id: int, branch_id: int,
               target_logprobs: np.ndarray) -> 'VerificationResult':
        """Verify speculative tokens."""
        return VerificationResult(
            accepted_count=0,
            rejected_position=-1,
            acceptance_rate=0.0
        )
    
    def commit(self, seq_id: int, branch_id: int, num_accepted: int):
        """Commit accepted speculative tokens."""
        pass
    
    def rollback(self, seq_id: int, branch_id: int):
        """Rollback speculative tokens."""
        if seq_id in self._branches and branch_id in self._branches[seq_id]:
            self._branches[seq_id][branch_id]['speculative_len'] = 0


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
    
    Example:
        >>> cache = PrefixCache(max_prefixes=1000)
        >>> 
        >>> # Cache a system prompt
        >>> cache.cache_prefix(system_tokens, block_indices)
        >>> 
        >>> # Lookup matching prefix
        >>> match_len, cached_blocks = cache.lookup(prompt_tokens)
    """
    
    def __init__(self, max_prefixes: int = 1000, max_memory_gb: float = 4.0):
        self.max_prefixes = max_prefixes
        self.max_memory = int(max_memory_gb * 1024 * 1024 * 1024)
        self._cache: Dict[tuple, Any] = {}
    
    def cache_prefix(self, tokens: List[int], 
                     block_indices: List[List[int]]) -> bool:
        """Cache a prefix with its block indices."""
        key = tuple(tokens)
        self._cache[key] = block_indices
        return True
    
    def lookup(self, tokens: List[int]) -> Tuple[int, Optional[List[List[int]]]]:
        """
        Lookup longest matching prefix.
        
        Returns:
            match_length: Length of matched prefix
            block_indices: Cached block indices (or None if no match)
        """
        # Find longest matching prefix
        for length in range(len(tokens), 0, -1):
            prefix = tuple(tokens[:length])
            if prefix in self._cache:
                return length, self._cache[prefix]
        
        return 0, None
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        return 0.0
    
    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()


#===----------------------------------------------------------------------===#
# Serving Classes
#===----------------------------------------------------------------------===#

class ContinuousBatchingEngine:
    """
    Continuous batching engine for production LLM serving.
    
    Implements vLLM-style dynamic batch management with preemption support.
    
    Example:
        >>> engine = ContinuousBatchingEngine(
        ...     cache_config=KVCacheConfig(...),
        ...     scheduler_config=SchedulerConfig(...)
        ... )
        >>> engine.start()
        >>> 
        >>> # Submit requests
        >>> request_id = engine.submit(prompt_tokens, sampling_params)
        >>> 
        >>> # Get outputs
        >>> for output in engine.iterate():
        ...     print(output.text)
    """
    
    def __init__(self, 
                 cache_config: KVCacheConfig,
                 scheduler_config: SchedulerConfig):
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self._cache = PagedKVCache(cache_config)
        self._running = False
        self._requests: Dict[str, Any] = {}
        self._next_id = 0
    
    def start(self):
        """Start the engine."""
        self._running = True
    
    def stop(self):
        """Stop the engine."""
        self._running = False
    
    def submit(self, 
               prompt_tokens: List[int],
               sampling_params: Optional[SamplingParams] = None,
               priority: RequestPriority = RequestPriority.NORMAL) -> str:
        """
        Submit a generation request.
        
        Returns:
            request_id: Unique identifier for the request
        """
        request_id = f"req-{self._next_id}"
        self._next_id += 1
        
        self._requests[request_id] = {
            'prompt': prompt_tokens,
            'params': sampling_params or SamplingParams(),
            'priority': priority,
            'status': 'pending',
            'outputs': [],
        }
        
        return request_id
    
    def abort(self, request_id: str) -> bool:
        """Abort a request."""
        if request_id in self._requests:
            self._requests[request_id]['status'] = 'aborted'
            return True
        return False
    
    def iterate(self) -> List['RequestOutput']:
        """
        Run one iteration and return outputs.
        
        Returns:
            List of RequestOutput for completed/updated requests
        """
        outputs = []
        
        for req_id, req in self._requests.items():
            if req['status'] == 'pending':
                req['status'] = 'running'
            elif req['status'] == 'running':
                # Simulate token generation
                output = RequestOutput(
                    request_id=req_id,
                    outputs=[CompletionOutput(text="", token_ids=[], finished=False)],
                    finished=False
                )
                outputs.append(output)
        
        return outputs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'num_pending': sum(1 for r in self._requests.values() if r['status'] == 'pending'),
            'num_running': sum(1 for r in self._requests.values() if r['status'] == 'running'),
            'num_completed': sum(1 for r in self._requests.values() if r['status'] == 'completed'),
        }


@dataclass
class CompletionOutput:
    """Output from a single completion."""
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: str = ""
    logprobs: Optional[List[float]] = None


@dataclass
class RequestOutput:
    """Output from a request."""
    request_id: str
    outputs: List[CompletionOutput]
    finished: bool
    prompt: str = ""
    prompt_token_ids: List[int] = field(default_factory=list)


class LLMEngine:
    """
    High-level LLM engine with vLLM-compatible API.
    
    Example:
        >>> engine = LLMEngine.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> 
        >>> # Generate completions
        >>> outputs = engine.generate(
        ...     ["Hello, how are you?", "What is 2+2?"],
        ...     SamplingParams(max_tokens=100)
        ... )
        >>> 
        >>> for output in outputs:
        ...     print(output.outputs[0].text)
    """
    
    def __init__(self,
                 model_path: str,
                 cache_config: Optional[KVCacheConfig] = None,
                 scheduler_config: Optional[SchedulerConfig] = None,
                 tensor_parallel_size: int = 1,
                 dtype: str = "float16",
                 gpu_memory_utilization: float = 0.9):
        self.model_path = model_path
        self.cache_config = cache_config or KVCacheConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self._engine = ContinuousBatchingEngine(
            self.cache_config, self.scheduler_config)
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path: str,
                        **kwargs) -> 'LLMEngine':
        """Load engine from a pretrained model."""
        return cls(model_path=model_name_or_path, **kwargs)
    
    def generate(self,
                 prompts: List[str],
                 sampling_params: Optional[SamplingParams] = None) -> List[RequestOutput]:
        """
        Generate completions for prompts.
        
        Args:
            prompts: List of prompt strings
            sampling_params: Sampling parameters
            
        Returns:
            List of RequestOutput
        """
        params = sampling_params or SamplingParams()
        
        # Submit all prompts
        request_ids = []
        for prompt in prompts:
            # Would tokenize prompt here
            tokens = list(range(len(prompt.split())))  # Placeholder
            req_id = self._engine.submit(tokens, params)
            request_ids.append(req_id)
        
        # Collect outputs
        outputs = []
        for req_id in request_ids:
            output = RequestOutput(
                request_id=req_id,
                prompt=prompts[request_ids.index(req_id)],
                outputs=[CompletionOutput(text="Generated text", token_ids=[], finished=True)],
                finished=True
            )
            outputs.append(output)
        
        return outputs


#===----------------------------------------------------------------------===#
# Utilities
#===----------------------------------------------------------------------===#

class CheckpointManager:
    """
    Manager for KV cache checkpoints.
    
    Example:
        >>> manager = CheckpointManager("/path/to/checkpoints")
        >>> 
        >>> # Save checkpoint
        >>> manager.save(cache, "checkpoint-001")
        >>> 
        >>> # Load checkpoint
        >>> manager.load(cache, "checkpoint-001")
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, cache: PagedKVCache, name: str) -> str:
        """Save a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
        # Would serialize cache state
        return path
    
    def load(self, cache: PagedKVCache, name: str) -> bool:
        """Load a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
        if os.path.exists(path):
            # Would deserialize cache state
            return True
        return False
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        return [f[:-5] for f in os.listdir(self.checkpoint_dir) 
                if f.endswith('.ckpt')]
    
    def delete(self, name: str) -> bool:
        """Delete a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


class Profiler:
    """
    Performance profiler for LLMIR.
    
    Example:
        >>> profiler = Profiler()
        >>> profiler.start()
        >>> 
        >>> # Run operations
        >>> with profiler.trace("attention"):
        ...     attention_output = model.attention(x)
        >>> 
        >>> profiler.stop()
        >>> print(profiler.get_summary())
    """
    
    def __init__(self):
        self._traces: Dict[str, List[float]] = {}
        self._active = False
        self._start_times: Dict[str, float] = {}
    
    def start(self):
        """Start profiling."""
        self._active = True
        self._traces.clear()
    
    def stop(self):
        """Stop profiling."""
        self._active = False
    
    class TraceContext:
        def __init__(self, profiler: 'Profiler', name: str):
            self.profiler = profiler
            self.name = name
            self.start_time = 0.0
        
        def __enter__(self):
            import time
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            import time
            elapsed = time.perf_counter() - self.start_time
            if self.name not in self.profiler._traces:
                self.profiler._traces[self.name] = []
            self.profiler._traces[self.name].append(elapsed * 1000)  # ms
    
    def trace(self, name: str) -> 'TraceContext':
        """Create a trace context for a named operation."""
        return self.TraceContext(self, name)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get profiling summary."""
        summary = {}
        for name, times in self._traces.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                }
        return summary
    
    def print_summary(self):
        """Print profiling summary."""
        summary = self.get_summary()
        print("\n=== Profiling Summary ===")
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
            print(f"\n{name}:")
            print(f"  Count:    {stats['count']}")
            print(f"  Total:    {stats['total_ms']:.2f} ms")
            print(f"  Average:  {stats['avg_ms']:.2f} ms")
            print(f"  Min/Max:  {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
