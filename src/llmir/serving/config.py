"""
LLMIR Serving Configuration Classes

Configuration for LLM serving and continuous batching.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional


class SchedulingPolicy(Enum):
    """Scheduling policy for continuous batching."""
    FCFS = auto()           # First-Come-First-Served
    SHORTEST_FIRST = auto() # Shortest remaining first
    PRIORITY_BASED = auto() # Priority-based scheduling
    FAIR_SHARE = auto()     # Fair share scheduling
    ADAPTIVE = auto()       # Adaptive policy


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class SamplingParams:
    """
    Sampling parameters for generation (vLLM compatible).
    
    Args:
        n: Number of completions to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling
        top_k: Top-k sampling (-1 to disable)
        max_tokens: Maximum tokens to generate
        stop: Stop strings
        stop_token_ids: Stop token IDs
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        repetition_penalty: Repetition penalty
    
    Example:
        >>> params = SamplingParams(
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     max_tokens=256
        ... )
    """
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
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SamplingParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerConfig:
    """
    Configuration for continuous batching scheduler.
    
    Args:
        policy: Scheduling policy
        max_batch_size: Maximum sequences in a batch
        max_num_seqs: Maximum concurrent sequences
        max_batch_tokens: Maximum tokens per batch
        chunk_size: Chunk size for chunked prefill
        enable_preemption: Enable preemption under memory pressure
        preemption_threshold: Memory threshold for preemption
    
    Example:
        >>> config = SchedulerConfig(
        ...     policy=SchedulingPolicy.ADAPTIVE,
        ...     max_batch_size=256,
        ...     enable_preemption=True
        ... )
    """
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    max_batch_size: int = 256
    max_num_seqs: int = 256
    max_batch_tokens: int = 8192
    chunk_size: int = 512
    enable_preemption: bool = True
    preemption_threshold: float = 0.9
    delay_factor: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy': self.policy.name,
            'max_batch_size': self.max_batch_size,
            'max_num_seqs': self.max_num_seqs,
            'max_batch_tokens': self.max_batch_tokens,
            'chunk_size': self.chunk_size,
            'enable_preemption': self.enable_preemption,
            'preemption_threshold': self.preemption_threshold,
            'delay_factor': self.delay_factor,
        }


@dataclass
class EngineConfig:
    """
    Configuration for LLM engine.
    
    Args:
        model_path: Path to the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of stages for pipeline parallelism
        dtype: Data type for inference
        gpu_memory_utilization: Target GPU memory utilization
        max_model_len: Maximum model context length
    """
    model_path: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_path': self.model_path,
            'tensor_parallel_size': self.tensor_parallel_size,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'dtype': self.dtype,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'max_model_len': self.max_model_len,
            'trust_remote_code': self.trust_remote_code,
        }
