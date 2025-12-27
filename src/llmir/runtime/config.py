"""
LLMIR Runtime Configuration Classes

Configuration dataclasses for KV cache, quantization, and other runtime components.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional


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


class QuantizationStrategy(Enum):
    """Quantization granularity strategy."""
    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    PER_GROUP = auto()


@dataclass
class KVCacheConfig:
    """Configuration for PagedKVCache.
    
    Attributes:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads (KV heads for GQA)
        head_dim: Dimension of each attention head
        block_size: Number of tokens per cache block
        max_seq_len: Maximum sequence length supported
        dtype: Data type for cache storage
        enable_gpu: Whether to use GPU memory
    
    Example:
        >>> config = KVCacheConfig(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     block_size=16,
        ...     max_seq_len=4096
        ... )
    """
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 16
    max_seq_len: int = 4096
    dtype: str = "float16"
    enable_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'block_size': self.block_size,
            'max_seq_len': self.max_seq_len,
            'dtype': self.dtype,
            'enable_gpu': self.enable_gpu,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KVCacheConfig':
        """Create configuration from dictionary."""
        return cls(**d)
    
    def memory_per_token(self) -> int:
        """Calculate memory usage per token in bytes."""
        dtype_size = 2 if 'float16' in self.dtype or 'bf16' in self.dtype else 4
        # 2 for K and V
        return 2 * self.num_layers * self.num_heads * self.head_dim * dtype_size
    
    def memory_per_sequence(self, seq_len: int) -> int:
        """Calculate memory usage per sequence in bytes."""
        return self.memory_per_token() * seq_len


@dataclass
class QuantizationConfig:
    """Configuration for quantized KV cache.
    
    Attributes:
        quant_type: Quantization type (INT8, INT4, FP8)
        strategy: Quantization granularity (per-tensor, per-channel, per-group)
        symmetric: Whether to use symmetric quantization
        group_size: Group size for per-group quantization
        dynamic_range: Whether to use dynamic range quantization
    
    Example:
        >>> config = QuantizationConfig(
        ...     quant_type=QuantizationType.INT8,
        ...     strategy=QuantizationStrategy.PER_CHANNEL
        ... )
    """
    quant_type: QuantizationType = QuantizationType.INT8
    strategy: QuantizationStrategy = QuantizationStrategy.PER_TENSOR
    symmetric: bool = True
    group_size: int = 128
    dynamic_range: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'quant_type': self.quant_type.name,
            'strategy': self.strategy.name,
            'symmetric': self.symmetric,
            'group_size': self.group_size,
            'dynamic_range': self.dynamic_range,
        }
    
    @property
    def compression_ratio(self) -> float:
        """Get expected compression ratio."""
        if self.quant_type == QuantizationType.INT8:
            return 4.0
        elif self.quant_type == QuantizationType.INT4:
            return 8.0
        elif self.quant_type == QuantizationType.FP8:
            return 2.0
        return 1.0


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.
    
    Attributes:
        max_draft_tokens: Maximum number of draft tokens per step
        max_branches: Maximum number of speculation branches
        enable_tree_attention: Whether to use tree attention
        acceptance_threshold: Threshold for accepting draft tokens
    
    Example:
        >>> config = SpeculativeConfig(
        ...     max_draft_tokens=8,
        ...     enable_tree_attention=True
        ... )
    """
    max_draft_tokens: int = 8
    max_branches: int = 4
    enable_tree_attention: bool = True
    acceptance_threshold: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_draft_tokens': self.max_draft_tokens,
            'max_branches': self.max_branches,
            'enable_tree_attention': self.enable_tree_attention,
            'acceptance_threshold': self.acceptance_threshold,
        }


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix caching.
    
    Attributes:
        max_prefixes: Maximum number of cached prefixes
        max_memory_gb: Maximum memory for prefix cache in GB
        enable_radix_tree: Whether to use radix tree for matching
        min_prefix_length: Minimum prefix length to cache
    """
    max_prefixes: int = 1000
    max_memory_gb: float = 4.0
    enable_radix_tree: bool = True
    min_prefix_length: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_prefixes': self.max_prefixes,
            'max_memory_gb': self.max_memory_gb,
            'enable_radix_tree': self.enable_radix_tree,
            'min_prefix_length': self.min_prefix_length,
        }


@dataclass
class ShardingConfig:
    """Configuration for distributed KV cache.
    
    Attributes:
        strategy: Sharding strategy
        num_devices: Number of devices to shard across
        device_ids: List of device IDs
        enable_nccl: Whether to use NCCL for communication
    """
    strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE
    num_devices: int = 1
    device_ids: List[int] = field(default_factory=list)
    enable_nccl: bool = True
    
    def __post_init__(self):
        if not self.device_ids:
            self.device_ids = list(range(self.num_devices))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'strategy': self.strategy.name,
            'num_devices': self.num_devices,
            'device_ids': self.device_ids,
            'enable_nccl': self.enable_nccl,
        }
