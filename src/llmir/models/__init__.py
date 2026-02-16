"""
LLMIR Models Module

Model-specific optimizations for popular LLM architectures.
"""

from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from llmir.runtime.config import KVCacheConfig, QuantizationConfig, QuantizationType

__all__ = [
    'ModelArchitecture',
    'ModelConfig',
    'ModelOptimizer',
    'LlamaOptimizer',
    'MistralOptimizer',
    'PhiOptimizer',
    'QwenOptimizer',
    'GemmaOptimizer',
    'FalconOptimizer',
    'ModelRegistry',
    'ModelMemoryEstimator',
]


class ModelArchitecture(Enum):
    """Supported model architectures."""
    LLAMA = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    MISTRAL = auto()
    MIXTRAL = auto()
    PHI = auto()
    PHI3 = auto()
    QWEN = auto()
    QWEN2 = auto()
    GEMMA = auto()
    FALCON = auto()
    CUSTOM = auto()


class AttentionType(Enum):
    """Attention type used by the model."""
    MULTI_HEAD = auto()
    MULTI_QUERY = auto()
    GROUPED_QUERY = auto()
    SLIDING_WINDOW = auto()


@dataclass
class ModelConfig:
    """Configuration for a specific model architecture."""
    architecture: ModelArchitecture = ModelArchitecture.LLAMA
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    
    # Model dimensions
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    
    # RoPE parameters
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    
    # Sliding window
    sliding_window_size: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "architecture": self.architecture.name,
            "attention_type": self.attention_type.name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling_factor": self.rope_scaling_factor,
            "sliding_window_size": self.sliding_window_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        d = dict(d)
        if "architecture" in d and isinstance(d["architecture"], str):
            d["architecture"] = ModelArchitecture[d["architecture"]]
        if "attention_type" in d and isinstance(d["attention_type"], str):
            d["attention_type"] = AttentionType[d["attention_type"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def get_head_dim(self) -> int:
        """Get head dimension."""
        if self.head_dim > 0:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads
    
    def is_gqa(self) -> bool:
        """Check if using grouped query attention."""
        return self.num_key_value_heads > 0 and self.num_key_value_heads < self.num_attention_heads
    
    def get_num_queries_per_kv(self) -> int:
        """Get number of query heads per KV head."""
        if self.num_key_value_heads <= 0:
            return 1
        return self.num_attention_heads // self.num_key_value_heads


class ModelOptimizer:
    """Base class for model-specific optimizations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def get_optimized_kv_cache_config(self) -> KVCacheConfig:
        """Get optimized KV cache configuration."""
        return KVCacheConfig(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_key_value_heads or self.config.num_attention_heads,
            head_dim=self.config.get_head_dim(),
            block_size=self.get_optimized_block_size(),
            max_seq_len=self.config.max_position_embeddings,
        )
    
    def get_recommended_quant_config(self) -> QuantizationConfig:
        """Get recommended quantization config."""
        # Larger models benefit more from INT4
        if self.config.num_layers >= 80:
            return QuantizationConfig(quant_type=QuantizationType.INT4)
        return QuantizationConfig(quant_type=QuantizationType.INT8)
    
    def get_optimized_block_size(self) -> int:
        """Get optimized block size."""
        head_dim = self.config.get_head_dim()
        if head_dim <= 64:
            return 32
        elif head_dim <= 128:
            return 16
        return 8
    
    def get_recommended_batch_size(self, gpu_memory_gb: float = 80) -> int:
        """Get recommended batch size based on GPU memory."""
        estimator = ModelMemoryEstimator(self.config)
        return estimator.find_max_batch_size(
            int(gpu_memory_gb * 1e9 * 0.9),  # 90% utilization
            self.config.max_position_embeddings
        )
    
    def estimate_memory(self, batch_size: int, seq_len: int) -> int:
        """Estimate memory usage in bytes."""
        estimator = ModelMemoryEstimator(self.config)
        return estimator.estimate_total_memory(batch_size, seq_len)


class LlamaOptimizer(ModelOptimizer):
    """Optimizations for Llama models."""
    
    @classmethod
    def for_llama_7b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=32,
            head_dim=128, intermediate_size=11008,
            vocab_size=32000, max_position_embeddings=2048,
        ))
    
    @classmethod
    def for_llama_13b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 13B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA,
            num_layers=40, hidden_size=5120,
            num_attention_heads=40, num_key_value_heads=40,
            head_dim=128, intermediate_size=13824,
            vocab_size=32000, max_position_embeddings=2048,
        ))
    
    @classmethod
    def for_llama2_7b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 2 7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA2,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=32,
            head_dim=128, intermediate_size=11008,
            vocab_size=32000, max_position_embeddings=4096,
        ))
    
    @classmethod
    def for_llama2_70b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 2 70B (GQA)."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA2,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8,
            head_dim=128, intermediate_size=28672,
            vocab_size=32000, max_position_embeddings=4096,
        ))
    
    @classmethod
    def for_llama3_8b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 3 8B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, intermediate_size=14336,
            vocab_size=128256, max_position_embeddings=8192,
            rope_theta=500000.0,
        ))
    
    @classmethod
    def for_llama3_70b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 3 70B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8,
            head_dim=128, intermediate_size=28672,
            vocab_size=128256, max_position_embeddings=8192,
            rope_theta=500000.0,
        ))
    
    @classmethod
    def for_llama31_8b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 3.1 8B (128K context)."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, intermediate_size=14336,
            vocab_size=128256, max_position_embeddings=131072,
            rope_theta=500000.0, rope_scaling_factor=8.0,
        ))
    
    @classmethod
    def for_llama31_70b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 3.1 70B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8,
            head_dim=128, intermediate_size=28672,
            vocab_size=128256, max_position_embeddings=131072,
            rope_theta=500000.0, rope_scaling_factor=8.0,
        ))
    
    @classmethod
    def for_llama31_405b(cls) -> 'LlamaOptimizer':
        """Create optimizer for Llama 3.1 405B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.LLAMA3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=126, hidden_size=16384,
            num_attention_heads=128, num_key_value_heads=8,
            head_dim=128, intermediate_size=53248,
            vocab_size=128256, max_position_embeddings=131072,
            rope_theta=500000.0, rope_scaling_factor=8.0,
        ))


class MistralOptimizer(ModelOptimizer):
    """Optimizations for Mistral models."""
    
    @classmethod
    def for_mistral_7b(cls) -> 'MistralOptimizer':
        """Create optimizer for Mistral 7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.MISTRAL,
            attention_type=AttentionType.SLIDING_WINDOW,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, intermediate_size=14336,
            vocab_size=32000, max_position_embeddings=32768,
            sliding_window_size=4096,
        ))
    
    @classmethod
    def for_mixtral_8x7b(cls) -> 'MistralOptimizer':
        """Create optimizer for Mixtral 8x7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.MIXTRAL,
            attention_type=AttentionType.SLIDING_WINDOW,
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, intermediate_size=14336,
            vocab_size=32000, max_position_embeddings=32768,
            sliding_window_size=4096,
            rope_theta=1000000.0,
        ))
    
    @classmethod
    def for_mixtral_8x22b(cls) -> 'MistralOptimizer':
        """Create optimizer for Mixtral 8x22B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.MIXTRAL,
            attention_type=AttentionType.SLIDING_WINDOW,
            num_layers=56, hidden_size=6144,
            num_attention_heads=48, num_key_value_heads=8,
            head_dim=128, intermediate_size=16384,
            vocab_size=32768, max_position_embeddings=65536,
            sliding_window_size=4096,
            rope_theta=1000000.0,
        ))
    
    def get_optimized_block_size(self) -> int:
        """Smaller block size for sliding window efficiency."""
        return 8
    
    def uses_sliding_window(self) -> bool:
        """Check if model uses sliding window attention."""
        return self.config.sliding_window_size < self.config.max_position_embeddings


class PhiOptimizer(ModelOptimizer):
    """Optimizations for Phi models."""
    
    @classmethod
    def for_phi2(cls) -> 'PhiOptimizer':
        """Create optimizer for Phi-2."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.PHI,
            num_layers=32, hidden_size=2560,
            num_attention_heads=32, num_key_value_heads=32,
            head_dim=80, intermediate_size=10240,
            vocab_size=51200, max_position_embeddings=2048,
        ))
    
    @classmethod
    def for_phi3_mini(cls) -> 'PhiOptimizer':
        """Create optimizer for Phi-3 Mini."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.PHI3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=32, hidden_size=3072,
            num_attention_heads=32, num_key_value_heads=8,
            head_dim=96, intermediate_size=8192,
            vocab_size=32064, max_position_embeddings=131072,
        ))
    
    @classmethod
    def for_phi3_medium(cls) -> 'PhiOptimizer':
        """Create optimizer for Phi-3 Medium."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.PHI3,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=40, hidden_size=5120,
            num_attention_heads=40, num_key_value_heads=10,
            head_dim=128, intermediate_size=17920,
            vocab_size=32064, max_position_embeddings=131072,
        ))
    
    def get_optimized_block_size(self) -> int:
        """Larger blocks for smaller head dimensions."""
        if self.config.head_dim <= 80:
            return 32
        return 16


class QwenOptimizer(ModelOptimizer):
    """Optimizations for Qwen models."""

    @classmethod
    def for_qwen2_0_5b(cls) -> 'QwenOptimizer':
        """Create optimizer for Qwen2 0.5B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.QWEN2,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=24, hidden_size=896,
            num_attention_heads=14, num_key_value_heads=2,
            head_dim=64, intermediate_size=4864,
            vocab_size=151936, max_position_embeddings=131072,
        ))

    @classmethod
    def for_qwen2_1_5b(cls) -> 'QwenOptimizer':
        """Create optimizer for Qwen2 1.5B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.QWEN2,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=28, hidden_size=1536,
            num_attention_heads=12, num_key_value_heads=2,
            head_dim=128, intermediate_size=8960,
            vocab_size=151936, max_position_embeddings=131072,
        ))

    @classmethod
    def for_qwen2_7b(cls) -> 'QwenOptimizer':
        """Create optimizer for Qwen2 7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.QWEN2,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=28, hidden_size=3584,
            num_attention_heads=28, num_key_value_heads=4,
            head_dim=128, intermediate_size=18944,
            vocab_size=152064, max_position_embeddings=32768,
        ))

    @classmethod
    def for_qwen2_72b(cls) -> 'QwenOptimizer':
        """Create optimizer for Qwen2 72B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.QWEN2,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8,
            head_dim=128, intermediate_size=29568,
            vocab_size=152064, max_position_embeddings=32768,
        ))


class GemmaOptimizer(ModelOptimizer):
    """Optimizations for Gemma models."""

    @classmethod
    def for_gemma_2b(cls) -> 'GemmaOptimizer':
        """Create optimizer for Gemma 2B (MQA)."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.GEMMA,
            attention_type=AttentionType.MULTI_QUERY,
            num_layers=18, hidden_size=2048,
            num_attention_heads=8, num_key_value_heads=1,
            head_dim=256, intermediate_size=16384,
            vocab_size=256000, max_position_embeddings=8192,
        ))

    @classmethod
    def for_gemma_7b(cls) -> 'GemmaOptimizer':
        """Create optimizer for Gemma 7B."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.GEMMA,
            num_layers=28, hidden_size=3072,
            num_attention_heads=16, num_key_value_heads=16,
            head_dim=256, intermediate_size=24576,
            vocab_size=256000, max_position_embeddings=8192,
        ))

    def get_optimized_block_size(self) -> int:
        """Gemma uses large head_dim (256); smaller blocks."""
        return 8


class FalconOptimizer(ModelOptimizer):
    """Optimizations for Falcon models."""

    @classmethod
    def for_falcon_7b(cls) -> 'FalconOptimizer':
        """Create optimizer for Falcon 7B (MHA)."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.FALCON,
            num_layers=32, hidden_size=4544,
            num_attention_heads=71, num_key_value_heads=71,
            head_dim=64, intermediate_size=18176,
            vocab_size=65024, max_position_embeddings=2048,
        ))

    @classmethod
    def for_falcon_40b(cls) -> 'FalconOptimizer':
        """Create optimizer for Falcon 40B (GQA)."""
        return cls(ModelConfig(
            architecture=ModelArchitecture.FALCON,
            attention_type=AttentionType.GROUPED_QUERY,
            num_layers=60, hidden_size=8192,
            num_attention_heads=128, num_key_value_heads=8,
            head_dim=64, intermediate_size=32768,
            vocab_size=65024, max_position_embeddings=2048,
        ))

    def get_optimized_block_size(self) -> int:
        """Falcon head_dim=64; use larger blocks."""
        return 32


class ModelRegistry:
    """Registry for model configurations and optimizers."""
    
    _instance: Optional['ModelRegistry'] = None
    _configs: Dict[str, ModelConfig] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_builtin_models()
        return cls._instance
    
    def _register_builtin_models(self):
        """Register built-in model configurations."""
        self._configs = {
            # Llama
            'llama-7b': LlamaOptimizer.for_llama_7b().config,
            'llama-13b': LlamaOptimizer.for_llama_13b().config,
            'llama2-7b': LlamaOptimizer.for_llama2_7b().config,
            'llama2-70b': LlamaOptimizer.for_llama2_70b().config,
            'llama3-8b': LlamaOptimizer.for_llama3_8b().config,
            'llama3-70b': LlamaOptimizer.for_llama3_70b().config,
            'llama3.1-8b': LlamaOptimizer.for_llama31_8b().config,
            'llama3.1-70b': LlamaOptimizer.for_llama31_70b().config,
            'llama3.1-405b': LlamaOptimizer.for_llama31_405b().config,
            # Mistral / Mixtral
            'mistral-7b': MistralOptimizer.for_mistral_7b().config,
            'mixtral-8x7b': MistralOptimizer.for_mixtral_8x7b().config,
            'mixtral-8x22b': MistralOptimizer.for_mixtral_8x22b().config,
            # Phi
            'phi-2': PhiOptimizer.for_phi2().config,
            'phi-3-mini': PhiOptimizer.for_phi3_mini().config,
            'phi-3-medium': PhiOptimizer.for_phi3_medium().config,
            # Qwen
            'qwen2-0.5b': QwenOptimizer.for_qwen2_0_5b().config,
            'qwen2-1.5b': QwenOptimizer.for_qwen2_1_5b().config,
            'qwen2-7b': QwenOptimizer.for_qwen2_7b().config,
            'qwen2-72b': QwenOptimizer.for_qwen2_72b().config,
            # Gemma
            'gemma-2b': GemmaOptimizer.for_gemma_2b().config,
            'gemma-7b': GemmaOptimizer.for_gemma_7b().config,
            # Falcon
            'falcon-7b': FalconOptimizer.for_falcon_7b().config,
            'falcon-40b': FalconOptimizer.for_falcon_40b().config,
        }
    
    def register(self, name: str, config: ModelConfig):
        """Register a model configuration."""
        self._configs[name] = config
    
    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._configs.get(name)
    
    def get_optimizer(self, name: str) -> Optional[ModelOptimizer]:
        """Create optimizer for a model."""
        config = self.get(name)
        if not config:
            return None
        
        if config.architecture in (ModelArchitecture.LLAMA,
                                   ModelArchitecture.LLAMA2,
                                   ModelArchitecture.LLAMA3):
            return LlamaOptimizer(config)
        elif config.architecture in (ModelArchitecture.MISTRAL,
                                     ModelArchitecture.MIXTRAL):
            return MistralOptimizer(config)
        elif config.architecture in (ModelArchitecture.PHI,
                                     ModelArchitecture.PHI3):
            return PhiOptimizer(config)
        elif config.architecture == ModelArchitecture.QWEN2:
            return QwenOptimizer(config)
        elif config.architecture == ModelArchitecture.GEMMA:
            return GemmaOptimizer(config)
        elif config.architecture == ModelArchitecture.FALCON:
            return FalconOptimizer(config)
        return ModelOptimizer(config)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._configs.keys())
    
    def has_model(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._configs


class ModelMemoryEstimator:
    """Memory usage estimator for models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def _get_dtype_size(self, dtype: str = "float16") -> int:
        """Get size of data type in bytes."""
        if dtype in ("float32", "fp32"):
            return 4
        elif dtype in ("float16", "fp16", "bfloat16", "bf16"):
            return 2
        elif dtype == "int8":
            return 1
        return 2
    
    def estimate_weight_memory(self, dtype: str = "float16") -> int:
        """Estimate model weight memory in bytes."""
        dtype_size = self._get_dtype_size(dtype)
        
        # Embedding
        params = self.config.vocab_size * self.config.hidden_size
        
        # Per-layer parameters
        per_layer = 0
        # Attention: Q, K, V, O projections
        per_layer += 4 * self.config.hidden_size * self.config.hidden_size
        # MLP: gate, up, down
        per_layer += 3 * self.config.hidden_size * self.config.intermediate_size
        # Layer norms
        per_layer += 2 * self.config.hidden_size
        
        params += per_layer * self.config.num_layers
        params += self.config.hidden_size  # Final norm
        
        return params * dtype_size
    
    def estimate_kv_cache_memory(self, batch_size: int, seq_len: int,
                                  dtype: str = "float16") -> int:
        """Estimate KV cache memory in bytes."""
        dtype_size = self._get_dtype_size(dtype)
        kv_heads = self.config.num_key_value_heads or self.config.num_attention_heads
        
        # 2 for K and V
        per_layer = 2 * batch_size * seq_len * kv_heads * self.config.get_head_dim()
        return per_layer * self.config.num_layers * dtype_size
    
    def estimate_activation_memory(self, batch_size: int, seq_len: int,
                                    dtype: str = "float16") -> int:
        """Estimate activation memory in bytes."""
        dtype_size = self._get_dtype_size(dtype)
        activation = batch_size * seq_len * self.config.hidden_size * dtype_size
        return activation * 3  # Conservative estimate
    
    def estimate_total_memory(self, batch_size: int, seq_len: int,
                               dtype: str = "float16") -> int:
        """Estimate total memory in bytes."""
        return (self.estimate_weight_memory(dtype) +
                self.estimate_kv_cache_memory(batch_size, seq_len, dtype) +
                self.estimate_activation_memory(batch_size, seq_len, dtype))
    
    def find_max_batch_size(self, memory_budget: int, seq_len: int,
                             dtype: str = "float16") -> int:
        """Find maximum batch size for given memory budget."""
        weight_mem = self.estimate_weight_memory(dtype)
        if weight_mem >= memory_budget:
            return 0
        
        available = memory_budget - weight_mem
        
        # Binary search
        low, high = 1, 1024
        result = 0
        
        while low <= high:
            mid = (low + high) // 2
            kv_mem = self.estimate_kv_cache_memory(mid, seq_len, dtype)
            act_mem = self.estimate_activation_memory(mid, seq_len, dtype)
            
            if kv_mem + act_mem <= available:
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return result
    
    def print_breakdown(self, batch_size: int, seq_len: int,
                        dtype: str = "float16"):
        """Print memory breakdown."""
        weight = self.estimate_weight_memory(dtype)
        kv = self.estimate_kv_cache_memory(batch_size, seq_len, dtype)
        act = self.estimate_activation_memory(batch_size, seq_len, dtype)
        total = weight + kv + act
        
        print(f"\n{'='*40}")
        print(f"Memory Breakdown (batch={batch_size}, seq={seq_len})")
        print(f"{'='*40}")
        print(f"Weights:     {weight/1e9:>6.2f} GB ({100*weight/total:>5.1f}%)")
        print(f"KV Cache:    {kv/1e9:>6.2f} GB ({100*kv/total:>5.1f}%)")
        print(f"Activations: {act/1e9:>6.2f} GB ({100*act/total:>5.1f}%)")
        print(f"{'='*40}")
        print(f"Total:       {total/1e9:>6.2f} GB")
