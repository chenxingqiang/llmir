"""
LLMIR - Large Language Model Intermediate Representation

A compiler infrastructure for optimizing LLM inference based on MLIR.

Features:
- PagedKVCache: Efficient key-value cache with block-based memory management
- Quantization: INT8/INT4 support for 4-8x memory reduction
- Speculative Decoding: KV cache branching for 2-3x faster generation
- Prefix Caching: Radix tree-based prefix matching for prompt reuse
- Continuous Batching: vLLM-style dynamic batch management
- Model Optimizations: Pre-configured settings for Llama, Mistral, Phi

Example:
    >>> from llmir import PagedKVCache, KVCacheConfig
    >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
    >>> cache = PagedKVCache(config)
    >>>
    >>> # Append KV pairs
    >>> block_indices = cache.append(keys, values, seq_ids)
    >>>
    >>> # Lookup KV pairs
    >>> k, v = cache.lookup(block_indices, seq_lens)

For serving:
    >>> from llmir import LLMEngine, SamplingParams
    >>> engine = LLMEngine.from_pretrained("meta-llama/Llama-3.1-8B")
    >>> outputs = engine.generate(["Hello!"], SamplingParams(max_tokens=100))
"""

# Disable TensorFlow in transformers to avoid protobuf/tensorflow dependency conflict.
# Must be set before any transformers import. See: transformers.utils.import_utils
import os

os.environ.setdefault("USE_TORCH", "1")

__version__ = "0.1.0"
__author__ = "Xingqiang Chen"
__email__ = "chenxingqiang@turingai.cc"

# Core KV Cache classes
from llmir.runtime.kv_cache import (
    PagedKVCache,
    QuantizedKVCache,
    DistributedKVCache,
    SpeculativeKVCache,
    PrefixCache,
)

# Configuration classes
from llmir.runtime.config import (
    KVCacheConfig,
    QuantizationConfig,
    QuantizationType,
    ShardingStrategy,
    SpeculativeConfig,
)

# Serving components
from llmir.serving.engine import (
    LLMEngine,
    ContinuousBatchingEngine,
)

from llmir.serving.config import (
    SamplingParams,
    SchedulerConfig,
    SchedulingPolicy,
    RequestPriority,
)

# Profiling tools
from llmir.profiling import (
    Profiler,
    MemoryProfiler,
    LatencyProfiler,
    ThroughputMonitor,
)

# Model optimizations
from llmir.models import (
    ModelOptimizer,
    LlamaOptimizer,
    MistralOptimizer,
    PhiOptimizer,
    ModelRegistry,
    ModelMemoryEstimator,
)

# HuggingFace integration (optional; requires transformers)
try:
    from llmir.integration.huggingface import from_pretrained
except ImportError:
    from_pretrained = None  # type: ignore[misc, assignment]

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # KV Cache
    "PagedKVCache",
    "QuantizedKVCache",
    "DistributedKVCache",
    "SpeculativeKVCache",
    "PrefixCache",

    # Configuration
    "KVCacheConfig",
    "QuantizationConfig",
    "QuantizationType",
    "ShardingStrategy",
    "SpeculativeConfig",

    # Serving
    "LLMEngine",
    "ContinuousBatchingEngine",
    "SamplingParams",
    "SchedulerConfig",
    "SchedulingPolicy",
    "RequestPriority",

    # Profiling
    "Profiler",
    "MemoryProfiler",
    "LatencyProfiler",
    "ThroughputMonitor",

    # Models
    "ModelOptimizer",
    "LlamaOptimizer",
    "MistralOptimizer",
    "PhiOptimizer",
    "ModelRegistry",
    "ModelMemoryEstimator",
    "from_pretrained",
]
