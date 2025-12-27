# LLMIR Design Overview

## Introduction

LLMIR (Large Language Model Intermediate Representation) is a compiler infrastructure based on MLIR designed to optimize LLM inference workloads. This document provides an overview of the system design and architecture.

## Core Design Principles

### 1. Multi-Level Abstraction
LLMIR uses MLIR's multi-level IR capabilities to represent LLM operations at different abstraction levels:
- **High-level**: LLM-specific operations (attention, KV cache)
- **Mid-level**: Tensor operations and memory management
- **Low-level**: Hardware-specific code generation

### 2. Production-Ready Serving
The design prioritizes production deployment with:
- Continuous batching for high throughput
- Memory pressure monitoring and preemption
- vLLM-compatible APIs for easy integration

### 3. Model-Aware Optimization
Optimizations are tailored for specific model architectures:
- Llama family (7B to 405B)
- Mistral/Mixtral with sliding window attention
- Phi models with efficient block sizes

## Architecture Components

### LLM Dialect
Custom MLIR dialect with specialized types and operations:

```
Types:
- !llm.paged_kv_cache<element, layers, heads, dim, block, maxlen>
- !llm.sharded_tensor<type, dim, shards, index>
- !llm.quantized_tensor<type, shape, symmetric, perchannel, axis, group, bits>

Operations:
- llm.attention, llm.paged_attention
- llm.append_kv, llm.lookup_kv
- llm.quantize, llm.dequantize
- llm.branch_kv, llm.commit_kv (speculative decoding)
- llm.prefix_lookup (prefix caching)
```

### Runtime Library
C++ implementation of core components:

| Component | Description |
|-----------|-------------|
| PagedKVCache | Block-based KV cache with dynamic allocation |
| QuantizedKVCache | INT8/INT4 quantization support |
| DistributedKVCache | Multi-GPU sharding with NCCL |
| SpeculativeKVCache | Branching/rollback for draft tokens |
| PrefixCache | Radix tree-based prefix matching |
| ContinuousBatching | Dynamic batch management |
| VLLMIntegration | vLLM API compatibility layer |

### Optimization Passes

1. **KV Cache Optimization**
   - Block size optimization
   - Operation fusion
   - Cache sharing analysis

2. **Multi-Precision Computation**
   - Selective quantization
   - Mixed precision optimization
   - Quantization-aware fusion

3. **Parallelization**
   - Tensor parallelism
   - Pipeline parallelism
   - Communication optimization

## Key Innovations

### 1. IR-Level PagedAttention
First compiler-level representation of PagedAttention, enabling:
- Static analysis of memory access patterns
- Compiler-driven block allocation optimization
- Cross-sequence cache sharing

### 2. Speculative Decoding Support
KV cache branching mechanism for draft token verification:
- Create branches for speculation
- Verify against target model
- Commit or rollback efficiently

### 3. Prefix Caching
Efficient reuse of common prompt prefixes:
- Radix tree for O(log n) prefix matching
- LRU eviction with pinning support
- System prompt caching

### 4. Adaptive Block Management
Dynamic block size adjustment:
- Workload analysis and statistics
- Multiple block size pools
- Predictive allocation policies

## Integration with Existing Frameworks

### vLLM Integration
- BlockSpaceManagerAdapter
- SchedulerAdapter
- PagedAttentionWrapper
- LLMEngineAdapter
- C API for Python bindings

### Framework Compatibility
- Torch-MLIR for PyTorch model import
- IREE for hardware abstraction
- XLA/StableHLO for optimization reuse

## Performance Characteristics

| Feature | Improvement |
|---------|-------------|
| PagedAttention | 1.5× throughput |
| INT8 Quantization | 4× memory reduction |
| INT4 Quantization | 8× memory reduction |
| Speculative Decoding | 2-3× faster generation |
| Prefix Caching | Up to 80% compute savings |
| Continuous Batching | 2× throughput vs static |
| 8-GPU Scaling | 94.5% efficiency |

## Future Directions

1. **HuggingFace Integration**: Direct model loading and optimization
2. **Distributed Training**: Extend compilation support for training
3. **Auto-Scaling**: Kubernetes-native deployment support
4. **New Architectures**: Mamba, RWKV, RetNet support

## References

- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
- [vLLM: PagedAttention](https://github.com/vllm-project/vllm)
- [SGLang: Structured Generation](https://github.com/sgl-project/sglang)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
