# LLMIR (Large Language Model Intermediate Representation)

LLMIR is a compiler infrastructure for large language models based on MLIR (Multi-Level Intermediate Representation), designed to optimize and accelerate LLM inference through compilation techniques.

**Project Website**: [https://chenxingqiang.github.io/llmir-www/](https://chenxingqiang.github.io/llmir-www/)

## Overview

LLMIR provides a unified intermediate representation layer for large language models, enhancing inference performance through specialized optimizations. It integrates capabilities from high-performance inference frameworks like vLLM and SGLang with MLIR's compilation infrastructure.

Key objectives:
- Build a unified intermediate representation for LLM inference
- Provide cross-framework compilation and optimization
- Support critical optimizations like KV cache management, attention fusion, and quantization
- Enable efficient deployment across diverse hardware backends (GPU, CPU, accelerators)

## Key Features

- **PagedKVCache**: Efficient key-value cache implementation for optimized attention computation
- **MLIR Dialect for LLMs**: Custom operations and types for language model inference
- **Memory Optimizations**: Block-based memory management for efficient, low-fragmentation memory usage
- **Multi-sequence Support**: Handle multiple concurrent sequences with varying lengths
- **Multi-layer Management**: Efficiently manage KV cache for all layers in transformer models

## Architecture

LLMIR follows a layered architecture:

```
                       ┌─────────────────┐
                       │   Application   │
                       │ vLLM / SGLang   │
                       └────────┬────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────┐
│                    LLMIR Compiler                │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ Front-end    │ → │  MLIR Optimization     │   │
│  │ Converters   │    │  Pipeline             │   │
│  └──────────────┘    └───────────┬───────────┘   │
│                                  │               │
│                      ┌───────────▼───────────┐   │
│                      │    Backend Generators  │   │
│                      └───────────────────────┘   │
└──────────────────────────┬───────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │       Execution Layer       │
            │ CUDA / ROCm / LLVM / Accel  │
            └─────────────────────────────┘
```

### Core Components

1. **LLM Dialect**: MLIR dialect with specialized operations for language models
   - Custom types (PagedKVCache, ShardedTensor, QuantizedTensor)
   - Operations for attention, KV cache management, etc.

2. **Runtime Library**: Support libraries for efficient runtime execution
   - PagedKVCache runtime implementation
   - Block-based memory management
   - Optimized attention computation

3. **Optimization Passes**: MLIR passes for LLM-specific optimizations
   - KV cache optimization
   - Attention computation fusion
   - Quantization support

## Resources

- **Project Website**: [https://chenxingqiang.github.io/llmir-www/](https://chenxingqiang.github.io/llmir-www/)
- **GitHub Repository**: [https://github.com/chenxingqiang/llmir](https://github.com/chenxingqiang/llmir)
- **Documentation**: Available on the [project website](https://chenxingqiang.github.io/llmir-www/)
- **Community**: 
  - Forums: LLVM forums LLMIR section
  - Chat: LLVM Discord server (LLMIR channel)

## Getting Started

### Prerequisites

- LLVM/MLIR development environment
- CMake 3.13.4 or higher
- C++17 compatible compiler
- Python 3.7 or higher (for Python bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/llmir.git
cd llmir

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -G Ninja ..

# Build
ninja

# Run tests
ninja check-llmir
```

## Usage Examples

### KV Cache in MLIR

```mlir
// Create a paged KV cache type
!kv_cache_t = !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>

// Append key-value pairs to the cache
%new_kv, %block_indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!kv_cache_t, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
    -> (!kv_cache_t, tensor<2x1xi32>)

// Perform paged attention with the KV cache
%output = llm.paged_attention %query, %new_kv, %block_indices, %seq_lens {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<2x1x16x64xf16>, !kv_cache_t, tensor<2x128xi32>, tensor<2xi32>) 
    -> tensor<2x1x16x64xf16>
```

### C++ Runtime API

```cpp
// Create a KV cache for a model
PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType);

// Append new KV pairs to the cache
cache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);

// Lookup existing KV pairs
cache.lookupKV(blockIndices, seqLens, batchSize, outputKeys, outputValues);
```

## Project Structure

```
include/mlir/Dialect/LLM/       # MLIR dialect definitions
  ├── IR/                       # MLIR operations and types
  └── Runtime/                  # Runtime support headers

lib/Dialect/LLM/                # Implementation
  ├── IR/                       # MLIR operation implementations
  └── Runtime/                  # Runtime library implementations

test/Dialect/LLM/               # Tests
  ├── IR/                       # MLIR operation tests
  └── Runtime/                  # Runtime tests

examples/                       # Example applications
  └── kv_cache_example.cpp      # KV cache example
```

## Development Roadmap

LLMIR follows a phased development approach:

1. **Phase 1**: Basic infrastructure
   - LLM dialect design and implementation
   - Basic type system
   - Core operations

2. **Phase 2**: Core optimizations
   - KV cache management (current focus)
   - Attention computation optimizations
   - Memory management

3. **Phase 3**: Advanced features
   - Quantization support
   - Tensor parallelism
   - Pipeline parallelism
   - Backend code generation

## Attention Optimization Benchmarks

The LLMIR project includes comprehensive benchmarks for various attention optimization techniques, which are critical for LLM inference performance. These benchmarks evaluate different approaches to optimizing the attention mechanism, a crucial component in transformer models.

### Optimization Techniques

Three key attention optimization techniques have been implemented and evaluated:

1. **Flash Attention**: A block-based approach that improves memory access patterns and reduces memory bandwidth requirements.
2. **Multi-Query Attention (MQA)**: Uses multiple query heads but shares a single key-value head across all queries, significantly reducing memory usage.
3. **Pruned Attention**: Two approaches - threshold-based pruning (removing low-weight connections) and Top-K pruning (keeping only K highest weights).

### Performance Results

Benchmark results on Mac M3 ARM processor show significant performance improvements:

| Technique | Speedup Range | Memory Reduction | Accuracy Impact |
|-----------|---------------|------------------|-----------------|
| Flash Attention | 1.28-1.69x | Minimal | Very Low |
| Multi-Query Attention | 1.12-1.38x | 60-70% | Moderate |
| Threshold Pruning | 1.96-2.09x | Moderate | Higher |
| Top-K Pruning | 1.52-1.73x | Moderate | Controllable |

All optimization techniques show better performance as sequence length increases, making them particularly valuable for LLM inference with long context windows.

### Key Findings

- **For maximum speed**: Threshold-based pruning provides the highest speedup (up to 2.09x)
- **For memory efficiency**: Multi-Query Attention offers significant memory savings (60-70%)
- **For accuracy-speed balance**: Flash Attention provides good speedup with minimal accuracy impact
- **For scalability**: Benefits of all optimization techniques increase with sequence length

For detailed information, see the [attention optimization test procedure](docs/attention_optimization_test_procedure.md), [comprehensive benchmark report](docs/attention_optimization_test_report.md), and [benchmark source code](benchmark/attention/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 with LLVM Exceptions - see the LICENSE.TXT file for details.

