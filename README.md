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
- **Quantized KV Cache**: INT8/INT4 quantization support for 4-8x memory reduction
- **Multi-GPU Sharding**: Distributed KV cache with layer-wise, head-wise, and sequence-wise sharding
- **Checkpoint Support**: Serialization/deserialization for long-running sessions
- **Speculative Decoding**: KV cache branching and rollback for draft token verification
- **Prefix Caching**: Efficient reuse of common prompt prefixes across sequences
- **Adaptive Block Management**: Dynamic block size adjustment based on workload patterns

## Architecture

LLMIR follows a layered architecture:

```
                       ┌─────────────────┐
                       │   Application   │
                       │  vLLM / SGLang  │
                       └────────┬────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────┐
│                    LLMIR Compiler                │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ Front-end    │ →  │  MLIR Optimization    │   │
│  │ Converters   │    │  Pipeline             │   │
│  └──────────────┘    └───────────┬───────────┘   │
│                                  │               │
│                      ┌───────────▼───────────┐   │
│                      │    Backend Generators │   │
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

### Quantized KV Cache

```cpp
// Create a quantized KV cache with INT8 quantization
QuantizationConfig config(QuantizationType::INT8, 
                          QuantizationStrategy::PER_TENSOR);
QuantizedPagedKVCache qCache(numLayers, numHeads, headDim, blockSize, 
                             maxSeqLen, config, elementType);

// Use the quantized cache (automatic quantization/dequantization)
qCache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);

// Get compression ratio
float ratio = qCache.getCompressionRatio();  // ~4x for INT8, ~8x for INT4
```

### Multi-GPU Distributed KV Cache

```cpp
// Configure sharding across 4 GPUs
ShardingConfig config;
config.strategy = ShardingStrategy::LAYER_WISE;
config.numDevices = 4;
config.deviceIds = {0, 1, 2, 3};

// Create distributed cache
DistributedPagedKVCache distCache(numLayers, numHeads, headDim, blockSize,
                                   maxSeqLen, elementType, config);

// Operations are automatically distributed
distCache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);
```

### Checkpointing

```cpp
// Create checkpoint manager
CheckpointManager manager("/path/to/checkpoints");

// Save checkpoint
manager.createCheckpoint(cache, "checkpoint_001");

// Load checkpoint
manager.loadCheckpoint(cache, "checkpoint_001");

// Auto-cleanup old checkpoints (keep 5 most recent)
manager.cleanupCheckpoints(5);
```

### Speculative Decoding

```cpp
// Create speculative KV cache
SpeculativeConfig specConfig;
specConfig.maxDraftTokens = 8;
specConfig.enableTreeAttention = true;

SpeculativeKVCache specCache(numLayers, numHeads, headDim, blockSize,
                              maxSeqLen, elementType, specConfig);

// Create a branch for speculation
int32_t branchId;
specCache.createBranch(sequenceId, branchId);

// Append speculative (draft) KV
specCache.appendSpeculativeKV(keyData, valueData, sequenceId, numDraftTokens, branchId);

// Verify and commit accepted tokens
VerificationResult result;
specCache.verifySpeculation(sequenceId, branchId, targetLogProbs, numTokens, result);
specCache.commitSpeculation(sequenceId, branchId, result.acceptedCount);
```

### Prefix Caching

```cpp
// Create prefix-aware KV cache
PrefixCacheConfig prefixConfig;
prefixConfig.maxCachedPrefixes = 1000;
prefixConfig.enableRadixTree = true;

PrefixAwareKVCache prefixCache(numLayers, numHeads, headDim, blockSize,
                                maxSeqLen, elementType, prefixConfig);

// Initialize sequence with automatic prefix reuse
int64_t cachedLength = prefixCache.initializeSequence(sequenceId, 
                                                       promptTokens, promptLength);
// cachedLength tokens loaded from cache, only need to compute remaining

// Cache system prompts
SystemPromptCache systemCache(prefixCache.getPrefixCache());
systemCache.registerSystemPrompt("assistant", systemTokens, systemLength);
```

### Adaptive Block Management

```cpp
// Create adaptive block manager
BlockSizeConfig blockConfig;
blockConfig.primaryBlockSize = 16;
blockConfig.smallBlockSize = 4;
blockConfig.largeBlockSize = 64;

AdaptationConfig adaptConfig;
adaptConfig.policy = AdaptationPolicy::PREDICTIVE;
adaptConfig.enableAutoTuning = true;

AdaptiveBlockManager manager(numLayers, numHeads, headDim,
                              blockConfig, adaptConfig);

// Allocate blocks with automatic size selection
std::vector<KVBlock*> blocks;
manager.allocateBlocksForSequence(sequenceId, expectedLength, blocks);

// Record workload for adaptation
manager.recordSequenceComplete(sequenceId, finalLength);

// Get recommended configuration based on workload
BlockSizeConfig recommended = manager.getRecommendedConfig();
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

1. **Phase 1**: Basic infrastructure ✅
   - LLM dialect design and implementation
   - Basic type system
   - Core operations

2. **Phase 2**: Core optimizations ✅
   - KV cache management
   - Attention computation optimizations
   - Memory management

3. **Phase 3**: Advanced features ✅
   - Quantization support (INT8/INT4)
   - Tensor parallelism
   - Pipeline parallelism
   - Multi-GPU sharding
   - Checkpoint/serialization support

4. **Phase 4**: Production & Integration (Current)
   - Framework integration (HuggingFace, vLLM)
   - Speculative decoding support
   - Prefix caching optimization
   - Performance tuning and benchmarks

## Attention Optimization Benchmarks

The LLMIR project includes comprehensive benchmarks for various attention optimization techniques, which are critical for LLM inference performance. These benchmarks evaluate different approaches to optimizing the attention mechanism, a crucial component in transformer models.

### Optimization Techniques

Four key attention optimization techniques have been implemented and evaluated:

1. **Flash Attention**: A block-based approach that improves memory access patterns and reduces memory bandwidth requirements. The implementation uses tiled matrix multiplications and on-chip memory to minimize HBM accesses.

2. **Fused Softmax Attention**: Combines softmax normalization with attention matrix multiplication in a single pass, eliminating the need to materialize the full attention matrix in memory.

3. **Optimized Masked Attention**: Provides specialized implementations for common attention mask patterns (causal masking, sliding window) that avoid unnecessary computation for masked-out tokens.

4. **Sliding Window Attention**: Optimizes attention for long sequences by limiting the context window, dramatically reducing computational complexity from O(n²) to O(n×w) where n is sequence length and w is window size.

#### Implementation Details

All optimization techniques are integrated with PagedKVCache to provide seamless operation with the KV cache architecture:

- **FlashAttentionImpl**: Implements the algorithm from ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), using block-based processing for better memory locality.

- **FusedSoftmaxAttentionImpl**: Uses a three-pass algorithm (max computation, exp normalization, weighted sum) that eliminates the need to store the full attention matrix.

- **OptimizedMaskedAttentionImpl**: Dynamically selects specialized implementations based on mask pattern detection, skipping computation for masked-out tokens entirely.

- **SlidingWindowAttentionImpl**: Optimizes memory access by only loading keys and values within the sliding window, using efficient memory gathering from the paged KV cache.

### Performance Results

Benchmark results on Mac M3 ARM processor show significant performance improvements:

| Technique | Speedup Range | Memory Reduction | Accuracy Impact |
|-----------|---------------|------------------|-----------------|
| Flash Attention | 1.28-1.69x | Minimal | Very Low |
| Fused Softmax | 1.36-1.48x | 30-40% | None |
| Optimized Masked | 1.42-1.92x | Varies by mask | None |
| Sliding Window | 1.52-2.15x | 40-70% | Controlled by window size |

All optimization techniques show better performance as sequence length increases, making them particularly valuable for LLM inference with long context windows.

### Key Findings

- **For maximum speed**: Threshold-based pruning with sliding window attention provides the highest speedup (up to 2.15x)
- **For memory efficiency**: Multi-Query Attention with fused softmax offers significant memory savings (60-70%)
- **For accuracy-speed balance**: Flash Attention provides good speedup with minimal accuracy impact
- **For scalability**: Benefits of all optimization techniques increase with sequence length

For detailed information, see the [attention optimization test procedure](docs/attention_optimization_test_procedure.md), [comprehensive benchmark report](docs/attention_optimization_test_report.md), and [benchmark source code](benchmark/attention/).

## Benchmarks

### KVCache Performance Benchmark

The repository includes a benchmark for measuring the performance of the PagedKVCache implementation on different hardware backends:

```bash
cd benchmark/LLM
./run_kvcache_benchmark.sh
```

This will run the benchmark with various configurations and generate performance reports.

### Llama-3.1 Model Benchmark with LLMIR

Benchmark the meta-llama/Llama-3.1-8B-Instruct model with LLMIR optimizations to measure performance improvements:

```bash
# Setup the environment
cd benchmark/LLM
./setup_llama31_benchmark.sh
source venv/bin/activate_llmir

# Run the benchmark
./run_llama31_benchmark.sh
```

This compares the baseline vLLM performance with LLMIR-optimized performance across different batch sizes and sequence lengths. For more details, see [benchmark/LLM/README_LLAMA31.md](benchmark/LLM/README_LLAMA31.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 with LLVM Exceptions - see the LICENSE.TXT file for details.

