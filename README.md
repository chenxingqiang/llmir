# LLMIR (Large Language Model Intermediate Representation)

LLMIR is a compiler infrastructure for large language models based on MLIR (Multi-Level Intermediate Representation), designed to optimize and accelerate LLM inference through compilation techniques.

**Project Website**: [https://chenxingqiang.github.io/llmir-www/](https://chenxingqiang.github.io/llmir-www/)

---

## Quick Start

```bash
# Install
pip install llmir
# or for HuggingFace support: pip install llmir[full]

# Try it
python -c "from llmir import PagedKVCache, KVCacheConfig; c = KVCacheConfig(num_layers=8, num_heads=8, head_dim=64); print(PagedKVCache(c))"

# Benchmark KV cache (registry or HuggingFace model ID)
llmir-benchmark --model llama3-8b --batch-sizes 1,4
llmir-benchmark --model Qwen/Qwen2-0.5B

# List supported models
llmir-list-models

# Run tests
pytest tests/ -v
```

| What | Where |
|------|-------|
| Python package | `src/llmir/` |
| MLIR dialect (C++) | `include/mlir/Dialect/LLM/`, `lib/Dialect/LLM/` |
| Benchmarks | `benchmark/`, `scripts/` |
| Docs | `docs/` |
| Tests | `tests/` (Python), `test/` (MLIR lit) |

---

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
- **Continuous Batching**: vLLM-style dynamic batch management for production serving
- **vLLM Integration**: Drop-in compatibility layer for vLLM-based applications
- **Python Bindings**: Full Python API for KV cache, profiling, and engine management
- **Model Optimizations**: Pre-configured optimizations for Llama, Mistral, Phi, Qwen, Gemma, Falcon; `from_pretrained()` supports all decoder-only architectures in HuggingFace Transformers
- **Performance Profiling**: Comprehensive profiling with latency, memory, and throughput tracking

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

## Installation

### PyPI Installation (Recommended for Python users)

Install the LLMIR Python package from PyPI:

```bash
pip install llmir
```

For development features:

```bash
pip install llmir[dev]    # Development tools (pytest, black, mypy)
pip install llmir[full]   # Full stack with torch and transformers
```

### Quick Start with Python

```python
import llmir

# Create a PagedKVCache
config = llmir.KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
cache = llmir.PagedKVCache(config)

# Use model-specific optimizations
optimizer = llmir.LlamaOptimizer.for_llama3_8b()
kv_config = optimizer.get_optimized_kv_cache_config()

# Or load from HuggingFace (requires: pip install llmir[full])
if llmir.from_pretrained:
    optimizer = llmir.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    kv_config = optimizer.get_optimized_kv_cache_config()

# Profile performance
profiler = llmir.Profiler()
profiler.start()
with profiler.trace("attention"):
    # Your code here
    pass
profiler.stop()
report = profiler.get_report()
report.print_summary()
```

### Building from Source

#### Prerequisites

- LLVM/MLIR development environment
- CMake 3.13.4 or higher
- C++17 compatible compiler
- Python 3.8 or higher (for Python bindings)

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

### Python Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all Python tests (no network)
pytest tests/ -v

# Run HuggingFace integration tests (requires network)
pytest tests/test_integration_hf.py -v -m network

# Run KV cache benchmark (Python)
llmir-benchmark --model llama3-8b --batch-sizes 1,4,8 --output results.json
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

### vLLM Integration

```cpp
// Create vLLM-compatible engine
vllm::LLMEngineAdapter::EngineConfig config;
config.maxNumSeqs = 256;
config.maxNumBatchedTokens = 8192;
config.blockSize = 16;
config.gpuMemoryUtilization = 0.9f;

vllm::LLMEngineAdapter engine(config);
engine.initialize();

// Submit request with vLLM-style parameters
vllm::SamplingParams params;
params.maxTokens = 256;
params.temperature = 0.7f;
params.topP = 0.9f;

std::string requestId = engine.addRequest("What is LLMIR?", params);

// Process requests
while (engine.hasPendingRequests()) {
    auto outputs = engine.step();
    for (const auto& output : outputs) {
        if (output.finished) {
            std::cout << output.outputs[0].text << std::endl;
        }
    }
}
```

### Continuous Batching

```cpp
// Create continuous batching engine
SchedulerConfig schedConfig;
schedConfig.maxBatchSize = 256;
schedConfig.maxBatchTokens = 8192;
schedConfig.enablePreemption = true;
schedConfig.policy = SchedulingPolicy::ADAPTIVE;

ContinuousBatchingEngine engine(cache, schedConfig);
engine.start();

// Submit multiple requests
for (const auto& prompt : prompts) {
    engine.submitRequest(tokenize(prompt), genConfig, RequestPriority::NORMAL);
}

// Engine runs in background, outputs delivered via callback
engine.setOutputCallback([](int32_t groupId, const std::vector<int32_t>& tokens, 
                            bool isFinished) {
    std::cout << "Request " << groupId << ": " << tokens.size() << " tokens"
              << (isFinished ? " [DONE]" : "") << std::endl;
});
```

### Python Bindings

```python
from mlir.dialects.llm import (
    PagedKVCache, QuantizedKVCache, KVCacheConfig, 
    QuantizationConfig, QuantizationType,
    LLMEngine, SamplingParams, ContinuousBatchingEngine,
    Profiler, LatencyProfiler, ThroughputMonitor
)

# Create KV cache
config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
cache = PagedKVCache(config)

# Use quantized cache for memory efficiency
quant_config = QuantizationConfig(quant_type=QuantizationType.INT8)
quant_cache = QuantizedKVCache(config, quant_config)
print(f"Compression ratio: {quant_cache.get_compression_ratio()}x")

# High-level LLM engine with vLLM-compatible API
engine = LLMEngine.from_pretrained("meta-llama/Llama-3.1-8B")
outputs = engine.generate(
    ["Hello, world!", "What is LLMIR?"],
    SamplingParams(max_tokens=100, temperature=0.7)
)

# Performance profiling
profiler = Profiler()
profiler.start()

with profiler.trace("attention"):
    # Run attention computation
    pass

profiler.stop()
profiler.get_report().print_summary()
```

### Model-Specific Optimizations (C++)

```cpp
// Use model-specific optimizer for Llama 3.1 70B
auto optimizer = LlamaOptimizer::forLlama31_70B();

// Get optimized configuration
auto kvConfig = optimizer.getOptimizedKVCacheConfig();
auto quantConfig = optimizer.getRecommendedQuantConfig();
int64_t blockSize = optimizer.getOptimizedBlockSize();

// Create optimized cache
auto cache = optimizer.createOptimizedKVCache(/*enableGPU=*/true);

// Estimate memory usage
size_t memUsage = optimizer.estimateMemoryUsage(batchSize, seqLen);

// For Mistral with sliding window
auto mistralOptimizer = MistralOptimizer::forMistral7B();
auto windowCache = mistralOptimizer.createSlidingWindowCache(
    mistralOptimizer.getSlidingWindowSize());

// Use model registry for any model
auto& registry = ModelRegistry::getInstance();
auto modelOptimizer = registry.createOptimizer("llama3.1-8b");
auto config = registry.getConfig("mixtral-8x7b");
```

### Memory Estimation

```cpp
// Estimate memory requirements before deployment
ModelMemoryEstimator estimator(LlamaOptimizer::forLlama31_70B().config_);

// Memory breakdown
estimator.printMemoryBreakdown(batchSize, seqLen);

// Find optimal batch size for available memory
size_t gpuMemory = 80ULL * 1024 * 1024 * 1024;  // 80 GB
int64_t maxBatch = estimator.findMaxBatchSize(gpuMemory, maxSeqLen);

// Find maximum sequence length for given batch
int64_t maxSeq = estimator.findMaxSeqLen(gpuMemory, batchSize);
```

## Project Structure

```
├── src/llmir/                  # Python package (pip install -e .)
│   ├── runtime/                # KV cache, config
│   ├── models/                 # Model optimizers (Llama, Mistral, Phi)
│   ├── serving/                # LLMEngine, ContinuousBatching
│   ├── integration/            # HuggingFace from_pretrained
│   └── cli/                    # llmir-profile, llmir-benchmark
│
├── include/mlir/Dialect/LLM/    # MLIR dialect (C++)
│   ├── IR/                     # Operations and types
│   └── Runtime/                # PagedKVCache, QuantizedKVCache, etc.
│
├── lib/Dialect/LLM/             # C++ implementation
├── benchmark/                   # C++ and Python benchmarks
│   └── LLM/                    # Llama 3.1 benchmark
├── scripts/                     # Benchmark and utility scripts
├── tests/                       # Python tests (pytest)
├── test/                        # MLIR lit tests
├── docs/                        # Documentation
├── examples/                    # Example applications
├── pyproject.toml               # Python package config
├── docker-compose.yml           # Docker benchmark
└── CMakeLists.txt               # MLIR build
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

4. **Phase 4**: Advanced Optimizations ✅
   - Speculative decoding with KV cache branching
   - Prefix caching with radix tree
   - Dynamic block size adjustment
   - Adaptive block management

5. **Phase 5**: Production & Integration ✅
   - Comprehensive benchmark suite
   - Continuous batching for production serving
   - vLLM integration layer with full API compatibility
   - Memory pressure monitoring and preemption

6. **Phase 6**: Developer Tools & Model Support ✅
   - Python bindings for runtime
   - Performance profiling tools
   - Model-specific optimizations (Llama, Mistral, Phi)
   - Model registry with presets
   - Memory estimation utilities

7. **Phase 7**: Future Enhancements (Planned)
   - HuggingFace Transformers integration
   - Distributed training support
   - Auto-scaling and Kubernetes support

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

### Docker Benchmark

```bash
./scripts/docker_run_benchmark.sh   # Requires NVIDIA GPU, HUGGINGFACE_TOKEN
# or: docker-compose up llama31-benchmark
```

### Additional Scripts

See [scripts/README.md](scripts/README.md) for PyTorch/vLLM/SGLang comparison and other benchmarks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 with LLVM Exceptions - see the LICENSE.TXT file for details.

