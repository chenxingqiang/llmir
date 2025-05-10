//===- AttentionOptBenchmark.cpp - Benchmarks for attention optimizations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements benchmarks for attention optimizations in LLM inference.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "benchmark/benchmark.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

using namespace mlir;
using namespace mlir::llm::runtime;

namespace {

// A simple Type implementation for testing
struct TestType : public Type {
  int getIntOrFloatBitWidth() const { return 16; }  // f16
};

// Configuration for a single benchmark run
struct BenchmarkConfig {
  int64_t batchSize;
  int64_t seqLen;
  int64_t contextLen;
  int64_t numHeads;
  int64_t headDim;
  AttentionMaskType maskType;
  AttentionOptLevel optLevel;
  bool useFusedSoftmax;
  int64_t windowSize;  // For sliding window attention
  
  // Calculate total elements in a tensor
  int64_t totalElements(int64_t b, int64_t s, int64_t h, int64_t d) const {
    return b * s * h * d;
  }
  
  // Calculate query tensor size
  int64_t querySize() const {
    return totalElements(batchSize, seqLen, numHeads, headDim);
  }
  
  // Calculate key/value tensor size
  int64_t keyValueSize() const {
    return totalElements(batchSize, contextLen, numHeads, headDim);
  }
  
  // Calculate output tensor size
  int64_t outputSize() const {
    return totalElements(batchSize, seqLen, numHeads, headDim);
  }
};

// Base fixture for all attention benchmarks
class AttentionBenchmarkBase {
protected:
  void SetUp(const BenchmarkConfig& config) {
    type = TestType();
    this->config = config;
    
    // Create attention configuration
    attConfig.numHeads = config.numHeads;
    attConfig.headDim = config.headDim;
    attConfig.scale = 1.0f / std::sqrt(static_cast<float>(config.headDim));
    attConfig.maskType = config.maskType;
    attConfig.optLevel = config.optLevel;
    attConfig.fuseSoftmax = config.useFusedSoftmax;
    attConfig.windowSize = config.windowSize;
    
    // Allocate data
    queryData.resize(config.querySize(), 0.0f);
    keyData.resize(config.keyValueSize(), 0.0f);
    valueData.resize(config.keyValueSize(), 0.0f);
    outputData.resize(config.outputSize(), 0.0f);
    
    // Fill with random data
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : queryData) v = dist(rng);
    for (auto& v : keyData) v = dist(rng);
    for (auto& v : valueData) v = dist(rng);
  }
  
  void TearDown() {
    attImpl.reset();
    kvCache.reset();
  }

  TestType type;
  BenchmarkConfig config;
  AttentionConfig attConfig;
  std::unique_ptr<AttentionImpl> attImpl;
  std::unique_ptr<PagedKVCache> kvCache;
  
  std::vector<float> queryData;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData;
};

// Benchmark for regular attention (non-KV-cache based)
class RegularAttentionBenchmark : public AttentionBenchmarkBase {
public:
  void SetUp(const BenchmarkConfig& config) {
    AttentionBenchmarkBase::SetUp(config);
    
    // Create attention implementation
    attImpl = createAttentionImpl(attConfig, type);
  }
  
  void RunBenchmark() {
    // Call the attention implementation
    attImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        config.batchSize,
        config.seqLen,
        config.contextLen);
  }
};

// Benchmark for KV-cache based attention
class KVCacheAttentionBenchmark : public AttentionBenchmarkBase {
public:
  void SetUp(const BenchmarkConfig& config) {
    AttentionBenchmarkBase::SetUp(config);
    
    // Initialize the KV cache
    const int64_t numLayers = 1;  // Single layer for benchmark simplicity
    const int64_t blockSize = 16;
    const int64_t maxSeqLen = config.contextLen * 2;  // Ensure enough space
    
    kvCache = std::make_unique<PagedKVCache>(
        numLayers, config.numHeads, config.headDim, blockSize, maxSeqLen, type);
    
    // Configure the cache
    kvCache->configureBlockAllocatorsAdvanced(
        config.contextLen / 2,  // Avg sequence length
        config.batchSize * 2,   // Max concurrent sequences
        true, 1);               // Enable eviction with balanced strategy
    
    // Configure attention
    kvCache->configureAttentionOpt(attConfig);
    
    // Prepare block indices and sequence IDs
    seqIds.resize(config.batchSize);
    blockIndices.resize(config.batchSize * numLayers);
    seqLens.resize(config.batchSize, config.contextLen);
    
    for (int64_t i = 0; i < config.batchSize; i++) {
      seqIds[i] = 100 + i;
    }
    
    // Fill the KV cache
    LogicalResult result = kvCache->appendKV(
        keyData.data(), valueData.data(),
        config.batchSize, config.contextLen, 
        seqIds.data(), blockIndices.data());
    
    if (failed(result)) {
      std::cerr << "Failed to fill KV cache for benchmark" << std::endl;
    }
  }
  
  void RunBenchmark() {
    // Run attention computation using the KV cache
    kvCache->computeAttention(
        outputData.data(),
        queryData.data(),
        blockIndices.data(),
        seqLens.data(),
        config.batchSize,
        config.seqLen);
  }
  
private:
  std::vector<int32_t> seqIds;
  std::vector<int32_t> blockIndices;
  std::vector<int32_t> seqLens;
};

// Define benchmark function for regular attention
void BM_RegularAttention(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t contextLen = state.range(2);
  const int64_t numHeads = state.range(3);
  const int64_t headDim = state.range(4);
  const AttentionMaskType maskType = static_cast<AttentionMaskType>(state.range(5));
  const AttentionOptLevel optLevel = static_cast<AttentionOptLevel>(state.range(6));
  
  BenchmarkConfig config = {
    batchSize,
    seqLen,
    contextLen,
    numHeads,
    headDim,
    maskType,
    optLevel,
    true,  // Use fused softmax
    256    // Default window size
  };
  
  RegularAttentionBenchmark benchmark;
  benchmark.SetUp(config);
  
  // Track total FLOPs
  int64_t flops_per_iter = 0;
  
  // Attention FLOPs estimation:
  // 1. QK^T: 2 * batchSize * numHeads * seqLen * contextLen * headDim
  // 2. Softmax: 5 * batchSize * numHeads * seqLen * contextLen
  // 3. AV: 2 * batchSize * numHeads * seqLen * contextLen * headDim
  flops_per_iter += 2 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen * config.headDim;  // QK^T
  flops_per_iter += 5 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen;  // Softmax (approx)
  flops_per_iter += 2 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen * config.headDim;  // AV
                   
  // Run the benchmark
  for (auto _ : state) {
    benchmark.RunBenchmark();
  }
  
  // Report metrics
  state.counters["FLOPs"] = benchmark::Counter(
      static_cast<double>(flops_per_iter),
      benchmark::Counter::kIsIterationInvariant | 
      benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1K);
      
  benchmark.TearDown();
}

// Define benchmark function for KV-cache based attention
void BM_KVCacheAttention(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t contextLen = state.range(2);
  const int64_t numHeads = state.range(3);
  const int64_t headDim = state.range(4);
  const AttentionMaskType maskType = static_cast<AttentionMaskType>(state.range(5));
  const AttentionOptLevel optLevel = static_cast<AttentionOptLevel>(state.range(6));
  
  BenchmarkConfig config = {
    batchSize,
    seqLen,
    contextLen,
    numHeads,
    headDim,
    maskType,
    optLevel,
    true,  // Use fused softmax
    256    // Default window size
  };
  
  KVCacheAttentionBenchmark benchmark;
  benchmark.SetUp(config);
  
  // Track total FLOPs - slightly different from regular attention
  // because we're not recomputing K and V for each query
  int64_t flops_per_iter = 0;
  
  // Attention FLOPs estimation:
  // 1. QK^T: 2 * batchSize * numHeads * seqLen * contextLen * headDim
  // 2. Softmax: 5 * batchSize * numHeads * seqLen * contextLen
  // 3. AV: 2 * batchSize * numHeads * seqLen * contextLen * headDim
  flops_per_iter += 2 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen * config.headDim;  // QK^T
  flops_per_iter += 5 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen;  // Softmax (approx)
  flops_per_iter += 2 * config.batchSize * config.numHeads * config.seqLen * 
                   config.contextLen * config.headDim;  // AV
                   
  // Run the benchmark
  for (auto _ : state) {
    benchmark.RunBenchmark();
  }
  
  // Report metrics
  state.counters["FLOPs"] = benchmark::Counter(
      static_cast<double>(flops_per_iter),
      benchmark::Counter::kIsIterationInvariant | 
      benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1K);
      
  benchmark.TearDown();
}

// Register benchmarks with various configurations

// Regular attention benchmarks
// Small model (comparable to GPT-2 small)
BENCHMARK(BM_RegularAttention)
    ->Args({1, 32, 32, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 32, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Medium model (comparable to GPT-2 medium)
BENCHMARK(BM_RegularAttention)
    ->Args({1, 32, 1024, 24, 96, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 32, 2048, 24, 96, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Large model (comparable to GPT-2 large)
BENCHMARK(BM_RegularAttention)
    ->Args({1, 32, 2048, 36, 128, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Test sliding window attention
BENCHMARK(BM_RegularAttention)
    ->Args({1, 32, 2048, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::SLIDING_WINDOW),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Batch size variations
BENCHMARK(BM_RegularAttention)
    ->Args({4, 32, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({8, 32, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// KV-cache attention benchmarks
// Small model (comparable to GPT-2 small)
BENCHMARK(BM_KVCacheAttention)
    ->Args({1, 1, 32, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 1, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 1, 1024, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Medium model (comparable to GPT-2 medium)
BENCHMARK(BM_KVCacheAttention)
    ->Args({1, 1, 1024, 24, 96, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({1, 1, 2048, 24, 96, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Large model (comparable to GPT-2 large)
BENCHMARK(BM_KVCacheAttention)
    ->Args({1, 1, 2048, 36, 128, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Test sliding window attention with KV cache
BENCHMARK(BM_KVCacheAttention)
    ->Args({1, 1, 4096, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::SLIDING_WINDOW),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Batch size variations with KV cache
BENCHMARK(BM_KVCacheAttention)
    ->Args({4, 1, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)})
    ->Args({8, 1, 512, 12, 64, 
            static_cast<int64_t>(AttentionMaskType::CAUSAL),
            static_cast<int64_t>(AttentionOptLevel::BASIC)});

// Benchmark function for different attention variants
void BM_AttentionVariants(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t contextLen = state.range(2);
  const int64_t numHeads = state.range(3);
  const int64_t headDim = state.range(4);
  const AttentionVariant variant = static_cast<AttentionVariant>(state.range(5));
  
  // Configure benchmark
  TestType type;
  AttentionConfig config;
  config.numHeads = numHeads;
  config.headDim = headDim;
  config.scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  config.variant = variant;
  config.configureForVariant(variant);
  
  // Create implementation
  auto attImpl = createAttentionImpl(config, type);
  
  // Allocate data
  std::vector<float> queryData(batchSize * seqLen * numHeads * headDim, 0.0f);
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData(batchSize * seqLen * numHeads * headDim, 0.0f);
  
  // Size depends on variant
  if (variant == AttentionVariant::MULTI_QUERY) {
    keyData.resize(batchSize * contextLen * 1 * headDim, 0.0f);
    valueData.resize(batchSize * contextLen * 1 * headDim, 0.0f);
  } else if (variant == AttentionVariant::GROUPED_QUERY) {
    keyData.resize(batchSize * contextLen * config.numKVHeads * headDim, 0.0f);
    valueData.resize(batchSize * contextLen * config.numKVHeads * headDim, 0.0f);
  } else {
    keyData.resize(batchSize * contextLen * numHeads * headDim, 0.0f);
    valueData.resize(batchSize * contextLen * numHeads * headDim, 0.0f);
  }
  
  // Fill with random data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  
  for (auto& v : queryData) v = dist(rng);
  for (auto& v : keyData) v = dist(rng);
  for (auto& v : valueData) v = dist(rng);
  
  // Run benchmark
  for (auto _ : state) {
    attImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
  }
  
  // Report memory usage
  size_t memUsage = 0;
  memUsage += queryData.size() * sizeof(float);
  memUsage += keyData.size() * sizeof(float);
  memUsage += valueData.size() * sizeof(float);
  memUsage += outputData.size() * sizeof(float);
  
  // Convert to MB
  double memoryMB = static_cast<double>(memUsage) / (1024 * 1024);
  state.counters["MemoryMB"] = memoryMB;
}

// Register benchmarks for different attention variants
BENCHMARK(BM_AttentionVariants)
    // Standard MHA (12 heads, 64 dim)
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionVariant::STANDARD)})
    // Multi-Query Attention (12 heads, 64 dim, but 1 KV head)
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionVariant::MULTI_QUERY)})
    // Grouped-Query Attention (12 heads, 64 dim, 4 KV heads)
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionVariant::GROUPED_QUERY)});

// Benchmark function for pruned attention
void BM_PrunedAttention(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t contextLen = state.range(2);
  const int64_t numHeads = state.range(3);
  const int64_t headDim = state.range(4);
  const AttentionPruningStrategy strategy = 
      static_cast<AttentionPruningStrategy>(state.range(5));
  
  // Configure benchmark
  TestType type;
  AttentionConfig config;
  config.numHeads = numHeads;
  config.headDim = headDim;
  config.scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  config.pruningStrategy = strategy;
  
  // Set strategy-specific parameters
  switch (strategy) {
    case AttentionPruningStrategy::THRESHOLD:
      config.pruningThreshold = 0.01f;
      break;
    case AttentionPruningStrategy::TOP_K:
      config.pruningTopK = contextLen / 4;  // Keep 25% of tokens
      break;
    case AttentionPruningStrategy::BLOCK_SPARSE:
      config.pruningBlockSize = 16;
      config.pruningRatio = 0.5f;  // Prune 50% of blocks
      break;
    default:
      break;
  }
  
  // Create implementation
  auto attImpl = createAttentionImpl(config, type);
  
  // Allocate data
  std::vector<float> queryData(batchSize * seqLen * numHeads * headDim, 0.0f);
  std::vector<float> keyData(batchSize * contextLen * numHeads * headDim, 0.0f);
  std::vector<float> valueData(batchSize * contextLen * numHeads * headDim, 0.0f);
  std::vector<float> outputData(batchSize * seqLen * numHeads * headDim, 0.0f);
  
  // Fill with random data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  
  for (auto& v : queryData) v = dist(rng);
  for (auto& v : keyData) v = dist(rng);
  for (auto& v : valueData) v = dist(rng);
  
  // Run benchmark
  for (auto _ : state) {
    attImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
  }
  
  // Calculate FLOPs - adjusted for pruning
  int64_t flops_per_iter = 0;
  
  // QK^T computation (adjusted for pruning ratio)
  double computeRatio = 1.0;
  if (strategy == AttentionPruningStrategy::TOP_K) {
    computeRatio = static_cast<double>(config.pruningTopK) / contextLen;
  } else if (strategy == AttentionPruningStrategy::BLOCK_SPARSE) {
    computeRatio = 1.0 - config.pruningRatio;
  }
  
  // Standard attention FLOPs (adjusted for pruning)
  flops_per_iter += 2 * batchSize * numHeads * seqLen * contextLen * headDim;  // QK^T
  flops_per_iter += 5 * batchSize * numHeads * seqLen * contextLen;  // Softmax
  flops_per_iter += 2 * batchSize * numHeads * seqLen * contextLen * headDim * computeRatio;  // AV (pruned)
  
  // Report metrics
  state.counters["FLOPs"] = benchmark::Counter(
      static_cast<double>(flops_per_iter),
      benchmark::Counter::kIsIterationInvariant | 
      benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1K);
  
  // Also report pruning ratio
  state.counters["PruningRatio"] = computeRatio;
}

// Register benchmarks for pruned attention
BENCHMARK(BM_PrunedAttention)
    // No pruning
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionPruningStrategy::NONE)})
    // Threshold pruning
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionPruningStrategy::THRESHOLD)})
    // Top-K pruning
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionPruningStrategy::TOP_K)})
    // Block-sparse pruning
    ->Args({1, 32, 1024, 12, 64, 
            static_cast<int64_t>(AttentionPruningStrategy::BLOCK_SPARSE)});

} // namespace

// BENCHMARK_MAIN() expands to a main function that runs all registered benchmarks
BENCHMARK_MAIN(); 