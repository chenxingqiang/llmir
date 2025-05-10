//===- LargeModelBenchmark.cpp - Benchmarks for large model attention ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements benchmarks for Flash Attention with large models and
// long sequences.
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

// Configuration for large model benchmark
struct LargeModelConfig {
  int64_t batchSize;
  int64_t seqLen;
  int64_t contextLen;
  int64_t numHeads;
  int64_t headDim;
  AttentionMaskType maskType;
  AttentionOptLevel optLevel;
  bool useFlashAttention;
  int64_t blockSizeM;
  int64_t blockSizeN;
  
  // Calculate total elements in a tensor
  int64_t totalElements(int64_t b, int64_t s, int64_t h, int64_t d) const {
    return b * s * h * d;
  }
  
  // Calculate total memory size for a benchmark run
  size_t estimateMemoryUsage() const {
    // Query, key, value, and output tensors (4 bytes per float)
    size_t querySize = totalElements(batchSize, seqLen, numHeads, headDim) * 4;
    size_t keySize = totalElements(batchSize, contextLen, numHeads, headDim) * 4;
    size_t valueSize = keySize;
    size_t outputSize = querySize;
    
    // Attention matrices (4 bytes per float)
    size_t attentionSize = batchSize * numHeads * seqLen * contextLen * 4;
    
    // Total size in bytes
    return querySize + keySize + valueSize + outputSize + attentionSize;
  }
};

// Base fixture for large model benchmarks
class LargeModelBenchmarkBase {
protected:
  LargeModelBenchmarkBase() : type(TestType()) {}
  
  void SetUp(const LargeModelConfig& config) {
    this->config = config;
    
    // Create attention configuration
    AttentionConfig attConfig;
    attConfig.numHeads = config.numHeads;
    attConfig.headDim = config.headDim;
    attConfig.scale = 1.0f / std::sqrt(static_cast<float>(config.headDim));
    attConfig.maskType = config.maskType;
    attConfig.optLevel = config.optLevel;
    attConfig.fuseSoftmax = true;
    attConfig.useFlashAttention = config.useFlashAttention;
    attConfig.blockSizeM = config.blockSizeM;
    attConfig.blockSizeN = config.blockSizeN;
    
    // Allocate data (memory usage is tracked here)
    size_t querySize = config.totalElements(config.batchSize, config.seqLen, 
                                           config.numHeads, config.headDim);
    size_t keyValueSize = config.totalElements(config.batchSize, config.contextLen, 
                                              config.numHeads, config.headDim);
    size_t outputSize = querySize;
    
    queryData.resize(querySize, 0.0f);
    keyData.resize(keyValueSize, 0.0f);
    valueData.resize(keyValueSize, 0.0f);
    outputData.resize(outputSize, 0.0f);
    
    // Fill with random data
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (auto& v : queryData) v = dist(rng);
    for (auto& v : keyData) v = dist(rng);
    for (auto& v : valueData) v = dist(rng);
    
    // Create attention implementation
    attImpl = createAttentionImpl(attConfig, type);
  }
  
  void TearDown() {
    attImpl.reset();
    
    // Clear data to free memory
    std::vector<float>().swap(queryData);
    std::vector<float>().swap(keyData);
    std::vector<float>().swap(valueData);
    std::vector<float>().swap(outputData);
  }
  
  void RunBenchmark() {
    // Run attention computation
    attImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        config.batchSize,
        config.seqLen,
        config.contextLen);
  }

  TestType type;
  LargeModelConfig config;
  std::unique_ptr<AttentionImpl> attImpl;
  
  std::vector<float> queryData;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData;
};

// Benchmark comparing Flash Attention vs Regular Attention for large models
void BM_FlashVsRegularAttention(benchmark::State& state) {
  // Parameters from benchmark arguments
  const int64_t modelType = state.range(0);        // 0=Small, 1=Medium, 2=Large, 3=XLarge
  const int64_t seqLen = state.range(1);           // Sequence length
  const int64_t useFlashAttention = state.range(2); // 0=Regular, 1=Flash
  
  // Model size configurations
  // Based on common model sizes: Small ~100M, Med ~300M, Large ~1B, XLarge ~7B+
  const std::vector<std::pair<int64_t, int64_t>> modelConfigs = {
    {12, 64},    // Small (12 heads, 64 dim)
    {16, 96},    // Medium (16 heads, 96 dim)
    {32, 128},   // Large (32 heads, 128 dim)
    {64, 128}    // XLarge (64 heads, 128 dim)
  };
  
  // Get model configuration
  int64_t numHeads = modelConfigs[modelType].first;
  int64_t headDim = modelConfigs[modelType].second;
  
  // Configure the benchmark
  LargeModelConfig config;
  config.batchSize = 1;
  config.seqLen = seqLen;
  config.contextLen = seqLen; // Context length = sequence length for benchmark
  config.numHeads = numHeads;
  config.headDim = headDim;
  config.maskType = AttentionMaskType::CAUSAL;
  config.optLevel = AttentionOptLevel::ADVANCED;
  config.useFlashAttention = (useFlashAttention == 1);
  config.blockSizeM = 64;
  config.blockSizeN = 64;
  
  // Print memory usage estimate
  size_t estimatedMemoryMB = config.estimateMemoryUsage() / (1024 * 1024);
  state.SetLabel(std::string("Mem: ") + std::to_string(estimatedMemoryMB) + " MB");
  
  // Create benchmark fixture
  LargeModelBenchmarkBase benchmark;
  benchmark.SetUp(config);
  
  // Calculate FLOPs
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
  
  // Run benchmark
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

// Benchmark for measuring scaling with sequence length
void BM_SequenceLengthScaling(benchmark::State& state) {
  // Parameters
  const int64_t seqLen = state.range(0);          // Sequence length
  const bool useFlashAttention = state.range(1);  // Use Flash Attention
  
  // Configure the benchmark
  LargeModelConfig config;
  config.batchSize = 1;
  config.seqLen = seqLen;
  config.contextLen = seqLen;
  config.numHeads = 16;
  config.headDim = 64;
  config.maskType = AttentionMaskType::CAUSAL;
  config.optLevel = AttentionOptLevel::ADVANCED;
  config.useFlashAttention = useFlashAttention;
  config.blockSizeM = 64;
  config.blockSizeN = 64;
  
  // Create benchmark fixture
  LargeModelBenchmarkBase benchmark;
  benchmark.SetUp(config);
  
  // Calculate theoretical complexity
  // Regular attention: O(N²) in sequence length
  // Flash attention: O(N²) in computation but improved memory efficiency
  int64_t n_sq = seqLen * seqLen;
  
  // Run benchmark
  for (auto _ : state) {
    benchmark.RunBenchmark();
  }
  
  // Report N² as a counter to check scaling
  state.counters["N^2"] = static_cast<double>(n_sq);
  
  benchmark.TearDown();
}

// Benchmark for very long sequences (up to 16K)
void BM_VeryLongSequences(benchmark::State& state) {
  // Parameters
  const int64_t seqLen = state.range(0);          // Sequence length
  const bool useFlashAttention = state.range(1);  // Use Flash Attention
  
  // Configure the benchmark - use smaller model for very long sequences
  // to avoid excessive memory usage
  LargeModelConfig config;
  config.batchSize = 1;
  config.seqLen = seqLen;
  config.contextLen = seqLen;
  config.numHeads = 8;  // Reduced model size
  config.headDim = 64;
  config.maskType = AttentionMaskType::SLIDING_WINDOW; // Use sliding window for long seqs
  config.optLevel = AttentionOptLevel::ADVANCED;
  config.useFlashAttention = useFlashAttention;
  config.blockSizeM = 64;
  config.blockSizeN = 64;
  config.windowSize = 1024;  // Fixed window size
  
  // Estimate and print memory usage
  size_t estimatedMemoryMB = config.estimateMemoryUsage() / (1024 * 1024);
  state.SetLabel(std::string("Mem: ") + std::to_string(estimatedMemoryMB) + " MB");
  
  // Create benchmark fixture
  LargeModelBenchmarkBase benchmark;
  benchmark.SetUp(config);
  
  // Run benchmark
  for (auto _ : state) {
    benchmark.RunBenchmark();
  }
  
  benchmark.TearDown();
}

// Register benchmarks

// Flash vs Regular Attention benchmark
// Args: modelType (0-3), seqLen, useFlashAttention (0-1)
// Small model with varying sequence lengths
BENCHMARK(BM_FlashVsRegularAttention)
    ->Args({0, 512, 0})    // Small model, 512 seq len, Regular attention
    ->Args({0, 512, 1})    // Small model, 512 seq len, Flash attention
    ->Args({0, 1024, 0})   // Small model, 1024 seq len, Regular attention
    ->Args({0, 1024, 1})   // Small model, 1024 seq len, Flash attention
    ->Args({0, 2048, 0})   // Small model, 2048 seq len, Regular attention
    ->Args({0, 2048, 1});  // Small model, 2048 seq len, Flash attention
    
// Medium model with varying sequence lengths
BENCHMARK(BM_FlashVsRegularAttention)
    ->Args({1, 512, 0})    // Medium model, 512 seq len, Regular attention
    ->Args({1, 512, 1})    // Medium model, 512 seq len, Flash attention
    ->Args({1, 1024, 0})   // Medium model, 1024 seq len, Regular attention
    ->Args({1, 1024, 1})   // Medium model, 1024 seq len, Flash attention
    ->Args({1, 2048, 0})   // Medium model, 2048 seq len, Regular attention
    ->Args({1, 2048, 1});  // Medium model, 2048 seq len, Flash attention

// Large model with varying sequence lengths (might require significant memory)
BENCHMARK(BM_FlashVsRegularAttention)
    ->Args({2, 512, 0})    // Large model, 512 seq len, Regular attention
    ->Args({2, 512, 1})    // Large model, 512 seq len, Flash attention
    ->Args({2, 1024, 0})   // Large model, 1024 seq len, Regular attention
    ->Args({2, 1024, 1})   // Large model, 1024 seq len, Flash attention
    ->Args({2, 2048, 1});  // Large model, 2048, Flash attention only (regular might OOM)

// XLarge model with Flash Attention only (regular would likely OOM)
BENCHMARK(BM_FlashVsRegularAttention)
    ->Args({3, 512, 1})    // XLarge model, 512 seq len, Flash attention
    ->Args({3, 1024, 1})   // XLarge model, 1024 seq len, Flash attention
    ->Args({3, 2048, 1});  // XLarge model, 2048 seq len, Flash attention

// Sequence length scaling benchmark
// Args: seqLen, useFlashAttention
BENCHMARK(BM_SequenceLengthScaling)
    ->Args({128, 0})     // Regular attention, 128 tokens
    ->Args({128, 1})     // Flash attention, 128 tokens
    ->Args({256, 0})     // Regular attention, 256 tokens
    ->Args({256, 1})     // Flash attention, 256 tokens
    ->Args({512, 0})     // Regular attention, 512 tokens
    ->Args({512, 1})     // Flash attention, 512 tokens
    ->Args({1024, 0})    // Regular attention, 1024 tokens
    ->Args({1024, 1})    // Flash attention, 1024 tokens
    ->Args({2048, 0})    // Regular attention, 2048 tokens
    ->Args({2048, 1})    // Flash attention, 2048 tokens
    ->Args({4096, 0})    // Regular attention, 4096 tokens
    ->Args({4096, 1});   // Flash attention, 4096 tokens

// Very long sequence benchmark - for these extremely long sequences,
// Flash Attention is much more efficient with memory usage
BENCHMARK(BM_VeryLongSequences)
    ->Args({4096, 1})     // Flash attention, 4K tokens
    ->Args({8192, 1})     // Flash attention, 8K tokens
    ->Args({16384, 1});   // Flash attention, 16K tokens

} // namespace

// BENCHMARK_MAIN() expands to a main function that runs all registered benchmarks
BENCHMARK_MAIN(); 