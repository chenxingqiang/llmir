//===- llama3_kvcache_benchmark.cpp - KVCache benchmark with Llama-3 3B ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a benchmark for PagedKVCache using the Llama-3 3B model.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Dialect/LLM/Runtime/GPUMemoryUtils.h"
#include <benchmark/benchmark.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

using namespace mlir::llm::runtime;

// Llama-3 3B model parameters
constexpr int64_t NUM_LAYERS = 26;      // Number of transformer layers
constexpr int64_t NUM_HEADS = 32;       // Number of attention heads
constexpr int64_t HEAD_DIM = 128;       // Dimension of each attention head
constexpr int64_t HIDDEN_DIM = 4096;    // Hidden dimension size
constexpr int64_t MAX_SEQ_LEN = 4096;   // Maximum sequence length

// Test parameters
constexpr int DEFAULT_BATCH_SIZE = 4;   // Default batch size for benchmarks
constexpr int DEFAULT_SEQ_LEN = 1024;   // Default sequence length for benchmarks

// Block size configurations to test
const std::vector<int64_t> BLOCK_SIZES = {16, 32, 64, 128, 256};

// GPU memory configurations to test
struct GPUMemConfig {
  bool enablePool;
  size_t unifiedMemThreshold;
  size_t initialPoolSize;
  std::string name;
};

const std::vector<GPUMemConfig> GPU_CONFIGS = {
  {false, 0, 0, "No optimizations"},                                   // No optimizations
  {true, 128 * 1024, 1024 * 1024 * 1024, "Pool + Unified(128KB)"},     // With pool and 128KB unified threshold
  {true, 256 * 1024, 1024 * 1024 * 1024, "Pool + Unified(256KB)"},     // With pool and 256KB unified threshold
  {true, 0, 1024 * 1024 * 1024, "Pool only"},                          // With pool only
  {false, 128 * 1024, 0, "Unified(128KB) only"}                        // With unified memory only
};

// Helper function to generate random data
template<typename T>
void fillRandom(std::vector<T>& data) {
  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<T>(dist(rng));
  }
}

// Benchmark for appending to KV cache
static void BM_KVCacheAppend(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t blockSize = state.range(2);
  const int gpuConfigIdx = state.range(3);
  
  // Create type for benchmarking
  mlir::Type elementType;
  
  // Initialize PagedKVCache
  auto kvCache = std::make_unique<PagedKVCache>(
      NUM_LAYERS, NUM_HEADS, HEAD_DIM, blockSize, MAX_SEQ_LEN, elementType, true);
  
  // Configure GPU memory options
  const auto& gpuConfig = GPU_CONFIGS[gpuConfigIdx];
  kvCache->configureGPUMemoryOptions(
      gpuConfig.enablePool, gpuConfig.unifiedMemThreshold, gpuConfig.initialPoolSize);
  
  // Create test data
  std::vector<float> keyData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> valueData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<int32_t> seqIds(batchSize);
  std::vector<int32_t> blockIndices(NUM_LAYERS * batchSize * seqLen);
  
  // Fill with random data
  fillRandom(keyData);
  fillRandom(valueData);
  
  // Set sequence IDs
  for (int i = 0; i < batchSize; i++) {
    seqIds[i] = 1000 + i;
  }
  
  // Warmup
  for (int i = 0; i < 3; i++) {
    kvCache->appendKV(
        keyData.data(), valueData.data(), batchSize, seqLen,
        seqIds.data(), blockIndices.data());
    kvCache->reset(); // Reset after warmup
  }
  
  // Benchmark loop
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Append to KV cache
    auto result = kvCache->appendKV(
        keyData.data(), valueData.data(), batchSize, seqLen,
        seqIds.data(), blockIndices.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    state.SetIterationTime(elapsed.count() / 1e9); // Convert to seconds
    
    // Reset after each iteration
    kvCache->reset();
    
    if (failed(result)) {
      state.SkipWithError("Failed to append to KV cache");
      break;
    }
  }
  
  // Set throughput metrics
  const double tokensProcessed = batchSize * seqLen;
  state.counters["Tokens/s"] = benchmark::Counter(
      tokensProcessed, benchmark::Counter::kIsRate);
  
  const size_t bytesProcessed = batchSize * seqLen * NUM_HEADS * HEAD_DIM * 
                               sizeof(float) * 2; // Keys + Values
  state.counters["MB/s"] = benchmark::Counter(
      bytesProcessed / (1024.0 * 1024.0), benchmark::Counter::kIsRate);
  
  // Report GPU memory stats
  std::cout << "=== GPU Memory Stats (" << gpuConfig.name << ", BlockSize=" << blockSize << ") ===" << std::endl;
  std::cout << kvCache->getGPUMemoryStats() << std::endl;
}

// Benchmark for looking up from KV cache
static void BM_KVCacheLookup(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t blockSize = state.range(2);
  const int gpuConfigIdx = state.range(3);
  
  // Create type for benchmarking
  mlir::Type elementType;
  
  // Initialize PagedKVCache
  auto kvCache = std::make_unique<PagedKVCache>(
      NUM_LAYERS, NUM_HEADS, HEAD_DIM, blockSize, MAX_SEQ_LEN, elementType, true);
  
  // Configure GPU memory options
  const auto& gpuConfig = GPU_CONFIGS[gpuConfigIdx];
  kvCache->configureGPUMemoryOptions(
      gpuConfig.enablePool, gpuConfig.unifiedMemThreshold, gpuConfig.initialPoolSize);
  
  // Create test data
  std::vector<float> keyData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> valueData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<int32_t> seqIds(batchSize);
  std::vector<int32_t> blockIndices(NUM_LAYERS * batchSize * seqLen);
  std::vector<int32_t> seqLens(batchSize, seqLen);
  
  // Output buffers for lookup
  std::vector<float> outputKeys(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> outputValues(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  
  // Fill with random data
  fillRandom(keyData);
  fillRandom(valueData);
  
  // Set sequence IDs
  for (int i = 0; i < batchSize; i++) {
    seqIds[i] = 1000 + i;
  }
  
  // Append data to cache once before benchmark
  auto appendResult = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
  
  if (failed(appendResult)) {
    state.SkipWithError("Failed to append to KV cache during setup");
    return;
  }
  
  // Warmup
  for (int i = 0; i < 3; i++) {
    kvCache->lookupKV(
        blockIndices.data(), seqLens.data(), batchSize,
        outputKeys.data(), outputValues.data());
  }
  
  // Benchmark loop
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Lookup from KV cache
    auto result = kvCache->lookupKV(
        blockIndices.data(), seqLens.data(), batchSize,
        outputKeys.data(), outputValues.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    state.SetIterationTime(elapsed.count() / 1e9); // Convert to seconds
    
    if (failed(result)) {
      state.SkipWithError("Failed to lookup from KV cache");
      break;
    }
  }
  
  // Set throughput metrics
  const double tokensProcessed = batchSize * seqLen;
  state.counters["Tokens/s"] = benchmark::Counter(
      tokensProcessed, benchmark::Counter::kIsRate);
  
  const size_t bytesProcessed = batchSize * seqLen * NUM_HEADS * HEAD_DIM * 
                               sizeof(float) * 2; // Keys + Values
  state.counters["MB/s"] = benchmark::Counter(
      bytesProcessed / (1024.0 * 1024.0), benchmark::Counter::kIsRate);
}

// Benchmark for attention computation with KV cache
static void BM_KVCacheAttention(benchmark::State& state) {
  const int64_t batchSize = state.range(0);
  const int64_t seqLen = state.range(1);
  const int64_t blockSize = state.range(2);
  const int gpuConfigIdx = state.range(3);
  
  // Create type for benchmarking
  mlir::Type elementType;
  
  // Initialize PagedKVCache
  auto kvCache = std::make_unique<PagedKVCache>(
      NUM_LAYERS, NUM_HEADS, HEAD_DIM, blockSize, MAX_SEQ_LEN, elementType, true);
  
  // Configure GPU memory options
  const auto& gpuConfig = GPU_CONFIGS[gpuConfigIdx];
  kvCache->configureGPUMemoryOptions(
      gpuConfig.enablePool, gpuConfig.unifiedMemThreshold, gpuConfig.initialPoolSize);
  
  // Configure attention optimization
  AttentionConfig attConfig;
  attConfig.numHeads = NUM_HEADS;
  attConfig.headDim = HEAD_DIM;
  attConfig.maskType = AttentionMaskType::CAUSAL;
  attConfig.optLevel = AttentionOptLevel::FLASH;
  attConfig.useFlashAttention = true;
  attConfig.useFusedSoftmax = true;
  kvCache->configureAttentionOpt(attConfig);
  
  // Create test data
  std::vector<float> keyData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> valueData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> queryData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<float> outputData(batchSize * seqLen * NUM_HEADS * HEAD_DIM);
  std::vector<int32_t> seqIds(batchSize);
  std::vector<int32_t> blockIndices(NUM_LAYERS * batchSize * seqLen);
  std::vector<int32_t> seqLens(batchSize, seqLen);
  
  // Fill with random data
  fillRandom(keyData);
  fillRandom(valueData);
  fillRandom(queryData);
  
  // Set sequence IDs
  for (int i = 0; i < batchSize; i++) {
    seqIds[i] = 1000 + i;
  }
  
  // Append data to cache once before benchmark
  auto appendResult = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
  
  if (failed(appendResult)) {
    state.SkipWithError("Failed to append to KV cache during setup");
    return;
  }
  
  // Warmup
  for (int i = 0; i < 3; i++) {
    kvCache->computeAttention(
        outputData.data(), queryData.data(), blockIndices.data(),
        seqLens.data(), batchSize, seqLen);
  }
  
  // Benchmark loop
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Compute attention
    auto result = kvCache->computeAttention(
        outputData.data(), queryData.data(), blockIndices.data(),
        seqLens.data(), batchSize, seqLen);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    state.SetIterationTime(elapsed.count() / 1e9); // Convert to seconds
    
    if (failed(result)) {
      state.SkipWithError("Failed to compute attention");
      break;
    }
  }
  
  // Set throughput metrics
  const double tokensProcessed = batchSize * seqLen;
  state.counters["Tokens/s"] = benchmark::Counter(
      tokensProcessed, benchmark::Counter::kIsRate);
  
  // Calculate FLOPs for attention operation
  // Attention FLOPs: 2 * B * H * S * S * D + B * H * S * S + B * H * S * S * D
  // = B * H * S * (2 * S * D + S + S * D) = B * H * S * S * (3 * D + 1)
  const double flops = batchSize * NUM_HEADS * seqLen * seqLen * (3 * HEAD_DIM + 1);
  state.counters["GFLOP/s"] = benchmark::Counter(
      flops / 1e9, benchmark::Counter::kIsRate);
}

// Define benchmarks
void registerKVCacheBenchmarks() {
  // Append benchmarks
  for (size_t configIdx = 0; configIdx < GPU_CONFIGS.size(); configIdx++) {
    for (int64_t blockSize : BLOCK_SIZES) {
      std::string name = "BM_KVCacheAppend/" + 
                         GPU_CONFIGS[configIdx].name + "/BlockSize=" + 
                         std::to_string(blockSize);
                         
      benchmark::RegisterBenchmark(name.c_str(), BM_KVCacheAppend)
          ->Args({DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, blockSize, static_cast<int>(configIdx)})
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime()
          ->Iterations(10)
          ->Repetitions(3)
          ->DisplayAggregatesOnly(true);
    }
  }
  
  // Lookup benchmarks
  for (size_t configIdx = 0; configIdx < GPU_CONFIGS.size(); configIdx++) {
    for (int64_t blockSize : BLOCK_SIZES) {
      std::string name = "BM_KVCacheLookup/" + 
                         GPU_CONFIGS[configIdx].name + "/BlockSize=" + 
                         std::to_string(blockSize);
                         
      benchmark::RegisterBenchmark(name.c_str(), BM_KVCacheLookup)
          ->Args({DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, blockSize, static_cast<int>(configIdx)})
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime()
          ->Iterations(10)
          ->Repetitions(3)
          ->DisplayAggregatesOnly(true);
    }
  }
  
  // Attention benchmarks
  for (size_t configIdx = 0; configIdx < GPU_CONFIGS.size(); configIdx++) {
    for (int64_t blockSize : BLOCK_SIZES) {
      std::string name = "BM_KVCacheAttention/" + 
                         GPU_CONFIGS[configIdx].name + "/BlockSize=" + 
                         std::to_string(blockSize);
                         
      benchmark::RegisterBenchmark(name.c_str(), BM_KVCacheAttention)
          ->Args({DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, blockSize, static_cast<int>(configIdx)})
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime()
          ->Iterations(10)
          ->Repetitions(3)
          ->DisplayAggregatesOnly(true);
    }
  }
  
  // Additional benchmarks for different batch sizes and sequence lengths
  for (int batchSize : {1, 2, 4, 8, 16}) {
    for (int seqLen : {256, 512, 1024, 2048, 4096}) {
      std::string name = "BM_KVCacheAppend/BatchSize=" + 
                         std::to_string(batchSize) + "/SeqLen=" + 
                         std::to_string(seqLen);
                         
      benchmark::RegisterBenchmark(name.c_str(), BM_KVCacheAppend)
          ->Args({batchSize, seqLen, 64, 1}) // Using 64 block size and optimal config
          ->Unit(benchmark::kMillisecond)
          ->UseManualTime()
          ->Iterations(5)
          ->Repetitions(3)
          ->DisplayAggregatesOnly(true);
    }
  }
}

// Main function
int main(int argc, char** argv) {
  registerKVCacheBenchmarks();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
} 