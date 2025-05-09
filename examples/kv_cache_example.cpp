//===- kv_cache_example.cpp - Example of using PagedKVCache -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an example of using the PagedKVCache runtime library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include <random>

using namespace mlir::llm::runtime;

// Mock Type class for testing in a standalone example
namespace mlir {
class Type {
public:
  Type() = default;
  int getIntOrFloatBitWidth() const { return 16; } // Simulate f16 type
  bool isF16() const { return true; }
};

// Simplified LogicalResult definition
class LogicalResult {
  bool success;
  
public:
  LogicalResult(bool success = true) : success(success) {}
  
  bool succeeded() const { return success; }
  bool failed() const { return !success; }
  
  static LogicalResult success() { return LogicalResult(true); }
  static LogicalResult failure() { return LogicalResult(false); }
};

inline LogicalResult success() { return LogicalResult::success(); }
inline LogicalResult failure() { return LogicalResult::failure(); }
inline bool succeeded(LogicalResult result) { return result.succeeded(); }
inline bool failed(LogicalResult result) { return result.failed(); }
} // namespace mlir

// Helper function to print tensor information
template <typename T>
void printTensorInfo(const std::vector<T>& data, const std::string& name, int64_t batchSize, int64_t seqLen, int64_t numHeads, int64_t headDim) {
  std::cout << name << " tensor shape: [" << batchSize << ", " << seqLen << ", " << numHeads << ", " << headDim << "]" << std::endl;
  std::cout << "First few values: ";
  for (size_t i = 0; i < std::min(size_t(5), data.size()); i++) {
    std::cout << data[i] << " ";
  }
  std::cout << "..." << std::endl;
}

int main() {
  // Configuration for the KV cache
  const int64_t numLayers = 2;
  const int64_t numHeads = 12;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 4096;
  const bool useGPU = false; // Use CPU for this example
  
  // Create a PagedKVCache
  mlir::Type elementType;
  auto kvCache = std::make_unique<PagedKVCache>(
      numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType, useGPU);
  
  std::cout << "Created PagedKVCache with:" << std::endl;
  std::cout << "  Layers: " << numLayers << std::endl;
  std::cout << "  Heads: " << numHeads << std::endl;
  std::cout << "  Head Dimension: " << headDim << std::endl;
  std::cout << "  Block Size: " << blockSize << std::endl;
  std::cout << "  Max Sequence Length: " << maxSeqLen << std::endl;
  
  // Simulate a batch of queries for autoregressive generation
  const int64_t batchSize = 2;
  const int64_t seqLen = 1; // Autoregressive generation generates one token at a time
  
  // Create random query data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  
  std::vector<float> keyData(batchSize * seqLen * numHeads * headDim);
  std::vector<float> valueData(batchSize * seqLen * numHeads * headDim);
  
  // Fill with random data
  for (size_t i = 0; i < keyData.size(); i++) {
    keyData[i] = dist(gen);
    valueData[i] = dist(gen);
  }
  
  // Print tensor information
  printTensorInfo(keyData, "Key", batchSize, seqLen, numHeads, headDim);
  printTensorInfo(valueData, "Value", batchSize, seqLen, numHeads, headDim);
  
  // Sequence IDs for the batch
  std::vector<int32_t> seqIds = {100, 101};
  
  // Block indices will be filled by appendKV
  std::vector<int32_t> blockIndices(numLayers * batchSize * seqLen);
  
  // Simulate generating multiple tokens autoregressively
  const int numGenerationSteps = 10;
  
  // Store block indices for each step and layer
  std::vector<std::vector<std::vector<int32_t>>> savedBlockIndices(
      numLayers, std::vector<std::vector<int32_t>>(
          batchSize, std::vector<int32_t>(numGenerationSteps)));
  
  for (int step = 0; step < numGenerationSteps; step++) {
    std::cout << "\nGeneration step " << step + 1 << std::endl;
    
    // Append the current KV pairs to the cache
    mlir::LogicalResult result = kvCache->appendKV(
        keyData.data(), valueData.data(), batchSize, seqLen,
        seqIds.data(), blockIndices.data());
    
    if (mlir::failed(result)) {
      std::cerr << "Failed to append KV pairs!" << std::endl;
      return 1;
    }
    
    // Save the block indices for later lookup
    for (int layer = 0; layer < numLayers; layer++) {
      for (int b = 0; b < batchSize; b++) {
        int idx = layer * batchSize * seqLen + b * seqLen + 0; // 0 because seqLen=1
        savedBlockIndices[layer][b][step] = blockIndices[idx];
      }
    }
    
    // Print sequence lengths after append
    for (int i = 0; i < batchSize; i++) {
      std::cout << "Sequence " << seqIds[i] << " length: " 
                << kvCache->getSequenceLength(seqIds[i]) << std::endl;
    }
    
    // Generate new random KV pairs for the next step
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = dist(gen);
      valueData[i] = dist(gen);
    }
  }
  
  // Lookup the full cached sequences
  std::vector<int32_t> seqLens = {numGenerationSteps, numGenerationSteps};
  std::vector<int32_t> allBlockIndices(numLayers * batchSize * numGenerationSteps);
  std::vector<float> outputKeys(batchSize * numGenerationSteps * numHeads * headDim);
  std::vector<float> outputValues(batchSize * numGenerationSteps * numHeads * headDim);
  
  // Build a block indices tensor using the saved block indices
  for (int layer = 0; layer < numLayers; layer++) {
    for (int b = 0; b < batchSize; b++) {
      for (int t = 0; t < numGenerationSteps; t++) {
        int32_t blockIdx = savedBlockIndices[layer][b][t];
        allBlockIndices[layer * batchSize * numGenerationSteps + b * numGenerationSteps + t] = blockIdx;
      }
    }
  }
  
  // Lookup all KV pairs
  mlir::LogicalResult lookupResult = kvCache->lookupKV(
      allBlockIndices.data(), seqLens.data(), batchSize,
      outputKeys.data(), outputValues.data());
  
  if (mlir::failed(lookupResult)) {
    std::cerr << "Failed to lookup KV pairs!" << std::endl;
    return 1;
  }
  
  std::cout << "\nSuccessfully looked up KV pairs for " << numGenerationSteps 
            << " tokens per sequence" << std::endl;
  
  // Print memory usage
  std::cout << "\nTotal memory usage: " << kvCache->getTotalMemoryUsage() 
            << " bytes" << std::endl;
  
  // Clean up by clearing sequences
  for (int32_t seqId : seqIds) {
    kvCache->clearSequence(seqId);
  }
  
  std::cout << "Sequences cleared. Total sequences remaining: " 
            << kvCache->getNumSequences() << std::endl;
  
  std::cout << "Example completed successfully!" << std::endl;
  return 0;
} 