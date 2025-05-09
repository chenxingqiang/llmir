//===- kv_cache_example.cpp - Example using KV cache runtime -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an example of using the LLM KV cache runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace mlir;
using namespace mlir::llm::runtime;

// A simple Type implementation for testing
class SimpleType : public mlir::Type {
public:
  SimpleType() : mlir::Type() {}
};

void printLine(const char* message) {
  std::cout << "----------------------------------------" << std::endl;
  std::cout << message << std::endl;
  std::cout << "----------------------------------------" << std::endl;
}

int main() {
  printLine("LLM KV Cache Example");
  
  // Create a simple f16 type for testing
  SimpleType f16Type;
  
  // Configuration
  const int64_t numLayers = 12;
  const int64_t numHeads = 16;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 4096;
  const int64_t batchSize = 2;
  const int64_t seqLen = 1;
  
  std::cout << "Creating KV cache with configuration:" << std::endl;
  std::cout << "  Layers: " << numLayers << std::endl;
  std::cout << "  Heads: " << numHeads << std::endl;
  std::cout << "  Head Dimension: " << headDim << std::endl;
  std::cout << "  Block Size: " << blockSize << std::endl;
  std::cout << "  Max Sequence Length: " << maxSeqLen << std::endl;
  
  // Create the paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Prepare test data - two different sequences
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData = malloc(keyValueSize);
  void* valueData = malloc(keyValueSize);
  
  // Set some pattern in the data for verification
  for (size_t i = 0; i < keyValueSize; i++) {
    static_cast<unsigned char*>(keyData)[i] = i % 256;
    static_cast<unsigned char*>(valueData)[i] = (i + 128) % 256;
  }
  
  // Two different sequence IDs
  int32_t seqIds[batchSize] = {100, 200};
  
  // Output block indices
  int32_t blockIndices[batchSize * seqLen * numLayers];
  
  printLine("Appending KV pairs to the cache");
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqIds, blockIndices);
  
  if (failed(appendResult)) {
    std::cerr << "Failed to append KV pairs!" << std::endl;
    return 1;
  }
  
  std::cout << "Successfully appended KV pairs" << std::endl;
  std::cout << "Block indices for the first token of the first sequence: " << blockIndices[0] << std::endl;
  std::cout << "Block indices for the first token of the second sequence: " << blockIndices[1] << std::endl;
  
  // Add one more token to the first sequence
  printLine("Appending one more token to the first sequence");
  
  int32_t seqId2[1] = {100};
  int32_t blockIndices2[1 * seqLen * numLayers];
  
  // Append KV pairs
  LogicalResult appendResult2 = cache.appendKV(
      keyData, valueData, 1, seqLen, seqId2, blockIndices2);
  
  if (failed(appendResult2)) {
    std::cerr << "Failed to append KV pairs for the second token!" << std::endl;
    return 1;
  }
  
  std::cout << "Successfully appended the second token" << std::endl;
  std::cout << "Block indices for the second token of the first sequence: " << blockIndices2[0] << std::endl;
  
  // Retrieve the KV pairs for the sequences
  printLine("Looking up KV pairs from the cache");
  
  // Allocate memory for lookup results
  const size_t lookupSize = batchSize * 2 * numHeads * headDim * sizeof(float); // 2 tokens for each sequence
  void* lookupKeys = malloc(lookupSize);
  void* lookupValues = malloc(lookupSize);
  memset(lookupKeys, 0, lookupSize);
  memset(lookupValues, 0, lookupSize);
  
  // Sequence lengths for lookup - first sequence has 2 tokens, second has 1
  int32_t seqLens[batchSize] = {2, 1};
  
  // Combine block indices for lookup
  int32_t combinedIndices[batchSize * 2]; // Space for 2 tokens per sequence
  combinedIndices[0] = blockIndices[0];  // First token of first sequence
  combinedIndices[1] = blockIndices2[0]; // Second token of first sequence
  combinedIndices[2] = blockIndices[1];  // First token of second sequence
  combinedIndices[3] = 0;                // Padding
  
  // Look up KV pairs
  LogicalResult lookupResult = cache.lookupKV(
      combinedIndices, seqLens, batchSize, lookupKeys, lookupValues);
  
  if (failed(lookupResult)) {
    std::cerr << "Failed to lookup KV pairs!" << std::endl;
    return 1;
  }
  
  std::cout << "Successfully looked up KV pairs" << std::endl;
  
  // Verify a few values
  unsigned char* keysData = static_cast<unsigned char*>(lookupKeys);
  unsigned char* valuesData = static_cast<unsigned char*>(lookupValues);
  
  std::cout << "First byte of first token key: " << static_cast<int>(keysData[0]) << std::endl;
  std::cout << "First byte of first token value: " << static_cast<int>(valuesData[0]) << std::endl;
  
  // Calculate offset to second token of first sequence
  size_t tokenSize = numHeads * headDim * sizeof(float);
  std::cout << "First byte of second token key: " << static_cast<int>(keysData[tokenSize]) << std::endl;
  std::cout << "First byte of second token value: " << static_cast<int>(valuesData[tokenSize]) << std::endl;
  
  // Clean up
  free(keyData);
  free(valueData);
  free(lookupKeys);
  free(lookupValues);
  
  printLine("KV Cache Example Complete");
  
  return 0;
} 