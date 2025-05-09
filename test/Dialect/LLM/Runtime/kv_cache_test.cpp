//===- kv_cache_test.cpp - Tests for LLM KV cache runtime -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::llm::runtime;

namespace {

// Mock MLIRContext for testing
class MLIRContextFixture {
public:
  MLIRContextFixture() : context() {}
  MLIRContext context;
};

class KVCacheTest : public ::testing::Test, public MLIRContextFixture {
protected:
  void SetUp() override {
    // Common test setup
    f16Type = FloatType::getF16(&context);
  }

  FloatType f16Type;
};

// Test KVBlock functionality
TEST_F(KVCacheTest, KVBlockBasic) {
  const int64_t blockSize = 16;
  const int64_t headDim = 64;
  
  // Allocate memory for key and value
  const size_t memSize = blockSize * headDim * sizeof(float);
  void* keyPtr = malloc(memSize);
  void* valuePtr = malloc(memSize);
  
  // Create a KV block
  KVBlock block(keyPtr, valuePtr, blockSize, headDim);
  
  // Test basic getters
  EXPECT_EQ(block.getKeyPtr(), keyPtr);
  EXPECT_EQ(block.getValuePtr(), valuePtr);
  EXPECT_EQ(block.getBlockSize(), blockSize);
  EXPECT_EQ(block.getHeadDim(), headDim);
  
  // Clean up
  free(keyPtr);
  free(valuePtr);
}

// Test BlockAllocator functionality
TEST_F(KVCacheTest, BlockAllocatorBasic) {
  const int64_t blockSize = 16;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  
  // Create a block allocator
  BlockAllocator allocator(blockSize, numHeads, headDim, f16Type);
  
  // Check initial state (should pre-allocate some blocks)
  EXPECT_GT(allocator.getNumFreeBlocks(), 0);
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), 0);
  
  // Allocate a block
  KVBlock* block = allocator.allocateBlock();
  EXPECT_NE(block, nullptr);
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), 1);
  EXPECT_EQ(allocator.getNumFreeBlocks(), allocator.getNumFreeBlocks());
  
  // Free the block
  allocator.freeBlock(block);
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), 0);
  EXPECT_EQ(allocator.getNumFreeBlocks(), allocator.getNumFreeBlocks() + 1);
  
  // Allocate multiple blocks
  const int numBlocksToAllocate = 5;
  std::vector<KVBlock*> blocks;
  for (int i = 0; i < numBlocksToAllocate; i++) {
    blocks.push_back(allocator.allocateBlock());
    EXPECT_NE(blocks.back(), nullptr);
  }
  
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), numBlocksToAllocate);
  
  // Free the blocks
  for (auto* b : blocks) {
    allocator.freeBlock(b);
  }
  
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), 0);
}

// Test PagedKVCache functionality
TEST_F(KVCacheTest, PagedKVCacheBasic) {
  const int64_t numLayers = 2;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 1024;
  const int64_t batchSize = 2;
  const int64_t seqLen = 1;
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Test getters
  EXPECT_EQ(cache.getNumLayers(), numLayers);
  EXPECT_EQ(cache.getNumHeads(), numHeads);
  EXPECT_EQ(cache.getHeadDim(), headDim);
  EXPECT_EQ(cache.getBlockSize(), blockSize);
  EXPECT_EQ(cache.getMaxSeqLen(), maxSeqLen);
  EXPECT_EQ(cache.getElementType(), f16Type);
  
  // Prepare test data
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData = malloc(keyValueSize);
  void* valueData = malloc(keyValueSize);
  memset(keyData, 1, keyValueSize);  // Fill with 1s
  memset(valueData, 2, keyValueSize); // Fill with 2s
  
  // Sequence IDs
  int32_t seqIds[batchSize] = {1, 2};
  
  // Output block indices
  int32_t blockIndices[batchSize * seqLen];
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqIds, blockIndices);
  EXPECT_TRUE(succeeded(appendResult));
  
  // Allocate memory for lookup results
  void* outputKeys = malloc(keyValueSize);
  void* outputValues = malloc(keyValueSize);
  memset(outputKeys, 0, keyValueSize);
  memset(outputValues, 0, keyValueSize);
  
  // Sequence lengths for lookup
  int32_t seqLens[batchSize] = {seqLen, seqLen};
  
  // Look up KV pairs
  LogicalResult lookupResult = cache.lookupKV(
      blockIndices, seqLens, batchSize, outputKeys, outputValues);
  EXPECT_TRUE(succeeded(lookupResult));
  
  // Verify data - first byte of keys should be 1, first byte of values should be 2
  EXPECT_EQ(static_cast<unsigned char*>(outputKeys)[0], 1);
  EXPECT_EQ(static_cast<unsigned char*>(outputValues)[0], 2);
  
  // Clean up
  free(keyData);
  free(valueData);
  free(outputKeys);
  free(outputValues);
}

// Test appending multiple tokens to PagedKVCache
TEST_F(KVCacheTest, PagedKVCacheMultipleTokens) {
  const int64_t numLayers = 1;
  const int64_t numHeads = 4;
  const int64_t headDim = 32;
  const int64_t blockSize = 8;
  const int64_t maxSeqLen = 128;
  const int64_t batchSize = 1;
  const int64_t seqLen = 4; // Adding 4 tokens at once
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Prepare test data
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData = malloc(keyValueSize);
  void* valueData = malloc(keyValueSize);
  
  // Fill with recognizable patterns
  for (size_t i = 0; i < keyValueSize; i++) {
    static_cast<unsigned char*>(keyData)[i] = i % 256;
    static_cast<unsigned char*>(valueData)[i] = (i + 128) % 256;
  }
  
  // Sequence IDs
  int32_t seqIds[batchSize] = {42}; // Just one sequence
  
  // Output block indices
  int32_t blockIndices[batchSize * seqLen];
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqIds, blockIndices);
  EXPECT_TRUE(succeeded(appendResult));
  
  // Allocate memory for lookup results
  void* outputKeys = malloc(keyValueSize);
  void* outputValues = malloc(keyValueSize);
  memset(outputKeys, 0, keyValueSize);
  memset(outputValues, 0, keyValueSize);
  
  // Sequence lengths for lookup
  int32_t seqLens[batchSize] = {seqLen};
  
  // Look up KV pairs
  LogicalResult lookupResult = cache.lookupKV(
      blockIndices, seqLens, batchSize, outputKeys, outputValues);
  EXPECT_TRUE(succeeded(lookupResult));
  
  // Verify data - should match our original patterns
  for (size_t i = 0; i < keyValueSize; i++) {
    EXPECT_EQ(static_cast<unsigned char*>(outputKeys)[i], i % 256);
    EXPECT_EQ(static_cast<unsigned char*>(outputValues)[i], (i + 128) % 256);
  }
  
  // Clean up
  free(keyData);
  free(valueData);
  free(outputKeys);
  free(outputValues);
}

// Test PagedKVCache with multiple layers
TEST_F(KVCacheTest, PagedKVCacheMultipleLayers) {
  const int64_t numLayers = 3;
  const int64_t numHeads = 4;
  const int64_t headDim = 32;
  const int64_t blockSize = 8;
  const int64_t maxSeqLen = 128;
  const int64_t batchSize = 1;
  const int64_t seqLen = 1;
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Prepare test data
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData = malloc(keyValueSize);
  void* valueData = malloc(keyValueSize);
  
  // Fill with layer-specific values
  memset(keyData, 0xAA, keyValueSize); // Pattern for keys
  memset(valueData, 0xBB, keyValueSize); // Pattern for values
  
  // Sequence IDs
  int32_t seqIds[batchSize] = {7};
  
  // Output block indices
  int32_t blockIndices[batchSize * seqLen];
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqIds, blockIndices);
  EXPECT_TRUE(succeeded(appendResult));
  
  // Allocate memory for lookup results
  void* outputKeys = malloc(keyValueSize);
  void* outputValues = malloc(keyValueSize);
  memset(outputKeys, 0, keyValueSize);
  memset(outputValues, 0, keyValueSize);
  
  // Sequence lengths for lookup
  int32_t seqLens[batchSize] = {seqLen};
  
  // Look up KV pairs
  LogicalResult lookupResult = cache.lookupKV(
      blockIndices, seqLens, batchSize, outputKeys, outputValues);
  EXPECT_TRUE(succeeded(lookupResult));
  
  // Verify data patterns
  for (size_t i = 0; i < keyValueSize; i++) {
    EXPECT_EQ(static_cast<unsigned char*>(outputKeys)[i], 0xAA);
    EXPECT_EQ(static_cast<unsigned char*>(outputValues)[i], 0xBB);
  }
  
  // Clean up
  free(keyData);
  free(valueData);
  free(outputKeys);
  free(outputValues);
}

} // namespace 