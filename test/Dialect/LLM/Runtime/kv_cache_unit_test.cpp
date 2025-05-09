//===- kv_cache_unit_test.cpp - Unit tests for LLM KV cache runtime ---*- C++ -*-===//
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

class KVCacheUnitTest : public ::testing::Test, public MLIRContextFixture {
protected:
  void SetUp() override {
    // Common test setup
    f16Type = FloatType::getF16(&context);
  }

  FloatType f16Type;
};

// Test KVBlock memory management
TEST_F(KVCacheUnitTest, KVBlockMemoryManagement) {
  const int64_t blockSize = 16;
  const int64_t headDim = 64;
  
  // Allocate memory for key and value
  const size_t memSize = blockSize * headDim * sizeof(float);
  void* keyPtr = malloc(memSize);
  void* valuePtr = malloc(memSize);
  
  // Fill with recognizable patterns
  memset(keyPtr, 0xA5, memSize);
  memset(valuePtr, 0x5A, memSize);
  
  // Create a KV block
  KVBlock block(keyPtr, valuePtr, blockSize, headDim);
  
  // Test that memory is accessible and correct
  EXPECT_EQ(static_cast<unsigned char*>(block.getKeyPtr())[0], 0xA5);
  EXPECT_EQ(static_cast<unsigned char*>(block.getValuePtr())[0], 0x5A);
  
  // Clean up
  free(keyPtr);
  free(valuePtr);
}

// Test BlockAllocator pool management
TEST_F(KVCacheUnitTest, BlockAllocatorPoolManagement) {
  const int64_t blockSize = 16;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  
  // Create a block allocator with a specific initial pool size
  BlockAllocator allocator(blockSize, numHeads, headDim, f16Type);
  
  // Test initial state
  int64_t initialFreeBlocks = allocator.getNumFreeBlocks();
  EXPECT_GT(initialFreeBlocks, 0);
  
  // Exhaust the initial pool
  std::vector<KVBlock*> blocks;
  for (int i = 0; i < initialFreeBlocks; i++) {
    blocks.push_back(allocator.allocateBlock());
    EXPECT_NE(blocks.back(), nullptr);
  }
  
  // Verify the pool is empty
  EXPECT_EQ(allocator.getNumFreeBlocks(), 0);
  
  // Allocate one more block - should create a new one
  KVBlock* newBlock = allocator.allocateBlock();
  EXPECT_NE(newBlock, nullptr);
  blocks.push_back(newBlock);
  
  // Return all blocks to the pool
  for (auto* block : blocks) {
    allocator.freeBlock(block);
  }
  
  // Check that all blocks were returned to the pool
  EXPECT_EQ(allocator.getNumFreeBlocks(), initialFreeBlocks + 1);
  EXPECT_EQ(allocator.getNumAllocatedBlocks(), 0);
}

// Test PagedKVCache sequence mapping
TEST_F(KVCacheUnitTest, PagedKVCacheSequenceMapping) {
  const int64_t numLayers = 1;
  const int64_t numHeads = 4;
  const int64_t headDim = 32;
  const int64_t blockSize = 8;
  const int64_t maxSeqLen = 128;
  const int64_t batchSize = 2;
  const int64_t seqLen = 1;
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Prepare test data - two different sequences
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData = malloc(keyValueSize);
  void* valueData = malloc(keyValueSize);
  memset(keyData, 0x33, keyValueSize);
  memset(valueData, 0x44, keyValueSize);
  
  // Two different sequence IDs
  int32_t seqIds[batchSize] = {100, 200};
  
  // Output block indices
  int32_t blockIndices[batchSize * seqLen];
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqIds, blockIndices);
  EXPECT_TRUE(succeeded(appendResult));
  
  // Add more tokens to the first sequence
  int32_t seqIds2[1] = {100};
  int32_t blockIndices2[1];
  
  // Reuse the same data but for a single sequence
  LogicalResult appendResult2 = cache.appendKV(
      keyData, valueData, 1, 1, seqIds2, blockIndices2);
  EXPECT_TRUE(succeeded(appendResult2));
  
  // Verify that the block indices are correct
  // First token in seq 100 should have a different block index than first token in seq 200
  EXPECT_NE(blockIndices[0], blockIndices[1]);
  
  // First and second token in seq 100 should be in the same block (if blockSize > 1)
  if (blockSize > 1) {
    EXPECT_EQ(blockIndices[0], blockIndices2[0]);
  }
  
  // Clean up
  free(keyData);
  free(valueData);
}

// Test PagedKVCache with multi-layer configuration
TEST_F(KVCacheUnitTest, PagedKVCacheMultiLayerConfig) {
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
  
  // Fill with layer-specific patterns
  memset(keyData, 0x77, keyValueSize);
  memset(valueData, 0x88, keyValueSize);
  
  // Sequence ID
  int32_t seqId[batchSize] = {42};
  
  // Output block indices - one per layer
  int32_t blockIndices[batchSize * seqLen * numLayers];
  
  // Append KV pairs
  LogicalResult appendResult = cache.appendKV(
      keyData, valueData, batchSize, seqLen, seqId, blockIndices);
  EXPECT_TRUE(succeeded(appendResult));
  
  // Verify that each layer gets different block indices
  for (int64_t i = 1; i < numLayers; i++) {
    // Block indices should be different for each layer
    EXPECT_NE(blockIndices[0], blockIndices[i]);
  }
  
  // Clean up
  free(keyData);
  free(valueData);
}

// Test PagedKVCache crossing block boundaries
TEST_F(KVCacheUnitTest, PagedKVCacheCrossBlockBoundaries) {
  const int64_t numLayers = 1;
  const int64_t numHeads = 4;
  const int64_t headDim = 32;
  const int64_t blockSize = 4;  // Small block size to test crossing boundaries
  const int64_t maxSeqLen = 128;
  const int64_t batchSize = 1;
  const int64_t seqLen = 3;  // Add 3 tokens at once
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // First append: 3 tokens
  const size_t keyValueSize = batchSize * seqLen * numHeads * headDim * sizeof(float);
  void* keyData1 = malloc(keyValueSize);
  void* valueData1 = malloc(keyValueSize);
  
  // Fill with recognizable patterns
  for (size_t i = 0; i < keyValueSize; i++) {
    static_cast<unsigned char*>(keyData1)[i] = i % 256;
    static_cast<unsigned char*>(valueData1)[i] = (i + 128) % 256;
  }
  
  // Sequence ID
  int32_t seqId[batchSize] = {33};
  
  // Block indices for the first append
  int32_t blockIndices1[batchSize * seqLen];
  
  // Append first batch of KV pairs
  LogicalResult appendResult1 = cache.appendKV(
      keyData1, valueData1, batchSize, seqLen, seqId, blockIndices1);
  EXPECT_TRUE(succeeded(appendResult1));
  
  // Second append: 3 more tokens, should cross block boundary
  void* keyData2 = malloc(keyValueSize);
  void* valueData2 = malloc(keyValueSize);
  
  // Different pattern for the second batch
  for (size_t i = 0; i < keyValueSize; i++) {
    static_cast<unsigned char*>(keyData2)[i] = (i + 64) % 256;
    static_cast<unsigned char*>(valueData2)[i] = (i + 192) % 256;
  }
  
  // Block indices for the second append
  int32_t blockIndices2[batchSize * seqLen];
  
  // Append second batch of KV pairs to the same sequence
  LogicalResult appendResult2 = cache.appendKV(
      keyData2, valueData2, batchSize, seqLen, seqId, blockIndices2);
  EXPECT_TRUE(succeeded(appendResult2));
  
  // Since blockSize = 4, and we added 3 tokens first, then 3 more:
  // - First 3 tokens should be in block 0
  // - Next 1 token should be in block 0
  // - Last 2 tokens should be in block 1
  
  // First 3 tokens should be in the same block
  EXPECT_EQ(blockIndices1[0], blockIndices1[1]);
  EXPECT_EQ(blockIndices1[1], blockIndices1[2]);
  
  // Next token should still be in the first block
  EXPECT_EQ(blockIndices1[0], blockIndices2[0]);
  
  // Last 2 tokens should be in a new block
  EXPECT_EQ(blockIndices2[1], blockIndices2[2]);
  EXPECT_NE(blockIndices1[0], blockIndices2[1]);
  
  // Clean up
  free(keyData1);
  free(valueData1);
  free(keyData2);
  free(valueData2);
}

// Test PagedKVCache lookupKV with multiple blocks
TEST_F(KVCacheUnitTest, PagedKVCacheLookupMultipleBlocks) {
  const int64_t numLayers = 1;
  const int64_t numHeads = 2;
  const int64_t headDim = 4;
  const int64_t blockSize = 2;  // Small block size to test multiple blocks
  const int64_t maxSeqLen = 16;
  const int64_t batchSize = 1;
  const int64_t totalSeqLen = 5;  // Will span 3 blocks
  
  // Create a paged KV cache
  PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, f16Type);
  
  // Prepare test data
  const size_t tokenSize = numHeads * headDim * sizeof(float);
  const size_t fullDataSize = totalSeqLen * tokenSize;
  void* keyData = malloc(fullDataSize);
  void* valueData = malloc(fullDataSize);
  
  // Fill with recognizable patterns
  for (int i = 0; i < totalSeqLen; i++) {
    // Fill each token position with a unique pattern
    memset(static_cast<char*>(keyData) + i * tokenSize, 0x10 + i, tokenSize);
    memset(static_cast<char*>(valueData) + i * tokenSize, 0x20 + i, tokenSize);
  }
  
  // Sequence ID
  int32_t seqId[batchSize] = {42};
  
  // Block indices
  int32_t blockIndices[totalSeqLen];
  
  // Append tokens one by one to ensure specific block allocation
  for (int i = 0; i < totalSeqLen; i++) {
    void* keyToken = static_cast<char*>(keyData) + i * tokenSize;
    void* valueToken = static_cast<char*>(valueData) + i * tokenSize;
    
    LogicalResult appendResult = cache.appendKV(
        keyToken, valueToken, batchSize, 1, seqId, &blockIndices[i]);
    EXPECT_TRUE(succeeded(appendResult));
  }
  
  // Now lookup the entire sequence
  void* lookupKeys = malloc(fullDataSize);
  void* lookupValues = malloc(fullDataSize);
  memset(lookupKeys, 0, fullDataSize);
  memset(lookupValues, 0, fullDataSize);
  
  int32_t seqLen[batchSize] = {totalSeqLen};
  
  LogicalResult lookupResult = cache.lookupKV(
      blockIndices, seqLen, batchSize, lookupKeys, lookupValues);
  EXPECT_TRUE(succeeded(lookupResult));
  
  // Verify the lookup data matches original data
  for (int i = 0; i < totalSeqLen; i++) {
    size_t offset = i * tokenSize;
    // Check first byte of each token
    EXPECT_EQ(static_cast<unsigned char*>(lookupKeys)[offset], 0x10 + i);
    EXPECT_EQ(static_cast<unsigned char*>(lookupValues)[offset], 0x20 + i);
  }
  
  // Clean up
  free(keyData);
  free(valueData);
  free(lookupKeys);
  free(lookupValues);
}

} // namespace 