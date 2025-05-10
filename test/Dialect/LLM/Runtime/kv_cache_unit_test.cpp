//===- kv_cache_unit_test.cpp - Tests for KV cache runtime ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the PagedKVCache runtime library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "gtest/gtest.h"

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

// Mock Type class for testing
namespace mlir {
class Type {
public:
  Type() = default;
  int getIntOrFloatBitWidth() const { return 16; } // Simulate f16 type
  bool isF16() const { return true; }
};

// Actual LogicalResult definition for tests
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

using namespace mlir::llm::runtime;

namespace {

class KVBlockTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a mock block
    keyBuffer = std::make_unique<float[]>(blockSize * numHeads * headDim);
    valueBuffer = std::make_unique<float[]>(blockSize * numHeads * headDim);
    block = std::make_unique<KVBlock>(keyBuffer.get(), valueBuffer.get(), blockSize, headDim);
  }
  
  void TearDown() override {
    block.reset();
    keyBuffer.reset();
    valueBuffer.reset();
  }
  
  const int64_t blockSize = 16;
  const int64_t numHeads = 12;
  const int64_t headDim = 64;
  
  std::unique_ptr<float[]> keyBuffer;
  std::unique_ptr<float[]> valueBuffer;
  std::unique_ptr<KVBlock> block;
};

TEST_F(KVBlockTest, BasicProperties) {
  EXPECT_EQ(block->getBlockSize(), blockSize);
  EXPECT_EQ(block->getHeadDim(), headDim);
  EXPECT_EQ(block->getKeyPtr(), keyBuffer.get());
  EXPECT_EQ(block->getValuePtr(), valueBuffer.get());
}

TEST_F(KVBlockTest, UsageTrackingAndRefCount) {
  EXPECT_EQ(block->getUsedSlots(), 0);
  EXPECT_FALSE(block->isFull());
  
  // Add some tokens
  block->incrementUsedSlots(5);
  EXPECT_EQ(block->getUsedSlots(), 5);
  EXPECT_FALSE(block->isFull());
  
  // Fill the block
  block->incrementUsedSlots(11);
  EXPECT_EQ(block->getUsedSlots(), 16);
  EXPECT_TRUE(block->isFull());
  
  // Test ref counting
  EXPECT_EQ(block->getRefCount(), 0);
  block->incrementRefCount();
  EXPECT_EQ(block->getRefCount(), 1);
  
  // Decrement ref count
  EXPECT_TRUE(block->decrementRefCount());
  EXPECT_EQ(block->getRefCount(), 0);
  
  // Should return false when already at zero
  EXPECT_FALSE(block->decrementRefCount());
  
  // Reset the block
  block->resetUsedSlots();
  EXPECT_EQ(block->getUsedSlots(), 0);
  EXPECT_FALSE(block->isFull());
}

class BlockAllocatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    mlir::Type type;
    allocator = std::make_unique<BlockAllocator>(blockSize, numHeads, headDim, type, false);
  }
  
  void TearDown() override {
    allocator.reset();
  }
  
  const int64_t blockSize = 16;
  const int64_t numHeads = 12;
  const int64_t headDim = 64;
  
  std::unique_ptr<BlockAllocator> allocator;
};

TEST_F(BlockAllocatorTest, InitialState) {
  // Should have pre-allocated blocks in the free list
  EXPECT_GT(allocator->getNumFreeBlocks(), 0);
  EXPECT_EQ(allocator->getNumAllocatedBlocks(), 0);
}

TEST_F(BlockAllocatorTest, AllocateAndFreeBlocks) {
  // Allocate a block
  KVBlock* block = allocator->allocateBlock();
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(allocator->getNumAllocatedBlocks(), 1);
  
  // Free the block
  allocator->freeBlock(block);
  EXPECT_EQ(allocator->getNumAllocatedBlocks(), 0);
  EXPECT_GT(allocator->getNumFreeBlocks(), 0);
}

TEST_F(BlockAllocatorTest, AllocateMultipleBlocks) {
  std::vector<KVBlock*> blocks;
  
  // Allocate 10 blocks
  for (int i = 0; i < 10; i++) {
    blocks.push_back(allocator->allocateBlock());
  }
  
  EXPECT_EQ(allocator->getNumAllocatedBlocks(), 10);
  
  // Free the blocks
  for (auto* block : blocks) {
    allocator->freeBlock(block);
  }
  
  EXPECT_EQ(allocator->getNumAllocatedBlocks(), 0);
}

TEST_F(BlockAllocatorTest, GetBlockByIndex) {
  // Allocate a block
  KVBlock* block = allocator->allocateBlock();
  ASSERT_NE(block, nullptr);
  
  // Get the block by index
  KVBlock* retrievedBlock = allocator->getBlock(0);
  EXPECT_EQ(retrievedBlock, block);
  
  // Invalid index
  EXPECT_EQ(allocator->getBlock(99), nullptr);
  
  // Free the block
  allocator->freeBlock(block);
}

class PagedKVCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    mlir::Type type;
    kvCache = std::make_unique<PagedKVCache>(numLayers, numHeads, headDim, 
                                          blockSize, maxSeqLen, type, false);
    
    // Create test data (f16 simulation with 2-byte floats)
    keyData.resize(batchSize * seqLen * numHeads * headDim);
    valueData.resize(batchSize * seqLen * numHeads * headDim);
    seqIds.resize(batchSize);
    
    // Fill with test pattern
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = i * 0.1f;
      valueData[i] = i * 0.2f;
    }
    
    // Set sequence IDs
    for (int i = 0; i < batchSize; i++) {
      seqIds[i] = 100 + i;
    }
    
    // Allocate space for block indices
    blockIndices.resize(numLayers * batchSize * seqLen);
  }
  
  void TearDown() override {
    kvCache.reset();
  }
  
  const int64_t numLayers = 2;
  const int64_t numHeads = 12;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 4096;
  const int64_t batchSize = 2;
  const int64_t seqLen = 8;
  
  std::unique_ptr<PagedKVCache> kvCache;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<int32_t> seqIds;
  std::vector<int32_t> blockIndices;
};

TEST_F(PagedKVCacheTest, BasicProperties) {
  EXPECT_EQ(kvCache->getNumLayers(), numLayers);
  EXPECT_EQ(kvCache->getNumHeads(), numHeads);
  EXPECT_EQ(kvCache->getHeadDim(), headDim);
  EXPECT_EQ(kvCache->getBlockSize(), blockSize);
  EXPECT_EQ(kvCache->getMaxSeqLen(), maxSeqLen);
}

TEST_F(PagedKVCacheTest, AppendKV) {
  // Append KV pairs
  mlir::LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Check sequence lengths
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), seqLen);
  }
  
  // Verify block indices are valid
  for (size_t i = 0; i < blockIndices.size(); i++) {
    EXPECT_GE(blockIndices[i], 0);
  }
}

TEST_F(PagedKVCacheTest, LookupKV) {
  // First append some KV pairs
  mlir::LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Create arrays for sequence lengths and output data
  std::vector<int32_t> seqLens(batchSize, seqLen);
  std::vector<float> outputKeys(batchSize * seqLen * numHeads * headDim);
  std::vector<float> outputValues(batchSize * seqLen * numHeads * headDim);
  
  // Lookup the KV pairs
  result = kvCache->lookupKV(
      blockIndices.data(), seqLens.data(), batchSize,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Verify the output data
  // Since we're simulating actual memory access in a test,
  // we'll just check a few values to make sure they're reasonable
  for (size_t i = 0; i < 10; i++) {
    EXPECT_FLOAT_EQ(outputKeys[i], keyData[i]);
    EXPECT_FLOAT_EQ(outputValues[i], valueData[i]);
  }
}

TEST_F(PagedKVCacheTest, CrossBlockBoundary) {
  // Create a sequence longer than one block
  int64_t longSeqLen = blockSize + 4;
  std::vector<float> longKeyData(batchSize * longSeqLen * numHeads * headDim);
  std::vector<float> longValueData(batchSize * longSeqLen * numHeads * headDim);
  std::vector<int32_t> longBlockIndices(numLayers * batchSize * longSeqLen);
  
  // Fill with test pattern
  for (size_t i = 0; i < longKeyData.size(); i++) {
    longKeyData[i] = i * 0.1f;
    longValueData[i] = i * 0.2f;
  }
  
  // Append KV pairs
  mlir::LogicalResult result = kvCache->appendKV(
      longKeyData.data(), longValueData.data(), batchSize, longSeqLen,
      seqIds.data(), longBlockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Check sequence lengths
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), longSeqLen);
  }
  
  // Create arrays for lookup
  std::vector<int32_t> seqLens(batchSize, longSeqLen);
  std::vector<float> outputKeys(batchSize * longSeqLen * numHeads * headDim);
  std::vector<float> outputValues(batchSize * longSeqLen * numHeads * headDim);
  
  // Lookup the KV pairs
  result = kvCache->lookupKV(
      longBlockIndices.data(), seqLens.data(), batchSize,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Verify the output data, especially across the block boundary
  int64_t boundaryIndex = blockSize * numHeads * headDim;
  EXPECT_FLOAT_EQ(outputKeys[boundaryIndex-1], longKeyData[boundaryIndex-1]);
  EXPECT_FLOAT_EQ(outputKeys[boundaryIndex], longKeyData[boundaryIndex]);
  EXPECT_FLOAT_EQ(outputValues[boundaryIndex-1], longValueData[boundaryIndex-1]);
  EXPECT_FLOAT_EQ(outputValues[boundaryIndex], longValueData[boundaryIndex]);
}

TEST_F(PagedKVCacheTest, ClearSequence) {
  // First append some KV pairs
  mlir::LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Clear one sequence
  int32_t seqToClear = seqIds[0];
  result = kvCache->clearSequence(seqToClear);
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Verify the sequence is gone
  EXPECT_EQ(kvCache->getSequenceLength(seqToClear), 0);
  
  // Other sequence should still be there
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[1]), seqLen);
}

TEST_F(PagedKVCacheTest, ResetCache) {
  // First append some KV pairs
  mlir::LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Reset the cache
  kvCache->reset();
  
  // Verify all sequences are gone
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), 0);
  }
  
  // Number of sequences should be zero
  EXPECT_EQ(kvCache->getNumSequences(), 0);
}

TEST_F(PagedKVCacheTest, MultipleAppends) {
  // Append KV pairs for the first time
  mlir::LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Append more KV pairs for the same sequences
  result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(mlir::succeeded(result));
  
  // Check sequence lengths
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), seqLen * 2);
  }
}

} // anonymous namespace 