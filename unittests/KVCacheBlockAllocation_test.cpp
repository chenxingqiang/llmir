//===- KVCacheBlockAllocation_test.cpp - Tests for KV Cache Block Allocation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <algorithm>

using namespace mlir;
using namespace mlir::llm::runtime;

namespace {

// A simple Type implementation for testing
struct TestType : public Type {
  int getIntOrFloatBitWidth() const { return 16; }  // f16
};

// Test fixture for block allocation optimization tests
class KVCacheBlockAllocationTest : public ::testing::Test {
protected:
  void SetUp() override {
    type = TestType();
    blockAllocator = std::make_unique<BlockAllocator>(
        blockSize, numHeads, headDim, type);
  }

  void TearDown() override {
    blockAllocator.reset();
  }

  const int64_t blockSize = 16;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  TestType type;
  std::unique_ptr<BlockAllocator> blockAllocator;
};

// Test preallocation functionality
TEST_F(KVCacheBlockAllocationTest, Preallocation) {
  // Test basic preallocation
  EXPECT_GT(blockAllocator->getNumFreeBlocks(), 0);
  
  // Clear existing free blocks
  size_t initialFreeBlocks = blockAllocator->getNumFreeBlocks();
  for (size_t i = 0; i < initialFreeBlocks; i++) {
    KVBlock* block = blockAllocator->allocateBlock();
    EXPECT_NE(block, nullptr);
  }
  EXPECT_EQ(blockAllocator->getNumFreeBlocks(), 0);
  
  // Test preallocation with sequence statistics
  int64_t avgSeqLen = 512;
  int64_t maxNumSequences = 4;
  
  blockAllocator->preallocateBlocks(0, avgSeqLen, maxNumSequences);
  
  // We should have preallocated enough blocks for the sequences
  int64_t expectedBlocks = (avgSeqLen * maxNumSequences) / blockSize;
  EXPECT_GE(blockAllocator->getNumFreeBlocks(), expectedBlocks);
  
  // Check metrics
  auto metrics = blockAllocator->getMetrics();
  EXPECT_GT(metrics.numPreallocated, 0);
}

// Test LRU eviction policy
TEST_F(KVCacheBlockAllocationTest, LRUEvictionPolicy) {
  // Allocate some blocks and mark different access times
  std::vector<KVBlock*> blocks;
  for (int i = 0; i < 10; i++) {
    KVBlock* block = blockAllocator->allocateBlock();
    ASSERT_NE(block, nullptr);
    // Set access times with different time gaps
    block->updateAccessTime(i * 100);
    blocks.push_back(block);
  }
  
  // Set up LRU policy
  LRUEvictionPolicy lruPolicy;
  
  // Get blocks to evict
  std::vector<int32_t> evictedBlocks = lruPolicy.selectBlocksForEviction(*blockAllocator, 3);
  
  // Should get 3 blocks with earliest access times
  ASSERT_EQ(evictedBlocks.size(), 3);
  EXPECT_EQ(evictedBlocks[0], 0); // First block (oldest)
  EXPECT_EQ(evictedBlocks[1], 1); // Second block
  EXPECT_EQ(evictedBlocks[2], 2); // Third block
}

// Test custom eviction policy (FragmentationAwareLRU)
TEST_F(KVCacheBlockAllocationTest, FragmentationAwareEvictionPolicy) {
  // Allocate blocks with different fragmentation levels
  std::vector<KVBlock*> blocks;
  
  for (int i = 0; i < 10; i++) {
    KVBlock* block = blockAllocator->allocateBlock();
    ASSERT_NE(block, nullptr);
    
    // Set access times
    block->updateAccessTime(i * 100);
    
    // Set different levels of utilization
    // Even blocks: high utilization (low fragmentation)
    // Odd blocks: low utilization (high fragmentation)
    if (i % 2 == 0) {
      // Fill block almost completely
      block->incrementUsedSlots(blockSize - 2);
    } else {
      // Fill block minimally
      block->incrementUsedSlots(blockSize / 4);
    }
    
    // Update fragmentation metric
    block->updateFragmentation();
    blocks.push_back(block);
  }
  
  // Set up fragmentation-aware policy with high weight on fragmentation
  FragmentationAwareLRUPolicy fragPolicy(0.8); // 80% weight on fragmentation
  
  // Get blocks to evict
  std::vector<int32_t> evictedBlocks = fragPolicy.selectBlocksForEviction(*blockAllocator, 3);
  
  // Should prioritize high-fragmentation (odd-indexed) blocks
  ASSERT_EQ(evictedBlocks.size(), 3);
  
  // Check if at least 2 of the 3 selected blocks are odd-indexed (high fragmentation)
  int oddCount = 0;
  for (int32_t idx : evictedBlocks) {
    if (idx % 2 == 1) {
      oddCount++;
    }
  }
  
  EXPECT_GE(oddCount, 2);
}

// Test block coalescing
TEST_F(KVCacheBlockAllocationTest, BlockCoalescing) {
  // First, allocate some blocks and partially fill them
  std::vector<KVBlock*> blocks;
  
  for (int i = 0; i < 6; i++) {
    KVBlock* block = blockAllocator->allocateBlock();
    ASSERT_NE(block, nullptr);
    
    // Make different fill patterns for blocks
    if (i < 3) {
      // Fill first 3 blocks at 30% capacity
      block->incrementUsedSlots(blockSize * 3 / 10);
    } else {
      // Fill last 3 blocks at 60% capacity
      block->incrementUsedSlots(blockSize * 6 / 10);
    }
    
    // Update fragmentation metric
    block->updateFragmentation();
    blocks.push_back(block);
  }
  
  // Now run coalescing
  int64_t coalesced = blockAllocator->coalesceBlocks(0.5, 5, true);
  
  // We should have coalesced some blocks
  EXPECT_GT(coalesced, 0);
  
  // The remaining blocks should have higher utilization
  double avgFragmentation = 0.0;
  for (size_t i = 0; i < blockAllocator->getNumAllocatedBlocks(); i++) {
    KVBlock* block = blockAllocator->getBlock(i);
    if (block) {
      avgFragmentation += block->getFragmentation();
    }
  }
  avgFragmentation /= blockAllocator->getNumAllocatedBlocks();
  
  // Average fragmentation should be lower than before
  EXPECT_LT(avgFragmentation, 0.5);
}

// Test for PagedKVCache with advanced block allocation
class PagedKVCacheBlockAllocationTest : public ::testing::Test {
protected:
  void SetUp() override {
    type = TestType();
    kvCache = std::make_unique<PagedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, type);
    
    // Configure with advanced settings
    kvCache->configureBlockAllocatorsAdvanced(
        avgSeqLen, maxConcurrentSeqs, true, 1); // Balanced strategy
  }

  void TearDown() override {
    kvCache.reset();
  }

  const int64_t numLayers = 2;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 1024;
  const int64_t avgSeqLen = 128;
  const int64_t maxConcurrentSeqs = 4;
  TestType type;
  std::unique_ptr<PagedKVCache> kvCache;
};

// Test advanced block allocation features in PagedKVCache
TEST_F(PagedKVCacheBlockAllocationTest, AdvancedBlockAllocation) {
  // Enable auto-coalescing
  kvCache->enableAutoCoalescing(0.4);
  
  // Get metrics before operations
  auto initialMetrics = kvCache->getAllBlockMetrics();
  ASSERT_FALSE(initialMetrics.empty());
  
  // Append some data
  const int64_t batchSize = 2;
  const int64_t seqLen = 8;
  
  // Create test data (dummy arrays, not actually accessed)
  std::vector<float> keyData(batchSize * seqLen * numHeads * headDim, 1.0f);
  std::vector<float> valueData(batchSize * seqLen * numHeads * headDim, 2.0f);
  std::vector<int32_t> seqIds = {10, 20};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV pairs
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen, seqIds.data(), blockIndices.data());
  
  EXPECT_TRUE(succeeded(result));
  
  // Run advanced block coalescing
  int64_t coalesced = kvCache->runAdvancedBlockCoalescing(0.3, 5, true);
  
  // May or may not coalesce depending on the state
  // Just make sure it's not negative
  EXPECT_GE(coalesced, 0);
  
  // Get metrics after operations
  auto finalMetrics = kvCache->getAllBlockMetrics();
  ASSERT_FALSE(finalMetrics.empty());
  
  // We should see some activity in the metrics
  EXPECT_GT(finalMetrics[0].totalTokensStored, 0);
}

} // anonymous namespace 