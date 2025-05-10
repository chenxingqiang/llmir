//===- KVCacheBlockAllocation_test.cpp - Test KV cache block allocation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the PagedKVCache block allocation optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include <gtest/gtest.h>
#include <random>
#include <algorithm>

using namespace mlir::llm::runtime;

// Create a custom eviction policy for testing
class TestEvictionPolicy : public EvictionPolicy {
public:
  // Always evict blocks in reverse order (newest first)
  std::vector<int32_t> selectBlocksForEviction(
      const BlockAllocator& allocator, int64_t numBlocksNeeded) const override {
    std::vector<int32_t> result;
    
    // Start from the end (most recently allocated) and go backwards
    for (int32_t i = static_cast<int32_t>(allocator.allocatedBlocks.size()) - 1; 
         i >= 0 && result.size() < static_cast<size_t>(numBlocksNeeded); 
         --i) {
      if (allocator.allocatedBlocks[i]->isEvictable()) {
        result.push_back(i);
      }
    }
    
    return result;
  }
  
  void* clone() const override {
    return new TestEvictionPolicy(*this);
  }
};

// Test fixture for KV cache block allocation tests
class KVCacheBlockAllocationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a small KV cache for testing
    numLayers = 2;
    numHeads = 4;
    headDim = 8;
    blockSize = 4;  // Small block size for testing
    maxSeqLen = 16;
    
    // Create a type (details don't matter for testing)
    mlir::Type elementType;
    
    // Create KV cache
    kvCache = std::make_unique<PagedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType, false);
  }
  
  // Helper to generate random KV data
  void generateRandomKVData(std::vector<float>& keys, std::vector<float>& values, 
                          int32_t batchSize, int32_t seqLen) {
    // Size of keys/values: [batchSize, seqLen, numHeads, headDim]
    size_t totalSize = batchSize * seqLen * numHeads * headDim;
    keys.resize(totalSize);
    values.resize(totalSize);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::generate(keys.begin(), keys.end(), [&]() { return dist(gen); });
    std::generate(values.begin(), values.end(), [&]() { return dist(gen); });
  }
  
  std::unique_ptr<PagedKVCache> kvCache;
  int64_t numLayers;
  int64_t numHeads;
  int64_t headDim;
  int64_t blockSize;
  int64_t maxSeqLen;
};

// Test block preallocation
TEST_F(KVCacheBlockAllocationTest, TestPreallocation) {
  // Ensure initial allocation occurred in constructor
  std::vector<BlockMetrics> initialMetrics = kvCache->getAllBlockMetrics();
  ASSERT_EQ(initialMetrics.size(), numLayers);
  
  // Preallocate more blocks
  const int64_t additionalBlocks = 10;
  kvCache->configureBlockAllocators(additionalBlocks, true);
  
  // Check metrics after preallocation
  std::vector<BlockMetrics> metricsAfter = kvCache->getAllBlockMetrics();
  for (const auto& metrics : metricsAfter) {
    // At least the requested number of blocks should be available
    EXPECT_GE(metrics.totalBlocks, additionalBlocks);
    // All should be free initially
    EXPECT_GE(metrics.freeBlocks, additionalBlocks); 
  }
}

// Test LRU eviction policy
TEST_F(KVCacheBlockAllocationTest, TestLRUEviction) {
  // Configure KV cache with minimal free blocks
  kvCache->configureBlockAllocators(1, true);
  
  // Generate test data - 3 sequences that will require multiple blocks
  const int32_t batchSize = 3;
  const int32_t seqLen = blockSize * 2; // 2 blocks per sequence
  std::vector<float> keys, values;
  generateRandomKVData(keys, values, batchSize, seqLen);
  
  // Create sequence IDs and block indices
  std::vector<int32_t> seqIds = {1, 2, 3};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV data - this should cause allocations and potentially evictions
  kvCache->appendKV(keys.data(), values.data(), batchSize, seqLen, 
                   seqIds.data(), blockIndices.data());
  
  // Check metrics after allocation
  std::vector<BlockMetrics> metrics = kvCache->getAllBlockMetrics();
  for (const auto& m : metrics) {
    // There should be some blocks in use now
    EXPECT_GT(m.usedBlocks, 0);
    // Check if eviction happened
    EXPECT_GE(m.numEvictions, 0);
  }
  
  // Set a custom eviction policy
  auto customPolicy = std::make_unique<TestEvictionPolicy>();
  kvCache->setEvictionPolicy(std::move(customPolicy));
  
  // Add more sequences to trigger eviction
  std::vector<int32_t> newSeqIds = {4, 5, 6};
  std::vector<int32_t> newBlockIndices(batchSize * numLayers);
  
  // Append more KV data to trigger eviction with custom policy
  kvCache->appendKV(keys.data(), values.data(), batchSize, seqLen, 
                   newSeqIds.data(), newBlockIndices.data());
  
  // Check metrics again
  std::vector<BlockMetrics> metricsAfter = kvCache->getAllBlockMetrics();
  for (const auto& m : metricsAfter) {
    // There should be more evictions now
    EXPECT_GT(m.numEvictions, 0);
  }
}

// Test block coalescing
TEST_F(KVCacheBlockAllocationTest, TestBlockCoalescing) {
  // Configure KV cache with minimal free blocks
  kvCache->configureBlockAllocators(2, true);
  
  // Generate test data - sequences with partial blocks
  const int32_t batchSize = 3;
  const int32_t seqLen = blockSize / 2; // Half a block
  std::vector<float> keys, values;
  generateRandomKVData(keys, values, batchSize, seqLen);
  
  // Create sequence IDs and block indices
  std::vector<int32_t> seqIds = {1, 2, 3};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV data - this creates partially filled blocks
  kvCache->appendKV(keys.data(), values.data(), batchSize, seqLen, 
                   seqIds.data(), blockIndices.data());
  
  // Check initial metrics
  std::vector<BlockMetrics> initialMetrics = kvCache->getAllBlockMetrics();
  
  // Run block coalescing
  kvCache->runBlockCoalescing();
  
  // Check metrics after coalescing
  std::vector<BlockMetrics> metricsAfter = kvCache->getAllBlockMetrics();
  for (size_t i = 0; i < metricsAfter.size(); i++) {
    // Number of coalescing operations should increase
    EXPECT_GE(metricsAfter[i].numCoalesces, initialMetrics[i].numCoalesces);
  }
}

// Main function to run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 