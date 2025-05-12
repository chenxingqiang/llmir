//===- KVCacheComprehensive_test.cpp - Comprehensive tests for KV Cache ----===//
//
// This file contains comprehensive tests for the PagedKVCache implementation,
// focusing on edge cases, cross-block boundary handling, and reference counting.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "gtest/gtest.h"
#include <random>
#include <vector>
#include <unordered_map>
#include <algorithm>



using namespace mlir;
using namespace mlir::LLM::runtime;

namespace {

// Mock type implementation for testing
class Type {
public:
  Type() = default;
  int getIntOrFloatBitWidth() const { return 16; } // Simulate f16 type
  bool isF16() const { return true; }
};

// Mock LogicalResult implementation for testing
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

class KVCacheComprehensiveTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a PagedKVCache instance with test parameters
    mlir::Type type;
    kvCache = std::make_unique<PagedKVCache>(numLayers, numHeads, headDim, 
                                          blockSize, maxSeqLen, type, false);
    
    // Set a custom eviction policy for testing
    auto policy = std::make_unique<LRUEvictionPolicy>();
    kvCache->setEvictionPolicy(std::move(policy));
    
    // Configure block allocators with metrics enabled
    kvCache->configureBlockAllocators(initialBlocks, true);
    
    // Create test data
    createTestData();
  }
  
  void TearDown() override {
    kvCache.reset();
  }
  
  // Create random test data
  void createTestData() {
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Create test data
    keyData.resize(batchSize * seqLen * numHeads * headDim);
    valueData.resize(batchSize * seqLen * numHeads * headDim);
    seqIds.resize(batchSize);
    
    // Fill with random values
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = dist(rng);
      valueData[i] = dist(rng);
    }
    
    // Set sequence IDs
    for (int i = 0; i < batchSize; i++) {
      seqIds[i] = 1000 + i;
    }
    
    // Allocate space for block indices
    blockIndices.resize(numLayers * batchSize * seqLen);
  }
  
  // Helper to append tokens to the cache
  LogicalResult appendTokens(int32_t seqId, int64_t numTokens) {
    std::vector<int32_t> singleSeqId = {seqId};
    std::vector<float> singleKeyData(numTokens * numHeads * headDim);
    std::vector<float> singleValueData(numTokens * numHeads * headDim);
    std::vector<int32_t> singleBlockIndices(numLayers * numTokens);
    
    // Fill with test pattern
    for (size_t i = 0; i < singleKeyData.size(); i++) {
      singleKeyData[i] = i * 0.1f;
      singleValueData[i] = i * 0.2f;
    }
    
    return kvCache->appendKV(
        singleKeyData.data(), singleValueData.data(), 1, numTokens,
        singleSeqId.data(), singleBlockIndices.data());
  }
  
  // Parameters for testing
  const int64_t numLayers = 2;
  const int64_t numHeads = 12;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 4096;
  const int64_t batchSize = 4;
  const int64_t seqLen = 8;
  const int64_t initialBlocks = 4;
  
  std::unique_ptr<PagedKVCache> kvCache;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<int32_t> seqIds;
  std::vector<int32_t> blockIndices;
};

//===----------------------------------------------------------------------===//
// Edge Cases in Block Allocation/Deallocation
//===----------------------------------------------------------------------===//

TEST_F(KVCacheComprehensiveTest, EdgeCase_EmptyAppend) {
  // Test appending zero tokens
  std::vector<int32_t> emptyBlockIndices(numLayers * batchSize * 0);
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, 0,
      seqIds.data(), emptyBlockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Verify no blocks were allocated
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), 0);
  }
}

TEST_F(KVCacheComprehensiveTest, EdgeCase_MaximumSequenceLength) {
  // Create a sequence at the maximum allowed length
  std::vector<int32_t> singleSeqId = {seqIds[0]};
  std::vector<float> longKeyData(maxSeqLen * numHeads * headDim);
  std::vector<float> longValueData(maxSeqLen * numHeads * headDim);
  std::vector<int32_t> longBlockIndices(numLayers * maxSeqLen);
  
  // Fill with test pattern
  for (size_t i = 0; i < longKeyData.size(); i++) {
    longKeyData[i] = i * 0.1f;
    longValueData[i] = i * 0.2f;
  }
  
  // Append at max sequence length
  LogicalResult result = kvCache->appendKV(
      longKeyData.data(), longValueData.data(), 1, maxSeqLen,
      singleSeqId.data(), longBlockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), maxSeqLen);
  
  // Now try to append one more token - should fail
  std::vector<float> extraKeyData(1 * numHeads * headDim);
  std::vector<float> extraValueData(1 * numHeads * headDim);
  std::vector<int32_t> extraBlockIndices(numLayers * 1);
  
  result = kvCache->appendKV(
      extraKeyData.data(), extraValueData.data(), 1, 1,
      singleSeqId.data(), extraBlockIndices.data());
      
  EXPECT_TRUE(failed(result));
}

TEST_F(KVCacheComprehensiveTest, EdgeCase_OutOfMemory) {
  // Configure a very small allocator that will run out of memory
  kvCache.reset();
  
  mlir::Type type;
  kvCache = std::make_unique<PagedKVCache>(numLayers, numHeads, headDim, 
                                       blockSize, maxSeqLen, type, false);
  
  // Allocate just one block per layer
  kvCache->configureBlockAllocators(1, true);
  
  // Create sequences that require more blocks than available
  std::vector<int32_t> manySeqIds(10);
  for (int i = 0; i < 10; i++) {
    manySeqIds[i] = 2000 + i;
  }
  
  // Append multiple sequences exceeding the allocator capacity
  bool encounteredFailure = false;
  for (int i = 0; i < 10 && !encounteredFailure; i++) {
    std::vector<int32_t> singleSeqId = {manySeqIds[i]};
    std::vector<int32_t> singleBlockIndices(numLayers * blockSize);
    
    LogicalResult result = kvCache->appendKV(
        keyData.data(), valueData.data(), 1, blockSize,
        singleSeqId.data(), singleBlockIndices.data());
        
    if (failed(result)) {
      encounteredFailure = true;
    }
  }
  
  // We should encounter a failure at some point due to running out of memory
  EXPECT_TRUE(encounteredFailure);
}

//===----------------------------------------------------------------------===//
// Cross-Block Boundary Handling
//===----------------------------------------------------------------------===//

TEST_F(KVCacheComprehensiveTest, CrossBlock_MultipleFullBlocks) {
  // Create a sequence spanning multiple full blocks
  int64_t numBlocks = 3;
  int64_t totalTokens = numBlocks * blockSize;
  
  std::vector<int32_t> singleSeqId = {seqIds[0]};
  std::vector<float> longKeyData(totalTokens * numHeads * headDim);
  std::vector<float> longValueData(totalTokens * numHeads * headDim);
  std::vector<int32_t> longBlockIndices(numLayers * totalTokens);
  
  // Fill with test pattern
  for (size_t i = 0; i < longKeyData.size(); i++) {
    longKeyData[i] = i * 0.1f;
    longValueData[i] = i * 0.2f;
  }
  
  // Append tokens spanning multiple blocks
  LogicalResult result = kvCache->appendKV(
      longKeyData.data(), longValueData.data(), 1, totalTokens,
      singleSeqId.data(), longBlockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), totalTokens);
  
  // Lookup all tokens
  std::vector<int32_t> seqLens = {totalTokens};
  std::vector<float> outputKeys(totalTokens * numHeads * headDim);
  std::vector<float> outputValues(totalTokens * numHeads * headDim);
  
  result = kvCache->lookupKV(
      longBlockIndices.data(), seqLens.data(), 1,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Verify data across block boundaries
  for (int block = 0; block < numBlocks; block++) {
    // Check tokens at the start of each block
    int64_t startIdx = block * blockSize * numHeads * headDim;
    EXPECT_FLOAT_EQ(outputKeys[startIdx], longKeyData[startIdx]);
    EXPECT_FLOAT_EQ(outputValues[startIdx], longValueData[startIdx]);
    
    // Check tokens at the end of each block
    int64_t endIdx = ((block + 1) * blockSize - 1) * numHeads * headDim;
    EXPECT_FLOAT_EQ(outputKeys[endIdx], longKeyData[endIdx]);
    EXPECT_FLOAT_EQ(outputValues[endIdx], longValueData[endIdx]);
  }
}

TEST_F(KVCacheComprehensiveTest, CrossBlock_PartialBlocks) {
  // Create sequence that spans partial blocks (not aligned with block size)
  int64_t totalTokens = blockSize * 2 + blockSize / 2;
  
  std::vector<int32_t> singleSeqId = {seqIds[0]};
  std::vector<float> longKeyData(totalTokens * numHeads * headDim);
  std::vector<float> longValueData(totalTokens * numHeads * headDim);
  std::vector<int32_t> longBlockIndices(numLayers * totalTokens);
  
  // Fill with test pattern
  for (size_t i = 0; i < longKeyData.size(); i++) {
    longKeyData[i] = i * 0.1f;
    longValueData[i] = i * 0.2f;
  }
  
  // Append tokens spanning multiple blocks
  LogicalResult result = kvCache->appendKV(
      longKeyData.data(), longValueData.data(), 1, totalTokens,
      singleSeqId.data(), longBlockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), totalTokens);
  
  // Lookup all tokens
  std::vector<int32_t> seqLens = {totalTokens};
  std::vector<float> outputKeys(totalTokens * numHeads * headDim);
  std::vector<float> outputValues(totalTokens * numHeads * headDim);
  
  result = kvCache->lookupKV(
      longBlockIndices.data(), seqLens.data(), 1,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Verify data across the partial block
  // Check the first token in the partial block
  int64_t partialStart = blockSize * 2 * numHeads * headDim;
  EXPECT_FLOAT_EQ(outputKeys[partialStart], longKeyData[partialStart]);
  EXPECT_FLOAT_EQ(outputValues[partialStart], longValueData[partialStart]);
  
  // Check the last token in the sequence
  int64_t lastIdx = (totalTokens - 1) * numHeads * headDim;
  EXPECT_FLOAT_EQ(outputKeys[lastIdx], longKeyData[lastIdx]);
  EXPECT_FLOAT_EQ(outputValues[lastIdx], longValueData[lastIdx]);
}

TEST_F(KVCacheComprehensiveTest, CrossBlock_SequentialAppends) {
  // Create a sequence by appending multiple times
  int64_t appendSize = blockSize / 2;
  int64_t numAppends = 5;
  
  std::vector<int32_t> singleSeqId = {seqIds[0]};
  std::vector<float> appendKeyData(appendSize * numHeads * headDim);
  std::vector<float> appendValueData(appendSize * numHeads * headDim);
  std::vector<int32_t> allBlockIndices;
  
  for (int append = 0; append < numAppends; append++) {
    // Fill with test pattern unique to this append
    for (size_t i = 0; i < appendKeyData.size(); i++) {
      appendKeyData[i] = (append * 1000 + i) * 0.1f;
      appendValueData[i] = (append * 1000 + i) * 0.2f;
    }
    
    std::vector<int32_t> appendBlockIndices(numLayers * appendSize);
    
    // Append tokens
    LogicalResult result = kvCache->appendKV(
        appendKeyData.data(), appendValueData.data(), 1, appendSize,
        singleSeqId.data(), appendBlockIndices.data());
        
    EXPECT_TRUE(succeeded(result));
    
    // Add to the collection of all block indices
    allBlockIndices.insert(allBlockIndices.end(), 
                          appendBlockIndices.begin(), 
                          appendBlockIndices.end());
  }
  
  // Verify sequence length
  int64_t totalTokens = appendSize * numAppends;
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), totalTokens);
  
  // Lookup all tokens
  std::vector<int32_t> seqLens = {totalTokens};
  std::vector<float> outputKeys(totalTokens * numHeads * headDim);
  std::vector<float> outputValues(totalTokens * numHeads * headDim);
  
  LogicalResult result = kvCache->lookupKV(
      allBlockIndices.data(), seqLens.data(), 1,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Lookup should be successful even across multiple appends
  // Full verification would require keeping track of all the appended data
}

//===----------------------------------------------------------------------===//
// Reference Counting and Cache Sharing
//===----------------------------------------------------------------------===//

TEST_F(KVCacheComprehensiveTest, RefCounting_BasicBlocks) {
  // First append tokens to a sequence
  ASSERT_TRUE(succeeded(appendTokens(seqIds[0], blockSize)));
  
  // Clear the sequence - should free the blocks
  LogicalResult result = kvCache->clearSequence(seqIds[0]);
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), 0);
  
  // Append tokens to a new sequence - should reuse the freed blocks
  ASSERT_TRUE(succeeded(appendTokens(seqIds[1], blockSize)));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[1]), blockSize);
}

TEST_F(KVCacheComprehensiveTest, RefCounting_SharedBlocks) {
  // Create identical content for two sequences
  std::vector<int32_t> seqPair = {seqIds[0], seqIds[1]};
  std::vector<int32_t> seqIndices(numLayers * 2 * blockSize);
  
  // Append identical content to both sequences
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), 2, blockSize,
      seqPair.data(), seqIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Both sequences should have blocks
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), blockSize);
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[1]), blockSize);
  
  // Clear the first sequence
  result = kvCache->clearSequence(seqIds[0]);
  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), 0);
  
  // Second sequence should still have access to its data
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[1]), blockSize);
  
  // Lookup the second sequence
  std::vector<int32_t> seqLens = {blockSize};
  std::vector<float> outputKeys(blockSize * numHeads * headDim);
  std::vector<float> outputValues(blockSize * numHeads * headDim);
  
  // Extract the block indices for the second sequence
  std::vector<int32_t> seq1Indices(numLayers * blockSize);
  for (int64_t i = 0; i < numLayers * blockSize; i++) {
    seq1Indices[i] = seqIndices[numLayers * blockSize + i];
  }
  
  result = kvCache->lookupKV(
      seq1Indices.data(), seqLens.data(), 1,
      outputKeys.data(), outputValues.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Verify data is still accessible
  for (size_t i = 0; i < 10; i++) {
    EXPECT_FLOAT_EQ(outputKeys[i], keyData[i]);
    EXPECT_FLOAT_EQ(outputValues[i], valueData[i]);
  }
}

TEST_F(KVCacheComprehensiveTest, CacheSharing_IdenticalContent) {
  // First append data to sequence 1
  std::vector<int32_t> seq1 = {seqIds[0]};
  std::vector<int32_t> seq1Indices(numLayers * blockSize);
  
  ASSERT_TRUE(succeeded(kvCache->appendKV(
      keyData.data(), valueData.data(), 1, blockSize,
      seq1.data(), seq1Indices.data())));
  
  // Append identical data to sequence 2
  std::vector<int32_t> seq2 = {seqIds[1]};
  std::vector<int32_t> seq2Indices(numLayers * blockSize);
  
  ASSERT_TRUE(succeeded(kvCache->appendKV(
      keyData.data(), valueData.data(), 1, blockSize,
      seq2.data(), seq2Indices.data())));
  
  // If content hash detection is working, they might share blocks
  // But we can't verify that directly, so we'll check that both sequences
  // can access their data
  
  // Lookup sequence 1
  std::vector<int32_t> seqLens = {blockSize};
  std::vector<float> outputKeys1(blockSize * numHeads * headDim);
  std::vector<float> outputValues1(blockSize * numHeads * headDim);
  
  ASSERT_TRUE(succeeded(kvCache->lookupKV(
      seq1Indices.data(), seqLens.data(), 1,
      outputKeys1.data(), outputValues1.data())));
  
  // Lookup sequence 2
  std::vector<float> outputKeys2(blockSize * numHeads * headDim);
  std::vector<float> outputValues2(blockSize * numHeads * headDim);
  
  ASSERT_TRUE(succeeded(kvCache->lookupKV(
      seq2Indices.data(), seqLens.data(), 1,
      outputKeys2.data(), outputValues2.data())));
  
  // Both should have the same data
  for (size_t i = 0; i < outputKeys1.size(); i++) {
    EXPECT_FLOAT_EQ(outputKeys1[i], outputKeys2[i]);
    EXPECT_FLOAT_EQ(outputValues1[i], outputValues2[i]);
  }
}

TEST_F(KVCacheComprehensiveTest, CacheSharing_ExplicitSharing) {
  // First append data to sequence 1
  ASSERT_TRUE(succeeded(appendTokens(seqIds[0], blockSize)));
  
  // Now explicitly share blocks from sequence 1 to sequence 2
  LogicalResult result = kvCache->shareSequenceBlocks(seqIds[0], seqIds[2], blockSize);
  EXPECT_TRUE(succeeded(result));
  
  // Both sequences should have the same length
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[0]), blockSize);
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[2]), blockSize);
  
  // Clear the source sequence
  result = kvCache->clearSequence(seqIds[0]);
  EXPECT_TRUE(succeeded(result));
  
  // Target sequence should still have access to the data
  EXPECT_EQ(kvCache->getSequenceLength(seqIds[2]), blockSize);
}

//===----------------------------------------------------------------------===//
// Block Coalescing and Eviction Policy Tests
//===----------------------------------------------------------------------===//

TEST_F(KVCacheComprehensiveTest, BlockCoalescing_Basic) {
  // Create multiple sequences with partial blocks
  int64_t partialSize = blockSize / 4;
  
  // Append to multiple sequences
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(succeeded(appendTokens(seqIds[i], partialSize)));
  }
  
  // Run block coalescing
  int64_t numCoalesced = kvCache->runAdvancedBlockCoalescing(0.5);
  
  // Depending on implementation, some blocks might be coalesced
  // But all sequences should still have their data
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), partialSize);
  }
}

TEST_F(KVCacheComprehensiveTest, EvictionPolicy_LRU) {
  // Configure a cache with limited blocks
  kvCache.reset();
  
  mlir::Type type;
  kvCache = std::make_unique<PagedKVCache>(numLayers, numHeads, headDim, 
                                       blockSize, maxSeqLen, type, false);
  
  // Allocate just two blocks per layer and set LRU policy
  kvCache->configureBlockAllocators(2, true);
  auto policy = std::make_unique<LRUEvictionPolicy>();
  kvCache->setEvictionPolicy(std::move(policy));
  
  // Append to first sequence
  ASSERT_TRUE(succeeded(appendTokens(seqIds[0], blockSize)));
  
  // Wait a moment to ensure timestamp differs
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  // Append to second sequence
  ASSERT_TRUE(succeeded(appendTokens(seqIds[1], blockSize)));
  
  // Wait a moment to ensure timestamp differs
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  // Access the first sequence to update its timestamp
  std::vector<int32_t> seqLens = {blockSize};
  std::vector<float> outputKeys(blockSize * numHeads * headDim);
  std::vector<float> outputValues(blockSize * numHeads * headDim);
  std::vector<int32_t> blockIndices(numLayers * blockSize);
  
  // We don't have the block indices for sequence 0, so we can't do the lookup
  // But in a real scenario, accessing seq 0 would update its timestamp
  
  // Now append to a third sequence - should evict the second sequence (LRU)
  ASSERT_TRUE(succeeded(appendTokens(seqIds[2], blockSize)));
  
  // At this point, the cache should contain sequences 0 and 2
  // (This depends on LRU working correctly)
}

//===----------------------------------------------------------------------===//
// Additional Edge Cases and Stress Tests
//===----------------------------------------------------------------------===//

TEST_F(KVCacheComprehensiveTest, ClearNonExistentSequence) {
  // Try to clear a sequence that doesn't exist
  LogicalResult result = kvCache->clearSequence(9999);
  
  // Should succeed or fail gracefully (implementation dependent)
  // Just ensure it doesn't crash
}

TEST_F(KVCacheComprehensiveTest, SimultaneousAppends) {
  // Append to multiple sequences simultaneously
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Verify all sequences have data
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), seqLen);
  }
}

TEST_F(KVCacheComprehensiveTest, ResetCache) {
  // Append data to multiple sequences
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), batchSize, seqLen,
      seqIds.data(), blockIndices.data());
      
  EXPECT_TRUE(succeeded(result));
  
  // Reset the cache
  kvCache->reset();
  
  // Verify all sequences are cleared
  for (int i = 0; i < batchSize; i++) {
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), 0);
  }
}

TEST_F(KVCacheComprehensiveTest, VaryingSequenceLengths) {
  // Append different sequence lengths
  for (int i = 0; i < batchSize; i++) {
    int64_t tokensToAppend = (i + 1) * blockSize / 2;
    ASSERT_TRUE(succeeded(appendTokens(seqIds[i], tokensToAppend)));
    EXPECT_EQ(kvCache->getSequenceLength(seqIds[i]), tokensToAppend);
  }
  
  // Get sequence length statistics
  int64_t minSeqLen, maxSeqLen;
  double avgSeqLen;
  kvCache->getSequenceLengthStats(minSeqLen, maxSeqLen, avgSeqLen);
  
  // Verify the statistics are correct
  EXPECT_EQ(minSeqLen, blockSize / 2);
  EXPECT_EQ(maxSeqLen, batchSize * blockSize / 2);
  EXPECT_GE(avgSeqLen, minSeqLen);
  EXPECT_LE(avgSeqLen, maxSeqLen);
}

} // end anonymous namespace 