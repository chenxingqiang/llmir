//===- KVCacheAttentionOpt_test.cpp - Tests for KV Cache with attention opts -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::llm::runtime;

namespace {

// A simple Type implementation for testing
struct TestType : public Type {
  int getIntOrFloatBitWidth() const { return 16; }  // f16
};

// Test fixture for KV cache with attention optimization tests
class KVCacheAttentionOptTest : public ::testing::Test {
protected:
  void SetUp() override {
    type = TestType();
    
    // Initialize KV cache
    kvCache = std::make_unique<PagedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, type);
    kvCache->configureBlockAllocatorsAdvanced(
        avgSeqLen, maxConcurrentSeqs, true, 1);
    
    // Create and configure attention
    AttentionConfig config;
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    config.maskType = AttentionMaskType::CAUSAL;
    config.optLevel = AttentionOptLevel::BASIC;
    config.fuseSoftmax = true;
    
    kvCache->configureAttentionOpt(config);
  }

  void TearDown() override {
    kvCache.reset();
  }
  
  // Create test data for attention computation
  void createTestData() {
    // Initialize query, key, value data
    queryData.resize(batchSize * seqLen * numHeads * headDim, 0.5f);
    keyData.resize(batchSize * seqLen * numHeads * headDim, 0.5f);
    valueData.resize(batchSize * seqLen * numHeads * headDim, 0.5f);
    outputData.resize(batchSize * seqLen * numHeads * headDim, 0.0f);
    
    // Fill with some test pattern
    for (size_t i = 0; i < queryData.size(); i++) {
      queryData[i] = 0.01f * (i % 100);
    }
    
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = 0.01f * ((i + 27) % 100);
    }
    
    for (size_t i = 0; i < valueData.size(); i++) {
      valueData[i] = 0.01f * ((i + 53) % 100);
    }
  }

  const int64_t numLayers = 2;
  const int64_t numHeads = 8;
  const int64_t headDim = 64;
  const int64_t blockSize = 16;
  const int64_t maxSeqLen = 1024;
  const int64_t avgSeqLen = 128;
  const int64_t maxConcurrentSeqs = 4;
  
  // Test data parameters
  const int64_t batchSize = 2;
  const int64_t seqLen = 4;
  
  TestType type;
  std::unique_ptr<PagedKVCache> kvCache;
  
  // Test data
  std::vector<float> queryData;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData;
};

// Test basic integration of attention optimizations with KV cache
TEST_F(KVCacheAttentionOptTest, BasicAttentionIntegration) {
  createTestData();
  
  // Add some KV data to the cache
  std::vector<int32_t> seqIds = {10, 20};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV pairs to cache
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), 
      batchSize, seqLen, seqIds.data(), blockIndices.data());
  
  ASSERT_TRUE(succeeded(result));
  
  // Now compute attention using the queries and cached KV
  std::vector<int32_t> seqLens = {seqLen, seqLen};
  
  result = kvCache->computeAttention(
      outputData.data(),
      queryData.data(),
      blockIndices.data(),
      seqLens.data(),
      batchSize,
      seqLen);
  
  ASSERT_TRUE(succeeded(result));
}

// Test different attention mask types with KV cache
TEST_F(KVCacheAttentionOptTest, DifferentMaskTypes) {
  createTestData();
  
  // Add KV data to the cache
  std::vector<int32_t> seqIds = {10, 20};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  std::vector<int32_t> seqLens = {seqLen, seqLen};
  
  // Append KV pairs to cache
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), 
      batchSize, seqLen, seqIds.data(), blockIndices.data());
  
  ASSERT_TRUE(succeeded(result));
  
  // Test causal masking
  {
    AttentionConfig config;
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.maskType = AttentionMaskType::CAUSAL;
    config.optLevel = AttentionOptLevel::BASIC;
    
    kvCache->configureAttentionOpt(config);
    
    result = kvCache->computeAttention(
        outputData.data(),
        queryData.data(),
        blockIndices.data(),
        seqLens.data(),
        batchSize,
        seqLen);
    
    ASSERT_TRUE(succeeded(result));
  }
  
  // Test sliding window masking
  {
    AttentionConfig config;
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.maskType = AttentionMaskType::SLIDING_WINDOW;
    config.windowSize = 2;
    config.optLevel = AttentionOptLevel::BASIC;
    
    kvCache->configureAttentionOpt(config);
    
    result = kvCache->computeAttention(
        outputData.data(),
        queryData.data(),
        blockIndices.data(),
        seqLens.data(),
        batchSize,
        seqLen);
    
    ASSERT_TRUE(succeeded(result));
  }
}

// Test attention with block eviction
TEST_F(KVCacheAttentionOptTest, AttentionWithBlockEviction) {
  createTestData();
  
  // Set up eviction policy
  kvCache->setEvictionPolicy(std::make_unique<LRUEvictionPolicy>());
  
  // Create multiple sequences that will exceed cache capacity
  const int numSequences = 10;
  std::vector<int32_t> seqIds(numSequences);
  std::vector<int32_t> blockIndices(numSequences * numLayers);
  
  for (int i = 0; i < numSequences; i++) {
    seqIds[i] = 100 + i;
    
    // Append KV pairs to cache
    LogicalResult result = kvCache->appendKV(
        keyData.data(), valueData.data(), 
        1, seqLen, &seqIds[i], &blockIndices[i * numLayers]);
    
    ASSERT_TRUE(succeeded(result));
    
    // Compute attention for each sequence
    int32_t currentSeqLen = seqLen;
    
    result = kvCache->computeAttention(
        outputData.data(),
        queryData.data(),
        &blockIndices[i * numLayers],
        &currentSeqLen,
        1,
        seqLen);
    
    ASSERT_TRUE(succeeded(result));
  }
  
  // Check metrics to verify that some eviction occurred
  auto metrics = kvCache->getAllBlockMetrics();
  bool evictionOccurred = false;
  
  for (const auto& m : metrics) {
    if (m.numBlocksEvicted > 0) {
      evictionOccurred = true;
      break;
    }
  }
  
  // This test may be flaky depending on the initial cache size
  // In a real implementation, we would set up a more controlled environment
  EXPECT_TRUE(evictionOccurred);
}

} // anonymous namespace 