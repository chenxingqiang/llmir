//===- AttentionOpt_test.cpp - Tests for attention optimizations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/KVCache.h"
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

// Test fixture for attention optimization tests
class AttentionOptTest : public ::testing::Test {
protected:
  void SetUp() override {
    type = TestType();
    
    // Create default attention config
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    config.maskType = AttentionMaskType::CAUSAL;
    config.optLevel = AttentionOptLevel::BASIC;
    config.fuseSoftmax = true;
    
    // Create attention implementation
    attImpl = createAttentionImpl(config, type);
    
    // Initialize KV cache
    kvCache = std::make_unique<PagedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, type);
    kvCache->configureBlockAllocatorsAdvanced(
        avgSeqLen, maxConcurrentSeqs, true, 1);
    
    // Configure attention in KV cache
    kvCache->configureAttentionOpt(config);
  }

  void TearDown() override {
    attImpl.reset();
    kvCache.reset();
  }
  
  // Create test data for attention computation
  void createTestData() {
    // Initialize query, key, value data
    queryData.resize(batchSize * seqLen * numHeads * headDim, 0.5f);
    keyData.resize(batchSize * contextLen * numHeads * headDim, 0.5f);
    valueData.resize(batchSize * contextLen * numHeads * headDim, 0.5f);
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
  const int64_t contextLen = 8;
  
  TestType type;
  AttentionConfig config;
  std::unique_ptr<AttentionImpl> attImpl;
  std::unique_ptr<PagedKVCache> kvCache;
  
  // Test data
  std::vector<float> queryData;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData;
};

// Test creating different attention implementations
TEST_F(AttentionOptTest, CreateAttentionImpl) {
  // Test creating fused softmax attention
  {
    AttentionConfig config;
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.maskType = AttentionMaskType::CAUSAL;
    config.optLevel = AttentionOptLevel::BASIC;
    
    auto impl = createAttentionImpl(config, type);
    ASSERT_NE(impl, nullptr);
    
    // Check if it's the correct type (using dynamic_cast)
    auto fusedImpl = dynamic_cast<FusedSoftmaxAttentionImpl*>(impl.get());
    EXPECT_NE(fusedImpl, nullptr);
  }
  
  // Test creating sliding window attention
  {
    AttentionConfig config;
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.maskType = AttentionMaskType::SLIDING_WINDOW;
    config.windowSize = 256;
    config.optLevel = AttentionOptLevel::BASIC;
    
    auto impl = createAttentionImpl(config, type);
    ASSERT_NE(impl, nullptr);
    
    // Check if it's the correct type
    auto slidingImpl = dynamic_cast<SlidingWindowAttentionImpl*>(impl.get());
    EXPECT_NE(slidingImpl, nullptr);
  }
}

// Test basic attention computation
TEST_F(AttentionOptTest, BasicAttentionComputation) {
  createTestData();
  
  // Run attention computation
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Since this is just a stub implementation, we can only verify
  // that the function completed without errors
  // In a real test, we would check the output values
}

// Test paged attention computation with KV cache
TEST_F(AttentionOptTest, PagedAttentionComputation) {
  createTestData();
  
  // First, append KV pairs to the cache
  std::vector<int32_t> seqIds = {10, 20};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV pairs to cache
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), 
      batchSize, contextLen, seqIds.data(), blockIndices.data());
  
  ASSERT_TRUE(succeeded(result));
  
  // Now compute attention using the KV cache
  std::vector<int32_t> seqLens = {contextLen, contextLen};
  
  result = kvCache->computeAttention(
      outputData.data(),
      queryData.data(),
      blockIndices.data(),
      seqLens.data(),
      batchSize,
      seqLen);
  
  ASSERT_TRUE(succeeded(result));
  
  // In a real test, we would validate output values
}

// Test different attention mask types
TEST_F(AttentionOptTest, AttentionMaskTypes) {
  createTestData();
  
  // Test causal masking
  {
    config.maskType = AttentionMaskType::CAUSAL;
    auto impl = createAttentionImpl(config, type);
    
    impl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // In a real test, we would check that future tokens are properly masked
  }
  
  // Test sliding window masking
  {
    config.maskType = AttentionMaskType::SLIDING_WINDOW;
    config.windowSize = 2;
    auto impl = createAttentionImpl(config, type);
    
    impl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // In a real test, we would check that tokens outside the window are masked
  }
}

// Test attention optimization levels
TEST_F(AttentionOptTest, OptimizationLevels) {
  createTestData();
  
  // Test basic optimization level
  {
    config.optLevel = AttentionOptLevel::BASIC;
    auto impl = createAttentionImpl(config, type);
    
    impl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
  }
  
  // Test advanced optimization level (if implemented)
  {
    config.optLevel = AttentionOptLevel::ADVANCED;
    auto impl = createAttentionImpl(config, type);
    
    impl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
  }
}

// Test for PrunedAttention
TEST_F(AttentionOptTest, PrunedAttention) {
  createTestData();
  
  // Test different pruning strategies
  std::vector<AttentionPruningStrategy> strategies = {
    AttentionPruningStrategy::THRESHOLD,
    AttentionPruningStrategy::TOP_K,
    AttentionPruningStrategy::BLOCK_SPARSE
  };
  
  for (auto strategy : strategies) {
    // Configure for pruned attention
    AttentionConfig pruningConfig = config;
    pruningConfig.pruningStrategy = strategy;
    
    // Set strategy-specific parameters
    switch (strategy) {
      case AttentionPruningStrategy::THRESHOLD:
        pruningConfig.pruningThreshold = 0.01f;
        break;
      case AttentionPruningStrategy::TOP_K:
        pruningConfig.pruningTopK = contextLen / 4;  // Keep 25% of tokens
        break;
      case AttentionPruningStrategy::BLOCK_SPARSE:
        pruningConfig.pruningBlockSize = 4;
        pruningConfig.pruningRatio = 0.5f;  // Prune 50% of blocks
        break;
      default:
        break;
    }
    
    // Create implementation with this pruning strategy
    auto prunedImpl = createAttentionImpl(pruningConfig, type);
    
    // Clear output
    std::fill(outputData.begin(), outputData.end(), 0.0f);
    
    // Run computation
    prunedImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // Check for valid output
    bool allZeros = true;
    bool hasNaNs = false;
    
    for (float val : outputData) {
      if (val != 0.0f) allZeros = false;
      if (std::isnan(val)) hasNaNs = true;
    }
    
    EXPECT_FALSE(allZeros);
    EXPECT_FALSE(hasNaNs);
    
    // Also check that we're not getting the same result as non-pruned attention
    // (at least some of the values should be different)
    std::vector<float> regularOutput(outputData.size(), 0.0f);
    attImpl->compute(
        regularOutput.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // Check that at least some values are different
    int diffCount = 0;
    for (size_t i = 0; i < outputData.size(); i++) {
      if (std::abs(outputData[i] - regularOutput[i]) > 1e-4f) {
        diffCount++;
      }
    }
    
    // We expect some values to be different due to pruning
    EXPECT_GT(diffCount, 0);
  }
}

// Test Multi-Query Attention
TEST_F(AttentionOptTest, MultiQueryAttention) {
  createTestData();
  
  // Configure for Multi-Query Attention
  config.variant = AttentionVariant::MULTI_QUERY;
  config.configureForVariant(AttentionVariant::MULTI_QUERY);
  
  // Create new implementation
  attImpl = createAttentionImpl(config, type);
  
  // Since MQA needs different key/value shape, we need to reshape
  std::vector<float> mqaKeyData(batchSize * contextLen * 1 * headDim, 0.0f);
  std::vector<float> mqaValueData(batchSize * contextLen * 1 * headDim, 0.0f);
  
  // Average the keys/values across heads for this test
  for (int64_t b = 0; b < batchSize; b++) {
    for (int64_t c = 0; c < contextLen; c++) {
      for (int64_t d = 0; d < headDim; d++) {
        float keySum = 0.0f;
        float valueSum = 0.0f;
        
        for (int64_t h = 0; h < numHeads; h++) {
          int64_t srcIdx = ((b * numHeads + h) * contextLen + c) * headDim + d;
          keySum += keyData[srcIdx];
          valueSum += valueData[srcIdx];
        }
        
        int64_t dstIdx = (b * contextLen + c) * headDim + d;
        mqaKeyData[dstIdx] = keySum / numHeads;
        mqaValueData[dstIdx] = valueSum / numHeads;
      }
    }
  }
  
  // Run computation with MQA
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      mqaKeyData.data(),
      mqaValueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Check for valid output
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
}

// Test Grouped-Query Attention
TEST_F(AttentionOptTest, GroupedQueryAttention) {
  createTestData();
  
  // Configure for Grouped-Query Attention with 2 KV heads
  config.variant = AttentionVariant::GROUPED_QUERY;
  config.numKVHeads = 2;  // 2 KV heads for 8 query heads = 4:1 ratio
  config.configureForVariant(AttentionVariant::GROUPED_QUERY);
  
  // Create new implementation
  attImpl = createAttentionImpl(config, type);
  
  // Reshape keys/values for GQA
  std::vector<float> gqaKeyData(batchSize * contextLen * config.numKVHeads * headDim, 0.0f);
  std::vector<float> gqaValueData(batchSize * contextLen * config.numKVHeads * headDim, 0.0f);
  
  // Consolidate heads - each KV head is the average of several query heads
  int64_t headsPerGroup = numHeads / config.numKVHeads;
  
  for (int64_t b = 0; b < batchSize; b++) {
    for (int64_t c = 0; c < contextLen; c++) {
      for (int64_t g = 0; g < config.numKVHeads; g++) {
        for (int64_t d = 0; d < headDim; d++) {
          float keySum = 0.0f;
          float valueSum = 0.0f;
          
          // Average over the query heads in this group
          for (int64_t h = g * headsPerGroup; h < (g + 1) * headsPerGroup; h++) {
            int64_t srcIdx = ((b * numHeads + h) * contextLen + c) * headDim + d;
            keySum += keyData[srcIdx];
            valueSum += valueData[srcIdx];
          }
          
          int64_t dstIdx = ((b * config.numKVHeads + g) * contextLen + c) * headDim + d;
          gqaKeyData[dstIdx] = keySum / headsPerGroup;
          gqaValueData[dstIdx] = valueSum / headsPerGroup;
        }
      }
    }
  }
  
  // Run computation with GQA
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      gqaKeyData.data(),
      gqaValueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Check for valid output
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
}

} // anonymous namespace 