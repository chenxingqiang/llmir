//===- FlashAttention_test.cpp - Tests for Flash Attention implementation -===//
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
#include <random>

using namespace mlir;
using namespace mlir::llm::runtime;

namespace {

// A simple Type implementation for testing
struct TestType : public Type {
  int getIntOrFloatBitWidth() const { return 16; }  // f16
};

// Test fixture for Flash Attention tests
class FlashAttentionTest : public ::testing::Test {
protected:
  void SetUp() override {
    type = TestType();
    
    // Create default attention config with Flash Attention enabled
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    config.maskType = AttentionMaskType::CAUSAL;
    config.optLevel = AttentionOptLevel::ADVANCED;
    config.fuseSoftmax = true;
    config.useFlashAttention = true;
    config.blockSizeM = 64;
    config.blockSizeN = 64;
    
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
    queryData.resize(batchSize * seqLen * numHeads * headDim, 0.0f);
    keyData.resize(batchSize * contextLen * numHeads * headDim, 0.0f);
    valueData.resize(batchSize * contextLen * numHeads * headDim, 0.0f);
    outputData.resize(batchSize * seqLen * numHeads * headDim, 0.0f);
    
    // Fill with random test data
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (size_t i = 0; i < queryData.size(); i++) {
      queryData[i] = dist(rng);
    }
    
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = dist(rng);
    }
    
    for (size_t i = 0; i < valueData.size(); i++) {
      valueData[i] = dist(rng);
    }
  }
  
  // Compare Flash Attention results with regular attention
  void compareWithRegularAttention() {
    // Create non-flash attention config
    AttentionConfig regularConfig = config;
    regularConfig.useFlashAttention = false;
    
    // Create regular attention implementation
    auto regularImpl = createAttentionImpl(regularConfig, type);
    
    // Allocate output for regular attention
    std::vector<float> regularOutput(outputData.size(), 0.0f);
    
    // Compute with both implementations
    attImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    regularImpl->compute(
        regularOutput.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // Compare results - should be approximately equal
    for (size_t i = 0; i < outputData.size(); i++) {
      EXPECT_NEAR(outputData[i], regularOutput[i], 1e-4f);
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
  const int64_t seqLen = 16;
  const int64_t contextLen = 32;
  
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

// Test creating Flash Attention implementation
TEST_F(FlashAttentionTest, CreateFlashAttentionImpl) {
  AttentionConfig config;
  config.numHeads = numHeads;
  config.headDim = headDim;
  config.useFlashAttention = true;
  
  auto impl = createAttentionImpl(config, type);
  ASSERT_NE(impl, nullptr);
  
  // Check if it's the correct type
  auto flashImpl = dynamic_cast<FlashAttentionImpl*>(impl.get());
  EXPECT_NE(flashImpl, nullptr);
}

// Test basic Flash Attention computation
TEST_F(FlashAttentionTest, BasicFlashAttentionComputation) {
  createTestData();
  
  // Run attention computation with Flash Attention
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Verify some basic properties of the output - not zero, no NaNs
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
}

// Test Flash Attention with long sequences
TEST_F(FlashAttentionTest, LongSequenceFlashAttention) {
  // Use longer sequence length for this test
  const int64_t longSeqLen = 128;
  const int64_t longContextLen = 256;
  
  // Resize test data for longer sequences
  queryData.resize(batchSize * longSeqLen * numHeads * headDim, 0.0f);
  keyData.resize(batchSize * longContextLen * numHeads * headDim, 0.0f);
  valueData.resize(batchSize * longContextLen * numHeads * headDim, 0.0f);
  outputData.resize(batchSize * longSeqLen * numHeads * headDim, 0.0f);
  
  // Fill with simple test pattern
  for (size_t i = 0; i < queryData.size(); i++) {
    queryData[i] = 0.01f * (i % 100);
  }
  
  for (size_t i = 0; i < keyData.size(); i++) {
    keyData[i] = 0.01f * ((i + 27) % 100);
  }
  
  for (size_t i = 0; i < valueData.size(); i++) {
    valueData[i] = 0.01f * ((i + 53) % 100);
  }
  
  // Run attention computation
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      longSeqLen,
      longContextLen);
  
  // Verify basic properties
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
}

// Test Flash Attention with KV cache
TEST_F(FlashAttentionTest, FlashAttentionWithKVCache) {
  createTestData();
  
  // Add some KV data to the cache
  std::vector<int32_t> seqIds = {10, 20};
  std::vector<int32_t> blockIndices(batchSize * numLayers);
  
  // Append KV pairs to cache
  LogicalResult result = kvCache->appendKV(
      keyData.data(), valueData.data(), 
      batchSize, contextLen, seqIds.data(), blockIndices.data());
  
  ASSERT_TRUE(succeeded(result));
  
  // Set sequence lengths for attention
  std::vector<int32_t> seqLens = {contextLen, contextLen};
  
  // Compute attention with the KV cache
  result = kvCache->computeAttention(
      outputData.data(),
      queryData.data(),
      blockIndices.data(),
      seqLens.data(),
      batchSize,
      seqLen);
  
  ASSERT_TRUE(succeeded(result));
  
  // Check for basic output properties
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
}

// Test Flash Attention correctness compared to regular attention
TEST_F(FlashAttentionTest, FlashAttentionCorrectness) {
  createTestData();
  
  // This test compares Flash Attention results with regular attention
  // They should produce approximately the same results
  compareWithRegularAttention();
}

// Test Flash Attention with various block sizes
TEST_F(FlashAttentionTest, BlockSizeVariations) {
  createTestData();
  
  // Test different block size configurations
  const std::vector<int64_t> blockSizes = {32, 64, 128};
  
  for (int64_t blockM : blockSizes) {
    for (int64_t blockN : blockSizes) {
      // Create configuration with specific block sizes
      AttentionConfig testConfig = config;
      testConfig.blockSizeM = blockM;
      testConfig.blockSizeN = blockN;
      
      // Create implementation with this config
      auto testImpl = createAttentionImpl(testConfig, type);
      
      // Clear output
      std::fill(outputData.begin(), outputData.end(), 0.0f);
      
      // Run computation
      testImpl->compute(
          outputData.data(),
          queryData.data(),
          keyData.data(),
          valueData.data(),
          batchSize,
          seqLen,
          contextLen);
      
      // Check that results are reasonable
      bool allZeros = true;
      bool hasNaNs = false;
      
      for (float val : outputData) {
        if (val != 0.0f) allZeros = false;
        if (std::isnan(val)) hasNaNs = true;
      }
      
      EXPECT_FALSE(allZeros);
      EXPECT_FALSE(hasNaNs);
    }
  }
}

// Test Flash Attention with different mask types
TEST_F(FlashAttentionTest, DifferentMaskTypes) {
  createTestData();
  
  // Test with different mask types
  std::vector<AttentionMaskType> maskTypes = {
    AttentionMaskType::CAUSAL,
    AttentionMaskType::SLIDING_WINDOW,
    AttentionMaskType::BIDIRECTIONAL
  };
  
  for (auto maskType : maskTypes) {
    // Create configuration with this mask type
    AttentionConfig testConfig = config;
    testConfig.maskType = maskType;
    
    if (maskType == AttentionMaskType::SLIDING_WINDOW) {
      testConfig.windowSize = 8;  // Set window size for sliding window attention
    }
    
    // Create implementation with this config
    auto testImpl = createAttentionImpl(testConfig, type);
    
    // Clear output
    std::fill(outputData.begin(), outputData.end(), 0.0f);
    
    // Run computation
    testImpl->compute(
        outputData.data(),
        queryData.data(),
        keyData.data(),
        valueData.data(),
        batchSize,
        seqLen,
        contextLen);
    
    // Check that results are reasonable
    bool allZeros = true;
    bool hasNaNs = false;
    
    for (float val : outputData) {
      if (val != 0.0f) allZeros = false;
      if (std::isnan(val)) hasNaNs = true;
    }
    
    EXPECT_FALSE(allZeros);
    EXPECT_FALSE(hasNaNs);
  }
}

// Test CUDA-accelerated Flash Attention implementation
TEST_F(FlashAttentionTest, CUDAAcceleratedFlashAttention) {
  // Skip test if CUDA is not available
  if (!cuda::isCUDAAvailable()) {
    GTEST_SKIP() << "CUDA not available, skipping GPU test";
  }
  
  createTestData();
  
  // Enable CUDA acceleration in the config
  config.useCUDA = true;
  config.useHalfPrecision = false;  // Use float32 for more precise test
  
  // Create a new implementation with CUDA enabled
  attImpl = createAttentionImpl(config, type);
  
  // Run attention computation
  attImpl->compute(
      outputData.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Verify some basic properties of the output
  bool allZeros = true;
  bool hasNaNs = false;
  
  for (float val : outputData) {
    if (val != 0.0f) allZeros = false;
    if (std::isnan(val)) hasNaNs = true;
  }
  
  EXPECT_FALSE(allZeros);
  EXPECT_FALSE(hasNaNs);
  
  // Compare with CPU implementation for correctness
  // Disable CUDA for CPU implementation
  config.useCUDA = false;
  auto cpuImpl = createAttentionImpl(config, type);
  
  std::vector<float> cpuOutput(outputData.size(), 0.0f);
  
  cpuImpl->compute(
      cpuOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen);
  
  // Check that CPU and GPU results are close
  for (size_t i = 0; i < outputData.size(); i++) {
    EXPECT_NEAR(outputData[i], cpuOutput[i], 1e-4f);
  }
}

} // anonymous namespace 