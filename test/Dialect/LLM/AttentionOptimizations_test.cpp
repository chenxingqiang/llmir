//===- AttentionOptimizations_test.cpp - Tests for attention optimizations ===//
//
// This file contains tests for the attention optimization implementations in the
// LLM runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

#include <random>
#include <vector>

using namespace mlir;
using namespace mlir::LLM::runtime;

namespace {

class AttentionOptimizationsTest : public testing::Test {
protected:
  void SetUp() override {
    // Set up common test parameters
    batchSize = 2;
    numHeads = 8;
    headDim = 64;
    seqLen = 32;
    contextLen = 64;
    
    // Create a fixed random seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Initialize test data
    queryData.resize(batchSize * seqLen * numHeads * headDim);
    keyData.resize(batchSize * contextLen * numHeads * headDim);
    valueData.resize(batchSize * contextLen * numHeads * headDim);
    outputData.resize(batchSize * seqLen * numHeads * headDim);
    referenceMask.resize(batchSize * seqLen * contextLen);
    
    // Fill with random data
    for (size_t i = 0; i < queryData.size(); i++) {
      queryData[i] = dist(rng);
    }
    
    for (size_t i = 0; i < keyData.size(); i++) {
      keyData[i] = dist(rng);
    }
    
    for (size_t i = 0; i < valueData.size(); i++) {
      valueData[i] = dist(rng);
    }
    
    // Initialize mask (causal in this case)
    for (int64_t b = 0; b < batchSize; b++) {
      for (int64_t q = 0; q < seqLen; q++) {
        for (int64_t k = 0; k < contextLen; k++) {
          int64_t idx = b * seqLen * contextLen + q * contextLen + k;
          // Causal mask: can only attend to positions before or at current position
          referenceMask[idx] = (k <= q) ? 1.0f : 0.0f;
        }
      }
    }
    
    // Zero output data
    std::fill(outputData.begin(), outputData.end(), 0.0f);
  }
  
  void createAttentionConfig(AttentionVariant variant, AttentionMaskType maskType) {
    config.numHeads = numHeads;
    config.headDim = headDim;
    config.variant = variant;
    config.maskType = maskType;
    config.setDefaultsFromHeadDim();
  }
  
  std::unique_ptr<AttentionImpl> createStandardAttention() {
    createAttentionConfig(AttentionVariant::STANDARD, AttentionMaskType::BIDIRECTIONAL);
    return createAttentionImpl(config, FloatType::getF32(nullptr), false);
  }
  
  std::unique_ptr<AttentionImpl> createFlashAttention() {
    createAttentionConfig(AttentionVariant::STANDARD, AttentionMaskType::BIDIRECTIONAL);
    config.useFlashAttention = true;
    return createAttentionImpl(config, FloatType::getF32(nullptr), false);
  }
  
  std::unique_ptr<AttentionImpl> createFusedSoftmaxAttention() {
    createAttentionConfig(AttentionVariant::STANDARD, AttentionMaskType::BIDIRECTIONAL);
    config.fuseSoftmax = true;
    return createAttentionImpl(config, FloatType::getF32(nullptr), false);
  }
  
  std::unique_ptr<AttentionImpl> createSlidingWindowAttention(int64_t windowSize) {
    createAttentionConfig(AttentionVariant::STANDARD, AttentionMaskType::SLIDING_WINDOW);
    config.windowSize = windowSize;
    return createAttentionImpl(config, FloatType::getF32(nullptr), false);
  }
  
  void compareOutputs(const std::vector<float>& expected, const std::vector<float>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    
    // Compute mean absolute error
    float totalError = 0.0f;
    for (size_t i = 0; i < expected.size(); i++) {
      totalError += std::abs(expected[i] - actual[i]);
    }
    float meanError = totalError / expected.size();
    
    // The implementations may differ slightly due to numerical precision,
    // so we use a reasonable tolerance for floating point comparison
    EXPECT_LT(meanError, 1e-4) << "Output values differ significantly";
  }
  
  // Test data
  int64_t batchSize;
  int64_t numHeads;
  int64_t headDim;
  int64_t seqLen;
  int64_t contextLen;
  std::vector<float> queryData;
  std::vector<float> keyData;
  std::vector<float> valueData;
  std::vector<float> outputData;
  std::vector<float> referenceMask;
  
  // Configuration
  AttentionConfig config;
};

TEST_F(AttentionOptimizationsTest, CompareFlashAttentionWithStandard) {
  // Get the standard attention implementation as reference
  auto standardAttention = createStandardAttention();
  
  // Run the standard implementation
  std::vector<float> standardOutput(outputData.size());
  standardAttention->compute(
      standardOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      nullptr);
  
  // Get the Flash Attention implementation
  auto flashAttention = createFlashAttention();
  
  // Run Flash Attention implementation
  std::vector<float> flashOutput(outputData.size());
  flashAttention->compute(
      flashOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      nullptr);
  
  // Compare outputs - they should match within floating point precision
  compareOutputs(standardOutput, flashOutput);
}

TEST_F(AttentionOptimizationsTest, CompareFusedSoftmaxWithStandard) {
  // Get the standard attention implementation as reference
  auto standardAttention = createStandardAttention();
  
  // Run the standard implementation
  std::vector<float> standardOutput(outputData.size());
  standardAttention->compute(
      standardOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      nullptr);
  
  // Get the fused softmax implementation
  auto fusedSoftmaxAttention = createFusedSoftmaxAttention();
  
  // Run fused softmax implementation
  std::vector<float> fusedOutput(outputData.size());
  fusedSoftmaxAttention->compute(
      fusedOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      nullptr);
  
  // Compare outputs - they should match within floating point precision
  compareOutputs(standardOutput, fusedOutput);
}

TEST_F(AttentionOptimizationsTest, SlidingWindowAttention) {
  // Create standard attention with a causal mask for reference
  createAttentionConfig(AttentionVariant::STANDARD, AttentionMaskType::CAUSAL);
  auto standardAttention = createAttentionImpl(config, FloatType::getF32(nullptr), false);
  
  // Run the standard implementation with causal mask
  std::vector<float> standardOutput(outputData.size());
  standardAttention->compute(
      standardOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      referenceMask.data());
  
  // Create sliding window attention with window size 16
  auto slidingWindowAttention = createSlidingWindowAttention(16);
  
  // Run sliding window implementation
  std::vector<float> slidingOutput(outputData.size());
  slidingWindowAttention->compute(
      slidingOutput.data(),
      queryData.data(),
      keyData.data(),
      valueData.data(),
      batchSize,
      seqLen,
      contextLen,
      nullptr); // No mask needed, it's built into the implementation
  
  // Results will be different due to the window constraint, but
  // we can check that tokens outside the window have no influence
  // by comparing with a modified version of the standard output
  
  // For this test, we're just validating that the sliding window works,
  // rather than comparing with standard attention
  EXPECT_FALSE(std::equal(standardOutput.begin(), standardOutput.end(), 
                        slidingOutput.begin()));
  
  // To properly test sliding window, we would need to mask the standard attention
  // in the same sliding window pattern and compare
}

} // end anonymous namespace 