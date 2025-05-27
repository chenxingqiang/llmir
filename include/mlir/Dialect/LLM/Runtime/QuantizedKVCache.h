//===- QuantizedKVCache.h - Quantized KV Cache Support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines quantized KV cache support for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_QUANTIZEDKVCACHE_H_
#define MLIR_DIALECT_LLM_RUNTIME_QUANTIZEDKVCACHE_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Quantization Configuration
//===----------------------------------------------------------------------===//

enum class QuantizationType {
  INT8,    // 8-bit integer quantization
  INT4,    // 4-bit integer quantization
  FP8      // 8-bit floating point quantization (future)
};

enum class QuantizationStrategy {
  PER_TENSOR,   // Single scale/zero-point for entire tensor
  PER_CHANNEL,  // Per-channel quantization
  PER_GROUP     // Group-wise quantization (for INT4)
};

struct QuantizationConfig {
  QuantizationType type;
  QuantizationStrategy strategy;
  bool symmetric;        // Whether to use symmetric quantization
  int64_t groupSize;     // Group size for per-group quantization
  bool dynamicRange;     // Whether to use dynamic range quantization
  
  QuantizationConfig(QuantizationType type = QuantizationType::INT8,
                    QuantizationStrategy strategy = QuantizationStrategy::PER_TENSOR,
                    bool symmetric = true,
                    int64_t groupSize = 128,
                    bool dynamicRange = false)
      : type(type), strategy(strategy), symmetric(symmetric),
        groupSize(groupSize), dynamicRange(dynamicRange) {}
};

//===----------------------------------------------------------------------===//
// Quantization Parameters
//===----------------------------------------------------------------------===//

struct QuantizationParams {
  std::vector<float> scales;      // Quantization scales
  std::vector<int32_t> zeroPoints; // Zero points (for asymmetric quantization)
  int64_t numBits;                // Number of bits used for quantization
  bool isSigned;                  // Whether quantized values are signed
  
  QuantizationParams() : numBits(8), isSigned(true) {}
};

//===----------------------------------------------------------------------===//
// Quantized KV Block
//===----------------------------------------------------------------------===//

class QuantizedKVBlock {
public:
  QuantizedKVBlock(void* quantizedKeyPtr, void* quantizedValuePtr,
                   const QuantizationParams& keyParams,
                   const QuantizationParams& valueParams,
                   int64_t blockSize, int64_t headDim);
  
  ~QuantizedKVBlock();
  
  // Basic getters
  void* getQuantizedKeyPtr() const { return quantizedKeyPtr_; }
  void* getQuantizedValuePtr() const { return quantizedValuePtr_; }
  int64_t getBlockSize() const { return blockSize_; }
  int64_t getHeadDim() const { return headDim_; }
  
  // Quantization parameters
  const QuantizationParams& getKeyParams() const { return keyParams_; }
  const QuantizationParams& getValueParams() const { return valueParams_; }
  
  // Usage tracking
  int64_t getUsedSlots() const { return usedSlots_; }
  void incrementUsedSlots(int64_t count = 1) { usedSlots_ += count; }
  void resetUsedSlots() { usedSlots_ = 0; }
  bool isFull() const { return usedSlots_ >= blockSize_; }
  
  // Reference counting
  int32_t getRefCount() const { return refCount_; }
  void incrementRefCount() { refCount_++; }
  bool decrementRefCount() { return --refCount_ > 0; }
  
  // Access time tracking for LRU
  void updateAccessTime(int64_t timestamp) { lastAccessTime_ = timestamp; }
  int64_t getLastAccessTime() const { return lastAccessTime_; }
  
  // Quantization/Dequantization operations
  LogicalResult quantizeAndStore(const float* keyData, const float* valueData,
                                int64_t numTokens, int64_t startOffset);
  
  LogicalResult dequantizeAndLoad(float* keyData, float* valueData,
                                 int64_t numTokens, int64_t startOffset) const;
  
  // Memory usage calculation
  size_t getMemoryUsage() const;
  
private:
  void* quantizedKeyPtr_;
  void* quantizedValuePtr_;
  QuantizationParams keyParams_;
  QuantizationParams valueParams_;
  int64_t blockSize_;
  int64_t headDim_;
  int64_t usedSlots_;
  int32_t refCount_;
  int64_t lastAccessTime_;
  
  // Helper methods for quantization
  void quantizeTensor(const float* input, void* output,
                     const QuantizationParams& params,
                     int64_t numElements) const;
  
  void dequantizeTensor(const void* input, float* output,
                       const QuantizationParams& params,
                       int64_t numElements) const;
};

//===----------------------------------------------------------------------===//
// Quantized Block Allocator
//===----------------------------------------------------------------------===//

class QuantizedBlockAllocator {
public:
  QuantizedBlockAllocator(int64_t blockSize, int64_t headDim,
                         const QuantizationConfig& config,
                         bool enableGPU = false);
  
  ~QuantizedBlockAllocator();
  
  // Block allocation/deallocation
  QuantizedKVBlock* allocateBlock();
  void deallocateBlock(QuantizedKVBlock* block);
  
  // Configuration
  void configureQuantization(const QuantizationConfig& config);
  const QuantizationConfig& getQuantizationConfig() const { return config_; }
  
  // Memory management
  void preallocateBlocks(int64_t numBlocks);
  size_t getNumFreeBlocks() const { return freeBlocks_.size(); }
  size_t getTotalBlocks() const { return totalBlocks_; }
  
  // Quantization parameter calculation
  QuantizationParams calculateQuantizationParams(const float* data,
                                                 int64_t numElements) const;
  
  // Memory usage statistics
  size_t getTotalMemoryUsage() const;
  size_t getQuantizedMemoryUsage() const;
  float getCompressionRatio() const;
  
private:
  int64_t blockSize_;
  int64_t headDim_;
  QuantizationConfig config_;
  bool enableGPU_;
  
  std::vector<QuantizedKVBlock*> freeBlocks_;
  std::vector<std::unique_ptr<QuantizedKVBlock>> allBlocks_;
  size_t totalBlocks_;
  
  // Memory pools for quantized data
  void* keyMemoryPool_;
  void* valueMemoryPool_;
  size_t poolSize_;
  size_t poolOffset_;
  
  // Helper methods
  void allocateMemoryPools();
  void deallocateMemoryPools();
  size_t calculateBlockMemorySize() const;
  
  // Quantization parameter calculation helpers
  std::pair<float, float> calculateMinMax(const float* data, int64_t numElements) const;
  float calculateScale(float min, float max, int64_t numBits, bool symmetric) const;
  int32_t calculateZeroPoint(float min, float max, float scale, bool symmetric) const;
};

//===----------------------------------------------------------------------===//
// Quantized Paged KV Cache
//===----------------------------------------------------------------------===//

class QuantizedPagedKVCache {
public:
  QuantizedPagedKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                       int64_t blockSize, int64_t maxSeqLen,
                       const QuantizationConfig& config,
                       Type elementType, bool enableGPU = false);
  
  ~QuantizedPagedKVCache();
  
  // Basic getters
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  int64_t getBlockSize() const { return blockSize_; }
  int64_t getMaxSeqLen() const { return maxSeqLen_; }
  Type getElementType() const { return elementType_; }
  
  // Quantization configuration
  const QuantizationConfig& getQuantizationConfig() const { return config_; }
  void updateQuantizationConfig(const QuantizationConfig& config);
  
  // Core KV cache operations with quantization
  LogicalResult appendKV(const void* keyData, const void* valueData,
                        int32_t batchSize, int32_t seqLen,
                        const int32_t* seqIds, int32_t* blockIndices);
  
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int32_t batchSize, void* outputKeys, void* outputValues);
  
  // Sequence management
  LogicalResult clearSequence(int32_t seqId);
  void reset();
  int64_t getSequenceLength(int32_t seqId) const;
  int64_t getNumSequences() const { return sequenceTable_.size(); }
  
  // Memory and performance statistics
  size_t getTotalMemoryUsage() const;
  size_t getQuantizedMemoryUsage() const;
  float getCompressionRatio() const;
  float getAccuracyLoss() const; // Estimated accuracy loss from quantization
  
  // Configuration and optimization
  void enableDynamicQuantization(bool enable) { dynamicQuantization_ = enable; }
  void setAccuracyThreshold(float threshold) { accuracyThreshold_ = threshold; }
  
  // Benchmarking and profiling
  struct QuantizationMetrics {
    int64_t numQuantizations;
    int64_t numDequantizations;
    double totalQuantizationTime;
    double totalDequantizationTime;
    double averageAccuracyLoss;
    size_t memoryReduction;
  };
  
  QuantizationMetrics getQuantizationMetrics() const { return metrics_; }
  void resetMetrics() { metrics_ = QuantizationMetrics{}; }
  
private:
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t blockSize_;
  int64_t maxSeqLen_;
  QuantizationConfig config_;
  Type elementType_;
  bool enableGPU_;
  
  // Block allocators for each layer
  std::vector<std::unique_ptr<QuantizedBlockAllocator>> blockAllocators_;
  
  // Sequence tracking
  std::unordered_map<int32_t, std::vector<std::vector<QuantizedKVBlock*>>> sequenceTable_;
  
  // Dynamic quantization settings
  bool dynamicQuantization_;
  float accuracyThreshold_;
  
  // Performance metrics
  mutable QuantizationMetrics metrics_;
  
  // Helper methods
  void initializeBlockAllocators();
  LogicalResult validateQuantizationAccuracy(const float* original,
                                            const float* quantized,
                                            int64_t numElements) const;
  
  // Accuracy measurement helpers
  float calculateMSE(const float* a, const float* b, int64_t numElements) const;
  float calculateSNR(const float* original, const float* quantized, int64_t numElements) const;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_QUANTIZEDKVCACHE_H_ 