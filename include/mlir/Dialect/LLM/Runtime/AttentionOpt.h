//===- AttentionOpt.h - Runtime support for attention optimizations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines runtime support for optimized attention computations in LLM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_ATTENTIONOPT_H_
#define MLIR_DIALECT_LLM_RUNTIME_ATTENTIONOPT_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include <vector>
#include <memory>
#include <cmath>
#include <functional>
#include <unordered_map>

namespace mlir {
namespace llm {
namespace runtime {

/// Enum to specify different attention mask types
enum class AttentionMaskType {
  CAUSAL,      // Future tokens are masked (typical for decoder)
  BIDIRECTIONAL, // No masking (typical for encoder)
  SLIDING_WINDOW, // Only attend to a window of tokens
  CUSTOM       // Custom mask pattern
};

/// Enum to specify attention computation optimization levels
enum class AttentionOptLevel {
  NONE,        // No optimizations
  BASIC,       // Basic optimizations (e.g., fused softmax)
  ADVANCED,    // Advanced optimizations (e.g., flash attention)
  MAX          // Maximum optimizations (may have numerical tradeoffs)
};

/// Enum to specify different attention variants
enum class AttentionVariant {
  STANDARD,     // Standard multi-head attention (separate Q,K,V for each head)
  MULTI_QUERY,  // Multi-query attention (shared K,V across heads)
  GROUPED_QUERY, // Grouped-query attention (K,V shared within groups)
  CUSTOM        // Custom attention variant
};

/// Enum to specify attention pruning strategies
enum class AttentionPruningStrategy {
  NONE,           // No pruning
  THRESHOLD,      // Prune attention scores below threshold
  TOP_K,          // Keep only top-K scores per query
  BLOCK_SPARSE,   // Block-sparse attention pattern
  STATIC_PATTERN, // Use a fixed pattern mask
  LOCALITY_SENSITIVE // Locality-sensitive hashing based pruning
};

/// Attention configuration parameters
struct AttentionConfig {
  // Basic parameters
  int64_t numHeads = 0;       // Number of attention heads
  int64_t headDim = 0;        // Dimension of each head
  float scale = 0.0f;         // Attention scale factor (typically 1/sqrt(head_dim))
  float dropoutProb = 0.0f;   // Attention dropout probability

  // CUDA acceleration options
  bool useCUDA = false;           // Whether to use CUDA acceleration
  int cudaBlockSize = 256;        // CUDA block size for kernels
  int cudaNumSMs = 0;             // Number of streaming multiprocessors (0 = auto-detect)
  bool useTensorCores = true;     // Use Tensor Cores if available
  bool useHalfPrecision = false;  // Use FP16 computation when available

  // Attention variant
  AttentionVariant variant = AttentionVariant::STANDARD;  // Attention variant
  int64_t numKVHeads = 0;         // Number of key-value heads (for MQA/GQA)
  int64_t headGroupSize = 0;      // Number of query heads per KV head (for GQA)
  bool rotaryEmbedding = false;   // Whether to apply rotary embeddings
  
  // Attention masks
  AttentionMaskType maskType = AttentionMaskType::BIDIRECTIONAL;  // Type of attention mask
  
  // Optimization flags
  bool useFlashAttention = false;   // Use Flash Attention algorithm
  bool fuseSoftmax = false;         // Fuse softmax with attention matrix multiplication
  bool optimizeMaskedAttention = false;  // Use specialized masked attention implementations
  bool blockSparse = false;         // Use block-sparse computation for attention
  
  // Flash Attention parameters (aliased with block size params)
  int64_t flashBlockSizeM = 64;    // Block size for queries in Flash Attention
  int64_t flashBlockSizeN = 64;    // Block size for keys in Flash Attention
  int64_t flashBlockSizeK = 32;    // Block size for the head dimension
  int64_t blockSizeM = 64;         // Block size M (alias for flashBlockSizeM)
  int64_t blockSizeN = 64;         // Block size N (alias for flashBlockSizeN)
  
  // Sliding window attention parameters
  int64_t windowSize = 0;          // Window size for sliding window attention (0 = disabled)
  
  // Pruning strategy and parameters
  AttentionPruningStrategy pruningStrategy = AttentionPruningStrategy::NONE;
  float pruningThreshold = 0.01f;   // Threshold for THRESHOLD pruning strategy
  int64_t pruningTopK = 0;          // K value for TOP_K pruning (0 = auto: 20% of context)
  int64_t pruningBlockSize = 32;    // Block size for BLOCK_SPARSE pruning
  float pruningRatio = 0.5f;        // Ratio of blocks to prune (for BLOCK_SPARSE)
  const void* staticPruningMask = nullptr;  // Static mask for STATIC_PATTERN pruning
  
  // Helper method to set default scale if not provided
  void setDefaultsFromHeadDim() {
    if (scale <= 0.0f && headDim > 0) {
      scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    }
    
    // Set default numKVHeads based on variant if not specified
    if (numKVHeads <= 0) {
      switch (variant) {
        case AttentionVariant::MULTI_QUERY:
          numKVHeads = 1;
          break;
        case AttentionVariant::GROUPED_QUERY:
          numKVHeads = numHeads / 4;  // Common grouping factor, adjustable
          break;
        default:
          numKVHeads = numHeads;  // Standard attention has same number of K/V heads as Q
          break;
      }
    }
    
    // Set headGroupSize for GQA/MQA
    if (headGroupSize <= 0 && numKVHeads > 0) {
      headGroupSize = numHeads / numKVHeads;
    }
    
    // Sync block size aliases
    blockSizeM = flashBlockSizeM;
    blockSizeN = flashBlockSizeN;
    
    // Set optimal Flash Attention block sizes based on hardware if not specified
    if (useFlashAttention) {
      // These are reasonable defaults that work well on most hardware
      // In a real implementation, these would be tuned for specific architectures
      if (flashBlockSizeM <= 0) flashBlockSizeM = 64;
      if (flashBlockSizeN <= 0) flashBlockSizeN = 64;
      if (flashBlockSizeK <= 0) flashBlockSizeK = 32;
      blockSizeM = flashBlockSizeM;
      blockSizeN = flashBlockSizeN;
    }
  }
};

/// Abstract base class for attention implementations
class AttentionImpl {
public:
  virtual ~AttentionImpl() = default;
  
  /// Compute attention with the provided queries, keys, and values
  /// \param output Output tensor [batchSize, seqLen, numHeads, headDim]
  /// \param queries Query tensor [batchSize, seqLen, numHeads, headDim]
  /// \param keys Key tensor [batchSize, contextLen, numHeads, headDim]
  /// \param values Value tensor [batchSize, contextLen, numHeads, headDim]
  /// \param batchSize Batch size
  /// \param seqLen Sequence length of queries
  /// \param contextLen Context length of keys/values
  /// \param attentionMask Optional attention mask [batchSize, seqLen, contextLen]
  virtual void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask = nullptr) = 0;
      
  /// Compute paged attention using the KV cache
  /// \param output Output tensor [batchSize, seqLen, numHeads, headDim]
  /// \param queries Query tensor [batchSize, seqLen, numHeads, headDim]
  /// \param kvCache KV cache containing keys and values
  /// \param blockIndices Indices into the KV cache [batchSize, maxSeqLen]
  /// \param seqLens Actual sequence lengths [batchSize]
  /// \param batchSize Batch size
  /// \param seqLen Sequence length of queries
  virtual void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) = 0;
};

/// Factory function to create an appropriate attention implementation
/// based on the provided configuration
std::unique_ptr<AttentionImpl> createAttentionImpl(
    const AttentionConfig& config,
    Type elementType,
    bool useGPU = true);

//===----------------------------------------------------------------------===//
// Fused Softmax Attention Implementation
//===----------------------------------------------------------------------===//

/// Implementation of attention with fused softmax operation
class FusedSoftmaxAttentionImpl : public AttentionImpl {
public:
  FusedSoftmaxAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
  
  // Helper function to apply causal mask
  void applyCausalMask(void* attentionScores, int64_t batchSize, int64_t seqLen, int64_t contextLen);
  
  // Helper function to apply sliding window mask
  void applySlidingWindowMask(void* attentionScores, int64_t batchSize, int64_t seqLen, int64_t contextLen);
  
  // Fused softmax implementation
  void fusedSoftmax(void* attentionScores, void* attentionProbs, int64_t batchSize, int64_t seqLen, int64_t contextLen);
};

//===----------------------------------------------------------------------===//
// Sliding Window Attention Implementation
//===----------------------------------------------------------------------===//

/// Implementation of attention with sliding window optimization
class SlidingWindowAttentionImpl : public AttentionImpl {
public:
  SlidingWindowAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
  
  // Helper for computing windowed attention
  void computeWindowedAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      int64_t windowSize);
};

//===----------------------------------------------------------------------===//
// Pruned Attention Implementation
//===----------------------------------------------------------------------===//

/// Implementation of attention with pruning optimizations
class PrunedAttentionImpl : public AttentionImpl {
public:
  PrunedAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
  
  // Workspace for pruning masks
  std::vector<float> pruningScores;
  std::vector<char> dynamicPruningMask;
  
  // Helper methods for different pruning strategies
  void applyThresholdPruning(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void applyTopKPruning(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void applyBlockSparsePruning(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void applyLocalitySensitivePruning(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void applyStaticPatternPruning(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  // Helper for computing attention with pruning
  void computeWithPruning(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
      
  // Helper for computing attention scores
  void calculateAttentionScores(
      void* attentionScores,
      const void* queries,
      const void* keys,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  // Helper for applying standard masks
  void applyStandardMask(
      void* attentionScores,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  // Helper for applying custom masks
  void applyCustomMask(
      void* attentionScores,
      const void* attentionMask,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  // Helper for softmax with pruning mask
  void applySoftmaxWithPruning(
      const void* attentionScores,
      void* attentionProbs,
      const char* pruningMask,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  // Helper for computing attention output
  void computeAttentionOutput(
      void* output,
      const void* attentionProbs,
      const void* values,
      const char* pruningMask,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
};

//===----------------------------------------------------------------------===//
// Flash Attention Implementation
//===----------------------------------------------------------------------===//

/// Implementation of attention using the Flash Attention algorithm for improved memory efficiency
class FlashAttentionImpl : public AttentionImpl {
public:
  FlashAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  // Helper method to implement Flash Attention algorithm
  void flashAttentionTiled(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
      
  // Helper method to process a single block of Flash Attention
  void processFlashAttentionBlock(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchIdx,
      int64_t headIdx,
      int64_t queryStart,
      int64_t queryEnd,
      int64_t keyStart,
      int64_t keyEnd,
      const void* attentionMask,
      void* blockAccumulatorsPtr,
      void* blockMaxValuesPtr);
      
  // Specialized version for paged KV cache
  void flashAttentionWithPagedKVCache(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen);
      
  AttentionConfig config;
  Type elementType;
  bool useGPU;
};

//===----------------------------------------------------------------------===//
// Standard Attention Implementation
//===----------------------------------------------------------------------===//

/// Standard multi-head attention implementation
class StandardAttentionImpl : public AttentionImpl {
public:
  StandardAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
};

//===----------------------------------------------------------------------===//
// Multi-Query Attention Implementation
//===----------------------------------------------------------------------===//

/// Multi-Query Attention (MQA): Shared K,V across all query heads
class MultiQueryAttentionImpl : public AttentionImpl {
public:
  MultiQueryAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
  
  // CPU implementation
  void computeMultiQueryAttentionCPU(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
      
  // CUDA implementation
  void computeMultiQueryAttentionCUDA(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
};

//===----------------------------------------------------------------------===//
// Grouped-Query Attention Implementation
//===----------------------------------------------------------------------===//

/// Grouped-Query Attention (GQA): K,V shared within head groups
class GroupedQueryAttentionImpl : public AttentionImpl {
public:
  GroupedQueryAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
};

//===----------------------------------------------------------------------===//
// Optimized Masked Attention Implementation
//===----------------------------------------------------------------------===//

/// Optimized attention for specific mask patterns (causal, sliding window, etc.)
class OptimizedMaskedAttentionImpl : public AttentionImpl {
public:
  OptimizedMaskedAttentionImpl(const AttentionConfig& config, Type elementType, bool useGPU);
  
  void compute(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask) override;
      
  void computePaged(
      void* output,
      const void* queries,
      PagedKVCache* kvCache,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen) override;

private:
  AttentionConfig config;
  Type elementType;
  bool useGPU;
  
  // Specialized implementations for different mask types
  void computeCausalMaskedAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void computeWindowedAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      int64_t windowSize);
      
  void computeBidirectionalAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen);
      
  void computeBlockDiagonalAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
      
  void computeGeneralMaskedAttention(
      void* output,
      const void* queries,
      const void* keys,
      const void* values,
      int64_t batchSize,
      int64_t seqLen,
      int64_t contextLen,
      const void* attentionMask);
      
  // Mask pattern detection helpers
  bool isBlockDiagonalMask(const void* mask, int64_t batchSize, int64_t seqLen, int64_t contextLen);
  bool isLocalMask(const void* mask, int64_t batchSize, int64_t seqLen, int64_t contextLen);
  int64_t detectWindowSize(const void* mask, int64_t batchSize, int64_t seqLen, int64_t contextLen);
};

//===----------------------------------------------------------------------===//
// Attention Variant Factory Registration
//===----------------------------------------------------------------------===//

/// Function type for attention variant factories
using AttentionVariantFactory = std::function<std::unique_ptr<AttentionImpl>(
    const AttentionConfig&, Type, bool)>;

/// Register an attention variant factory
void registerAttentionVariant(AttentionVariant variant, AttentionVariantFactory factory);

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_ATTENTIONOPT_H_ 