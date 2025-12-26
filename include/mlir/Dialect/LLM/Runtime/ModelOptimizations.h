//===- ModelOptimizations.h - Model-specific optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines model-specific optimizations for popular LLM architectures
// including Llama, Mistral, Phi, and others.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_MODELOPTIMIZATIONS_H
#define MLIR_DIALECT_LLM_RUNTIME_MODELOPTIMIZATIONS_H

#include "mlir/Dialect/LLM/Runtime/PagedKVCache.h"
#include "mlir/Dialect/LLM/Runtime/QuantizedKVCache.h"
#include "mlir/Dialect/LLM/Runtime/PrefixCache.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// Model Architecture Types
//===----------------------------------------------------------------------===//

/// Supported model architectures
enum class ModelArchitecture {
  LLAMA,
  LLAMA2,
  LLAMA3,
  MISTRAL,
  MIXTRAL,
  PHI,
  PHI3,
  QWEN,
  QWEN2,
  GEMMA,
  GEMMA2,
  FALCON,
  MPT,
  CUSTOM
};

/// Attention type used by the model
enum class AttentionType {
  MULTI_HEAD,        // Standard MHA
  MULTI_QUERY,       // MQA (single KV head)
  GROUPED_QUERY,     // GQA (grouped KV heads)
  SLIDING_WINDOW,    // Sliding window attention
  HYBRID             // Combination of types
};

/// RoPE (Rotary Position Embedding) variant
enum class RoPEType {
  NONE,
  STANDARD,
  SCALED,     // Position interpolation
  YARN,       // YaRN scaling
  DYNAMIC,    // Dynamic NTK
  ALIBI       // ALiBi (not RoPE, but position encoding)
};

//===----------------------------------------------------------------------===//
// Model Configuration
//===----------------------------------------------------------------------===//

/// Configuration for a specific model architecture
struct ModelConfig {
  ModelArchitecture architecture;
  AttentionType attentionType;
  RoPEType ropeType;
  
  // Model dimensions
  int64_t numLayers;
  int64_t hiddenSize;
  int64_t numAttentionHeads;
  int64_t numKeyValueHeads;
  int64_t headDim;
  int64_t intermediateSize;
  int64_t vocabSize;
  int64_t maxPositionEmbeddings;
  
  // RoPE parameters
  float ropeTheta;
  float ropeScalingFactor;
  
  // Sliding window parameters
  int64_t slidingWindowSize;
  
  // Normalization
  bool useRMSNorm;
  float rmsNormEps;
  
  // Activation
  std::string hiddenAct;  // "silu", "gelu", "gelu_new", etc.
  
  // Tie embeddings
  bool tieWordEmbeddings;
  
  // Default constructor
  ModelConfig()
      : architecture(ModelArchitecture::LLAMA),
        attentionType(AttentionType::MULTI_HEAD),
        ropeType(RoPEType::STANDARD),
        numLayers(32), hiddenSize(4096),
        numAttentionHeads(32), numKeyValueHeads(32),
        headDim(128), intermediateSize(11008),
        vocabSize(32000), maxPositionEmbeddings(4096),
        ropeTheta(10000.0f), ropeScalingFactor(1.0f),
        slidingWindowSize(4096),
        useRMSNorm(true), rmsNormEps(1e-5f),
        hiddenAct("silu"), tieWordEmbeddings(false) {}
  
  /// Get head dimension if not explicitly set
  int64_t getHeadDim() const {
    if (headDim > 0) return headDim;
    return hiddenSize / numAttentionHeads;
  }
  
  /// Check if using grouped query attention
  bool isGQA() const {
    return numKeyValueHeads > 0 && numKeyValueHeads < numAttentionHeads;
  }
  
  /// Get number of query heads per KV head
  int64_t getNumQueriesPerKV() const {
    if (numKeyValueHeads <= 0) return 1;
    return numAttentionHeads / numKeyValueHeads;
  }
};

//===----------------------------------------------------------------------===//
// Model-Specific Optimizers
//===----------------------------------------------------------------------===//

/// Base class for model-specific optimizations
class ModelOptimizer {
public:
  explicit ModelOptimizer(const ModelConfig& config);
  virtual ~ModelOptimizer() = default;
  
  /// Get optimized KV cache configuration
  virtual KVCacheConfig getOptimizedKVCacheConfig() const;
  
  /// Get recommended quantization config
  virtual QuantizationConfig getRecommendedQuantConfig() const;
  
  /// Get optimized block size for the model
  virtual int64_t getOptimizedBlockSize() const;
  
  /// Get recommended batch size for prefill
  virtual int64_t getRecommendedPrefillBatchSize() const;
  
  /// Get recommended batch size for decode
  virtual int64_t getRecommendedDecodeBatchSize() const;
  
  /// Check if sliding window attention should be used
  virtual bool useSlidingWindow() const;
  
  /// Get sliding window size
  virtual int64_t getSlidingWindowSize() const;
  
  /// Create optimized KV cache for this model
  virtual std::unique_ptr<PagedKVCache> createOptimizedKVCache(
      bool enableGPU = true) const;
  
protected:
  ModelConfig config_;
};

//===----------------------------------------------------------------------===//
// Llama Optimizer
//===----------------------------------------------------------------------===//

/// Optimizations specific to Llama models
class LlamaOptimizer : public ModelOptimizer {
public:
  /// Create optimizer for Llama model
  explicit LlamaOptimizer(const ModelConfig& config);
  
  /// Create optimizer from model size
  static LlamaOptimizer forLlama7B();
  static LlamaOptimizer forLlama13B();
  static LlamaOptimizer forLlama70B();
  
  /// Llama 2 variants
  static LlamaOptimizer forLlama2_7B();
  static LlamaOptimizer forLlama2_13B();
  static LlamaOptimizer forLlama2_70B();
  
  /// Llama 3 variants
  static LlamaOptimizer forLlama3_8B();
  static LlamaOptimizer forLlama3_70B();
  static LlamaOptimizer forLlama31_8B();
  static LlamaOptimizer forLlama31_70B();
  static LlamaOptimizer forLlama31_405B();
  
  // Overrides
  KVCacheConfig getOptimizedKVCacheConfig() const override;
  QuantizationConfig getRecommendedQuantConfig() const override;
  int64_t getOptimizedBlockSize() const override;
  int64_t getRecommendedPrefillBatchSize() const override;
  int64_t getRecommendedDecodeBatchSize() const override;
  
  /// Get RoPE scaling parameters for extended context
  std::pair<float, float> getRoPEScalingParams(int64_t targetContext) const;
  
  /// Estimate memory usage for given batch size and sequence length
  size_t estimateMemoryUsage(int64_t batchSize, int64_t seqLen) const;
};

//===----------------------------------------------------------------------===//
// Mistral Optimizer
//===----------------------------------------------------------------------===//

/// Optimizations specific to Mistral models
class MistralOptimizer : public ModelOptimizer {
public:
  explicit MistralOptimizer(const ModelConfig& config);
  
  /// Create optimizer for Mistral model variants
  static MistralOptimizer forMistral7B();
  static MistralOptimizer forMistral7BInstruct();
  static MistralOptimizer forMixtral8x7B();
  static MistralOptimizer forMixtral8x22B();
  
  // Overrides
  KVCacheConfig getOptimizedKVCacheConfig() const override;
  QuantizationConfig getRecommendedQuantConfig() const override;
  int64_t getOptimizedBlockSize() const override;
  bool useSlidingWindow() const override;
  int64_t getSlidingWindowSize() const override;
  
  /// Get sliding window optimized cache
  std::unique_ptr<PagedKVCache> createSlidingWindowCache(
      int64_t windowSize, bool enableGPU = true) const;
  
  /// Check if this is a MoE model
  bool isMixtral() const;
  
  /// Get number of experts for MoE models
  int64_t getNumExperts() const;
  
  /// Get top-k experts for routing
  int64_t getTopKExperts() const;
  
private:
  int64_t numExperts_;
  int64_t topKExperts_;
};

//===----------------------------------------------------------------------===//
// Phi Optimizer
//===----------------------------------------------------------------------===//

/// Optimizations specific to Phi models
class PhiOptimizer : public ModelOptimizer {
public:
  explicit PhiOptimizer(const ModelConfig& config);
  
  static PhiOptimizer forPhi2();
  static PhiOptimizer forPhi3Mini();
  static PhiOptimizer forPhi3Small();
  static PhiOptimizer forPhi3Medium();
  
  // Overrides
  KVCacheConfig getOptimizedKVCacheConfig() const override;
  int64_t getOptimizedBlockSize() const override;
  int64_t getRecommendedDecodeBatchSize() const override;
};

//===----------------------------------------------------------------------===//
// Model Registry
//===----------------------------------------------------------------------===//

/// Registry for model configurations and optimizers
class ModelRegistry {
public:
  static ModelRegistry& getInstance();
  
  /// Register a model configuration
  void registerModel(const std::string& name, const ModelConfig& config);
  
  /// Get model configuration by name
  const ModelConfig* getConfig(const std::string& name) const;
  
  /// Create optimizer for a model
  std::unique_ptr<ModelOptimizer> createOptimizer(
      const std::string& name) const;
  
  /// List all registered models
  std::vector<std::string> listModels() const;
  
  /// Check if model is registered
  bool hasModel(const std::string& name) const;
  
  /// Parse model name to get architecture and size
  static std::pair<ModelArchitecture, int64_t> parseModelName(
      const std::string& name);
  
private:
  ModelRegistry();
  void registerBuiltinModels();
  
  std::unordered_map<std::string, ModelConfig> configs_;
};

//===----------------------------------------------------------------------===//
// Attention Optimizations
//===----------------------------------------------------------------------===//

/// Optimized attention implementation selector
class AttentionOptimizationSelector {
public:
  /// Select best attention implementation for model
  static std::string selectImplementation(
      const ModelConfig& config,
      int64_t batchSize,
      int64_t seqLen,
      bool hasFlashAttention = true);
  
  /// Check if Flash Attention is beneficial
  static bool shouldUseFlashAttention(
      const ModelConfig& config,
      int64_t seqLen);
  
  /// Check if paged attention is beneficial
  static bool shouldUsePagedAttention(
      const ModelConfig& config,
      int64_t numSequences);
  
  /// Get optimal tile sizes for attention
  static std::tuple<int64_t, int64_t, int64_t> getOptimalTileSizes(
      const ModelConfig& config,
      int64_t seqLen);
};

//===----------------------------------------------------------------------===//
// RoPE Optimizations
//===----------------------------------------------------------------------===//

/// RoPE (Rotary Position Embedding) optimizer
class RoPEOptimizer {
public:
  RoPEOptimizer(RoPEType type, float theta, int64_t maxLen);
  
  /// Get scaling factor for extended context
  float getScalingFactor(int64_t targetLen) const;
  
  /// Get NTK-aware scaling parameters
  std::pair<float, float> getNTKParams(int64_t targetLen) const;
  
  /// Get YaRN scaling parameters
  struct YaRNParams {
    float scale;
    float alpha;
    float beta;
    int64_t originalMaxLen;
  };
  YaRNParams getYaRNParams(int64_t targetLen) const;
  
  /// Check if dynamic scaling is needed
  bool needsDynamicScaling(int64_t seqLen) const;
  
  /// Create optimized RoPE embedding table
  std::vector<float> createEmbeddingTable(
      int64_t headDim, int64_t maxLen, float scalingFactor = 1.0f) const;
  
private:
  RoPEType type_;
  float theta_;
  int64_t maxLen_;
};

//===----------------------------------------------------------------------===//
// Memory Optimization Utilities
//===----------------------------------------------------------------------===//

/// Memory usage estimator for models
class ModelMemoryEstimator {
public:
  explicit ModelMemoryEstimator(const ModelConfig& config);
  
  /// Estimate model weight memory
  size_t estimateWeightMemory(const std::string& dtype = "float16") const;
  
  /// Estimate KV cache memory for given configuration
  size_t estimateKVCacheMemory(
      int64_t batchSize, int64_t seqLen,
      const std::string& dtype = "float16") const;
  
  /// Estimate activation memory for given batch/sequence
  size_t estimateActivationMemory(
      int64_t batchSize, int64_t seqLen,
      const std::string& dtype = "float16") const;
  
  /// Estimate total memory usage
  size_t estimateTotalMemory(
      int64_t batchSize, int64_t seqLen,
      const std::string& dtype = "float16") const;
  
  /// Find maximum batch size for given memory budget
  int64_t findMaxBatchSize(
      size_t memoryBudget, int64_t seqLen,
      const std::string& dtype = "float16") const;
  
  /// Find maximum sequence length for given memory budget
  int64_t findMaxSeqLen(
      size_t memoryBudget, int64_t batchSize,
      const std::string& dtype = "float16") const;
  
  /// Print memory breakdown
  void printMemoryBreakdown(
      int64_t batchSize, int64_t seqLen,
      const std::string& dtype = "float16") const;
  
private:
  ModelConfig config_;
  
  size_t getDtypeSize(const std::string& dtype) const;
};

} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_MODELOPTIMIZATIONS_H
