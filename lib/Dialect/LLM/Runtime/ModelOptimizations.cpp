//===- ModelOptimizations.cpp - Model-specific optimizations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/ModelOptimizations.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// ModelOptimizer Implementation
//===----------------------------------------------------------------------===//

ModelOptimizer::ModelOptimizer(const ModelConfig& config) : config_(config) {}

KVCacheConfig ModelOptimizer::getOptimizedKVCacheConfig() const {
  KVCacheConfig kvConfig;
  kvConfig.numLayers = config_.numLayers;
  kvConfig.numHeads = config_.numKeyValueHeads > 0 ? 
                      config_.numKeyValueHeads : config_.numAttentionHeads;
  kvConfig.headDim = config_.getHeadDim();
  kvConfig.blockSize = getOptimizedBlockSize();
  kvConfig.maxSeqLen = config_.maxPositionEmbeddings;
  return kvConfig;
}

QuantizationConfig ModelOptimizer::getRecommendedQuantConfig() const {
  QuantizationConfig quantConfig;
  
  // Default to INT8 for good balance of speed and accuracy
  quantConfig.type = QuantizationType::INT8;
  quantConfig.strategy = QuantizationStrategy::PER_TENSOR;
  quantConfig.symmetric = true;
  quantConfig.groupSize = 128;
  
  return quantConfig;
}

int64_t ModelOptimizer::getOptimizedBlockSize() const {
  // Default block size based on head dimension
  int64_t headDim = config_.getHeadDim();
  
  if (headDim <= 64) {
    return 32;
  } else if (headDim <= 128) {
    return 16;
  } else {
    return 8;
  }
}

int64_t ModelOptimizer::getRecommendedPrefillBatchSize() const {
  // Larger models need smaller batch sizes due to memory constraints
  if (config_.numLayers >= 80) {
    return 1;
  } else if (config_.numLayers >= 40) {
    return 4;
  } else if (config_.numLayers >= 20) {
    return 8;
  } else {
    return 16;
  }
}

int64_t ModelOptimizer::getRecommendedDecodeBatchSize() const {
  // Decode is more memory efficient, can use larger batches
  return getRecommendedPrefillBatchSize() * 8;
}

bool ModelOptimizer::useSlidingWindow() const {
  return config_.attentionType == AttentionType::SLIDING_WINDOW ||
         config_.slidingWindowSize < config_.maxPositionEmbeddings;
}

int64_t ModelOptimizer::getSlidingWindowSize() const {
  return config_.slidingWindowSize;
}

std::unique_ptr<PagedKVCache> ModelOptimizer::createOptimizedKVCache(
    bool enableGPU) const {
  auto kvConfig = getOptimizedKVCacheConfig();
  
  return std::make_unique<PagedKVCache>(
      kvConfig.numLayers,
      kvConfig.numHeads,
      kvConfig.headDim,
      kvConfig.blockSize,
      kvConfig.maxSeqLen,
      Float16Type::get(nullptr),  // Default to fp16
      enableGPU
  );
}

//===----------------------------------------------------------------------===//
// LlamaOptimizer Implementation
//===----------------------------------------------------------------------===//

LlamaOptimizer::LlamaOptimizer(const ModelConfig& config) 
    : ModelOptimizer(config) {}

LlamaOptimizer LlamaOptimizer::forLlama7B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA;
  config.attentionType = AttentionType::MULTI_HEAD;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 32;
  config.headDim = 128;
  config.intermediateSize = 11008;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 2048;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama13B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA;
  config.attentionType = AttentionType::MULTI_HEAD;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 40;
  config.hiddenSize = 5120;
  config.numAttentionHeads = 40;
  config.numKeyValueHeads = 40;
  config.headDim = 128;
  config.intermediateSize = 13824;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 2048;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama70B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 80;
  config.hiddenSize = 8192;
  config.numAttentionHeads = 64;
  config.numKeyValueHeads = 8;  // GQA: 8 KV heads
  config.headDim = 128;
  config.intermediateSize = 28672;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 4096;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama2_7B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA2;
  config.attentionType = AttentionType::MULTI_HEAD;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 32;
  config.headDim = 128;
  config.intermediateSize = 11008;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 4096;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama2_13B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA2;
  config.attentionType = AttentionType::MULTI_HEAD;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 40;
  config.hiddenSize = 5120;
  config.numAttentionHeads = 40;
  config.numKeyValueHeads = 40;
  config.headDim = 128;
  config.intermediateSize = 13824;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 4096;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama2_70B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA2;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 80;
  config.hiddenSize = 8192;
  config.numAttentionHeads = 64;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 28672;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 4096;
  config.ropeTheta = 10000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama3_8B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;  // GQA
  config.headDim = 128;
  config.intermediateSize = 14336;
  config.vocabSize = 128256;
  config.maxPositionEmbeddings = 8192;
  config.ropeTheta = 500000.0f;  // Higher theta for longer context
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama3_70B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 80;
  config.hiddenSize = 8192;
  config.numAttentionHeads = 64;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 28672;
  config.vocabSize = 128256;
  config.maxPositionEmbeddings = 8192;
  config.ropeTheta = 500000.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama31_8B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;  // Uses scaling for 128k context
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 14336;
  config.vocabSize = 128256;
  config.maxPositionEmbeddings = 131072;  // 128k context
  config.ropeTheta = 500000.0f;
  config.ropeScalingFactor = 8.0f;  // 128k / 16k
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama31_70B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;
  config.numLayers = 80;
  config.hiddenSize = 8192;
  config.numAttentionHeads = 64;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 28672;
  config.vocabSize = 128256;
  config.maxPositionEmbeddings = 131072;
  config.ropeTheta = 500000.0f;
  config.ropeScalingFactor = 8.0f;
  return LlamaOptimizer(config);
}

LlamaOptimizer LlamaOptimizer::forLlama31_405B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::LLAMA3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;
  config.numLayers = 126;
  config.hiddenSize = 16384;
  config.numAttentionHeads = 128;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 53248;
  config.vocabSize = 128256;
  config.maxPositionEmbeddings = 131072;
  config.ropeTheta = 500000.0f;
  config.ropeScalingFactor = 8.0f;
  return LlamaOptimizer(config);
}

KVCacheConfig LlamaOptimizer::getOptimizedKVCacheConfig() const {
  KVCacheConfig kvConfig = ModelOptimizer::getOptimizedKVCacheConfig();
  
  // Llama models with GQA have fewer KV heads
  if (config_.isGQA()) {
    kvConfig.numHeads = config_.numKeyValueHeads;
  }
  
  return kvConfig;
}

QuantizationConfig LlamaOptimizer::getRecommendedQuantConfig() const {
  QuantizationConfig quantConfig;
  
  // Larger models benefit more from INT4
  if (config_.numLayers >= 80) {
    quantConfig.type = QuantizationType::INT4;
    quantConfig.groupSize = 128;
  } else {
    quantConfig.type = QuantizationType::INT8;
    quantConfig.groupSize = 128;
  }
  
  quantConfig.strategy = QuantizationStrategy::PER_CHANNEL;
  quantConfig.symmetric = true;
  
  return quantConfig;
}

int64_t LlamaOptimizer::getOptimizedBlockSize() const {
  // Llama works well with block size 16
  return 16;
}

int64_t LlamaOptimizer::getRecommendedPrefillBatchSize() const {
  // Based on model size
  if (config_.numLayers >= 126) {  // 405B
    return 1;
  } else if (config_.numLayers >= 80) {  // 70B
    return 2;
  } else if (config_.numLayers >= 40) {  // 13B
    return 8;
  } else {  // 7B/8B
    return 16;
  }
}

int64_t LlamaOptimizer::getRecommendedDecodeBatchSize() const {
  // Decode can handle larger batches
  int64_t prefillBatch = getRecommendedPrefillBatchSize();
  
  if (config_.numLayers >= 80) {
    return prefillBatch * 4;
  } else {
    return prefillBatch * 8;
  }
}

std::pair<float, float> LlamaOptimizer::getRoPEScalingParams(
    int64_t targetContext) const {
  float originalMaxLen = static_cast<float>(config_.maxPositionEmbeddings);
  float targetLen = static_cast<float>(targetContext);
  
  float scalingFactor = targetLen / originalMaxLen;
  float adjustedTheta = config_.ropeTheta;
  
  // NTK-aware scaling
  if (scalingFactor > 1.0f) {
    // Increase theta proportionally for longer contexts
    adjustedTheta = config_.ropeTheta * std::pow(scalingFactor, 
        2.0f / (config_.getHeadDim() - 2.0f));
  }
  
  return {scalingFactor, adjustedTheta};
}

size_t LlamaOptimizer::estimateMemoryUsage(int64_t batchSize, 
                                            int64_t seqLen) const {
  ModelMemoryEstimator estimator(config_);
  return estimator.estimateTotalMemory(batchSize, seqLen);
}

//===----------------------------------------------------------------------===//
// MistralOptimizer Implementation
//===----------------------------------------------------------------------===//

MistralOptimizer::MistralOptimizer(const ModelConfig& config)
    : ModelOptimizer(config), numExperts_(0), topKExperts_(0) {}

MistralOptimizer MistralOptimizer::forMistral7B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::MISTRAL;
  config.attentionType = AttentionType::SLIDING_WINDOW;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;  // GQA
  config.headDim = 128;
  config.intermediateSize = 14336;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 32768;
  config.slidingWindowSize = 4096;  // Sliding window
  config.ropeTheta = 10000.0f;
  return MistralOptimizer(config);
}

MistralOptimizer MistralOptimizer::forMistral7BInstruct() {
  return forMistral7B();  // Same architecture
}

MistralOptimizer MistralOptimizer::forMixtral8x7B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::MIXTRAL;
  config.attentionType = AttentionType::SLIDING_WINDOW;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 14336;
  config.vocabSize = 32000;
  config.maxPositionEmbeddings = 32768;
  config.slidingWindowSize = 4096;
  config.ropeTheta = 1000000.0f;
  
  MistralOptimizer optimizer(config);
  optimizer.numExperts_ = 8;
  optimizer.topKExperts_ = 2;
  return optimizer;
}

MistralOptimizer MistralOptimizer::forMixtral8x22B() {
  ModelConfig config;
  config.architecture = ModelArchitecture::MIXTRAL;
  config.attentionType = AttentionType::SLIDING_WINDOW;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 56;
  config.hiddenSize = 6144;
  config.numAttentionHeads = 48;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 16384;
  config.vocabSize = 32768;
  config.maxPositionEmbeddings = 65536;
  config.slidingWindowSize = 4096;
  config.ropeTheta = 1000000.0f;
  
  MistralOptimizer optimizer(config);
  optimizer.numExperts_ = 8;
  optimizer.topKExperts_ = 2;
  return optimizer;
}

KVCacheConfig MistralOptimizer::getOptimizedKVCacheConfig() const {
  KVCacheConfig kvConfig = ModelOptimizer::getOptimizedKVCacheConfig();
  
  // Use sliding window size as effective max length for KV cache
  if (useSlidingWindow()) {
    kvConfig.maxSeqLen = std::min(kvConfig.maxSeqLen, 
                                   config_.slidingWindowSize);
  }
  
  return kvConfig;
}

QuantizationConfig MistralOptimizer::getRecommendedQuantConfig() const {
  QuantizationConfig quantConfig;
  
  // Mistral models work well with INT8
  quantConfig.type = QuantizationType::INT8;
  quantConfig.strategy = QuantizationStrategy::PER_CHANNEL;
  quantConfig.symmetric = true;
  quantConfig.groupSize = 128;
  
  return quantConfig;
}

int64_t MistralOptimizer::getOptimizedBlockSize() const {
  // Smaller block size for sliding window efficiency
  return 8;
}

bool MistralOptimizer::useSlidingWindow() const {
  return config_.slidingWindowSize > 0 && 
         config_.slidingWindowSize < config_.maxPositionEmbeddings;
}

int64_t MistralOptimizer::getSlidingWindowSize() const {
  return config_.slidingWindowSize;
}

std::unique_ptr<PagedKVCache> MistralOptimizer::createSlidingWindowCache(
    int64_t windowSize, bool enableGPU) const {
  auto kvConfig = getOptimizedKVCacheConfig();
  kvConfig.maxSeqLen = windowSize;
  
  return std::make_unique<PagedKVCache>(
      kvConfig.numLayers,
      kvConfig.numHeads,
      kvConfig.headDim,
      kvConfig.blockSize,
      kvConfig.maxSeqLen,
      Float16Type::get(nullptr),
      enableGPU
  );
}

bool MistralOptimizer::isMixtral() const {
  return config_.architecture == ModelArchitecture::MIXTRAL;
}

int64_t MistralOptimizer::getNumExperts() const {
  return numExperts_;
}

int64_t MistralOptimizer::getTopKExperts() const {
  return topKExperts_;
}

//===----------------------------------------------------------------------===//
// PhiOptimizer Implementation
//===----------------------------------------------------------------------===//

PhiOptimizer::PhiOptimizer(const ModelConfig& config) : ModelOptimizer(config) {}

PhiOptimizer PhiOptimizer::forPhi2() {
  ModelConfig config;
  config.architecture = ModelArchitecture::PHI;
  config.attentionType = AttentionType::MULTI_HEAD;
  config.ropeType = RoPEType::STANDARD;
  config.numLayers = 32;
  config.hiddenSize = 2560;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 32;
  config.headDim = 80;
  config.intermediateSize = 10240;
  config.vocabSize = 51200;
  config.maxPositionEmbeddings = 2048;
  config.ropeTheta = 10000.0f;
  config.hiddenAct = "gelu_new";
  return PhiOptimizer(config);
}

PhiOptimizer PhiOptimizer::forPhi3Mini() {
  ModelConfig config;
  config.architecture = ModelArchitecture::PHI3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;
  config.numLayers = 32;
  config.hiddenSize = 3072;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;
  config.headDim = 96;
  config.intermediateSize = 8192;
  config.vocabSize = 32064;
  config.maxPositionEmbeddings = 131072;
  config.ropeTheta = 10000.0f;
  config.ropeScalingFactor = 1.0f;
  config.hiddenAct = "silu";
  return PhiOptimizer(config);
}

PhiOptimizer PhiOptimizer::forPhi3Small() {
  ModelConfig config;
  config.architecture = ModelArchitecture::PHI3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;
  config.numLayers = 32;
  config.hiddenSize = 4096;
  config.numAttentionHeads = 32;
  config.numKeyValueHeads = 8;
  config.headDim = 128;
  config.intermediateSize = 14336;
  config.vocabSize = 100352;
  config.maxPositionEmbeddings = 131072;
  config.ropeTheta = 10000.0f;
  config.hiddenAct = "silu";
  return PhiOptimizer(config);
}

PhiOptimizer PhiOptimizer::forPhi3Medium() {
  ModelConfig config;
  config.architecture = ModelArchitecture::PHI3;
  config.attentionType = AttentionType::GROUPED_QUERY;
  config.ropeType = RoPEType::SCALED;
  config.numLayers = 40;
  config.hiddenSize = 5120;
  config.numAttentionHeads = 40;
  config.numKeyValueHeads = 10;
  config.headDim = 128;
  config.intermediateSize = 17920;
  config.vocabSize = 32064;
  config.maxPositionEmbeddings = 131072;
  config.ropeTheta = 10000.0f;
  config.hiddenAct = "silu";
  return PhiOptimizer(config);
}

KVCacheConfig PhiOptimizer::getOptimizedKVCacheConfig() const {
  return ModelOptimizer::getOptimizedKVCacheConfig();
}

int64_t PhiOptimizer::getOptimizedBlockSize() const {
  // Phi models have smaller head dimensions, use larger blocks
  if (config_.headDim <= 80) {
    return 32;
  }
  return 16;
}

int64_t PhiOptimizer::getRecommendedDecodeBatchSize() const {
  // Phi models are smaller, can handle larger batches
  return 256;
}

//===----------------------------------------------------------------------===//
// ModelRegistry Implementation
//===----------------------------------------------------------------------===//

ModelRegistry& ModelRegistry::getInstance() {
  static ModelRegistry instance;
  return instance;
}

ModelRegistry::ModelRegistry() {
  registerBuiltinModels();
}

void ModelRegistry::registerBuiltinModels() {
  // Register Llama models
  registerModel("llama-7b", LlamaOptimizer::forLlama7B().config_);
  registerModel("llama-13b", LlamaOptimizer::forLlama13B().config_);
  registerModel("llama-70b", LlamaOptimizer::forLlama70B().config_);
  registerModel("llama2-7b", LlamaOptimizer::forLlama2_7B().config_);
  registerModel("llama2-13b", LlamaOptimizer::forLlama2_13B().config_);
  registerModel("llama2-70b", LlamaOptimizer::forLlama2_70B().config_);
  registerModel("llama3-8b", LlamaOptimizer::forLlama3_8B().config_);
  registerModel("llama3-70b", LlamaOptimizer::forLlama3_70B().config_);
  registerModel("llama3.1-8b", LlamaOptimizer::forLlama31_8B().config_);
  registerModel("llama3.1-70b", LlamaOptimizer::forLlama31_70B().config_);
  registerModel("llama3.1-405b", LlamaOptimizer::forLlama31_405B().config_);
  
  // Register Mistral models
  registerModel("mistral-7b", MistralOptimizer::forMistral7B().config_);
  registerModel("mistral-7b-instruct", MistralOptimizer::forMistral7BInstruct().config_);
  registerModel("mixtral-8x7b", MistralOptimizer::forMixtral8x7B().config_);
  registerModel("mixtral-8x22b", MistralOptimizer::forMixtral8x22B().config_);
  
  // Register Phi models
  registerModel("phi-2", PhiOptimizer::forPhi2().config_);
  registerModel("phi-3-mini", PhiOptimizer::forPhi3Mini().config_);
  registerModel("phi-3-small", PhiOptimizer::forPhi3Small().config_);
  registerModel("phi-3-medium", PhiOptimizer::forPhi3Medium().config_);
}

void ModelRegistry::registerModel(const std::string& name, 
                                   const ModelConfig& config) {
  configs_[name] = config;
}

const ModelConfig* ModelRegistry::getConfig(const std::string& name) const {
  auto it = configs_.find(name);
  if (it != configs_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::unique_ptr<ModelOptimizer> ModelRegistry::createOptimizer(
    const std::string& name) const {
  const ModelConfig* config = getConfig(name);
  if (!config) {
    return nullptr;
  }
  
  switch (config->architecture) {
    case ModelArchitecture::LLAMA:
    case ModelArchitecture::LLAMA2:
    case ModelArchitecture::LLAMA3:
      return std::make_unique<LlamaOptimizer>(*config);
    case ModelArchitecture::MISTRAL:
    case ModelArchitecture::MIXTRAL:
      return std::make_unique<MistralOptimizer>(*config);
    case ModelArchitecture::PHI:
    case ModelArchitecture::PHI3:
      return std::make_unique<PhiOptimizer>(*config);
    default:
      return std::make_unique<ModelOptimizer>(*config);
  }
}

std::vector<std::string> ModelRegistry::listModels() const {
  std::vector<std::string> names;
  names.reserve(configs_.size());
  for (const auto& pair : configs_) {
    names.push_back(pair.first);
  }
  return names;
}

bool ModelRegistry::hasModel(const std::string& name) const {
  return configs_.find(name) != configs_.end();
}

std::pair<ModelArchitecture, int64_t> ModelRegistry::parseModelName(
    const std::string& name) {
  // Simple parser for model names like "llama-7b", "mistral-7b", etc.
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  
  ModelArchitecture arch = ModelArchitecture::CUSTOM;
  int64_t size = 0;
  
  if (lower.find("llama3.1") != std::string::npos) {
    arch = ModelArchitecture::LLAMA3;
  } else if (lower.find("llama3") != std::string::npos) {
    arch = ModelArchitecture::LLAMA3;
  } else if (lower.find("llama2") != std::string::npos) {
    arch = ModelArchitecture::LLAMA2;
  } else if (lower.find("llama") != std::string::npos) {
    arch = ModelArchitecture::LLAMA;
  } else if (lower.find("mixtral") != std::string::npos) {
    arch = ModelArchitecture::MIXTRAL;
  } else if (lower.find("mistral") != std::string::npos) {
    arch = ModelArchitecture::MISTRAL;
  } else if (lower.find("phi-3") != std::string::npos || 
             lower.find("phi3") != std::string::npos) {
    arch = ModelArchitecture::PHI3;
  } else if (lower.find("phi") != std::string::npos) {
    arch = ModelArchitecture::PHI;
  }
  
  // Extract size
  size_t pos = lower.find_first_of("0123456789");
  if (pos != std::string::npos) {
    size = std::stoll(lower.substr(pos));
  }
  
  return {arch, size};
}

//===----------------------------------------------------------------------===//
// AttentionOptimizationSelector Implementation
//===----------------------------------------------------------------------===//

std::string AttentionOptimizationSelector::selectImplementation(
    const ModelConfig& config,
    int64_t batchSize,
    int64_t seqLen,
    bool hasFlashAttention) {
  
  // Use Flash Attention for long sequences
  if (hasFlashAttention && shouldUseFlashAttention(config, seqLen)) {
    return "flash_attention";
  }
  
  // Use paged attention for serving many sequences
  if (shouldUsePagedAttention(config, batchSize)) {
    return "paged_attention";
  }
  
  // Use GQA-optimized kernel for GQA models
  if (config.isGQA()) {
    return "gqa_attention";
  }
  
  // Default to standard attention
  return "standard_attention";
}

bool AttentionOptimizationSelector::shouldUseFlashAttention(
    const ModelConfig& config, int64_t seqLen) {
  // Flash Attention is beneficial for longer sequences
  return seqLen >= 512;
}

bool AttentionOptimizationSelector::shouldUsePagedAttention(
    const ModelConfig& config, int64_t numSequences) {
  // Paged attention is beneficial for serving multiple sequences
  return numSequences >= 4;
}

std::tuple<int64_t, int64_t, int64_t> AttentionOptimizationSelector::getOptimalTileSizes(
    const ModelConfig& config, int64_t seqLen) {
  int64_t headDim = config.getHeadDim();
  
  // Tile sizes for Flash Attention
  int64_t blockM = 128;
  int64_t blockN = 64;
  int64_t blockK = headDim;
  
  // Adjust for smaller head dimensions
  if (headDim <= 64) {
    blockN = 128;
  }
  
  return {blockM, blockN, blockK};
}

//===----------------------------------------------------------------------===//
// RoPEOptimizer Implementation
//===----------------------------------------------------------------------===//

RoPEOptimizer::RoPEOptimizer(RoPEType type, float theta, int64_t maxLen)
    : type_(type), theta_(theta), maxLen_(maxLen) {}

float RoPEOptimizer::getScalingFactor(int64_t targetLen) const {
  if (targetLen <= maxLen_) {
    return 1.0f;
  }
  return static_cast<float>(targetLen) / static_cast<float>(maxLen_);
}

std::pair<float, float> RoPEOptimizer::getNTKParams(int64_t targetLen) const {
  float scale = getScalingFactor(targetLen);
  
  if (scale <= 1.0f) {
    return {1.0f, theta_};
  }
  
  // NTK-aware scaling
  float adjustedTheta = theta_ * std::pow(scale, 64.0f / 62.0f);
  
  return {scale, adjustedTheta};
}

RoPEOptimizer::YaRNParams RoPEOptimizer::getYaRNParams(int64_t targetLen) const {
  YaRNParams params;
  params.originalMaxLen = maxLen_;
  params.scale = getScalingFactor(targetLen);
  
  if (params.scale <= 1.0f) {
    params.alpha = 1.0f;
    params.beta = 1.0f;
    return params;
  }
  
  // YaRN parameters (from the paper)
  params.alpha = 1.0f;
  params.beta = 32.0f;
  
  return params;
}

bool RoPEOptimizer::needsDynamicScaling(int64_t seqLen) const {
  return seqLen > maxLen_;
}

std::vector<float> RoPEOptimizer::createEmbeddingTable(
    int64_t headDim, int64_t maxLen, float scalingFactor) const {
  std::vector<float> table(headDim * maxLen);
  
  float adjustedTheta = theta_;
  if (scalingFactor > 1.0f && type_ == RoPEType::SCALED) {
    adjustedTheta = theta_ * scalingFactor;
  }
  
  for (int64_t pos = 0; pos < maxLen; ++pos) {
    for (int64_t i = 0; i < headDim / 2; ++i) {
      float freq = 1.0f / std::pow(adjustedTheta, 
                                    2.0f * i / static_cast<float>(headDim));
      float angle = pos * freq;
      
      // Store cos and sin values
      table[pos * headDim + 2 * i] = std::cos(angle);
      table[pos * headDim + 2 * i + 1] = std::sin(angle);
    }
  }
  
  return table;
}

//===----------------------------------------------------------------------===//
// ModelMemoryEstimator Implementation
//===----------------------------------------------------------------------===//

ModelMemoryEstimator::ModelMemoryEstimator(const ModelConfig& config)
    : config_(config) {}

size_t ModelMemoryEstimator::getDtypeSize(const std::string& dtype) const {
  if (dtype == "float32" || dtype == "fp32") return 4;
  if (dtype == "float16" || dtype == "fp16") return 2;
  if (dtype == "bfloat16" || dtype == "bf16") return 2;
  if (dtype == "int8") return 1;
  if (dtype == "int4") return 1;  // Packed, but estimate as 1
  return 2;  // Default to fp16
}

size_t ModelMemoryEstimator::estimateWeightMemory(const std::string& dtype) const {
  size_t dtypeSize = getDtypeSize(dtype);
  size_t totalParams = 0;
  
  // Embedding
  totalParams += config_.vocabSize * config_.hiddenSize;
  
  // Per-layer parameters
  size_t perLayer = 0;
  
  // Self-attention
  // Q, K, V projections
  perLayer += config_.hiddenSize * config_.numAttentionHeads * config_.getHeadDim();
  perLayer += config_.hiddenSize * config_.numKeyValueHeads * config_.getHeadDim();
  perLayer += config_.hiddenSize * config_.numKeyValueHeads * config_.getHeadDim();
  // Output projection
  perLayer += config_.numAttentionHeads * config_.getHeadDim() * config_.hiddenSize;
  
  // MLP
  perLayer += config_.hiddenSize * config_.intermediateSize;  // Gate
  perLayer += config_.hiddenSize * config_.intermediateSize;  // Up
  perLayer += config_.intermediateSize * config_.hiddenSize;  // Down
  
  // Layer norms (2 per layer)
  perLayer += 2 * config_.hiddenSize;
  
  totalParams += perLayer * config_.numLayers;
  
  // Final layer norm
  totalParams += config_.hiddenSize;
  
  // LM head (if not tied)
  if (!config_.tieWordEmbeddings) {
    totalParams += config_.hiddenSize * config_.vocabSize;
  }
  
  return totalParams * dtypeSize;
}

size_t ModelMemoryEstimator::estimateKVCacheMemory(
    int64_t batchSize, int64_t seqLen, const std::string& dtype) const {
  size_t dtypeSize = getDtypeSize(dtype);
  
  // KV cache size per layer: 2 (K and V) * batch * seq * num_kv_heads * head_dim
  size_t perLayer = 2 * batchSize * seqLen * 
                    config_.numKeyValueHeads * config_.getHeadDim() * dtypeSize;
  
  return perLayer * config_.numLayers;
}

size_t ModelMemoryEstimator::estimateActivationMemory(
    int64_t batchSize, int64_t seqLen, const std::string& dtype) const {
  size_t dtypeSize = getDtypeSize(dtype);
  
  // Rough estimate: hidden states at each point
  size_t activation = batchSize * seqLen * config_.hiddenSize * dtypeSize;
  
  // Multiply by factor for intermediate activations
  return activation * 3;  // Conservative estimate
}

size_t ModelMemoryEstimator::estimateTotalMemory(
    int64_t batchSize, int64_t seqLen, const std::string& dtype) const {
  return estimateWeightMemory(dtype) + 
         estimateKVCacheMemory(batchSize, seqLen, dtype) +
         estimateActivationMemory(batchSize, seqLen, dtype);
}

int64_t ModelMemoryEstimator::findMaxBatchSize(
    size_t memoryBudget, int64_t seqLen, const std::string& dtype) const {
  size_t weightMemory = estimateWeightMemory(dtype);
  
  if (weightMemory >= memoryBudget) {
    return 0;  // Not enough memory for weights
  }
  
  size_t availableMemory = memoryBudget - weightMemory;
  
  // Binary search for max batch size
  int64_t low = 1, high = 1024;
  int64_t result = 0;
  
  while (low <= high) {
    int64_t mid = (low + high) / 2;
    size_t kvMemory = estimateKVCacheMemory(mid, seqLen, dtype);
    size_t actMemory = estimateActivationMemory(mid, seqLen, dtype);
    
    if (kvMemory + actMemory <= availableMemory) {
      result = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  
  return result;
}

int64_t ModelMemoryEstimator::findMaxSeqLen(
    size_t memoryBudget, int64_t batchSize, const std::string& dtype) const {
  size_t weightMemory = estimateWeightMemory(dtype);
  
  if (weightMemory >= memoryBudget) {
    return 0;
  }
  
  size_t availableMemory = memoryBudget - weightMemory;
  
  // Binary search for max sequence length
  int64_t low = 1, high = config_.maxPositionEmbeddings;
  int64_t result = 0;
  
  while (low <= high) {
    int64_t mid = (low + high) / 2;
    size_t kvMemory = estimateKVCacheMemory(batchSize, mid, dtype);
    size_t actMemory = estimateActivationMemory(batchSize, mid, dtype);
    
    if (kvMemory + actMemory <= availableMemory) {
      result = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  
  return result;
}

void ModelMemoryEstimator::printMemoryBreakdown(
    int64_t batchSize, int64_t seqLen, const std::string& dtype) const {
  size_t weightMem = estimateWeightMemory(dtype);
  size_t kvMem = estimateKVCacheMemory(batchSize, seqLen, dtype);
  size_t actMem = estimateActivationMemory(batchSize, seqLen, dtype);
  size_t totalMem = weightMem + kvMem + actMem;
  
  std::cout << "\n=== Memory Breakdown ===\n";
  std::cout << "Model weights:  " << weightMem / 1e9 << " GB ("
            << (100.0 * weightMem / totalMem) << "%)\n";
  std::cout << "KV Cache:       " << kvMem / 1e9 << " GB ("
            << (100.0 * kvMem / totalMem) << "%)\n";
  std::cout << "Activations:    " << actMem / 1e9 << " GB ("
            << (100.0 * actMem / totalMem) << "%)\n";
  std::cout << "Total:          " << totalMem / 1e9 << " GB\n";
}

} // namespace llm
} // namespace mlir
