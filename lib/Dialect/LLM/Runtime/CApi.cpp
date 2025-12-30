//===- CApi.cpp - LLMIR C API Implementation ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the C API for LLMIR runtime components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/CApi.h"
#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Dialect/LLM/Runtime/QuantizedKVCache.h"
#include "mlir/Dialect/LLM/Runtime/DistributedKVCache.h"
#include "mlir/Dialect/LLM/Runtime/SpeculativeKVCache.h"
#include "mlir/Dialect/LLM/Runtime/PrefixCache.h"
#include "mlir/Dialect/LLM/Runtime/ContinuousBatching.h"

#include <cstring>
#include <memory>
#include <unordered_map>

using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// Version Info
//===----------------------------------------------------------------------===//

extern "C" {

const char* llmir_get_version(void) {
  return "0.1.0";
}

bool llmir_has_cuda_support(void) {
#ifdef LLMIR_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

bool llmir_has_nccl_support(void) {
#ifdef LLMIR_ENABLE_NCCL
  return true;
#else
  return false;
#endif
}

bool llmir_has_metal_support(void) {
#ifdef LLMIR_ENABLE_METAL
  return true;
#else
  return false;
#endif
}

//===----------------------------------------------------------------------===//
// PagedKVCache Implementation
//===----------------------------------------------------------------------===//

LlmirKVCacheHandle llmir_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t dtype,
    bool enableGPU) {
  try {
    auto cache = std::make_unique<PagedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen);
    return static_cast<LlmirKVCacheHandle>(cache.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_kvcache_destroy(LlmirKVCacheHandle handle) {
  if (handle) {
    delete static_cast<PagedKVCache*>(handle);
  }
}

int32_t llmir_kvcache_append(
    LlmirKVCacheHandle handle,
    const void* keyData,
    const void* valueData,
    int32_t batchSize,
    int32_t seqLen,
    const int32_t* seqIds,
    int32_t* blockIndices) {
  if (!handle || !keyData || !valueData || !seqIds || !blockIndices) {
    return LLMIR_ERROR_INVALID_ARGUMENT;
  }
  
  try {
    auto* cache = static_cast<PagedKVCache*>(handle);
    
    for (int32_t i = 0; i < batchSize; ++i) {
      int32_t seqId = seqIds[i];
      std::vector<int64_t> indices = cache->appendSequence(
          seqId, 
          static_cast<const float*>(keyData) + i * seqLen * cache->getNumHeads() * cache->getHeadDim(),
          static_cast<const float*>(valueData) + i * seqLen * cache->getNumHeads() * cache->getHeadDim(),
          seqLen);
      
      for (size_t j = 0; j < indices.size() && j < static_cast<size_t>(cache->getNumLayers()); ++j) {
        blockIndices[i * cache->getNumLayers() + j] = static_cast<int32_t>(indices[j]);
      }
    }
    
    return LLMIR_SUCCESS;
  } catch (...) {
    return LLMIR_ERROR_INTERNAL;
  }
}

int32_t llmir_kvcache_lookup(
    LlmirKVCacheHandle handle,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int32_t batchSize,
    void* outputKeys,
    void* outputValues) {
  if (!handle || !blockIndices || !seqLens || !outputKeys || !outputValues) {
    return LLMIR_ERROR_INVALID_ARGUMENT;
  }
  
  try {
    auto* cache = static_cast<PagedKVCache*>(handle);
    
    for (int32_t i = 0; i < batchSize; ++i) {
      int32_t seqLen = seqLens[i];
      std::vector<int64_t> indices;
      int32_t numBlocks = (seqLen + cache->getBlockSize() - 1) / cache->getBlockSize();
      
      for (int32_t j = 0; j < numBlocks; ++j) {
        indices.push_back(blockIndices[i * cache->getNumLayers() + j]);
      }
      
      cache->lookupSequence(
          indices,
          static_cast<float*>(outputKeys) + i * seqLen * cache->getNumHeads() * cache->getHeadDim(),
          static_cast<float*>(outputValues) + i * seqLen * cache->getNumHeads() * cache->getHeadDim(),
          seqLen);
    }
    
    return LLMIR_SUCCESS;
  } catch (...) {
    return LLMIR_ERROR_INTERNAL;
  }
}

int32_t llmir_kvcache_clear_sequence(LlmirKVCacheHandle handle, int32_t seqId) {
  if (!handle) {
    return LLMIR_ERROR_INVALID_ARGUMENT;
  }
  
  try {
    auto* cache = static_cast<PagedKVCache*>(handle);
    cache->clearSequence(seqId);
    return LLMIR_SUCCESS;
  } catch (...) {
    return LLMIR_ERROR_INTERNAL;
  }
}

void llmir_kvcache_reset(LlmirKVCacheHandle handle) {
  if (handle) {
    auto* cache = static_cast<PagedKVCache*>(handle);
    cache->reset();
  }
}

int64_t llmir_kvcache_get_memory_usage(LlmirKVCacheHandle handle) {
  if (!handle) {
    return 0;
  }
  auto* cache = static_cast<PagedKVCache*>(handle);
  return cache->getMemoryUsage();
}

int32_t llmir_kvcache_get_num_sequences(LlmirKVCacheHandle handle) {
  if (!handle) {
    return 0;
  }
  auto* cache = static_cast<PagedKVCache*>(handle);
  return static_cast<int32_t>(cache->getNumActiveSequences());
}

int32_t llmir_kvcache_get_block_size(LlmirKVCacheHandle handle) {
  if (!handle) {
    return 0;
  }
  auto* cache = static_cast<PagedKVCache*>(handle);
  return static_cast<int32_t>(cache->getBlockSize());
}

int32_t llmir_kvcache_get_num_layers(LlmirKVCacheHandle handle) {
  if (!handle) {
    return 0;
  }
  auto* cache = static_cast<PagedKVCache*>(handle);
  return static_cast<int32_t>(cache->getNumLayers());
}

//===----------------------------------------------------------------------===//
// QuantizedKVCache Implementation
//===----------------------------------------------------------------------===//

LlmirQuantizedKVCacheHandle llmir_quantized_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t quantType,
    int32_t strategy,
    bool symmetric,
    int64_t groupSize,
    bool dynamicRange) {
  try {
    QuantizationType qType;
    switch (quantType) {
      case LLMIR_QUANT_INT8: qType = QuantizationType::INT8; break;
      case LLMIR_QUANT_INT4: qType = QuantizationType::INT4; break;
      case LLMIR_QUANT_FP8: qType = QuantizationType::FP8; break;
      default: qType = QuantizationType::None; break;
    }
    
    QuantizationConfig config;
    config.type = qType;
    config.symmetric = symmetric;
    config.groupSize = static_cast<int>(groupSize);
    
    auto cache = std::make_unique<QuantizedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, config);
    return static_cast<LlmirQuantizedKVCacheHandle>(cache.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_quantized_kvcache_destroy(LlmirQuantizedKVCacheHandle handle) {
  if (handle) {
    delete static_cast<QuantizedKVCache*>(handle);
  }
}

float llmir_quantized_kvcache_get_compression_ratio(
    LlmirQuantizedKVCacheHandle handle) {
  if (!handle) {
    return 1.0f;
  }
  auto* cache = static_cast<QuantizedKVCache*>(handle);
  return cache->getCompressionRatio();
}

float llmir_quantized_kvcache_get_accuracy_loss(
    LlmirQuantizedKVCacheHandle handle) {
  if (!handle) {
    return 0.0f;
  }
  auto* cache = static_cast<QuantizedKVCache*>(handle);
  return cache->estimateAccuracyLoss();
}

//===----------------------------------------------------------------------===//
// DistributedKVCache Implementation
//===----------------------------------------------------------------------===//

LlmirDistributedKVCacheHandle llmir_distributed_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t numDevices,
    const int32_t* deviceIds,
    int32_t strategy,
    bool enableNCCL) {
  try {
    ShardingStrategy shardStrategy;
    switch (strategy) {
      case LLMIR_SHARD_LAYER_WISE: shardStrategy = ShardingStrategy::LayerWise; break;
      case LLMIR_SHARD_HEAD_WISE: shardStrategy = ShardingStrategy::HeadWise; break;
      case LLMIR_SHARD_SEQUENCE_WISE: shardStrategy = ShardingStrategy::SequenceWise; break;
      default: shardStrategy = ShardingStrategy::Hybrid; break;
    }
    
    std::vector<int> devices;
    if (deviceIds) {
      devices.assign(deviceIds, deviceIds + numDevices);
    } else {
      for (int i = 0; i < numDevices; ++i) {
        devices.push_back(i);
      }
    }
    
    auto cache = std::make_unique<DistributedKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen,
        devices, shardStrategy);
    return static_cast<LlmirDistributedKVCacheHandle>(cache.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_distributed_kvcache_destroy(LlmirDistributedKVCacheHandle handle) {
  if (handle) {
    delete static_cast<DistributedKVCache*>(handle);
  }
}

void llmir_distributed_kvcache_get_per_device_memory(
    LlmirDistributedKVCacheHandle handle,
    int64_t* memoryPerDevice) {
  if (!handle || !memoryPerDevice) {
    return;
  }
  
  auto* cache = static_cast<DistributedKVCache*>(handle);
  auto usage = cache->getPerDeviceMemoryUsage();
  for (size_t i = 0; i < usage.size(); ++i) {
    memoryPerDevice[i] = usage[i];
  }
}

void llmir_distributed_kvcache_rebalance(LlmirDistributedKVCacheHandle handle) {
  if (handle) {
    auto* cache = static_cast<DistributedKVCache*>(handle);
    cache->rebalance();
  }
}

//===----------------------------------------------------------------------===//
// SpeculativeKVCache Implementation
//===----------------------------------------------------------------------===//

LlmirSpeculativeKVCacheHandle llmir_speculative_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t maxDraftTokens,
    int32_t maxBranches,
    bool enableTreeAttention) {
  try {
    SpeculativeConfig config;
    config.maxDraftTokens = maxDraftTokens;
    config.maxBranches = maxBranches;
    config.enableTreeAttention = enableTreeAttention;
    
    auto cache = std::make_unique<SpeculativeKVCache>(
        numLayers, numHeads, headDim, blockSize, maxSeqLen, config);
    return static_cast<LlmirSpeculativeKVCacheHandle>(cache.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_speculative_kvcache_destroy(LlmirSpeculativeKVCacheHandle handle) {
  if (handle) {
    delete static_cast<SpeculativeKVCache*>(handle);
  }
}

int32_t llmir_speculative_kvcache_create_branch(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId) {
  if (!handle) {
    return -1;
  }
  auto* cache = static_cast<SpeculativeKVCache*>(handle);
  return static_cast<int32_t>(cache->createBranch(seqId));
}

void llmir_speculative_kvcache_commit(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId,
    int32_t branchId,
    int32_t numAccepted) {
  if (handle) {
    auto* cache = static_cast<SpeculativeKVCache*>(handle);
    cache->commitBranch(seqId, branchId, numAccepted);
  }
}

void llmir_speculative_kvcache_rollback(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId,
    int32_t branchId) {
  if (handle) {
    auto* cache = static_cast<SpeculativeKVCache*>(handle);
    cache->rollbackBranch(seqId, branchId);
  }
}

//===----------------------------------------------------------------------===//
// PrefixCache Implementation
//===----------------------------------------------------------------------===//

LlmirPrefixCacheHandle llmir_prefix_cache_create(
    int32_t maxPrefixes,
    int64_t maxMemoryBytes,
    bool enableRadixTree,
    int32_t minPrefixLength) {
  try {
    auto cache = std::make_unique<PrefixCache>(
        maxPrefixes, maxMemoryBytes, minPrefixLength);
    return static_cast<LlmirPrefixCacheHandle>(cache.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_prefix_cache_destroy(LlmirPrefixCacheHandle handle) {
  if (handle) {
    delete static_cast<PrefixCache*>(handle);
  }
}

bool llmir_prefix_cache_insert(
    LlmirPrefixCacheHandle handle,
    const int32_t* tokens,
    int32_t numTokens,
    const int32_t* blockIndices,
    int32_t numBlocks) {
  if (!handle || !tokens || !blockIndices) {
    return false;
  }
  
  auto* cache = static_cast<PrefixCache*>(handle);
  std::vector<int64_t> tokenVec(tokens, tokens + numTokens);
  std::vector<int64_t> blockVec(blockIndices, blockIndices + numBlocks);
  
  return cache->cachePrefix(tokenVec, blockVec);
}

int32_t llmir_prefix_cache_lookup(
    LlmirPrefixCacheHandle handle,
    const int32_t* tokens,
    int32_t numTokens,
    int32_t* blockIndices,
    int32_t maxBlocks) {
  if (!handle || !tokens) {
    return 0;
  }
  
  auto* cache = static_cast<PrefixCache*>(handle);
  std::vector<int64_t> tokenVec(tokens, tokens + numTokens);
  
  auto result = cache->lookupPrefix(tokenVec);
  if (result.matchLength == 0) {
    return 0;
  }
  
  if (blockIndices) {
    size_t copySize = std::min(static_cast<size_t>(maxBlocks), result.blockIndices.size());
    for (size_t i = 0; i < copySize; ++i) {
      blockIndices[i] = static_cast<int32_t>(result.blockIndices[i]);
    }
  }
  
  return static_cast<int32_t>(result.matchLength);
}

float llmir_prefix_cache_get_hit_ratio(LlmirPrefixCacheHandle handle) {
  if (!handle) {
    return 0.0f;
  }
  auto* cache = static_cast<PrefixCache*>(handle);
  return cache->getHitRatio();
}

void llmir_prefix_cache_clear(LlmirPrefixCacheHandle handle) {
  if (handle) {
    auto* cache = static_cast<PrefixCache*>(handle);
    cache->clear();
  }
}

//===----------------------------------------------------------------------===//
// ContinuousBatchingEngine Implementation
//===----------------------------------------------------------------------===//

LlmirEngineHandle llmir_engine_create(
    LlmirKVCacheHandle cacheHandle,
    int32_t maxBatchSize,
    int32_t maxNumSeqs,
    int32_t chunkSize) {
  try {
    auto* cache = static_cast<PagedKVCache*>(cacheHandle);
    if (!cache) {
      return nullptr;
    }
    
    SchedulerConfig config;
    config.maxBatchSize = maxBatchSize;
    config.maxNumSeqs = maxNumSeqs;
    config.chunkSize = chunkSize;
    
    auto engine = std::make_unique<ContinuousBatchingEngine>(cache, config);
    return static_cast<LlmirEngineHandle>(engine.release());
  } catch (...) {
    return nullptr;
  }
}

void llmir_engine_destroy(LlmirEngineHandle handle) {
  if (handle) {
    delete static_cast<ContinuousBatchingEngine*>(handle);
  }
}

void llmir_engine_start(LlmirEngineHandle handle) {
  if (handle) {
    auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
    engine->start();
  }
}

void llmir_engine_stop(LlmirEngineHandle handle) {
  if (handle) {
    auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
    engine->stop();
  }
}

bool llmir_engine_is_running(LlmirEngineHandle handle) {
  if (!handle) {
    return false;
  }
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  return engine->isRunning();
}

int32_t llmir_engine_submit(
    LlmirEngineHandle handle,
    const int32_t* promptTokens,
    int32_t numTokens,
    float temperature,
    float topP,
    int32_t topK,
    int32_t maxTokens) {
  if (!handle || !promptTokens) {
    return -1;
  }
  
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  
  SamplingParams params;
  params.temperature = temperature;
  params.topP = topP;
  params.topK = topK;
  params.maxTokens = maxTokens;
  
  std::vector<int64_t> tokens(promptTokens, promptTokens + numTokens);
  
  return static_cast<int32_t>(engine->submitRequest(tokens, params));
}

bool llmir_engine_abort(LlmirEngineHandle handle, int32_t requestId) {
  if (!handle) {
    return false;
  }
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  return engine->abortRequest(requestId);
}

int32_t llmir_engine_step(LlmirEngineHandle handle) {
  if (!handle) {
    return 0;
  }
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  return static_cast<int32_t>(engine->step());
}

int32_t llmir_engine_get_output(
    LlmirEngineHandle handle,
    int32_t requestId,
    int32_t* outputTokens,
    int32_t maxTokens,
    bool* finished) {
  if (!handle) {
    return 0;
  }
  
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  auto output = engine->getOutput(requestId);
  
  if (finished) {
    *finished = output.finished;
  }
  
  size_t copySize = std::min(static_cast<size_t>(maxTokens), output.tokens.size());
  if (outputTokens) {
    for (size_t i = 0; i < copySize; ++i) {
      outputTokens[i] = static_cast<int32_t>(output.tokens[i]);
    }
  }
  
  return static_cast<int32_t>(copySize);
}

void llmir_engine_get_stats(
    LlmirEngineHandle handle,
    int32_t* totalRequests,
    int32_t* completedRequests,
    int32_t* pendingRequests,
    int64_t* totalTokens) {
  if (!handle) {
    return;
  }
  
  auto* engine = static_cast<ContinuousBatchingEngine*>(handle);
  auto stats = engine->getStats();
  
  if (totalRequests) *totalRequests = static_cast<int32_t>(stats.totalRequests);
  if (completedRequests) *completedRequests = static_cast<int32_t>(stats.completedRequests);
  if (pendingRequests) *pendingRequests = static_cast<int32_t>(stats.pendingRequests);
  if (totalTokens) *totalTokens = stats.totalTokens;
}

//===----------------------------------------------------------------------===//
// Profiler Implementation (Placeholder)
//===----------------------------------------------------------------------===//

LlmirProfilerHandle llmir_profiler_create(void) {
  // Placeholder - would create actual profiler
  return reinterpret_cast<LlmirProfilerHandle>(1);
}

void llmir_profiler_destroy(LlmirProfilerHandle handle) {
  // Placeholder
}

void llmir_profiler_start(LlmirProfilerHandle handle) {
  // Placeholder
}

void llmir_profiler_stop(LlmirProfilerHandle handle) {
  // Placeholder
}

void llmir_profiler_record_event(
    LlmirProfilerHandle handle,
    const char* name,
    int32_t eventType,
    float durationMs) {
  // Placeholder
}

bool llmir_profiler_export_trace(
    LlmirProfilerHandle handle,
    const char* filepath) {
  // Placeholder
  return false;
}

int32_t llmir_profiler_get_num_events(LlmirProfilerHandle handle) {
  // Placeholder
  return 0;
}

} // extern "C"
