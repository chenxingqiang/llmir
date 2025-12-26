//===- MultiQueryAttention.cpp - Multi-Query Attention implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Multi-Query Attention for optimized inference.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/CUDAKernels.h"
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Multi-Query Attention Implementation
//===----------------------------------------------------------------------===//

MultiQueryAttentionImpl::MultiQueryAttentionImpl(
    const AttentionConfig& config, Type elementType, bool useGPU)
    : config(config), elementType(elementType), useGPU(useGPU) {
  
  // Configure for multi-query attention if not already done
  if (config.variant != AttentionVariant::MULTI_QUERY) {
    this->config.variant = AttentionVariant::MULTI_QUERY;
    this->config.numKVHeads = 1;
    this->config.headGroupSize = config.numHeads;
  }
  
  // Ensure scale is set
  if (this->config.scale <= 0.0f && this->config.headDim > 0) {
    this->config.scale = 1.0f / std::sqrt(static_cast<float>(this->config.headDim));
  }
}

void MultiQueryAttentionImpl::compute(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    const void* attentionMask) {
  
  // Choose between GPU and CPU implementation
#if defined(LLMIR_ENABLE_CUDA)
  if (useGPU && cuda::isCUDAAvailable() && config.useCUDA) {
    computeMultiQueryAttentionCUDA(
        output, queries, keys, values,
        batchSize, seqLen, contextLen, attentionMask);
    return;
  }
#endif
  
  // Fall back to CPU implementation
  computeMultiQueryAttentionCPU(
      output, queries, keys, values,
      batchSize, seqLen, contextLen, attentionMask);
}

void MultiQueryAttentionImpl::computePaged(
    void* output,
    const void* queries,
    PagedKVCache* kvCache,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int64_t batchSize,
    int64_t seqLen) {
  
  if (!kvCache) {
    return;
  }
  
  // Find the maximum context length
  int64_t maxContextLen = 0;
  for (int64_t b = 0; b < batchSize; b++) {
    maxContextLen = std::max(maxContextLen, static_cast<int64_t>(seqLens[b]));
  }
  
  if (maxContextLen == 0) {
    return;
  }
  
  // Allocate temporary buffers for gathered keys and values
  // For MQA: keys and values have shape [batchSize, contextLen, 1, headDim]
  // (shared across all query heads)
  int64_t kvBufferSize = batchSize * maxContextLen * config.headDim * sizeof(float);
  std::vector<float> keysBuffer(batchSize * maxContextLen * config.headDim, 0.0f);
  std::vector<float> valuesBuffer(batchSize * maxContextLen * config.headDim, 0.0f);
  
  // Gather keys and values from KV cache for each batch
  for (int64_t b = 0; b < batchSize; b++) {
    int64_t contextLen = seqLens[b];
    if (contextLen == 0) continue;
    
    // Get sequence ID from block indices (stored in high bits)
    int32_t seqId = blockIndices[b * maxContextLen] >> 16;
    
    // Use KV cache's gather function to get the KV data
    float* keyDest = keysBuffer.data() + b * maxContextLen * config.headDim;
    float* valueDest = valuesBuffer.data() + b * maxContextLen * config.headDim;
    
    if (kvCache->gatherKVForAttention(
            keyDest, valueDest, seqId, 0, contextLen).failed()) {
      // If gather fails, zero out the buffers (already done during allocation)
      continue;
    }
  }
  
  // Compute attention with the gathered keys and values
  compute(
      output, queries, keysBuffer.data(), valuesBuffer.data(),
      batchSize, seqLen, maxContextLen,
      nullptr);
}

void MultiQueryAttentionImpl::computeMultiQueryAttentionCPU(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    const void* attentionMask) {
  
  // CPU implementation of multi-query attention
  // In MQA, each query head has its own parameters but keys and values are shared
  float* outputPtr = static_cast<float*>(output);
  const float* q = static_cast<const float*>(queries);
  const float* k = static_cast<const float*>(keys);
  const float* v = static_cast<const float*>(values);
  const float* mask = static_cast<const float*>(attentionMask);
  
  // Zero-initialize output
  std::memset(outputPtr, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
  
  // Process each batch
  for (int64_t b = 0; b < batchSize; b++) {
    // Process each query head (each head uses the same K,V)
    for (int64_t h = 0; h < config.numHeads; h++) {
      // For each query position
      for (int64_t queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        // Get query vector pointer
        // Query layout: [batch, seq, numHeads, headDim]
        const float* queryVec = q + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        
        // Output for this query
        float* outputVec = outputPtr + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        
        // Track max for numerical stability
        float maxVal = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(contextLen);
        
        // First pass: compute attention scores
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          // Apply causal mask if needed
          if (config.maskType == AttentionMaskType::CAUSAL && keyIdx > queryIdx) {
            scores[keyIdx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          
          // Apply sliding window mask if configured
          if (config.maskType == AttentionMaskType::SLIDING_WINDOW && 
              config.windowSize > 0 && 
              std::abs(keyIdx - queryIdx) > config.windowSize) {
            scores[keyIdx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          
          // Apply custom mask if provided
          if (config.maskType == AttentionMaskType::CUSTOM && mask) {
            int64_t maskIdx = b * seqLen * contextLen + queryIdx * contextLen + keyIdx;
            if (mask[maskIdx] == 0.0f) {
              scores[keyIdx] = -std::numeric_limits<float>::infinity();
              continue;
            }
          }
          
          // Get key vector pointer
          // Key layout for MQA: [batch, context, 1, headDim] - shared across heads
          // So we index only by batch and context position
          const float* keyVec = k + (b * contextLen + keyIdx) * config.headDim;
          
          // Compute dot product (Q·K)
          float score = 0.0f;
          for (int64_t d = 0; d < config.headDim; d++) {
            score += queryVec[d] * keyVec[d];
          }
          
          // Apply scaling factor
          score *= config.scale;
          
          // Update max value
          maxVal = std::max(maxVal, score);
          scores[keyIdx] = score;
        }
        
        // Second pass: compute softmax (exp and sum)
        float expSum = 0.0f;
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          if (scores[keyIdx] > -std::numeric_limits<float>::infinity() / 2) {
            scores[keyIdx] = std::exp(scores[keyIdx] - maxVal);
            expSum += scores[keyIdx];
          } else {
            scores[keyIdx] = 0.0f;
          }
        }
        
        // Third pass: compute weighted values
        if (expSum > 0.0f) {
          for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
            if (scores[keyIdx] == 0.0f) continue;
            
            // Get value vector pointer (shared across heads like keys)
            const float* valueVec = v + (b * contextLen + keyIdx) * config.headDim;
            
            // Normalize and apply to value vector
            float weight = scores[keyIdx] / expSum;
            
            // Accumulate weighted value to output
            for (int64_t d = 0; d < config.headDim; d++) {
              outputVec[d] += weight * valueVec[d];
            }
          }
        }
      }
    }
  }
}

void MultiQueryAttentionImpl::computeMultiQueryAttentionCUDA(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    const void* attentionMask) {
  
#if defined(LLMIR_ENABLE_CUDA)
  // GPU implementation using CUDA kernels
  cuda::launchMultiQueryAttentionKernel(
      output, queries, keys, values,
      batchSize, seqLen, contextLen,
      config.numHeads, config.headDim, config.scale,
      config.maskType, config.windowSize,
      config.cudaBlockSize, config.useTensorCores, config.useHalfPrecision);
#else
  // Fallback to CPU if CUDA is not available
  computeMultiQueryAttentionCPU(
      output, queries, keys, values,
      batchSize, seqLen, contextLen, attentionMask);
#endif
}

//===----------------------------------------------------------------------===//
// Grouped-Query Attention Implementation
//===----------------------------------------------------------------------===//

GroupedQueryAttentionImpl::GroupedQueryAttentionImpl(
    const AttentionConfig& config, Type elementType, bool useGPU)
    : config(config), elementType(elementType), useGPU(useGPU) {
  
  // Ensure proper GQA configuration
  if (this->config.numKVHeads <= 0) {
    this->config.numKVHeads = this->config.numHeads / 4; // Default grouping factor
  }
  if (this->config.headGroupSize <= 0) {
    this->config.headGroupSize = this->config.numHeads / this->config.numKVHeads;
  }
  
  // Ensure scale is set
  if (this->config.scale <= 0.0f && this->config.headDim > 0) {
    this->config.scale = 1.0f / std::sqrt(static_cast<float>(this->config.headDim));
  }
}

void GroupedQueryAttentionImpl::compute(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    const void* attentionMask) {
  
  // GQA implementation: K,V are shared within groups of query heads
  float* outputPtr = static_cast<float*>(output);
  const float* q = static_cast<const float*>(queries);
  const float* k = static_cast<const float*>(keys);
  const float* v = static_cast<const float*>(values);
  const float* mask = static_cast<const float*>(attentionMask);
  
  // Zero-initialize output
  std::memset(outputPtr, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
  
  int64_t headsPerGroup = config.headGroupSize;
  
  // Process each batch
  for (int64_t b = 0; b < batchSize; b++) {
    // Process each query head
    for (int64_t h = 0; h < config.numHeads; h++) {
      // Determine which KV head group this query head belongs to
      int64_t kvHeadIdx = h / headsPerGroup;
      
      // For each query position
      for (int64_t queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        // Get query vector pointer
        const float* queryVec = q + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        
        // Output for this query
        float* outputVec = outputPtr + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        
        // Track max for numerical stability
        float maxVal = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(contextLen);
        
        // First pass: compute attention scores
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          // Apply causal mask if needed
          if (config.maskType == AttentionMaskType::CAUSAL && keyIdx > queryIdx) {
            scores[keyIdx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          
          // Get key vector for this KV head group
          // Key layout for GQA: [batch, context, numKVHeads, headDim]
          const float* keyVec = k + ((b * contextLen + keyIdx) * config.numKVHeads + kvHeadIdx) * config.headDim;
          
          // Compute dot product
          float score = 0.0f;
          for (int64_t d = 0; d < config.headDim; d++) {
            score += queryVec[d] * keyVec[d];
          }
          
          // Apply scaling
          score *= config.scale;
          maxVal = std::max(maxVal, score);
          scores[keyIdx] = score;
        }
        
        // Second pass: softmax
        float expSum = 0.0f;
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          if (scores[keyIdx] > -std::numeric_limits<float>::infinity() / 2) {
            scores[keyIdx] = std::exp(scores[keyIdx] - maxVal);
            expSum += scores[keyIdx];
          } else {
            scores[keyIdx] = 0.0f;
          }
        }
        
        // Third pass: weighted sum
        if (expSum > 0.0f) {
          for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
            if (scores[keyIdx] == 0.0f) continue;
            
            const float* valueVec = v + ((b * contextLen + keyIdx) * config.numKVHeads + kvHeadIdx) * config.headDim;
            float weight = scores[keyIdx] / expSum;
            
            for (int64_t d = 0; d < config.headDim; d++) {
              outputVec[d] += weight * valueVec[d];
            }
          }
        }
      }
    }
  }
}

void GroupedQueryAttentionImpl::computePaged(
    void* output,
    const void* queries,
    PagedKVCache* kvCache,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int64_t batchSize,
    int64_t seqLen) {
  
  if (!kvCache) return;
  
  // Find max context length
  int64_t maxContextLen = 0;
  for (int64_t b = 0; b < batchSize; b++) {
    maxContextLen = std::max(maxContextLen, static_cast<int64_t>(seqLens[b]));
  }
  
  if (maxContextLen == 0) return;
  
  // Allocate temporary buffers for GQA: [batch, context, numKVHeads, headDim]
  std::vector<float> keysBuffer(batchSize * maxContextLen * config.numKVHeads * config.headDim, 0.0f);
  std::vector<float> valuesBuffer(batchSize * maxContextLen * config.numKVHeads * config.headDim, 0.0f);
  
  // Gather from KV cache
  for (int64_t b = 0; b < batchSize; b++) {
    int64_t contextLen = seqLens[b];
    if (contextLen == 0) continue;
    
    int32_t seqId = blockIndices[b * maxContextLen] >> 16;
    float* keyDest = keysBuffer.data() + b * maxContextLen * config.numKVHeads * config.headDim;
    float* valueDest = valuesBuffer.data() + b * maxContextLen * config.numKVHeads * config.headDim;
    
    kvCache->gatherKVForAttention(keyDest, valueDest, seqId, 0, contextLen);
  }
  
  compute(output, queries, keysBuffer.data(), valuesBuffer.data(),
          batchSize, seqLen, maxContextLen, nullptr);
}

//===----------------------------------------------------------------------===//
// Standard Attention Implementation
//===----------------------------------------------------------------------===//

StandardAttentionImpl::StandardAttentionImpl(
    const AttentionConfig& config, Type elementType, bool useGPU)
    : config(config), elementType(elementType), useGPU(useGPU) {
  
  if (this->config.scale <= 0.0f && this->config.headDim > 0) {
    this->config.scale = 1.0f / std::sqrt(static_cast<float>(this->config.headDim));
  }
}

void StandardAttentionImpl::compute(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    const void* attentionMask) {
  
  // Standard multi-head attention where each head has its own K,V
  float* outputPtr = static_cast<float*>(output);
  const float* q = static_cast<const float*>(queries);
  const float* k = static_cast<const float*>(keys);
  const float* v = static_cast<const float*>(values);
  const float* mask = static_cast<const float*>(attentionMask);
  
  std::memset(outputPtr, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
  
  for (int64_t b = 0; b < batchSize; b++) {
    for (int64_t h = 0; h < config.numHeads; h++) {
      for (int64_t queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        const float* queryVec = q + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        float* outputVec = outputPtr + ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
        
        float maxVal = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(contextLen);
        
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          if (config.maskType == AttentionMaskType::CAUSAL && keyIdx > queryIdx) {
            scores[keyIdx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          
          // Standard attention: each head has its own K,V
          const float* keyVec = k + ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
          
          float score = 0.0f;
          for (int64_t d = 0; d < config.headDim; d++) {
            score += queryVec[d] * keyVec[d];
          }
          score *= config.scale;
          maxVal = std::max(maxVal, score);
          scores[keyIdx] = score;
        }
        
        float expSum = 0.0f;
        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
          if (scores[keyIdx] > -std::numeric_limits<float>::infinity() / 2) {
            scores[keyIdx] = std::exp(scores[keyIdx] - maxVal);
            expSum += scores[keyIdx];
          } else {
            scores[keyIdx] = 0.0f;
          }
        }
        
        if (expSum > 0.0f) {
          for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
            if (scores[keyIdx] == 0.0f) continue;
            const float* valueVec = v + ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
            float weight = scores[keyIdx] / expSum;
            for (int64_t d = 0; d < config.headDim; d++) {
              outputVec[d] += weight * valueVec[d];
            }
          }
        }
      }
    }
  }
}

void StandardAttentionImpl::computePaged(
    void* output,
    const void* queries,
    PagedKVCache* kvCache,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int64_t batchSize,
    int64_t seqLen) {
  
  if (!kvCache) return;
  
  int64_t maxContextLen = 0;
  for (int64_t b = 0; b < batchSize; b++) {
    maxContextLen = std::max(maxContextLen, static_cast<int64_t>(seqLens[b]));
  }
  
  if (maxContextLen == 0) return;
  
  std::vector<float> keysBuffer(batchSize * maxContextLen * config.numHeads * config.headDim, 0.0f);
  std::vector<float> valuesBuffer(batchSize * maxContextLen * config.numHeads * config.headDim, 0.0f);
  
  for (int64_t b = 0; b < batchSize; b++) {
    int64_t contextLen = seqLens[b];
    if (contextLen == 0) continue;
    
    int32_t seqId = blockIndices[b * maxContextLen] >> 16;
    float* keyDest = keysBuffer.data() + b * maxContextLen * config.numHeads * config.headDim;
    float* valueDest = valuesBuffer.data() + b * maxContextLen * config.numHeads * config.headDim;
    
    kvCache->gatherKVForAttention(keyDest, valueDest, seqId, 0, contextLen);
  }
  
  compute(output, queries, keysBuffer.data(), valuesBuffer.data(),
          batchSize, seqLen, maxContextLen, nullptr);
}

//===----------------------------------------------------------------------===//
// Attention Variant Registration
//===----------------------------------------------------------------------===//

// Global registry for attention variant factories
static std::unordered_map<int, AttentionVariantFactory> attentionVariantRegistry;

void registerAttentionVariant(AttentionVariant variant, AttentionVariantFactory factory) {
  attentionVariantRegistry[static_cast<int>(variant)] = factory;
}

// Register factory functions
static bool registerAttentionVariants() {
  registerAttentionVariant(AttentionVariant::STANDARD, 
      [](const AttentionConfig& config, Type elementType, bool useGPU) {
        return std::make_unique<StandardAttentionImpl>(config, elementType, useGPU);
      });
  
  registerAttentionVariant(AttentionVariant::MULTI_QUERY, 
      [](const AttentionConfig& config, Type elementType, bool useGPU) {
        return std::make_unique<MultiQueryAttentionImpl>(config, elementType, useGPU);
      });
  
  registerAttentionVariant(AttentionVariant::GROUPED_QUERY, 
      [](const AttentionConfig& config, Type elementType, bool useGPU) {
        return std::make_unique<GroupedQueryAttentionImpl>(config, elementType, useGPU);
      });
  
  return true;
}

static bool variantsRegistered = registerAttentionVariants();

} // namespace runtime
} // namespace llm
} // namespace mlir
