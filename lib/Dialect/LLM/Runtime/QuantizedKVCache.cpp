//===- QuantizedKVCache.cpp - Quantized KV Cache Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements quantized KV cache support for the LLM dialect.
// It provides INT8 and INT4 quantization for reduced memory footprint.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/QuantizedKVCache.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <chrono>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// QuantizedKVBlock Implementation
//===----------------------------------------------------------------------===//

QuantizedKVBlock::QuantizedKVBlock(void* quantizedKeyPtr, void* quantizedValuePtr,
                                   const QuantizationParams& keyParams,
                                   const QuantizationParams& valueParams,
                                   int64_t blockSize, int64_t headDim)
    : quantizedKeyPtr_(quantizedKeyPtr), quantizedValuePtr_(quantizedValuePtr),
      keyParams_(keyParams), valueParams_(valueParams),
      blockSize_(blockSize), headDim_(headDim),
      usedSlots_(0), refCount_(0), lastAccessTime_(0) {}

QuantizedKVBlock::~QuantizedKVBlock() {
  // Memory is managed by the allocator, not the block itself
}

LogicalResult QuantizedKVBlock::quantizeAndStore(const float* keyData, 
                                                  const float* valueData,
                                                  int64_t numTokens, 
                                                  int64_t startOffset) {
  if (!keyData || !valueData) {
    return failure();
  }
  
  if (startOffset + numTokens > blockSize_) {
    return failure();
  }
  
  int64_t numElements = numTokens * headDim_;
  int64_t bytesPerElement = keyParams_.numBits == 8 ? 1 : 
                            (keyParams_.numBits == 4 ? 1 : 2); // INT4 packs 2 values per byte
  
  // Calculate offset in quantized storage
  int64_t byteOffset = startOffset * headDim_;
  if (keyParams_.numBits == 4) {
    byteOffset /= 2; // INT4 packs 2 values per byte
  }
  
  // Quantize and store keys
  int8_t* keyDst = static_cast<int8_t*>(quantizedKeyPtr_) + byteOffset;
  quantizeTensor(keyData, keyDst, keyParams_, numElements);
  
  // Quantize and store values
  int8_t* valueDst = static_cast<int8_t*>(quantizedValuePtr_) + byteOffset;
  quantizeTensor(valueData, valueDst, valueParams_, numElements);
  
  usedSlots_ += numTokens;
  return success();
}

LogicalResult QuantizedKVBlock::dequantizeAndLoad(float* keyData, 
                                                   float* valueData,
                                                   int64_t numTokens, 
                                                   int64_t startOffset) const {
  if (!keyData || !valueData) {
    return failure();
  }
  
  if (startOffset + numTokens > usedSlots_) {
    return failure();
  }
  
  int64_t numElements = numTokens * headDim_;
  
  // Calculate offset in quantized storage
  int64_t byteOffset = startOffset * headDim_;
  if (keyParams_.numBits == 4) {
    byteOffset /= 2;
  }
  
  // Dequantize keys
  const int8_t* keySrc = static_cast<const int8_t*>(quantizedKeyPtr_) + byteOffset;
  dequantizeTensor(keySrc, keyData, keyParams_, numElements);
  
  // Dequantize values
  const int8_t* valueSrc = static_cast<const int8_t*>(quantizedValuePtr_) + byteOffset;
  dequantizeTensor(valueSrc, valueData, valueParams_, numElements);
  
  return success();
}

size_t QuantizedKVBlock::getMemoryUsage() const {
  // Calculate memory per element based on quantization type
  size_t bitsPerElement = keyParams_.numBits;
  size_t totalElements = blockSize_ * headDim_ * 2; // Keys and values
  
  // Calculate bytes needed
  size_t dataBytes = (totalElements * bitsPerElement + 7) / 8;
  
  // Add scale and zero-point storage
  size_t paramBytes = keyParams_.scales.size() * sizeof(float) * 2 +
                      keyParams_.zeroPoints.size() * sizeof(int32_t) * 2;
  
  return dataBytes + paramBytes;
}

void QuantizedKVBlock::quantizeTensor(const float* input, void* output,
                                       const QuantizationParams& params,
                                       int64_t numElements) const {
  if (params.numBits == 8) {
    // INT8 quantization
    int8_t* out = static_cast<int8_t*>(output);
    float scale = params.scales.empty() ? 1.0f : params.scales[0];
    int32_t zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
    
    for (int64_t i = 0; i < numElements; i++) {
      // Quantize: q = clip(round(x / scale) + zero_point, qmin, qmax)
      float scaled = input[i] / scale + static_cast<float>(zeroPoint);
      int32_t quantized = static_cast<int32_t>(std::round(scaled));
      
      // Clip to INT8 range
      if (params.isSigned) {
        quantized = std::max(-128, std::min(127, quantized));
      } else {
        quantized = std::max(0, std::min(255, quantized));
      }
      
      out[i] = static_cast<int8_t>(quantized);
    }
  } else if (params.numBits == 4) {
    // INT4 quantization (pack 2 values per byte)
    uint8_t* out = static_cast<uint8_t*>(output);
    float scale = params.scales.empty() ? 1.0f : params.scales[0];
    int32_t zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
    
    for (int64_t i = 0; i < numElements; i += 2) {
      int8_t q1 = 0, q2 = 0;
      
      // First value (lower 4 bits)
      {
        float scaled = input[i] / scale + static_cast<float>(zeroPoint);
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        if (params.isSigned) {
          quantized = std::max(-8, std::min(7, quantized));
        } else {
          quantized = std::max(0, std::min(15, quantized));
        }
        q1 = static_cast<int8_t>(quantized);
      }
      
      // Second value (upper 4 bits)
      if (i + 1 < numElements) {
        float scaled = input[i + 1] / scale + static_cast<float>(zeroPoint);
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        if (params.isSigned) {
          quantized = std::max(-8, std::min(7, quantized));
        } else {
          quantized = std::max(0, std::min(15, quantized));
        }
        q2 = static_cast<int8_t>(quantized);
      }
      
      // Pack into byte
      out[i / 2] = static_cast<uint8_t>((q1 & 0x0F) | ((q2 & 0x0F) << 4));
    }
  }
}

void QuantizedKVBlock::dequantizeTensor(const void* input, float* output,
                                         const QuantizationParams& params,
                                         int64_t numElements) const {
  if (params.numBits == 8) {
    // INT8 dequantization
    const int8_t* in = static_cast<const int8_t*>(input);
    float scale = params.scales.empty() ? 1.0f : params.scales[0];
    int32_t zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
    
    for (int64_t i = 0; i < numElements; i++) {
      // Dequantize: x = (q - zero_point) * scale
      output[i] = (static_cast<float>(in[i]) - static_cast<float>(zeroPoint)) * scale;
    }
  } else if (params.numBits == 4) {
    // INT4 dequantization (unpack 2 values per byte)
    const uint8_t* in = static_cast<const uint8_t*>(input);
    float scale = params.scales.empty() ? 1.0f : params.scales[0];
    int32_t zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
    
    for (int64_t i = 0; i < numElements; i += 2) {
      uint8_t packed = in[i / 2];
      
      // Lower 4 bits
      int8_t q1 = packed & 0x0F;
      if (params.isSigned && (q1 & 0x08)) {
        q1 |= 0xF0; // Sign extend
      }
      output[i] = (static_cast<float>(q1) - static_cast<float>(zeroPoint)) * scale;
      
      // Upper 4 bits
      if (i + 1 < numElements) {
        int8_t q2 = (packed >> 4) & 0x0F;
        if (params.isSigned && (q2 & 0x08)) {
          q2 |= 0xF0; // Sign extend
        }
        output[i + 1] = (static_cast<float>(q2) - static_cast<float>(zeroPoint)) * scale;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// QuantizedBlockAllocator Implementation
//===----------------------------------------------------------------------===//

QuantizedBlockAllocator::QuantizedBlockAllocator(int64_t blockSize, int64_t headDim,
                                                  const QuantizationConfig& config,
                                                  bool enableGPU)
    : blockSize_(blockSize), headDim_(headDim), config_(config),
      enableGPU_(enableGPU), totalBlocks_(0),
      keyMemoryPool_(nullptr), valueMemoryPool_(nullptr),
      poolSize_(0), poolOffset_(0) {
  allocateMemoryPools();
}

QuantizedBlockAllocator::~QuantizedBlockAllocator() {
  deallocateMemoryPools();
}

QuantizedKVBlock* QuantizedBlockAllocator::allocateBlock() {
  // Check if we have free blocks
  if (!freeBlocks_.empty()) {
    QuantizedKVBlock* block = freeBlocks_.back();
    freeBlocks_.pop_back();
    block->resetUsedSlots();
    return block;
  }
  
  // Need to allocate a new block
  size_t blockMemSize = calculateBlockMemorySize();
  
  // Check if we have space in the pool
  if (poolOffset_ + blockMemSize * 2 > poolSize_) {
    // Need to grow the pool or return nullptr
    return nullptr;
  }
  
  // Allocate from the pool
  void* keyPtr = static_cast<char*>(keyMemoryPool_) + poolOffset_;
  void* valuePtr = static_cast<char*>(valueMemoryPool_) + poolOffset_;
  poolOffset_ += blockMemSize;
  
  // Calculate quantization parameters
  QuantizationParams keyParams, valueParams;
  keyParams.numBits = config_.type == QuantizationType::INT8 ? 8 : 4;
  keyParams.isSigned = true;
  keyParams.scales.push_back(1.0f); // Will be updated when data is stored
  keyParams.zeroPoints.push_back(0);
  
  valueParams = keyParams;
  
  // Create the block
  auto block = std::make_unique<QuantizedKVBlock>(
      keyPtr, valuePtr, keyParams, valueParams, blockSize_, headDim_);
  
  QuantizedKVBlock* blockPtr = block.get();
  allBlocks_.push_back(std::move(block));
  totalBlocks_++;
  
  return blockPtr;
}

void QuantizedBlockAllocator::deallocateBlock(QuantizedKVBlock* block) {
  if (block) {
    block->resetUsedSlots();
    freeBlocks_.push_back(block);
  }
}

void QuantizedBlockAllocator::configureQuantization(const QuantizationConfig& config) {
  config_ = config;
}

void QuantizedBlockAllocator::preallocateBlocks(int64_t numBlocks) {
  for (int64_t i = 0; i < numBlocks; i++) {
    QuantizedKVBlock* block = allocateBlock();
    if (block) {
      freeBlocks_.push_back(block);
    } else {
      break; // Out of memory
    }
  }
}

QuantizationParams QuantizedBlockAllocator::calculateQuantizationParams(
    const float* data, int64_t numElements) const {
  
  QuantizationParams params;
  params.numBits = config_.type == QuantizationType::INT8 ? 8 : 4;
  params.isSigned = true;
  
  // Calculate min and max
  auto [minVal, maxVal] = calculateMinMax(data, numElements);
  
  // Calculate scale and zero point
  float scale = calculateScale(minVal, maxVal, params.numBits, config_.symmetric);
  int32_t zeroPoint = calculateZeroPoint(minVal, maxVal, scale, config_.symmetric);
  
  if (config_.strategy == QuantizationStrategy::PER_TENSOR) {
    params.scales.push_back(scale);
    params.zeroPoints.push_back(zeroPoint);
  } else if (config_.strategy == QuantizationStrategy::PER_GROUP) {
    // Per-group quantization
    int64_t numGroups = (numElements + config_.groupSize - 1) / config_.groupSize;
    for (int64_t g = 0; g < numGroups; g++) {
      int64_t start = g * config_.groupSize;
      int64_t end = std::min(start + config_.groupSize, numElements);
      
      auto [gMin, gMax] = calculateMinMax(data + start, end - start);
      float gScale = calculateScale(gMin, gMax, params.numBits, config_.symmetric);
      int32_t gZeroPoint = calculateZeroPoint(gMin, gMax, gScale, config_.symmetric);
      
      params.scales.push_back(gScale);
      params.zeroPoints.push_back(gZeroPoint);
    }
  }
  
  return params;
}

size_t QuantizedBlockAllocator::getTotalMemoryUsage() const {
  return poolSize_ * 2; // Key and value pools
}

size_t QuantizedBlockAllocator::getQuantizedMemoryUsage() const {
  return poolOffset_ * 2;
}

float QuantizedBlockAllocator::getCompressionRatio() const {
  // Compare to FP32 storage
  size_t fp32Size = totalBlocks_ * blockSize_ * headDim_ * sizeof(float) * 2;
  size_t quantizedSize = getQuantizedMemoryUsage();
  
  if (quantizedSize == 0) return 1.0f;
  return static_cast<float>(fp32Size) / static_cast<float>(quantizedSize);
}

void QuantizedBlockAllocator::allocateMemoryPools() {
  // Calculate initial pool size (enough for 64 blocks)
  size_t blockMemSize = calculateBlockMemorySize();
  poolSize_ = blockMemSize * 64;
  
  if (enableGPU_) {
#if defined(LLMIR_ENABLE_CUDA)
    cudaMalloc(&keyMemoryPool_, poolSize_);
    cudaMalloc(&valueMemoryPool_, poolSize_);
#elif defined(LLMIR_ENABLE_HIP)
    hipMalloc(&keyMemoryPool_, poolSize_);
    hipMalloc(&valueMemoryPool_, poolSize_);
#else
    keyMemoryPool_ = std::malloc(poolSize_);
    valueMemoryPool_ = std::malloc(poolSize_);
#endif
  } else {
    keyMemoryPool_ = std::malloc(poolSize_);
    valueMemoryPool_ = std::malloc(poolSize_);
  }
  
  if (keyMemoryPool_) std::memset(keyMemoryPool_, 0, poolSize_);
  if (valueMemoryPool_) std::memset(valueMemoryPool_, 0, poolSize_);
}

void QuantizedBlockAllocator::deallocateMemoryPools() {
  if (enableGPU_) {
#if defined(LLMIR_ENABLE_CUDA)
    if (keyMemoryPool_) cudaFree(keyMemoryPool_);
    if (valueMemoryPool_) cudaFree(valueMemoryPool_);
#elif defined(LLMIR_ENABLE_HIP)
    if (keyMemoryPool_) hipFree(keyMemoryPool_);
    if (valueMemoryPool_) hipFree(valueMemoryPool_);
#else
    std::free(keyMemoryPool_);
    std::free(valueMemoryPool_);
#endif
  } else {
    std::free(keyMemoryPool_);
    std::free(valueMemoryPool_);
  }
  
  keyMemoryPool_ = nullptr;
  valueMemoryPool_ = nullptr;
}

size_t QuantizedBlockAllocator::calculateBlockMemorySize() const {
  // Bytes per element depends on quantization type
  size_t bitsPerElement = config_.type == QuantizationType::INT8 ? 8 : 4;
  size_t elementsPerBlock = blockSize_ * headDim_;
  
  // Calculate bytes (round up for INT4)
  return (elementsPerBlock * bitsPerElement + 7) / 8;
}

std::pair<float, float> QuantizedBlockAllocator::calculateMinMax(
    const float* data, int64_t numElements) const {
  
  float minVal = std::numeric_limits<float>::max();
  float maxVal = std::numeric_limits<float>::lowest();
  
  for (int64_t i = 0; i < numElements; i++) {
    minVal = std::min(minVal, data[i]);
    maxVal = std::max(maxVal, data[i]);
  }
  
  return {minVal, maxVal};
}

float QuantizedBlockAllocator::calculateScale(float min, float max, 
                                               int64_t numBits, bool symmetric) const {
  float qmin, qmax;
  
  if (symmetric) {
    // Symmetric quantization: range is [-2^(n-1), 2^(n-1)-1]
    qmax = static_cast<float>((1 << (numBits - 1)) - 1);
    qmin = -qmax - 1;
    
    // Use max absolute value for symmetric
    float absMax = std::max(std::abs(min), std::abs(max));
    return absMax / qmax;
  } else {
    // Asymmetric quantization: range is [0, 2^n - 1] or [-2^(n-1), 2^(n-1)-1]
    qmax = static_cast<float>((1 << numBits) - 1);
    qmin = 0.0f;
    
    return (max - min) / (qmax - qmin);
  }
}

int32_t QuantizedBlockAllocator::calculateZeroPoint(float min, float max, 
                                                     float scale, bool symmetric) const {
  if (symmetric) {
    return 0; // Symmetric quantization has zero point at 0
  }
  
  // Asymmetric: zero_point = round(-min / scale)
  if (scale == 0.0f) return 0;
  return static_cast<int32_t>(std::round(-min / scale));
}

//===----------------------------------------------------------------------===//
// QuantizedPagedKVCache Implementation
//===----------------------------------------------------------------------===//

QuantizedPagedKVCache::QuantizedPagedKVCache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    const QuantizationConfig& config,
    Type elementType, bool enableGPU)
    : numLayers_(numLayers), numHeads_(numHeads), headDim_(headDim),
      blockSize_(blockSize), maxSeqLen_(maxSeqLen), config_(config),
      elementType_(elementType), enableGPU_(enableGPU),
      dynamicQuantization_(false), accuracyThreshold_(0.01f) {
  
  initializeBlockAllocators();
  metrics_ = QuantizationMetrics{};
}

QuantizedPagedKVCache::~QuantizedPagedKVCache() {
  reset();
}

void QuantizedPagedKVCache::updateQuantizationConfig(const QuantizationConfig& config) {
  config_ = config;
  for (auto& allocator : blockAllocators_) {
    allocator->configureQuantization(config);
  }
}

LogicalResult QuantizedPagedKVCache::appendKV(
    const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  if (!keyData || !valueData || !seqIds || !blockIndices) {
    return failure();
  }
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  const float* keys = static_cast<const float*>(keyData);
  const float* values = static_cast<const float*>(valueData);
  
  // Process each sequence in the batch
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t seqId = seqIds[b];
    
    // Initialize sequence table entry if needed
    if (sequenceTable_.find(seqId) == sequenceTable_.end()) {
      sequenceTable_[seqId].resize(numLayers_);
    }
    
    // For each layer
    for (int64_t layer = 0; layer < numLayers_; layer++) {
      auto& allocator = blockAllocators_[layer];
      auto& seqBlocks = sequenceTable_[seqId][layer];
      
      // Get or allocate a block
      QuantizedKVBlock* block = nullptr;
      int64_t posInBlock = 0;
      
      if (!seqBlocks.empty()) {
        block = seqBlocks.back();
        posInBlock = block->getUsedSlots();
        
        // Check if current block has space
        if (posInBlock + seqLen > blockSize_) {
          // Need a new block
          block = allocator->allocateBlock();
          if (!block) return failure();
          seqBlocks.push_back(block);
          posInBlock = 0;
        }
      } else {
        // First block for this sequence/layer
        block = allocator->allocateBlock();
        if (!block) return failure();
        seqBlocks.push_back(block);
        posInBlock = 0;
      }
      
      // Calculate data offset for this batch and layer
      int64_t dataOffset = (b * numLayers_ + layer) * seqLen * numHeads_ * headDim_;
      
      // Store the block index for this batch/layer
      blockIndices[b * numLayers_ + layer] = static_cast<int32_t>(seqBlocks.size() - 1);
      
      // Quantize and store the KV data
      if (block->quantizeAndStore(
              keys + dataOffset, values + dataOffset,
              seqLen, posInBlock).failed()) {
        return failure();
      }
      
      block->updateAccessTime(metrics_.numQuantizations);
    }
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  
  metrics_.numQuantizations++;
  metrics_.totalQuantizationTime += elapsed;
  
  return success();
}

LogicalResult QuantizedPagedKVCache::lookupKV(
    const int32_t* blockIndices, const int32_t* seqLens,
    int32_t batchSize, void* outputKeys, void* outputValues) {
  
  if (!blockIndices || !seqLens || !outputKeys || !outputValues) {
    return failure();
  }
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  float* keys = static_cast<float*>(outputKeys);
  float* values = static_cast<float*>(outputValues);
  
  // Process each sequence in the batch
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t seqLen = seqLens[b];
    
    // For each layer
    for (int64_t layer = 0; layer < numLayers_; layer++) {
      int32_t blockIdx = blockIndices[b * numLayers_ + layer];
      
      // Calculate output offset
      int64_t outOffset = (b * numLayers_ + layer) * seqLen * numHeads_ * headDim_;
      
      // Look up the block (need to find the sequence ID first)
      // For simplicity, iterate through sequences
      bool found = false;
      for (auto& [seqId, layers] : sequenceTable_) {
        if (layer < static_cast<int64_t>(layers.size()) && 
            blockIdx < static_cast<int32_t>(layers[layer].size())) {
          auto* block = layers[layer][blockIdx];
          if (block) {
            if (block->dequantizeAndLoad(
                    keys + outOffset, values + outOffset,
                    seqLen, 0).failed()) {
              return failure();
            }
            found = true;
            break;
          }
        }
      }
      
      if (!found) {
        // Zero out if not found
        std::memset(keys + outOffset, 0, seqLen * numHeads_ * headDim_ * sizeof(float));
        std::memset(values + outOffset, 0, seqLen * numHeads_ * headDim_ * sizeof(float));
      }
    }
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  
  metrics_.numDequantizations++;
  metrics_.totalDequantizationTime += elapsed;
  
  return success();
}

LogicalResult QuantizedPagedKVCache::clearSequence(int32_t seqId) {
  auto it = sequenceTable_.find(seqId);
  if (it == sequenceTable_.end()) {
    return failure();
  }
  
  // Return all blocks to their allocators
  for (int64_t layer = 0; layer < numLayers_; layer++) {
    for (auto* block : it->second[layer]) {
      if (block) {
        blockAllocators_[layer]->deallocateBlock(block);
      }
    }
  }
  
  sequenceTable_.erase(it);
  return success();
}

void QuantizedPagedKVCache::reset() {
  // Clear all sequences
  for (auto& [seqId, layers] : sequenceTable_) {
    for (int64_t layer = 0; layer < numLayers_; layer++) {
      for (auto* block : layers[layer]) {
        if (block) {
          blockAllocators_[layer]->deallocateBlock(block);
        }
      }
    }
  }
  sequenceTable_.clear();
  resetMetrics();
}

int64_t QuantizedPagedKVCache::getSequenceLength(int32_t seqId) const {
  auto it = sequenceTable_.find(seqId);
  if (it == sequenceTable_.end()) {
    return 0;
  }
  
  // Sum up tokens across all blocks for layer 0
  int64_t totalTokens = 0;
  if (!it->second.empty()) {
    for (auto* block : it->second[0]) {
      if (block) {
        totalTokens += block->getUsedSlots();
      }
    }
  }
  return totalTokens;
}

size_t QuantizedPagedKVCache::getTotalMemoryUsage() const {
  size_t total = 0;
  for (const auto& allocator : blockAllocators_) {
    total += allocator->getTotalMemoryUsage();
  }
  return total;
}

size_t QuantizedPagedKVCache::getQuantizedMemoryUsage() const {
  size_t total = 0;
  for (const auto& allocator : blockAllocators_) {
    total += allocator->getQuantizedMemoryUsage();
  }
  return total;
}

float QuantizedPagedKVCache::getCompressionRatio() const {
  // Compare to FP32 storage
  size_t fp32Size = 0;
  for (const auto& [seqId, layers] : sequenceTable_) {
    for (const auto& blocks : layers) {
      for (const auto* block : blocks) {
        if (block) {
          fp32Size += block->getUsedSlots() * headDim_ * sizeof(float) * 2;
        }
      }
    }
  }
  
  size_t quantizedSize = getQuantizedMemoryUsage();
  if (quantizedSize == 0) return 1.0f;
  
  return static_cast<float>(fp32Size) / static_cast<float>(quantizedSize);
}

float QuantizedPagedKVCache::getAccuracyLoss() const {
  return static_cast<float>(metrics_.averageAccuracyLoss);
}

void QuantizedPagedKVCache::initializeBlockAllocators() {
  blockAllocators_.reserve(numLayers_);
  for (int64_t i = 0; i < numLayers_; i++) {
    blockAllocators_.push_back(std::make_unique<QuantizedBlockAllocator>(
        blockSize_, headDim_ * numHeads_, config_, enableGPU_));
    
    // Preallocate some blocks
    blockAllocators_.back()->preallocateBlocks(8);
  }
}

LogicalResult QuantizedPagedKVCache::validateQuantizationAccuracy(
    const float* original, const float* quantized, int64_t numElements) const {
  
  float mse = calculateMSE(original, quantized, numElements);
  float snr = calculateSNR(original, quantized, numElements);
  
  // SNR should be above a threshold (e.g., 30 dB for good quality)
  if (snr < 20.0f) {
    return failure();
  }
  
  return success();
}

float QuantizedPagedKVCache::calculateMSE(const float* a, const float* b, 
                                           int64_t numElements) const {
  double sum = 0.0;
  for (int64_t i = 0; i < numElements; i++) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return static_cast<float>(sum / numElements);
}

float QuantizedPagedKVCache::calculateSNR(const float* original, const float* quantized,
                                           int64_t numElements) const {
  double signalPower = 0.0;
  double noisePower = 0.0;
  
  for (int64_t i = 0; i < numElements; i++) {
    signalPower += static_cast<double>(original[i]) * static_cast<double>(original[i]);
    double diff = static_cast<double>(original[i]) - static_cast<double>(quantized[i]);
    noisePower += diff * diff;
  }
  
  if (noisePower == 0.0) return 100.0f; // Perfect match
  
  // SNR in dB
  return static_cast<float>(10.0 * std::log10(signalPower / noisePower));
}

} // namespace runtime
} // namespace llm
} // namespace mlir
