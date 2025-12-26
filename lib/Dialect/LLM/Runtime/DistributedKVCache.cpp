//===- DistributedKVCache.cpp - Distributed KV Cache --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements distributed KV cache support for multi-GPU environments.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/DistributedKVCache.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#if defined(LLMIR_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(LLMIR_ENABLE_NCCL)
#include <nccl.h>
#endif

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

// Get current timestamp in milliseconds
double getCurrentTimeMs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

// Calculate data size in bytes
size_t getDataSize(int64_t count, int64_t elementSize) {
  return static_cast<size_t>(count) * static_cast<size_t>(elementSize);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// NCCLCommunicationHandle Implementation
//===----------------------------------------------------------------------===//

NCCLCommunicationHandle::NCCLCommunicationHandle(
    const std::vector<int32_t>& deviceIds)
    : deviceIds_(deviceIds), ncclComms_(nullptr), initialized_(false) {}

NCCLCommunicationHandle::~NCCLCommunicationHandle() {
#if defined(LLMIR_ENABLE_NCCL)
  if (initialized_ && ncclComms_) {
    ncclComm_t* comms = static_cast<ncclComm_t*>(ncclComms_);
    for (size_t i = 0; i < deviceIds_.size(); i++) {
      ncclCommDestroy(comms[i]);
    }
    delete[] comms;
  }
#endif
}

LogicalResult NCCLCommunicationHandle::initialize() {
#if defined(LLMIR_ENABLE_NCCL)
  int numDevices = deviceIds_.size();
  ncclComm_t* comms = new ncclComm_t[numDevices];
  
  // Initialize NCCL communicators
  ncclResult_t result = ncclCommInitAll(comms, numDevices, deviceIds_.data());
  if (result != ncclSuccess) {
    delete[] comms;
    return failure();
  }
  
  ncclComms_ = comms;
  initialized_ = true;
  return success();
#else
  // NCCL not available, use fallback
  initialized_ = true;
  return success();
#endif
}

LogicalResult NCCLCommunicationHandle::allToAll(
    const void* sendBuf, void* recvBuf,
    int64_t count, int64_t elementSize) {
  
  if (!initialized_) return failure();
  
#if defined(LLMIR_ENABLE_NCCL)
  // NCCL doesn't have native all-to-all, implement using send/recv
  ncclComm_t* comms = static_cast<ncclComm_t*>(ncclComms_);
  int numDevices = deviceIds_.size();
  size_t sendSize = count / numDevices;
  
  for (int i = 0; i < numDevices; i++) {
    cudaSetDevice(deviceIds_[i]);
    ncclGroupStart();
    
    for (int j = 0; j < numDevices; j++) {
      const char* sendPtr = static_cast<const char*>(sendBuf) + 
                            j * sendSize * elementSize;
      char* recvPtr = static_cast<char*>(recvBuf) + 
                      i * sendSize * elementSize;
      
      if (i == j) {
        // Local copy
        cudaMemcpy(recvPtr, sendPtr, sendSize * elementSize, 
                   cudaMemcpyDeviceToDevice);
      } else {
        ncclSend(sendPtr, sendSize * elementSize, ncclInt8, j, comms[i], 
                 cudaStreamDefault);
        ncclRecv(recvPtr, sendSize * elementSize, ncclInt8, j, comms[i],
                 cudaStreamDefault);
      }
    }
    
    ncclGroupEnd();
  }
  
  return success();
#else
  // Fallback: just copy for single device
  std::memcpy(recvBuf, sendBuf, count * elementSize);
  return success();
#endif
}

LogicalResult NCCLCommunicationHandle::allReduce(
    const void* sendBuf, void* recvBuf,
    int64_t count, int64_t elementSize) {
  
  if (!initialized_) return failure();
  
#if defined(LLMIR_ENABLE_NCCL)
  ncclComm_t* comms = static_cast<ncclComm_t*>(ncclComms_);
  
  // Determine data type
  ncclDataType_t dtype = ncclFloat32;
  if (elementSize == 2) dtype = ncclFloat16;
  else if (elementSize == 8) dtype = ncclFloat64;
  
  for (size_t i = 0; i < deviceIds_.size(); i++) {
    cudaSetDevice(deviceIds_[i]);
    ncclResult_t result = ncclAllReduce(sendBuf, recvBuf, count, dtype,
                                         ncclSum, comms[i], cudaStreamDefault);
    if (result != ncclSuccess) return failure();
  }
  
  return success();
#else
  // Fallback: just copy for single device
  std::memcpy(recvBuf, sendBuf, count * elementSize);
  return success();
#endif
}

LogicalResult NCCLCommunicationHandle::send(
    const void* buf, int64_t count,
    int64_t elementSize, int32_t destDevice) {
  
  if (!initialized_) return failure();
  
#if defined(LLMIR_ENABLE_NCCL)
  ncclComm_t* comms = static_cast<ncclComm_t*>(ncclComms_);
  
  // Find current device index
  int currentDevice;
  cudaGetDevice(&currentDevice);
  
  int srcIdx = -1;
  for (size_t i = 0; i < deviceIds_.size(); i++) {
    if (deviceIds_[i] == currentDevice) {
      srcIdx = i;
      break;
    }
  }
  
  if (srcIdx < 0) return failure();
  
  ncclResult_t result = ncclSend(buf, count * elementSize, ncclInt8,
                                  destDevice, comms[srcIdx], cudaStreamDefault);
  return result == ncclSuccess ? success() : failure();
#else
  return success();
#endif
}

LogicalResult NCCLCommunicationHandle::recv(
    void* buf, int64_t count,
    int64_t elementSize, int32_t srcDevice) {
  
  if (!initialized_) return failure();
  
#if defined(LLMIR_ENABLE_NCCL)
  ncclComm_t* comms = static_cast<ncclComm_t*>(ncclComms_);
  
  int currentDevice;
  cudaGetDevice(&currentDevice);
  
  int dstIdx = -1;
  for (size_t i = 0; i < deviceIds_.size(); i++) {
    if (deviceIds_[i] == currentDevice) {
      dstIdx = i;
      break;
    }
  }
  
  if (dstIdx < 0) return failure();
  
  ncclResult_t result = ncclRecv(buf, count * elementSize, ncclInt8,
                                  srcDevice, comms[dstIdx], cudaStreamDefault);
  return result == ncclSuccess ? success() : failure();
#else
  return success();
#endif
}

LogicalResult NCCLCommunicationHandle::barrier() {
  if (!initialized_) return failure();
  
#if defined(LLMIR_ENABLE_CUDA)
  // Synchronize all devices
  for (int32_t deviceId : deviceIds_) {
    cudaSetDevice(deviceId);
    cudaDeviceSynchronize();
  }
#endif
  
  return success();
}

//===----------------------------------------------------------------------===//
// KVCacheShard Implementation
//===----------------------------------------------------------------------===//

KVCacheShard::KVCacheShard(int32_t deviceId, int64_t numLayers, int64_t numHeads,
                           int64_t headDim, int64_t blockSize, int64_t maxSeqLen,
                           Type elementType)
    : deviceId_(deviceId), numLayers_(numLayers), numHeads_(numHeads),
      headDim_(headDim), blockSize_(blockSize), maxSeqLen_(maxSeqLen),
      elementType_(elementType) {
  
  // Activate device and create cache
  activate();
  cache_ = std::make_unique<PagedKVCache>(
      numLayers, numHeads, headDim, blockSize, maxSeqLen,
      elementType, true /* enableGPU */);
}

KVCacheShard::~KVCacheShard() = default;

size_t KVCacheShard::getMemoryUsage() const {
  return cache_->getTotalMemoryUsage();
}

size_t KVCacheShard::getAvailableMemory() const {
#if defined(LLMIR_ENABLE_CUDA)
  size_t free, total;
  cudaSetDevice(deviceId_);
  cudaMemGetInfo(&free, &total);
  return free;
#else
  return 0;
#endif
}

LogicalResult KVCacheShard::activate() {
#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t err = cudaSetDevice(deviceId_);
  return err == cudaSuccess ? success() : failure();
#else
  return success();
#endif
}

//===----------------------------------------------------------------------===//
// DistributedPagedKVCache Implementation
//===----------------------------------------------------------------------===//

DistributedPagedKVCache::DistributedPagedKVCache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    Type elementType, const ShardingConfig& config)
    : numLayers_(numLayers), numHeads_(numHeads), headDim_(headDim),
      blockSize_(blockSize), maxSeqLen_(maxSeqLen),
      elementType_(elementType), config_(config) {
  
  initializeShards();
  initializeCommunication();
  
  deviceLoads_.resize(config_.numDevices, 0.0f);
  metrics_ = DistributedMetrics{};
  metrics_.perDeviceUtilization.resize(config_.numDevices, 0.0);
}

DistributedPagedKVCache::~DistributedPagedKVCache() {
  reset();
}

void DistributedPagedKVCache::initializeShards() {
  shards_.reserve(config_.numDevices);
  
  for (int64_t i = 0; i < config_.numDevices; i++) {
    int32_t deviceId = config_.deviceIds.empty() ? i : config_.deviceIds[i];
    
    // Calculate layers/heads for this shard based on strategy
    int64_t shardLayers = numLayers_;
    int64_t shardHeads = numHeads_;
    
    switch (config_.strategy) {
      case ShardingStrategy::LAYER_WISE:
        if (i < static_cast<int64_t>(config_.layerRanges.size())) {
          shardLayers = config_.layerRanges[i].second - config_.layerRanges[i].first;
        } else {
          shardLayers = (numLayers_ + config_.numDevices - 1) / config_.numDevices;
        }
        break;
        
      case ShardingStrategy::HEAD_WISE:
        if (i < static_cast<int64_t>(config_.headRanges.size())) {
          shardHeads = config_.headRanges[i].second - config_.headRanges[i].first;
        } else {
          shardHeads = (numHeads_ + config_.numDevices - 1) / config_.numDevices;
        }
        break;
        
      case ShardingStrategy::SEQUENCE_WISE:
      case ShardingStrategy::HYBRID:
        // Each shard handles all layers and heads
        break;
    }
    
    shards_.push_back(std::make_unique<KVCacheShard>(
        deviceId, shardLayers, shardHeads, headDim_, blockSize_, maxSeqLen_,
        elementType_));
  }
}

void DistributedPagedKVCache::initializeCommunication() {
  if (config_.useNCCL && config_.numDevices > 1) {
    auto ncclHandle = std::make_unique<NCCLCommunicationHandle>(config_.deviceIds);
    if (ncclHandle->initialize().succeeded()) {
      commHandle_ = std::move(ncclHandle);
    }
  }
}

LogicalResult DistributedPagedKVCache::appendKV(
    const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  if (!keyData || !valueData || !seqIds || !blockIndices) {
    return failure();
  }
  
  double startTime = getCurrentTimeMs();
  
  // Determine target device for each sequence
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t seqId = seqIds[b];
    int32_t targetDevice = getDeviceForSequence(seqId);
    
    // Track sequence to device mapping
    sequenceDeviceMap_[seqId] = targetDevice;
    
    // Activate target shard
    auto& shard = shards_[targetDevice];
    if (shard->activate().failed()) {
      return failure();
    }
    
    // Calculate data offset for this batch element
    size_t elementSize = 4; // Assume FP32
    size_t dataOffset = b * seqLen * numHeads_ * headDim_ * elementSize;
    
    const char* keyPtr = static_cast<const char*>(keyData) + dataOffset;
    const char* valuePtr = static_cast<const char*>(valueData) + dataOffset;
    
    // Append to the shard
    int32_t localBlockIdx;
    if (shard->getCache().appendKV(keyPtr, valuePtr, 1, seqLen,
                                    &seqId, &localBlockIdx).failed()) {
      return failure();
    }
    
    // Store block index with device information encoded
    blockIndices[b] = (targetDevice << 16) | (localBlockIdx & 0xFFFF);
    
    // Update load metrics
    deviceLoads_[targetDevice] += 1.0f;
  }
  
  double elapsed = getCurrentTimeMs() - startTime;
  metrics_.totalComputeTime += elapsed;
  
  return success();
}

LogicalResult DistributedPagedKVCache::lookupKV(
    const int32_t* blockIndices, const int32_t* seqLens,
    int32_t batchSize, void* outputKeys, void* outputValues) {
  
  if (!blockIndices || !seqLens || !outputKeys || !outputValues) {
    return failure();
  }
  
  double startTime = getCurrentTimeMs();
  
  char* keyOut = static_cast<char*>(outputKeys);
  char* valueOut = static_cast<char*>(outputValues);
  size_t elementSize = 4;
  
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t encodedIdx = blockIndices[b];
    int32_t deviceIdx = (encodedIdx >> 16) & 0xFFFF;
    int32_t localIdx = encodedIdx & 0xFFFF;
    int32_t seqLen = seqLens[b];
    
    // Activate the correct shard
    auto& shard = shards_[deviceIdx];
    if (shard->activate().failed()) {
      return failure();
    }
    
    size_t dataSize = seqLen * numHeads_ * headDim_ * elementSize;
    size_t outOffset = b * dataSize;
    
    // Lookup from the shard
    if (shard->getCache().lookupKV(&localIdx, &seqLen, 1,
                                    keyOut + outOffset,
                                    valueOut + outOffset).failed()) {
      return failure();
    }
  }
  
  double elapsed = getCurrentTimeMs() - startTime;
  metrics_.totalComputeTime += elapsed;
  
  return success();
}

LogicalResult DistributedPagedKVCache::computeDistributedAttention(
    const void* queries,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int32_t batchSize,
    void* output) {
  
  if (!queries || !blockIndices || !seqLens || !output) {
    return failure();
  }
  
  double startTime = getCurrentTimeMs();
  
  // For distributed attention, we need to:
  // 1. Gather KV from all devices
  // 2. Compute attention locally
  // 3. Reduce results if needed
  
  // Allocate temporary buffers for gathered KV
  size_t maxSeq = 0;
  for (int32_t b = 0; b < batchSize; b++) {
    maxSeq = std::max(maxSeq, static_cast<size_t>(seqLens[b]));
  }
  
  size_t elementSize = 4;
  size_t kvBufferSize = batchSize * maxSeq * numHeads_ * headDim_ * elementSize;
  
  std::vector<uint8_t> keyBuffer(kvBufferSize);
  std::vector<uint8_t> valueBuffer(kvBufferSize);
  
  // Lookup KV data
  if (lookupKV(blockIndices, seqLens, batchSize,
               keyBuffer.data(), valueBuffer.data()).failed()) {
    return failure();
  }
  
  // If using multiple devices, gather across devices
  if (config_.numDevices > 1 && commHandle_) {
    double commStart = getCurrentTimeMs();
    
    // All-gather KV data from all devices
    if (allGatherKV(0, keyBuffer.data(), kvBufferSize).failed()) {
      return failure();
    }
    
    double commElapsed = getCurrentTimeMs() - commStart;
    metrics_.totalCommunicationTime += commElapsed;
    metrics_.numCommunications++;
    metrics_.totalBytesTransferred += kvBufferSize * 2;
  }
  
  // Compute attention (simplified - actual implementation would use optimized kernels)
  // This is a placeholder for the actual attention computation
  float* queryPtr = static_cast<float*>(const_cast<void*>(queries));
  float* keyPtr = reinterpret_cast<float*>(keyBuffer.data());
  float* valuePtr = reinterpret_cast<float*>(valueBuffer.data());
  float* outPtr = static_cast<float*>(output);
  
  // Basic attention computation loop (would be GPU-accelerated in practice)
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t seqLen = seqLens[b];
    
    for (int64_t h = 0; h < numHeads_; h++) {
      // Compute attention scores
      // Q @ K^T -> scores
      // softmax(scores / sqrt(headDim)) -> weights  
      // weights @ V -> output
      
      // Placeholder: just copy values for now
      size_t offset = (b * numHeads_ + h) * headDim_;
      std::memcpy(outPtr + offset, valuePtr + offset,
                  headDim_ * sizeof(float));
    }
  }
  
  double elapsed = getCurrentTimeMs() - startTime;
  metrics_.totalComputeTime += elapsed;
  
  return success();
}

LogicalResult DistributedPagedKVCache::clearSequence(int32_t seqId) {
  auto it = sequenceDeviceMap_.find(seqId);
  if (it == sequenceDeviceMap_.end()) {
    return failure();
  }
  
  int32_t deviceIdx = it->second;
  auto& shard = shards_[deviceIdx];
  
  if (shard->activate().failed()) {
    return failure();
  }
  
  if (shard->getCache().clearSequence(seqId).failed()) {
    return failure();
  }
  
  sequenceDeviceMap_.erase(it);
  deviceLoads_[deviceIdx] -= 1.0f;
  
  return success();
}

void DistributedPagedKVCache::reset() {
  for (auto& shard : shards_) {
    shard->activate();
    shard->getCache().reset();
  }
  sequenceDeviceMap_.clear();
  std::fill(deviceLoads_.begin(), deviceLoads_.end(), 0.0f);
  resetMetrics();
}

int64_t DistributedPagedKVCache::getSequenceLength(int32_t seqId) const {
  auto it = sequenceDeviceMap_.find(seqId);
  if (it == sequenceDeviceMap_.end()) {
    return 0;
  }
  
  int32_t deviceIdx = it->second;
  return shards_[deviceIdx]->getCache().getSequenceLength(seqId);
}

size_t DistributedPagedKVCache::getTotalMemoryUsage() const {
  size_t total = 0;
  for (const auto& shard : shards_) {
    total += shard->getMemoryUsage();
  }
  return total;
}

std::vector<size_t> DistributedPagedKVCache::getPerDeviceMemoryUsage() const {
  std::vector<size_t> usage;
  usage.reserve(shards_.size());
  for (const auto& shard : shards_) {
    usage.push_back(shard->getMemoryUsage());
  }
  return usage;
}

LogicalResult DistributedPagedKVCache::rebalance() {
  // Calculate target load per device
  float totalLoad = std::accumulate(deviceLoads_.begin(), deviceLoads_.end(), 0.0f);
  float targetLoad = totalLoad / config_.numDevices;
  
  // Find overloaded and underloaded devices
  std::vector<int32_t> overloaded, underloaded;
  for (int64_t i = 0; i < config_.numDevices; i++) {
    if (deviceLoads_[i] > targetLoad * 1.2f) {
      overloaded.push_back(i);
    } else if (deviceLoads_[i] < targetLoad * 0.8f) {
      underloaded.push_back(i);
    }
  }
  
  // Move sequences from overloaded to underloaded devices
  // This is a placeholder - actual implementation would migrate data
  
  return success();
}

void DistributedPagedKVCache::updateLoadMetrics() {
  for (int64_t i = 0; i < config_.numDevices; i++) {
    size_t used = shards_[i]->getMemoryUsage();
    size_t total = shards_[i]->getAvailableMemory() + used;
    if (total > 0) {
      metrics_.perDeviceUtilization[i] = 
          static_cast<double>(used) / static_cast<double>(total);
    }
  }
}

std::vector<DeviceInfo> DistributedPagedKVCache::getDeviceInfo() const {
  std::vector<DeviceInfo> info;
  info.reserve(config_.numDevices);
  
  for (int64_t i = 0; i < config_.numDevices; i++) {
    DeviceInfo di;
    di.deviceId = config_.deviceIds.empty() ? i : config_.deviceIds[i];
    di.isActive = true;
    
#if defined(LLMIR_ENABLE_CUDA)
    cudaSetDevice(di.deviceId);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    di.totalMemory = total;
    di.availableMemory = free;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, di.deviceId);
    di.deviceName = prop.name;
    di.computeCapability = prop.major * 10 + prop.minor;
#else
    di.totalMemory = 0;
    di.availableMemory = 0;
    di.deviceName = "CPU";
    di.computeCapability = 0;
#endif
    
    info.push_back(di);
  }
  
  return info;
}

LogicalResult DistributedPagedKVCache::setActiveDevices(
    const std::vector<int32_t>& deviceIds) {
  // Validate all device IDs
  for (int32_t id : deviceIds) {
    bool found = false;
    for (int32_t existingId : config_.deviceIds) {
      if (existingId == id) {
        found = true;
        break;
      }
    }
    if (!found) return failure();
  }
  
  // Update active devices (would need to migrate data if shrinking)
  return success();
}

void DistributedPagedKVCache::resetMetrics() {
  metrics_ = DistributedMetrics{};
  metrics_.perDeviceUtilization.resize(config_.numDevices, 0.0);
}

int32_t DistributedPagedKVCache::getDeviceForLayer(int64_t layer) const {
  if (config_.strategy == ShardingStrategy::LAYER_WISE) {
    for (size_t i = 0; i < config_.layerRanges.size(); i++) {
      if (layer >= config_.layerRanges[i].first &&
          layer < config_.layerRanges[i].second) {
        return static_cast<int32_t>(i);
      }
    }
    // Default: divide layers evenly
    return static_cast<int32_t>(layer * config_.numDevices / numLayers_);
  }
  return 0;
}

int32_t DistributedPagedKVCache::getDeviceForHead(int64_t head) const {
  if (config_.strategy == ShardingStrategy::HEAD_WISE) {
    for (size_t i = 0; i < config_.headRanges.size(); i++) {
      if (head >= config_.headRanges[i].first &&
          head < config_.headRanges[i].second) {
        return static_cast<int32_t>(i);
      }
    }
    return static_cast<int32_t>(head * config_.numDevices / numHeads_);
  }
  return 0;
}

int32_t DistributedPagedKVCache::getDeviceForSequence(int32_t seqId) const {
  // Check if sequence already mapped
  auto it = sequenceDeviceMap_.find(seqId);
  if (it != sequenceDeviceMap_.end()) {
    return it->second;
  }
  
  if (config_.strategy == ShardingStrategy::SEQUENCE_WISE) {
    // Round-robin assignment or least-loaded device
    float minLoad = std::numeric_limits<float>::max();
    int32_t targetDevice = 0;
    for (int64_t i = 0; i < config_.numDevices; i++) {
      if (deviceLoads_[i] < minLoad) {
        minLoad = deviceLoads_[i];
        targetDevice = static_cast<int32_t>(i);
      }
    }
    return targetDevice;
  }
  
  // Default: hash-based assignment
  return seqId % config_.numDevices;
}

LogicalResult DistributedPagedKVCache::distributeData(
    const void* data, int64_t totalSize,
    std::vector<void*>& devicePtrs,
    std::vector<int64_t>& deviceSizes) {
  
  devicePtrs.resize(config_.numDevices);
  deviceSizes.resize(config_.numDevices);
  
  int64_t sizePerDevice = (totalSize + config_.numDevices - 1) / config_.numDevices;
  const char* src = static_cast<const char*>(data);
  
  for (int64_t i = 0; i < config_.numDevices; i++) {
    int64_t offset = i * sizePerDevice;
    int64_t size = std::min(sizePerDevice, totalSize - offset);
    deviceSizes[i] = size;
    
    if (size <= 0) {
      devicePtrs[i] = nullptr;
      continue;
    }
    
    shards_[i]->activate();
    
#if defined(LLMIR_ENABLE_CUDA)
    cudaMalloc(&devicePtrs[i], size);
    cudaMemcpy(devicePtrs[i], src + offset, size, cudaMemcpyHostToDevice);
#else
    devicePtrs[i] = std::malloc(size);
    std::memcpy(devicePtrs[i], src + offset, size);
#endif
  }
  
  return success();
}

LogicalResult DistributedPagedKVCache::gatherData(
    const std::vector<void*>& devicePtrs,
    const std::vector<int64_t>& deviceSizes,
    void* output, int64_t totalSize) {
  
  char* dst = static_cast<char*>(output);
  int64_t offset = 0;
  
  for (int64_t i = 0; i < config_.numDevices; i++) {
    if (devicePtrs[i] && deviceSizes[i] > 0) {
      shards_[i]->activate();
      
#if defined(LLMIR_ENABLE_CUDA)
      cudaMemcpy(dst + offset, devicePtrs[i], deviceSizes[i], 
                 cudaMemcpyDeviceToHost);
#else
      std::memcpy(dst + offset, devicePtrs[i], deviceSizes[i]);
#endif
      
      offset += deviceSizes[i];
    }
  }
  
  return success();
}

LogicalResult DistributedPagedKVCache::allGatherKV(
    int64_t layer, void* output, int64_t outputSize) {
  
  if (!commHandle_) return failure();
  
  // Each device contributes its portion
  int64_t sizePerDevice = outputSize / config_.numDevices;
  
  // Use all-gather to combine data
  // Placeholder: would use NCCL allGather
  return commHandle_->barrier();
}

LogicalResult DistributedPagedKVCache::scatterKV(
    int64_t layer, const void* input, int64_t inputSize) {
  
  if (!commHandle_) return failure();
  
  // Scatter input to appropriate devices
  int64_t sizePerDevice = inputSize / config_.numDevices;
  
  // Placeholder: would use NCCL scatter
  return commHandle_->barrier();
}

//===----------------------------------------------------------------------===//
// PipelineKVCache Implementation
//===----------------------------------------------------------------------===//

PipelineKVCache::PipelineKVCache(
    int64_t numStages, int64_t layersPerStage,
    int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    Type elementType, const std::vector<int32_t>& deviceIds)
    : numStages_(numStages), layersPerStage_(layersPerStage) {
  
  stageCaches_.reserve(numStages);
  for (int64_t i = 0; i < numStages; i++) {
    int32_t deviceId = i < static_cast<int64_t>(deviceIds.size()) ? 
                       deviceIds[i] : static_cast<int32_t>(i);
    stageCaches_.push_back(std::make_unique<KVCacheShard>(
        deviceId, layersPerStage, numHeads, headDim,
        blockSize, maxSeqLen, elementType));
  }
  
  if (numStages > 1) {
    auto ncclHandle = std::make_unique<NCCLCommunicationHandle>(deviceIds);
    if (ncclHandle->initialize().succeeded()) {
      commHandle_ = std::move(ncclHandle);
    }
  }
}

PipelineKVCache::~PipelineKVCache() = default;

KVCacheShard& PipelineKVCache::getStageCache(int64_t stage) {
  return *stageCaches_[stage];
}

const KVCacheShard& PipelineKVCache::getStageCache(int64_t stage) const {
  return *stageCaches_[stage];
}

LogicalResult PipelineKVCache::passActivations(
    int64_t fromStage, int64_t toStage,
    const void* data, int64_t size) {
  
  if (!commHandle_) return failure();
  
  int32_t srcDevice = stageCaches_[fromStage]->getDeviceId();
  int32_t dstDevice = stageCaches_[toStage]->getDeviceId();
  
  if (srcDevice == dstDevice) {
    return success(); // No transfer needed
  }
  
  // Send from source to destination
  return commHandle_->send(data, size, 1, dstDevice);
}

LogicalResult PipelineKVCache::appendKV(
    int64_t stage, const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  auto& shard = stageCaches_[stage];
  if (shard->activate().failed()) {
    return failure();
  }
  
  return shard->getCache().appendKV(keyData, valueData, batchSize, seqLen,
                                     seqIds, blockIndices);
}

LogicalResult PipelineKVCache::lookupKV(
    int64_t stage, const int32_t* blockIndices,
    const int32_t* seqLens, int32_t batchSize,
    void* outputKeys, void* outputValues) {
  
  auto& shard = stageCaches_[stage];
  if (shard->activate().failed()) {
    return failure();
  }
  
  return shard->getCache().lookupKV(blockIndices, seqLens, batchSize,
                                     outputKeys, outputValues);
}

//===----------------------------------------------------------------------===//
// TensorParallelKVCache Implementation
//===----------------------------------------------------------------------===//

TensorParallelKVCache::TensorParallelKVCache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    Type elementType, int64_t tensorParallelSize,
    const std::vector<int32_t>& deviceIds)
    : numLayers_(numLayers), totalHeads_(numHeads), headDim_(headDim),
      tensorParallelSize_(tensorParallelSize), localRank_(0) {
  
  localHeads_ = (numHeads + tensorParallelSize - 1) / tensorParallelSize;
  
  // Create local cache for this rank's heads
  int32_t deviceId = deviceIds.empty() ? 0 : deviceIds[0];
  localCache_ = std::make_unique<KVCacheShard>(
      deviceId, numLayers, localHeads_, headDim,
      blockSize, maxSeqLen, elementType);
  
  if (tensorParallelSize > 1) {
    auto ncclHandle = std::make_unique<NCCLCommunicationHandle>(deviceIds);
    if (ncclHandle->initialize().succeeded()) {
      commHandle_ = std::move(ncclHandle);
    }
  }
}

TensorParallelKVCache::~TensorParallelKVCache() = default;

KVCacheShard& TensorParallelKVCache::getLocalCache() {
  return *localCache_;
}

const KVCacheShard& TensorParallelKVCache::getLocalCache() const {
  return *localCache_;
}

LogicalResult TensorParallelKVCache::appendKVLocal(
    const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  if (localCache_->activate().failed()) {
    return failure();
  }
  
  return localCache_->getCache().appendKV(keyData, valueData, batchSize, seqLen,
                                           seqIds, blockIndices);
}

LogicalResult TensorParallelKVCache::allGatherKV(
    const int32_t* blockIndices, const int32_t* seqLens,
    int32_t batchSize, void* outputKeys, void* outputValues) {
  
  // First, lookup local KV
  size_t localSize = batchSize * localHeads_ * headDim_ * sizeof(float);
  std::vector<uint8_t> localKeys(localSize);
  std::vector<uint8_t> localValues(localSize);
  
  if (localCache_->getCache().lookupKV(blockIndices, seqLens, batchSize,
                                        localKeys.data(), 
                                        localValues.data()).failed()) {
    return failure();
  }
  
  if (!commHandle_ || tensorParallelSize_ == 1) {
    // No communication needed, just copy
    std::memcpy(outputKeys, localKeys.data(), localSize);
    std::memcpy(outputValues, localValues.data(), localSize);
    return success();
  }
  
  // All-gather across tensor parallel group
  size_t totalSize = localSize * tensorParallelSize_;
  
  // All-gather keys
  if (commHandle_->allToAll(localKeys.data(), outputKeys, 
                            totalSize, 1).failed()) {
    return failure();
  }
  
  // All-gather values
  if (commHandle_->allToAll(localValues.data(), outputValues,
                            totalSize, 1).failed()) {
    return failure();
  }
  
  return success();
}

LogicalResult TensorParallelKVCache::reduceScatterOutput(
    const void* input, void* output,
    int32_t batchSize, int32_t seqLen) {
  
  if (!commHandle_ || tensorParallelSize_ == 1) {
    size_t size = batchSize * seqLen * localHeads_ * headDim_ * sizeof(float);
    std::memcpy(output, input, size);
    return success();
  }
  
  // Reduce-scatter the attention output
  size_t totalSize = batchSize * seqLen * totalHeads_ * headDim_ * sizeof(float);
  
  return commHandle_->allReduce(input, output, totalSize, sizeof(float));
}

} // namespace runtime
} // namespace llm
} // namespace mlir
