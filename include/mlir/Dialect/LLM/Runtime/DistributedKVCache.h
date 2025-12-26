//===- DistributedKVCache.h - Distributed KV Cache Support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines distributed KV cache support for multi-GPU environments,
// enabling sharding of the cache across multiple devices for large models.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_DISTRIBUTEDKVCACHE_H_
#define MLIR_DIALECT_LLM_RUNTIME_DISTRIBUTEDKVCACHE_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Sharding Strategy
//===----------------------------------------------------------------------===//

enum class ShardingStrategy {
  LAYER_WISE,      // Each GPU handles specific layers
  HEAD_WISE,       // Each GPU handles specific attention heads
  SEQUENCE_WISE,   // Each GPU handles specific sequences
  HYBRID           // Combination of strategies
};

struct ShardingConfig {
  ShardingStrategy strategy;
  int64_t numDevices;
  std::vector<int32_t> deviceIds;
  
  // Layer-wise sharding config
  std::vector<std::pair<int64_t, int64_t>> layerRanges; // [start, end) per device
  
  // Head-wise sharding config
  std::vector<std::pair<int64_t, int64_t>> headRanges;  // [start, end) per device
  
  // Memory limits per device
  std::vector<size_t> memoryLimits;
  
  // Communication settings
  bool enableOverlap;          // Overlap computation with communication
  bool useNCCL;                // Use NCCL for GPU communication
  int64_t communicationBuffer; // Buffer size for async communication
  
  ShardingConfig()
      : strategy(ShardingStrategy::LAYER_WISE), numDevices(1),
        enableOverlap(true), useNCCL(true), communicationBuffer(1024 * 1024) {}
};

//===----------------------------------------------------------------------===//
// Device Information
//===----------------------------------------------------------------------===//

struct DeviceInfo {
  int32_t deviceId;
  int64_t totalMemory;
  int64_t availableMemory;
  std::string deviceName;
  int32_t computeCapability; // For CUDA devices
  bool isActive;
};

//===----------------------------------------------------------------------===//
// Communication Handle
//===----------------------------------------------------------------------===//

class CommunicationHandle {
public:
  virtual ~CommunicationHandle() = default;
  
  // All-to-all communication for attention
  virtual LogicalResult allToAll(const void* sendBuf, void* recvBuf,
                                 int64_t count, int64_t elementSize) = 0;
  
  // All-reduce for gradient synchronization
  virtual LogicalResult allReduce(const void* sendBuf, void* recvBuf,
                                  int64_t count, int64_t elementSize) = 0;
  
  // Point-to-point communication
  virtual LogicalResult send(const void* buf, int64_t count, 
                            int64_t elementSize, int32_t destDevice) = 0;
  virtual LogicalResult recv(void* buf, int64_t count,
                            int64_t elementSize, int32_t srcDevice) = 0;
  
  // Barrier synchronization
  virtual LogicalResult barrier() = 0;
};

//===----------------------------------------------------------------------===//
// NCCL Communication Handle
//===----------------------------------------------------------------------===//

class NCCLCommunicationHandle : public CommunicationHandle {
public:
  NCCLCommunicationHandle(const std::vector<int32_t>& deviceIds);
  ~NCCLCommunicationHandle() override;
  
  LogicalResult allToAll(const void* sendBuf, void* recvBuf,
                         int64_t count, int64_t elementSize) override;
  LogicalResult allReduce(const void* sendBuf, void* recvBuf,
                          int64_t count, int64_t elementSize) override;
  LogicalResult send(const void* buf, int64_t count,
                    int64_t elementSize, int32_t destDevice) override;
  LogicalResult recv(void* buf, int64_t count,
                    int64_t elementSize, int32_t srcDevice) override;
  LogicalResult barrier() override;
  
  // Initialize NCCL communicators
  LogicalResult initialize();
  
private:
  std::vector<int32_t> deviceIds_;
  void* ncclComms_; // Opaque pointer to NCCL communicators
  bool initialized_;
};

//===----------------------------------------------------------------------===//
// Shard
//===----------------------------------------------------------------------===//

// Represents a shard of the KV cache on a single device
class KVCacheShard {
public:
  KVCacheShard(int32_t deviceId, int64_t numLayers, int64_t numHeads,
               int64_t headDim, int64_t blockSize, int64_t maxSeqLen,
               Type elementType);
  ~KVCacheShard();
  
  // Basic getters
  int32_t getDeviceId() const { return deviceId_; }
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  
  // Access the underlying cache
  PagedKVCache& getCache() { return *cache_; }
  const PagedKVCache& getCache() const { return *cache_; }
  
  // Memory statistics
  size_t getMemoryUsage() const;
  size_t getAvailableMemory() const;
  
  // Activate device before operations
  LogicalResult activate();
  
private:
  int32_t deviceId_;
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t blockSize_;
  int64_t maxSeqLen_;
  Type elementType_;
  
  std::unique_ptr<PagedKVCache> cache_;
};

//===----------------------------------------------------------------------===//
// Distributed Paged KV Cache
//===----------------------------------------------------------------------===//

class DistributedPagedKVCache {
public:
  DistributedPagedKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                          int64_t blockSize, int64_t maxSeqLen,
                          Type elementType, const ShardingConfig& config);
  ~DistributedPagedKVCache();
  
  // Basic getters
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  int64_t getBlockSize() const { return blockSize_; }
  int64_t getMaxSeqLen() const { return maxSeqLen_; }
  int64_t getNumDevices() const { return config_.numDevices; }
  
  // Get sharding configuration
  const ShardingConfig& getShardingConfig() const { return config_; }
  
  // Core KV cache operations (distributed)
  LogicalResult appendKV(const void* keyData, const void* valueData,
                        int32_t batchSize, int32_t seqLen,
                        const int32_t* seqIds, int32_t* blockIndices);
  
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int32_t batchSize, void* outputKeys, void* outputValues);
  
  // Cross-device attention computation
  LogicalResult computeDistributedAttention(const void* queries,
                                            const int32_t* blockIndices,
                                            const int32_t* seqLens,
                                            int32_t batchSize,
                                            void* output);
  
  // Sequence management
  LogicalResult clearSequence(int32_t seqId);
  void reset();
  int64_t getSequenceLength(int32_t seqId) const;
  
  // Memory statistics across all devices
  size_t getTotalMemoryUsage() const;
  std::vector<size_t> getPerDeviceMemoryUsage() const;
  
  // Load balancing
  LogicalResult rebalance();
  void updateLoadMetrics();
  
  // Device management
  std::vector<DeviceInfo> getDeviceInfo() const;
  LogicalResult setActiveDevices(const std::vector<int32_t>& deviceIds);
  
  // Performance metrics
  struct DistributedMetrics {
    double totalCommunicationTime;
    double totalComputeTime;
    int64_t numCommunications;
    int64_t totalBytesTransferred;
    std::vector<double> perDeviceUtilization;
  };
  
  DistributedMetrics getMetrics() const { return metrics_; }
  void resetMetrics();
  
private:
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t blockSize_;
  int64_t maxSeqLen_;
  Type elementType_;
  ShardingConfig config_;
  
  // Shards on each device
  std::vector<std::unique_ptr<KVCacheShard>> shards_;
  
  // Communication handle
  std::unique_ptr<CommunicationHandle> commHandle_;
  
  // Sequence to device mapping
  std::unordered_map<int32_t, int32_t> sequenceDeviceMap_;
  
  // Load tracking
  std::vector<float> deviceLoads_;
  
  // Performance metrics
  mutable DistributedMetrics metrics_;
  
  // Helper methods
  void initializeShards();
  void initializeCommunication();
  
  // Sharding helpers
  int32_t getDeviceForLayer(int64_t layer) const;
  int32_t getDeviceForHead(int64_t head) const;
  int32_t getDeviceForSequence(int32_t seqId) const;
  
  // Data distribution
  LogicalResult distributeData(const void* data, int64_t totalSize,
                               std::vector<void*>& devicePtrs,
                               std::vector<int64_t>& deviceSizes);
  LogicalResult gatherData(const std::vector<void*>& devicePtrs,
                           const std::vector<int64_t>& deviceSizes,
                           void* output, int64_t totalSize);
  
  // Communication primitives
  LogicalResult allGatherKV(int64_t layer, void* output, int64_t outputSize);
  LogicalResult scatterKV(int64_t layer, const void* input, int64_t inputSize);
};

//===----------------------------------------------------------------------===//
// Pipeline Parallel Support
//===----------------------------------------------------------------------===//

// Support for pipeline parallelism with KV cache
class PipelineKVCache {
public:
  PipelineKVCache(int64_t numStages, int64_t layersPerStage,
                  int64_t numHeads, int64_t headDim,
                  int64_t blockSize, int64_t maxSeqLen,
                  Type elementType, const std::vector<int32_t>& deviceIds);
  ~PipelineKVCache();
  
  // Get cache for a specific pipeline stage
  KVCacheShard& getStageCache(int64_t stage);
  const KVCacheShard& getStageCache(int64_t stage) const;
  
  // Pipeline operations
  LogicalResult passActivations(int64_t fromStage, int64_t toStage,
                                const void* data, int64_t size);
  
  // Sequence management across pipeline
  LogicalResult appendKV(int64_t stage, const void* keyData, const void* valueData,
                        int32_t batchSize, int32_t seqLen,
                        const int32_t* seqIds, int32_t* blockIndices);
  
  LogicalResult lookupKV(int64_t stage, const int32_t* blockIndices,
                        const int32_t* seqLens, int32_t batchSize,
                        void* outputKeys, void* outputValues);
  
private:
  int64_t numStages_;
  int64_t layersPerStage_;
  std::vector<std::unique_ptr<KVCacheShard>> stageCaches_;
  std::unique_ptr<CommunicationHandle> commHandle_;
};

//===----------------------------------------------------------------------===//
// Tensor Parallel Support
//===----------------------------------------------------------------------===//

// Support for tensor parallelism where heads are split across devices
class TensorParallelKVCache {
public:
  TensorParallelKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                        int64_t blockSize, int64_t maxSeqLen,
                        Type elementType, int64_t tensorParallelSize,
                        const std::vector<int32_t>& deviceIds);
  ~TensorParallelKVCache();
  
  // Get local cache for current rank
  KVCacheShard& getLocalCache();
  const KVCacheShard& getLocalCache() const;
  
  // Tensor parallel operations
  LogicalResult appendKVLocal(const void* keyData, const void* valueData,
                              int32_t batchSize, int32_t seqLen,
                              const int32_t* seqIds, int32_t* blockIndices);
  
  // All-gather for full attention computation
  LogicalResult allGatherKV(const int32_t* blockIndices, const int32_t* seqLens,
                           int32_t batchSize, void* outputKeys, void* outputValues);
  
  // Reduce-scatter for output
  LogicalResult reduceScatterOutput(const void* input, void* output,
                                    int32_t batchSize, int32_t seqLen);
  
private:
  int64_t numLayers_;
  int64_t totalHeads_;
  int64_t localHeads_;
  int64_t headDim_;
  int64_t tensorParallelSize_;
  int64_t localRank_;
  
  std::unique_ptr<KVCacheShard> localCache_;
  std::unique_ptr<CommunicationHandle> commHandle_;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_DISTRIBUTEDKVCACHE_H_
