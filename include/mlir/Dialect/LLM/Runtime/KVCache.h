//===- KVCache.h - Runtime support for LLM KV cache ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines runtime support for the LLM PagedKVCache data structure.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_KVCACHE_H_
#define MLIR_DIALECT_LLM_RUNTIME_KVCACHE_H_

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm> // For std::find
#include <utility>   // For std::pair
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

// Forward declarations for MLIR classes
namespace mlir {
// Forward declare Type without including MLIR headers
struct Type {
  int getIntOrFloatBitWidth() const { return 16; } // Simplified for testing
};

// Forward declare LogicalResult without including MLIR headers
class LogicalResult {
private:
  bool succeeded_;
public:
  LogicalResult(bool succeeded = true) : succeeded_(succeeded) {}
  bool succeeded() const { return succeeded_; }
  bool failed() const { return !succeeded_; }
  
  static LogicalResult success() { return LogicalResult(true); }
  static LogicalResult failure() { return LogicalResult(false); }
};

inline LogicalResult success() { return LogicalResult::success(); }
inline LogicalResult failure() { return LogicalResult::failure(); }
inline bool succeeded(LogicalResult result) { return result.succeeded(); }
inline bool failed(LogicalResult result) { return result.failed(); }
} // namespace mlir

namespace mlir {
namespace llm {
namespace runtime {

/// Forward declarations
class PagedKVCache;
class KVBlock;
class BlockAllocator;
class EvictionPolicy;
class AttentionConfig;
class AttentionImpl;

// Hash function for std::pair<int32_t, int64_t>
namespace {
struct PairHash {
  std::size_t operator()(const std::pair<int32_t, int64_t>& p) const {
    auto h1 = std::hash<int32_t>{}(p.first);
    auto h2 = std::hash<int64_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

// Hash function for sequence content for duplicate detection
struct ContentHash {
  template<typename T>
  std::size_t operator()(const std::vector<T>& content) const {
    std::size_t hash = 0;
    for (const auto& val : content) {
      hash ^= std::hash<T>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};
} // anonymous namespace

/// Struct to track sequence information in the KV cache
struct SequenceInfo {
  // Current position in the sequence
  int64_t currentPos = 0;
  
  // Block indices and positions within blocks for this sequence
  std::vector<std::pair<int32_t, int64_t>> blockPositions;
  
  // Reference to the last active block
  int32_t lastBlockIdx = -1;
  
  // Position within the last block
  int64_t posInLastBlock = 0;
  
  // Content hash for duplicate detection
  std::size_t contentHash = 0;
  
  // Flag indicating if this sequence shares blocks with another sequence
  bool sharesBlocks = false;
  
  // If this sequence shares blocks, the source sequence ID
  int32_t sourceSeqId = -1;
  
  // Shared blocks position mapping (for partial sharing)
  std::unordered_map<int32_t, int32_t> sharedBlockMapping;
};

/// Struct to track block metrics
struct BlockMetrics {
  int64_t totalBlocks = 0;
  int64_t freeBlocks = 0;
  int64_t usedBlocks = 0;
  double avgFragmentation = 0.0;
  int64_t numEvictions = 0;
  int64_t numCoalesces = 0;
  // Enhanced metrics for optimization
  int64_t numPreallocated = 0;
  int64_t numCreated = 0;
  int64_t numReused = 0;
  int64_t totalTokensStored = 0;
  int64_t peakMemoryUsage = 0;
  double avgBlockUtilization = 0.0; // Average percentage of block capacity used
  int64_t numEvictionAttempts = 0;  // Number of times eviction was considered
};

/// Eviction policy interface for KV cache blocks
class EvictionPolicy {
public:
  virtual ~EvictionPolicy() = default;
  
  /// Select blocks for eviction when memory is constrained
  virtual std::vector<int32_t> selectBlocksForEviction(
      const BlockAllocator& allocator, int64_t numBlocksNeeded) const = 0;
      
  /// Clone this policy (for creating copies)
  virtual std::unique_ptr<EvictionPolicy> clone() const = 0;
};

/// LRU (Least Recently Used) eviction policy
class LRUEvictionPolicy : public EvictionPolicy {
public:
  std::vector<int32_t> selectBlocksForEviction(
      const BlockAllocator& allocator, int64_t numBlocksNeeded) const override;
      
  std::unique_ptr<EvictionPolicy> clone() const override {
    return std::make_unique<LRUEvictionPolicy>();
  }
};

/// Fragmentation-aware LRU eviction policy
class FragmentationAwareLRUPolicy : public EvictionPolicy {
public:
  FragmentationAwareLRUPolicy(double fragmentationWeight = 0.5)
    : fragmentationWeight(fragmentationWeight) {}
    
  std::vector<int32_t> selectBlocksForEviction(
      const BlockAllocator& allocator, int64_t numBlocksNeeded) const override;
      
  std::unique_ptr<EvictionPolicy> clone() const override {
    return std::make_unique<FragmentationAwareLRUPolicy>(fragmentationWeight);
  }
  
private:
  // Weight between 0.0 and 1.0 determining how much to prioritize
  // fragmented blocks (1.0) vs old blocks (0.0)
  double fragmentationWeight;
};

/// Enhanced block metadata to track LRU and fragmentation info
struct BlockMetadata {
  int64_t lastAccessTime = 0;  // Timestamp for LRU
  double fragmentation = 0.0;  // % of unused space in the block
  bool isEvictable = true;     // Can this block be evicted?
};

/// Represents a block of KV cache with a fixed block size
class KVBlock {
public:
  /// Constructor
  KVBlock(void* keyPtr, void* valuePtr, int64_t blockSize, int64_t headDim)
      : keyPtr(keyPtr), valuePtr(valuePtr), blockSize(blockSize), headDim(headDim),
        usedSlots(0), refCount(0) {}

  /// Get the pointer to key storage
  void* getKeyPtr() const { return keyPtr; }

  /// Get the pointer to value storage
  void* getValuePtr() const { return valuePtr; }

  /// Get the block size (number of tokens)
  int64_t getBlockSize() const { return blockSize; }

  /// Get head dimension size
  int64_t getHeadDim() const { return headDim; }
  
  /// Get number of used slots in this block
  int64_t getUsedSlots() const { return usedSlots; }
  
  /// Increment used slots count
  void incrementUsedSlots(int64_t count = 1) { usedSlots += count; }
  
  /// Reset used slots count
  void resetUsedSlots() { usedSlots = 0; }
  
  /// Check if block is full
  bool isFull() const { return usedSlots >= blockSize; }
  
  /// Get the reference count
  int32_t getRefCount() const { return refCount; }
  
  /// Increment reference count
  void incrementRefCount() { refCount++; }
  
  /// Decrement reference count
  bool decrementRefCount() { 
    if (refCount > 0) {
      refCount--;
      return true;
    }
    return false;
  }
  
  /// Update the last access time
  void updateAccessTime(int64_t timestamp) { metadata.lastAccessTime = timestamp; }
  
  /// Get the last access time
  int64_t getLastAccessTime() const { return metadata.lastAccessTime; }
  
  /// Set evictable status
  void setEvictable(bool evictable) { metadata.isEvictable = evictable; }
  
  /// Check if block is evictable
  bool isEvictable() const { return metadata.isEvictable && refCount == 0; }
  
  /// Update fragmentation metric
  void updateFragmentation() { 
    metadata.fragmentation = blockSize > 0 ? 
        static_cast<double>(blockSize - usedSlots) / blockSize : 0.0;
  }
  
  /// Get fragmentation metric
  double getFragmentation() const { return metadata.fragmentation; }

private:
  void* keyPtr;
  void* valuePtr;
  int64_t blockSize;
  int64_t headDim;
  int64_t usedSlots;
  int32_t refCount;
  BlockMetadata metadata;
};

/// Block allocator for KV cache
class BlockAllocator {
public:
  /// Constructor
  BlockAllocator(int64_t blockSize, int64_t numHeads, int64_t headDim, 
                Type elementType, bool useGPU = true);
  
  /// Destructor
  ~BlockAllocator();

  /// Allocate a new KV block
  KVBlock* allocateBlock();

  /// Free a KV block
  void freeBlock(KVBlock* block);
  
  /// Get block by index
  KVBlock* getBlock(int32_t blockIdx) const;

  /// Get number of allocated blocks
  int64_t getNumAllocatedBlocks() const { return allocatedBlocks.size(); }

  /// Get number of free blocks
  int64_t getNumFreeBlocks() const { return freeBlocks.size(); }
  
  /// Get element type size
  int64_t getElementTypeSize() const;
  
  /// Get block size
  int64_t getBlockSize() const { return blockSize; }
  
  /// Get number of heads
  int64_t getNumHeads() const { return numHeads; }
  
  /// Get head dimension
  int64_t getHeadDim() const { return headDim; }
  
  /// Get element type
  Type getElementType() const { return elementType; }
  
  /// Enhanced preallocation strategy
  /// Preallocate blocks based on analysis of sequence patterns and model configuration
  /// @param baseNumBlocks Minimum number of blocks to preallocate
  /// @param avgSeqLen Average sequence length for the workload
  /// @param maxNumSequences Maximum number of concurrent sequences expected
  void preallocateBlocks(int64_t baseNumBlocks, 
                         int64_t avgSeqLen = 0, 
                         int64_t maxNumSequences = 0);
  
  /// Simple preallocation (for backward compatibility)
  void preallocateBlocks(int64_t numBlocks) {
    preallocateBlocks(numBlocks, 0, 0);
  }
  
  /// Advanced block coalescing
  /// Attempts to merge contents of partially filled blocks to reduce fragmentation
  /// @param fragmentationThreshold Only consider blocks with fragmentation above this value
  /// @param maxBlocksToCoalesce Maximum number of blocks to process in one call
  /// @param preserveOrder Whether to maintain token order when coalescing
  /// @return Number of blocks coalesced
  int64_t coalesceBlocks(double fragmentationThreshold = 0.3, 
                        int64_t maxBlocksToCoalesce = 10,
                        bool preserveOrder = true);
  
  /// Evict blocks if memory is constrained
  void evictBlocksIfNeeded(int64_t numBlocksNeeded = 1);
  
  /// Get current block metrics
  BlockMetrics getMetrics() const;
  
  /// Enable or disable metrics collection
  void enableMetrics(bool enable) { collectMetrics = enable; }
  
  /// Set the eviction policy
  void setEvictionPolicy(std::unique_ptr<EvictionPolicy> policy) { 
    evictionPolicy = std::move(policy); 
  }
  
  /// Get current timestamp for LRU tracking
  int64_t getCurrentTimestamp() const { return timestampCounter++; }
  
  /// Get memory requirements (in bytes) for each block
  int64_t getBlockMemorySize() const {
    return blockSize * numHeads * headDim * getElementTypeSize() * 2; // key and value
  }
  
  /// Get total memory usage (in bytes)
  int64_t getTotalMemoryUsage() const {
    return getBlockMemorySize() * (allocatedBlocks.size() + freeBlocks.size());
  }
  
  /// Find the most suitable block for coalescing
  /// @param sourceBlock The block we want to merge from
  /// @return Index of the target block or -1 if none found
  int32_t findCoalescingTarget(KVBlock* sourceBlock) const;
  
  /// Copy data between blocks for coalescing
  /// @param sourceBlock Source block to copy from
  /// @param targetBlock Target block to copy to
  /// @param sourceStartPos Starting position in source block 
  /// @param targetStartPos Starting position in target block
  /// @param numTokens Number of tokens to copy
  /// @return Success if the copy was successful
  LogicalResult copyBetweenBlocks(KVBlock* sourceBlock, KVBlock* targetBlock,
                                 int64_t sourceStartPos, int64_t targetStartPos,
                                 int64_t numTokens);
  
  // Make allocatedBlocks and freeBlocks public for testing and policy access
  std::vector<KVBlock*> allocatedBlocks;
  std::vector<KVBlock*> freeBlocks;

private:
  int64_t blockSize;
  int64_t numHeads;
  int64_t headDim;
  Type elementType;
  bool useGPU;
  bool collectMetrics = true;
  
  // Metrics
  mutable BlockMetrics metrics;
  
  // Eviction policy
  std::unique_ptr<EvictionPolicy> evictionPolicy;
  
  // Timestamp counter for LRU
  mutable int64_t timestampCounter = 0;
  
  // Sequence length statistics for adaptive preallocation
  int64_t minObservedSeqLen = 0;
  int64_t maxObservedSeqLen = 0;
  double avgObservedSeqLen = 0.0;
  int64_t totalObservations = 0;

  // Helper method to create a new block
  KVBlock* createNewBlock();
  
  // Update metrics after allocation/deallocation
  void updateMetrics();
  
  // Calculate current fragmentation
  double calculateFragmentation() const;
  
  // Update sequence length statistics
  void updateSeqLengthStats(int64_t seqLen);
};

/// PagedKVCache provides runtime support for the paged KV cache
class PagedKVCache {
public:
  /// Constructor
  PagedKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
              int64_t blockSize, int64_t maxSeqLen, Type elementType,
              bool useGPU = true);

  /// Destructor
  ~PagedKVCache();

  /// Append new key-value pairs to the cache
  LogicalResult appendKV(const void* keyPtr, const void* valuePtr,
                        int64_t batchSize, int64_t seqLen, 
                        const int32_t* seqIds, int32_t* blockIndices);

  /// Lookup key-value pairs from the cache
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int64_t batchSize, void* outputKeys, void* outputValues);
                        
  /// Clear sequence data from the cache                    
  LogicalResult clearSequence(int32_t seqId);
  
  /// Reset the entire cache
  void reset();

  /// Get the number of layers
  int64_t getNumLayers() const { return numLayers; }

  /// Get the number of heads
  int64_t getNumHeads() const { return numHeads; }

  /// Get the head dimension
  int64_t getHeadDim() const { return headDim; }

  /// Get the block size
  int64_t getBlockSize() const { return blockSize; }

  /// Get the maximum sequence length
  int64_t getMaxSeqLen() const { return maxSeqLen; }

  /// Get the element type
  Type getElementType() const { return elementType; }
  
  /// Get the total memory usage in bytes
  int64_t getTotalMemoryUsage() const;
  
  /// Get the number of sequences currently in the cache
  int64_t getNumSequences() const;
  
  /// Get the number of tokens for a sequence
  int64_t getSequenceLength(int32_t seqId) const;

  /// Check if two sequences have identical prefixes up to a given length
  bool hasIdenticalPrefix(int32_t seqId1, int32_t seqId2, int64_t prefixLen) const;
  
  /// Share KV cache blocks between sequences (source to target)
  LogicalResult shareSequenceBlocks(int32_t sourceSeqId, int32_t targetSeqId, int64_t numTokens);
  
  /// Get content hash for a sequence
  std::size_t getSequenceContentHash(int32_t seqId) const;
  
  /// Generate content hash for input key-value data
  std::size_t generateContentHash(const void* keyPtr, const void* valuePtr, 
                                 int64_t seqLen) const;
  
  /// Find sequence with identical content
  int32_t findIdenticalSequence(const void* keyPtr, const void* valuePtr, 
                               int64_t seqLen) const;

  /// Configure the block allocators
  void configureBlockAllocators(int64_t initialBlocksPerLayer, bool enableMetrics);

  /// Enhanced configuration for block allocators
  /// @param avgSeqLen Expected average sequence length
  /// @param maxConcurrentSeqs Maximum number of concurrent sequences
  /// @param enableMetrics Whether to collect metrics
  /// @param preallocationStrategy Preallocation strategy (0: minimal, 1: balanced, 2: aggressive)
  void configureBlockAllocatorsAdvanced(int64_t avgSeqLen, int64_t maxConcurrentSeqs,
                                      bool enableMetrics, int preallocationStrategy = 1);
  
  /// Set eviction policy for all layers
  void setEvictionPolicy(std::unique_ptr<EvictionPolicy> policy);
  
  /// Run block coalescing on all layers
  void runBlockCoalescing();
  
  /// Run advanced block coalescing on all layers
  /// @param fragmentationThreshold Only consider blocks with fragmentation above this value
  /// @param maxBlocksToCoalesce Maximum number of blocks to process in one call
  /// @param preserveOrder Whether to maintain token order when coalescing
  /// @return Total number of blocks coalesced across all layers
  int64_t runAdvancedBlockCoalescing(double fragmentationThreshold = 0.3,
                                   int64_t maxBlocksToCoalesce = 10,
                                   bool preserveOrder = true);
  
  /// Enable auto-coalescing (will automatically run coalescing when fragmentation exceeds threshold)
  void enableAutoCoalescing(double fragmentationThreshold = 0.3);
  
  /// Disable auto-coalescing
  void disableAutoCoalescing();
  
  /// Get metrics for all layers
  std::vector<BlockMetrics> getAllBlockMetrics() const;
  
  /// Calculate optimal block size based on sequence patterns
  /// @param minBlockSize Minimum block size to consider
  /// @param maxBlockSize Maximum block size to consider
  /// @return Recommended block size
  int64_t calculateOptimalBlockSize(int64_t minBlockSize = 8, int64_t maxBlockSize = 128) const;
  
  /// Get sequence length statistics across all current sequences
  /// @param minSeqLen Output parameter for minimum sequence length
  /// @param maxSeqLen Output parameter for maximum sequence length
  /// @param avgSeqLen Output parameter for average sequence length
  void getSequenceLengthStats(int64_t& minSeqLen, int64_t& maxSeqLen, double& avgSeqLen) const;

  /// Configure attention optimization
  /// \param config Attention configuration parameters
  void configureAttentionOpt(const AttentionConfig& config);
  
  /// Efficiently gather keys and values from KV cache for attention computation
  /// \param outputKeys Output buffer for gathered keys [numLayers, numTokens, numHeads, headDim]
  /// \param outputValues Output buffer for gathered values [numLayers, numTokens, numHeads, headDim]
  /// \param seqId Sequence ID to gather from
  /// \param startPos Starting position in the sequence
  /// \param numTokens Number of tokens to gather
  LogicalResult gatherKVForAttention(
      void* outputKeys,
      void* outputValues,
      int32_t seqId,
      int64_t startPos,
      int64_t numTokens);
  
  /// Compute optimized attention using the KV cache
  /// \param output Output tensor [batchSize, seqLen, numHeads, headDim]
  /// \param queries Query tensor [batchSize, seqLen, numHeads, headDim]
  /// \param blockIndices Indices into the KV cache [batchSize, maxSeqLen]
  /// \param seqLens Actual sequence lengths [batchSize]
  /// \param batchSize Batch size
  /// \param seqLen Sequence length of queries
  LogicalResult computeAttention(
      void* output,
      const void* queries,
      const int32_t* blockIndices,
      const int32_t* seqLens,
      int64_t batchSize,
      int64_t seqLen);

private:
  int64_t numLayers;
  int64_t numHeads;
  int64_t headDim;
  int64_t blockSize;
  int64_t maxSeqLen;
  Type elementType;
  bool useGPU;
  
  // Auto-coalescing configuration
  bool autoCoalescingEnabled = false;
  double autoCoalescingThreshold = 0.3;
  
  // Map from layer index to block allocator
  std::vector<std::unique_ptr<BlockAllocator>> blockAllocators;
  
  // Map from sequence ID to sequence info for each layer
  std::vector<std::unordered_map<int32_t, SequenceInfo>> layerSeqInfo;
  
  // Map of content hash to sequence ID for duplicate detection
  std::unordered_map<std::size_t, std::vector<int32_t>> contentHashToSeqIds;
  
  // Helper function to copy data to a block
  LogicalResult copyToBlock(KVBlock* block, int64_t posInBlock,
                           const void* keyPtr, const void* valuePtr,
                           int64_t tokenOffset, int64_t numTokens);
  
  // Helper function to copy data from a block
  LogicalResult copyFromBlock(const KVBlock* block, int64_t posInBlock,
                             void* outputKeys, void* outputValues,
                             int64_t tokenOffset, int64_t numTokens);
  
  // Helper function to allocate a block for a sequence
  LogicalResult allocateBlockForSequence(int32_t seqId, int64_t layerIdx,
                                        int32_t& blockIdx);
                                        
  // Helper function to compare two sequences for identity
  bool compareSequenceContent(int32_t seqId1, int32_t seqId2, int64_t length) const;
  
  // Helper function to update content hash mapping
  void updateContentHashMapping(int32_t seqId, std::size_t contentHash);

  // Attention implementation
  std::unique_ptr<AttentionImpl> attentionImpl;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHE_H_ 