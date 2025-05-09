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

// Hash function for std::pair<int32_t, int64_t>
namespace {
struct PairHash {
  std::size_t operator()(const std::pair<int32_t, int64_t>& p) const {
    auto h1 = std::hash<int32_t>{}(p.first);
    auto h2 = std::hash<int64_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
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

private:
  void* keyPtr;
  void* valuePtr;
  int64_t blockSize;
  int64_t headDim;
  int64_t usedSlots;
  int32_t refCount;
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
  
  // Make allocatedBlocks and freeBlocks public for testing
  std::vector<KVBlock*> allocatedBlocks;
  std::vector<KVBlock*> freeBlocks;

private:
  int64_t blockSize;
  int64_t numHeads;
  int64_t headDim;
  Type elementType;
  bool useGPU;

  // Helper method to create a new block
  KVBlock* createNewBlock();
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

private:
  int64_t numLayers;
  int64_t numHeads;
  int64_t headDim;
  int64_t blockSize;
  int64_t maxSeqLen;
  Type elementType;
  bool useGPU;

  // One block allocator per layer
  std::vector<std::unique_ptr<BlockAllocator>> blockAllocators;

  // Maps sequence IDs to their information per layer
  // Key: (sequenceId, layerIdx)
  // Value: SequenceInfo containing current position and block mappings
  std::unordered_map<std::pair<int32_t, int64_t>, 
                     SequenceInfo,
                     PairHash> seqInfo;
                     
  // Helper methods
  
  // Copy key-value data to a specific position in a block
  LogicalResult copyToBlock(KVBlock* block, int64_t posInBlock,
                           const void* keyPtr, const void* valuePtr,
                           int64_t tokenOffset, int64_t numTokens);
                           
  // Copy key-value data from a specific position in a block
  LogicalResult copyFromBlock(const KVBlock* block, int64_t posInBlock,
                             void* outputKeys, void* outputValues,
                             int64_t tokenOffset, int64_t numTokens);
                             
  // Allocate a new block for a sequence in a specific layer
  LogicalResult allocateBlockForSequence(int32_t seqId, int64_t layerIdx,
                                        int32_t& blockIdx);
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHE_H_ 