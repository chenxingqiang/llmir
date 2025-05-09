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

// Forward declare Type from MLIR to avoid include errors
namespace mlir {
class Type;
class LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
inline LogicalResult failure() { return LogicalResult::failure(); }
inline bool succeeded(LogicalResult result) { return result.succeeded(); }
inline bool failed(LogicalResult result) { return result.failed(); }
} // namespace mlir

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm> // For std::find
#include <utility>   // For std::pair

namespace mlir {
namespace llm {
namespace runtime {

/// Forward declarations
class PagedKVCache;
class KVBlock;
class BlockAllocator;

// Hash function for std::pair<int32_t, int64_t>
struct PairHash {
  std::size_t operator()(const std::pair<int32_t, int64_t>& p) const {
    auto h1 = std::hash<int32_t>{}(p.first);
    auto h2 = std::hash<int64_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

/// Represents a block of KV cache with a fixed block size
class KVBlock {
public:
  /// Constructor
  KVBlock(void* keyPtr, void* valuePtr, int64_t blockSize, int64_t headDim)
      : keyPtr(keyPtr), valuePtr(valuePtr), blockSize(blockSize), headDim(headDim) {}

  /// Get the pointer to key storage
  void* getKeyPtr() const { return keyPtr; }

  /// Get the pointer to value storage
  void* getValuePtr() const { return valuePtr; }

  /// Get the block size (number of tokens)
  int64_t getBlockSize() const { return blockSize; }

  /// Get head dimension size
  int64_t getHeadDim() const { return headDim; }

private:
  void* keyPtr;
  void* valuePtr;
  int64_t blockSize;
  int64_t headDim;
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

  /// Get number of allocated blocks
  int64_t getNumAllocatedBlocks() const { return allocatedBlocks.size(); }

  /// Get number of free blocks
  int64_t getNumFreeBlocks() const { return freeBlocks.size(); }
  
  // Make allocatedBlocks public for testing
  std::vector<KVBlock*> allocatedBlocks;

private:
  int64_t blockSize;
  int64_t numHeads;
  int64_t headDim;
  Type elementType;
  bool useGPU;

  std::vector<KVBlock*> freeBlocks;

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

  // Maps sequence IDs to their block indices and positions
  // Key: (sequenceId, layerIdx)
  // Value: vector of (blockIdx, posInBlock) pairs for this sequence
  std::unordered_map<std::pair<int32_t, int64_t>, 
                     std::vector<std::pair<int32_t, int64_t>>, 
                     PairHash> seqToBlocks;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHE_H_ 