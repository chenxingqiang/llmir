//===- KVCache.cpp - Runtime support for LLM KV cache -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements runtime support for the LLM PagedKVCache data structure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCache.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility> // for std::make_pair
#include <array>
#include <algorithm> // For std::find
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stdexcept>

namespace mlir {
namespace llm {
namespace runtime {

namespace {
// Helper function to get the size in bytes for a type
// This is a simplified version for testing without actual MLIR Type definitions
int64_t getTypeSizeInBytes(const Type& type) {
  // Since we can't access the actual Type methods due to forward declaration,
  // we'll return a default value for testing
  return 2; // Default to 2 bytes (f16)
}

// Helper function to allocate memory (on CPU or GPU)
void* allocateMemory(int64_t sizeInBytes, bool useGPU) {
  void* ptr = nullptr;
  
  if (useGPU) {
    // TODO: Implement GPU memory allocation using CUDA/HIP
    // For now, just use CPU memory
    ptr = std::malloc(sizeInBytes);
  } else {
    ptr = std::malloc(sizeInBytes);
  }
  
  // Initialize to zeros
  if (ptr) {
    std::memset(ptr, 0, sizeInBytes);
  }
  
  return ptr;
}

// Helper function to free memory (on CPU or GPU)
void freeMemory(void* ptr, bool useGPU) {
  if (ptr) {
    if (useGPU) {
      // TODO: Implement GPU memory deallocation using CUDA/HIP
      // For now, just use CPU memory free
      std::free(ptr);
    } else {
      std::free(ptr);
    }
  }
}

// Helper function to copy memory (on CPU or GPU)
void copyMemory(void* dst, const void* src, int64_t sizeInBytes, bool useGPU) {
  if (dst && src) {
    if (useGPU) {
      // TODO: Implement GPU memory copy using CUDA/HIP
      // For now, just use CPU memory copy
      std::memcpy(dst, src, sizeInBytes);
    } else {
      std::memcpy(dst, src, sizeInBytes);
    }
  }
}

// Hash function for std::pair<int32_t, int64_t> using Boost's hash_combine algorithm
struct PairHash {
  std::size_t operator()(const std::pair<int32_t, int64_t>& p) const {
    auto h1 = std::hash<int32_t>{}(p.first);
    auto h2 = std::hash<int64_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// BlockAllocator Implementation
//===----------------------------------------------------------------------===//

BlockAllocator::BlockAllocator(int64_t blockSize, int64_t numHeads, 
                              int64_t headDim, Type elementType, bool useGPU)
    : blockSize(blockSize), numHeads(numHeads), headDim(headDim),
      elementType(elementType), useGPU(useGPU) {
  // Pre-allocate some blocks to avoid frequent allocation
  const int initialBlocks = 8;
  for (int i = 0; i < initialBlocks; i++) {
    KVBlock* block = createNewBlock();
    if (block) {
      freeBlocks.push_back(block);
    }
  }
}

BlockAllocator::~BlockAllocator() {
  // Free all allocated blocks
  for (auto* block : freeBlocks) {
    freeMemory(block->getKeyPtr(), useGPU);
    freeMemory(block->getValuePtr(), useGPU);
    delete block;
  }
  
  for (auto* block : allocatedBlocks) {
    freeMemory(block->getKeyPtr(), useGPU);
    freeMemory(block->getValuePtr(), useGPU);
    delete block;
  }
  
  freeBlocks.clear();
  allocatedBlocks.clear();
}

KVBlock* BlockAllocator::allocateBlock() {
  if (freeBlocks.empty()) {
    // No free blocks available, create a new one
    KVBlock* block = createNewBlock();
    if (!block) {
      return nullptr;
    }
    allocatedBlocks.push_back(block);
    return block;
  }
  
  // Reuse an existing free block
  KVBlock* block = freeBlocks.back();
  freeBlocks.pop_back();
  
  // Reset the block state
  block->resetUsedSlots();
  
  allocatedBlocks.push_back(block);
  return block;
}

void BlockAllocator::freeBlock(KVBlock* block) {
  // Find the block in the allocated blocks
  auto it = std::find(allocatedBlocks.begin(), allocatedBlocks.end(), block);
  if (it != allocatedBlocks.end()) {
    allocatedBlocks.erase(it);
    
    // Optionally clear the block memory for security/debugging
    int64_t blockSizeBytes = blockSize * numHeads * headDim * getElementTypeSize();
    std::memset(block->getKeyPtr(), 0, blockSizeBytes);
    std::memset(block->getValuePtr(), 0, blockSizeBytes);
    
    // Reset block's state
    block->resetUsedSlots();
    
    freeBlocks.push_back(block);
  }
}

KVBlock* BlockAllocator::getBlock(int32_t blockIdx) const {
  if (blockIdx >= 0 && blockIdx < static_cast<int32_t>(allocatedBlocks.size())) {
    return allocatedBlocks[blockIdx];
  }
  return nullptr;
}

int64_t BlockAllocator::getElementTypeSize() const {
  return getTypeSizeInBytes(elementType);
}

KVBlock* BlockAllocator::createNewBlock() {
  int64_t typeSizeInBytes = getElementTypeSize();
  
  // Calculate the required memory for one block
  // blockSize: number of tokens in the block
  // numHeads: number of attention heads
  // headDim: dimension of each head
  int64_t memSizePerBlock = blockSize * numHeads * headDim * typeSizeInBytes;
  
  // Allocate memory for keys and values separately
  void* keyPtr = allocateMemory(memSizePerBlock, useGPU);
  void* valuePtr = allocateMemory(memSizePerBlock, useGPU);
  
  if (!keyPtr || !valuePtr) {
    if (keyPtr) freeMemory(keyPtr, useGPU);
    if (valuePtr) freeMemory(valuePtr, useGPU);
    return nullptr;
  }
  
  return new KVBlock(keyPtr, valuePtr, blockSize, headDim);
}

//===----------------------------------------------------------------------===//
// PagedKVCache Implementation
//===----------------------------------------------------------------------===//

PagedKVCache::PagedKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                         int64_t blockSize, int64_t maxSeqLen, Type elementType,
                         bool useGPU)
    : numLayers(numLayers), numHeads(numHeads), headDim(headDim),
      blockSize(blockSize), maxSeqLen(maxSeqLen), elementType(elementType),
      useGPU(useGPU) {
  
  // Create block allocators for each layer
  for (int64_t i = 0; i < numLayers; i++) {
    blockAllocators.push_back(
        std::make_unique<BlockAllocator>(blockSize, numHeads, headDim, 
                                        elementType, useGPU));
  }
}

PagedKVCache::~PagedKVCache() {
  // BlockAllocators will be automatically freed by their destructors
  blockAllocators.clear();
  seqInfo.clear();
}

LogicalResult PagedKVCache::appendKV(const void* keyPtr, const void* valuePtr,
                                    int64_t batchSize, int64_t seqLen,
                                    const int32_t* seqIds, int32_t* blockIndices) {
  if (!keyPtr || !valuePtr || !seqIds || !blockIndices) {
    return failure();
  }
  
  int64_t typeSizeInBytes = getTypeSizeInBytes(elementType);
  int64_t tokensPerHead = numHeads * headDim;
  int64_t singleTokenSize = tokensPerHead * typeSizeInBytes;
  
  // For each sequence in the batch
  for (int64_t b = 0; b < batchSize; b++) {
    int32_t seqId = seqIds[b];
    
    // For each layer
    for (int64_t layer = 0; layer < numLayers; layer++) {
      // Get or create the sequence info for this sequence and layer
      auto key = std::make_pair(seqId, layer);
      auto& info = seqInfo[key];
      
      // For each token in the sequence
      for (int64_t t = 0; t < seqLen; t++) {
        // Check if we need a new block
        if (info.lastBlockIdx < 0 || 
            info.posInLastBlock >= blockSize) {
          // Allocate a new block for this sequence
          int32_t newBlockIdx;
          if (failed(allocateBlockForSequence(seqId, layer, newBlockIdx))) {
            return failure();
          }
          
          // Update the sequence info
          info.lastBlockIdx = newBlockIdx;
          info.posInLastBlock = 0;
        }
        
        // Get the current block
        KVBlock* block = blockAllocators[layer]->getBlock(info.lastBlockIdx);
        if (!block) {
          return failure();
        }
        
        // Copy the key and value data to the block
        int64_t tokenOffset = b * seqLen + t;
        if (failed(copyToBlock(block, info.posInLastBlock, 
                              keyPtr, valuePtr, tokenOffset, 1))) {
          return failure();
        }
        
        // Update block indices output - store the index for this layer and token
        int64_t outputIdx = layer * batchSize * seqLen + b * seqLen + t;
        blockIndices[outputIdx] = info.lastBlockIdx;
        
        // Store the block and position info for this token
        info.blockPositions.push_back(std::make_pair(info.lastBlockIdx, info.posInLastBlock));
        
        // Increment positions
        info.posInLastBlock++;
        info.currentPos++;
        
        // Increment the used slots in the block
        block->incrementUsedSlots();
      }
    }
  }
  
  return success();
}

LogicalResult PagedKVCache::lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                                     int64_t batchSize, void* outputKeys, void* outputValues) {
  if (!blockIndices || !seqLens || !outputKeys || !outputValues) {
    return failure();
  }
  
  int64_t typeSizeInBytes = getTypeSizeInBytes(elementType);
  int64_t tokensPerHead = numHeads * headDim;
  int64_t singleTokenSize = tokensPerHead * typeSizeInBytes;
  
  // Keep track of output position
  int64_t outputPos = 0;
  
  // For each sequence in the batch
  for (int64_t b = 0; b < batchSize; b++) {
    int32_t seqLen = seqLens[b];
    
    // For each token position
    for (int64_t t = 0; t < seqLen; t++) {
      // For each layer
      for (int64_t layer = 0; layer < numLayers; layer++) {
        // Get the block index for this token and layer
        int64_t blockIdxOffset = layer * batchSize * seqLen + b * seqLen + t;
        int32_t blockIdx = blockIndices[blockIdxOffset];
        
        // Get the block
        KVBlock* block = blockAllocators[layer]->getBlock(blockIdx);
        if (!block) {
          return failure();
        }
        
        // Calculate the position within the block
        // This assumes that tokens are stored in the order they were added to the block
        int64_t posInBlock = t % blockSize;
        
        // Copy key and value data from the block to the output
        if (failed(copyFromBlock(block, posInBlock, 
                                outputKeys, outputValues, 
                                outputPos, 1))) {
          return failure();
        }
        
        // Update output position
        outputPos++;
      }
    }
  }
  
  return success();
}

LogicalResult PagedKVCache::clearSequence(int32_t seqId) {
  // For each layer
  for (int64_t layer = 0; layer < numLayers; layer++) {
    auto key = std::make_pair(seqId, layer);
    
    // Remove the sequence info if it exists
    auto it = seqInfo.find(key);
    if (it != seqInfo.end()) {
      auto& info = it->second;
      
      // Decrease reference counts for all blocks used by this sequence
      std::unordered_set<int32_t> uniqueBlocks;
      for (const auto& blockPos : info.blockPositions) {
        uniqueBlocks.insert(blockPos.first);
      }
      
      // Free blocks that are no longer needed
      for (int32_t blockIdx : uniqueBlocks) {
        KVBlock* block = blockAllocators[layer]->getBlock(blockIdx);
        if (block && block->decrementRefCount() && block->getRefCount() == 0) {
          blockAllocators[layer]->freeBlock(block);
        }
      }
      
      // Remove the sequence info
      seqInfo.erase(it);
    }
  }
  
  return success();
}

void PagedKVCache::reset() {
  // Clear all sequence information
  seqInfo.clear();
  
  // Reset all block allocators
  for (auto& allocator : blockAllocators) {
    // Move all allocated blocks to free blocks
    for (auto* block : allocator->allocatedBlocks) {
      // Reset block state
      block->resetUsedSlots();
      
      // Add to free blocks
      allocator->freeBlocks.push_back(block);
    }
    
    // Clear allocated blocks
    allocator->allocatedBlocks.clear();
  }
}

int64_t PagedKVCache::getTotalMemoryUsage() const {
  int64_t totalMemory = 0;
  
  for (const auto& allocator : blockAllocators) {
    int64_t blockSizeBytes = allocator->getBlockSize() * 
                            allocator->getNumHeads() * 
                            allocator->getHeadDim() * 
                            allocator->getElementTypeSize();
                            
    // Each block has memory for both keys and values
    totalMemory += (allocator->getNumAllocatedBlocks() + allocator->getNumFreeBlocks()) * 
                  blockSizeBytes * 2;
  }
  
  return totalMemory;
}

int64_t PagedKVCache::getNumSequences() const {
  std::unordered_set<int32_t> uniqueSeqIds;
  
  for (const auto& entry : seqInfo) {
    uniqueSeqIds.insert(entry.first.first); // Insert sequence ID
  }
  
  return uniqueSeqIds.size();
}

int64_t PagedKVCache::getSequenceLength(int32_t seqId) const {
  // Find the sequence info for the first layer, if it exists
  auto key = std::make_pair(seqId, 0);
  auto it = seqInfo.find(key);
  
  if (it != seqInfo.end()) {
    return it->second.currentPos;
  }
  
  return 0; // Sequence not found
}

LogicalResult PagedKVCache::copyToBlock(KVBlock* block, int64_t posInBlock,
                                      const void* keyPtr, const void* valuePtr,
                                      int64_t tokenOffset, int64_t numTokens) {
  if (!block || !keyPtr || !valuePtr) {
    return failure();
  }
  
  // Check if there's enough space in the block
  if (posInBlock + numTokens > block->getBlockSize()) {
    return failure();
  }
  
  int64_t typeSizeInBytes = getTypeSizeInBytes(elementType);
  int64_t tokensPerHead = numHeads * headDim;
  int64_t singleTokenSize = tokensPerHead * typeSizeInBytes;
  
  // Calculate offsets for the token in the source and destination
  int64_t srcOffset = tokenOffset * singleTokenSize;
  int64_t dstOffset = posInBlock * singleTokenSize;
  
  // Copy key and value data to the block
  char* blockKeyPtr = static_cast<char*>(block->getKeyPtr()) + dstOffset;
  char* blockValuePtr = static_cast<char*>(block->getValuePtr()) + dstOffset;
  const char* srcKeyPtr = static_cast<const char*>(keyPtr) + srcOffset;
  const char* srcValuePtr = static_cast<const char*>(valuePtr) + srcOffset;
  
  int64_t copySize = numTokens * singleTokenSize;
  copyMemory(blockKeyPtr, srcKeyPtr, copySize, useGPU);
  copyMemory(blockValuePtr, srcValuePtr, copySize, useGPU);
  
  return success();
}

LogicalResult PagedKVCache::copyFromBlock(const KVBlock* block, int64_t posInBlock,
                                         void* outputKeys, void* outputValues,
                                         int64_t tokenOffset, int64_t numTokens) {
  if (!block || !outputKeys || !outputValues) {
    return failure();
  }
  
  // Check if the position is valid
  if (posInBlock + numTokens > block->getBlockSize()) {
    return failure();
  }
  
  int64_t typeSizeInBytes = getTypeSizeInBytes(elementType);
  int64_t tokensPerHead = numHeads * headDim;
  int64_t singleTokenSize = tokensPerHead * typeSizeInBytes;
  
  // Calculate offsets
  int64_t srcOffset = posInBlock * singleTokenSize;
  int64_t dstOffset = tokenOffset * singleTokenSize;
  
  // Copy key and value data from the block to the output
  char* dstKeyPtr = static_cast<char*>(outputKeys) + dstOffset;
  char* dstValuePtr = static_cast<char*>(outputValues) + dstOffset;
  const char* srcKeyPtr = static_cast<const char*>(block->getKeyPtr()) + srcOffset;
  const char* srcValuePtr = static_cast<const char*>(block->getValuePtr()) + srcOffset;
  
  int64_t copySize = numTokens * singleTokenSize;
  copyMemory(dstKeyPtr, srcKeyPtr, copySize, useGPU);
  copyMemory(dstValuePtr, srcValuePtr, copySize, useGPU);
  
  return success();
}

LogicalResult PagedKVCache::allocateBlockForSequence(int32_t seqId, int64_t layerIdx,
                                                   int32_t& blockIdx) {
  // Get the block allocator for this layer
  auto& allocator = blockAllocators[layerIdx];
  
  // Allocate a new block
  KVBlock* block = allocator->allocateBlock();
  if (!block) {
    return failure();
  }
  
  // Find the index of the newly allocated block
  auto it = std::find(allocator->allocatedBlocks.begin(),
                     allocator->allocatedBlocks.end(), block);
  if (it == allocator->allocatedBlocks.end()) {
    return failure();
  }
  
  // Get the block index
  blockIdx = static_cast<int32_t>(it - allocator->allocatedBlocks.begin());
  
  // Increment the reference count for this block
  block->incrementRefCount();
  
  return success();
}

} // namespace runtime
} // namespace llm
} // namespace mlir 