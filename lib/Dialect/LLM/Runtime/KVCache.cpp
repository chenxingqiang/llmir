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
#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/GPUMemoryUtils.h"

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
#include <cmath>
#include <sstream> // for std::ostringstream

// Include CUDA headers when GPU support is enabled
#if defined(LLMIR_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(LLMIR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

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
    // First check if this block should use unified memory (for small blocks)
    if (UnifiedMemoryManager::getInstance().shouldUseUnifiedMemory(sizeInBytes)) {
      ptr = GPUMemoryUtils::allocateUnified(sizeInBytes);
      if (ptr) return ptr;
      // Fall back to regular device memory if unified allocation fails
    }
    
    // Try to allocate from the memory pool
    ptr = GPUMemoryPool::getInstance().allocate(sizeInBytes);
    if (!ptr) {
      // Fall back to direct allocation if pool allocation fails
      ptr = GPUMemoryUtils::allocateDevice(sizeInBytes);
    }
    
    if (!ptr) {
      // Last resort: fall back to CPU if GPU allocation fails
      std::cerr << "GPU memory allocation failed, falling back to CPU" << std::endl;
      ptr = std::malloc(sizeInBytes);
      if (ptr) {
        std::memset(ptr, 0, sizeInBytes);
      }
    }
  } else {
    // For host memory, use pinned memory for better GPU transfer performance
    ptr = GPUMemoryUtils::allocateHostPinned(sizeInBytes);
    if (!ptr) {
      // Fall back to regular malloc if pinned allocation fails
      ptr = std::malloc(sizeInBytes);
      if (ptr) {
        std::memset(ptr, 0, sizeInBytes);
      }
    }
  }
  
  return ptr;
}

// Helper function to free memory (on CPU or GPU)
void freeMemory(void* ptr, bool useGPU) {
  if (!ptr) return;
  
  if (useGPU) {
    // Try the memory pool first
    GPUMemoryPool::getInstance().free(ptr);
    
    // Try unified memory manager
    UnifiedMemoryManager::getInstance().free(ptr);
    
    // The above calls are no-ops if the ptr is not managed by them
    // No need to explicitly call GPUMemoryUtils::freeDevice as
    // the pool and unified memory manager will handle this correctly
  } else {
    // For host memory, try pinned memory manager first
    PinnedMemoryManager::getInstance().free(ptr);
    // No need to call std::free as pinned memory manager will handle it correctly
    // if the ptr is not managed by it
  }
}

// Helper function to copy memory (on CPU or GPU)
void copyMemory(void* dst, const void* src, int64_t sizeInBytes, bool useGPU) {
  if (!dst || !src) return;
  
  if (useGPU) {
    // Determine the type of memory for src and dst for correct copy
    
    // For simplicity, assume device-to-device copy
    // In a more complex implementation, we would detect the memory type
    // and use the appropriate copy function
    LogicalResult result = GPUMemoryUtils::copyDeviceToDevice(
        dst, src, sizeInBytes);
    
    if (failed(result)) {
      std::cerr << "GPU memory copy failed" << std::endl;
    }
  } else {
    // CPU memory copy
    std::memcpy(dst, src, sizeInBytes);
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// BlockAllocator Implementation
//===----------------------------------------------------------------------===//

BlockAllocator::BlockAllocator(int64_t blockSize, int64_t numHeads, 
                              int64_t headDim, Type elementType, bool useGPU)
    : blockSize(blockSize), numHeads(numHeads), headDim(headDim),
      elementType(elementType), useGPU(useGPU) {
  
  // Set default eviction policy to LRU if none provided
  evictionPolicy = std::make_unique<LRUEvictionPolicy>();
  
  // Initialize GPU memory optimizations if using GPU
  if (useGPU) {
    // Enable memory pool
    GPUMemoryPool::getInstance().enable(true);
    
    // Set a reasonable size threshold for unified memory
    // Small blocks (<=128KB) will use unified memory for better performance
    UnifiedMemoryManager::getInstance().setThreshold(128 * 1024);
    
    // Preallocate some memory in the pool
    // Calculate memory requirements for a single block
    int64_t blockMemSize = blockSize * numHeads * headDim * getElementTypeSize() * 2;
    // Preallocate enough for 16 blocks by default
    GPUMemoryPool::getInstance().setInitialCapacity(blockMemSize * 16);
  }
  
  // Pre-allocate some blocks to avoid frequent allocation
  preallocateBlocks(8);
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
  // Check if we need to evict blocks before allocation
  if (freeBlocks.empty()) {
    // Try to evict blocks if memory is constrained
    if (collectMetrics) {
      metrics.numEvictionAttempts++;
    }
    evictBlocksIfNeeded(1);
  }
  
  if (freeBlocks.empty()) {
    // No free blocks available, create a new one
    KVBlock* block = createNewBlock();
    if (!block) {
      return nullptr;
    }
    
    // Update access time for LRU tracking
    block->updateAccessTime(getCurrentTimestamp());
    
    allocatedBlocks.push_back(block);
    
    // Update metrics
    if (collectMetrics) {
      metrics.numCreated++;
      updateMetrics();
    }
    
    return block;
  }
  
  // Reuse an existing free block
  KVBlock* block = freeBlocks.back();
  freeBlocks.pop_back();
  
  // Reset the block state
  block->resetUsedSlots();
  block->updateAccessTime(getCurrentTimestamp());
  
  allocatedBlocks.push_back(block);
  
  // Update metrics
  if (collectMetrics) {
    metrics.numReused++;
    updateMetrics();
  }
  
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
    
    // Update metrics
    if (collectMetrics) {
      updateMetrics();
    }
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

void BlockAllocator::preallocateBlocks(int64_t baseNumBlocks, 
                                int64_t avgSeqLen, 
                                int64_t maxNumSequences) {
  // Determine how many blocks to allocate
  int64_t blocksToAllocate = std::max(static_cast<int64_t>(0), baseNumBlocks);
  
  // If we have sequence statistics, use them to make a better estimate
  if (avgSeqLen > 0 && maxNumSequences > 0) {
    // Calculate total tokens we need to support
    int64_t totalTokens = avgSeqLen * maxNumSequences;
    
    // Calculate how many blocks we need for this many tokens
    int64_t blocksNeeded = (totalTokens + blockSize - 1) / blockSize; // Ceiling division
    
    // Add a safety margin of 20%
    blocksNeeded = static_cast<int64_t>(blocksNeeded * 1.2);
    
    // Take the maximum of our base allocation and the calculated need
    blocksToAllocate = std::max(blocksToAllocate, blocksNeeded);
  } else if (avgObservedSeqLen > 0 && totalObservations > 0) {
    // Use observed statistics if we have them and external stats weren't provided
    int64_t totalTokens = static_cast<int64_t>(avgObservedSeqLen * std::max(totalObservations, static_cast<int64_t>(4)));
    int64_t blocksNeeded = (totalTokens + blockSize - 1) / blockSize;
    blocksNeeded = static_cast<int64_t>(blocksNeeded * 1.2); // 20% safety margin
    blocksToAllocate = std::max(blocksToAllocate, blocksNeeded);
  }
  
  // Don't allocate more than a reasonable maximum (avoid excessive memory usage)
  const int64_t maxReasonableBlocks = 1000;
  blocksToAllocate = std::min(blocksToAllocate, maxReasonableBlocks);
  
  // Calculate how many additional blocks we need to allocate
  int64_t currentBlocks = freeBlocks.size() + allocatedBlocks.size();
  int64_t additionalBlocksNeeded = blocksToAllocate - currentBlocks;
  
  // Only allocate if we need more blocks
  if (additionalBlocksNeeded <= 0) {
    return;
  }
  
  // Update metrics
  if (collectMetrics) {
    metrics.numPreallocated += additionalBlocksNeeded;
  }
  
  // Allocate the blocks
  for (int64_t i = 0; i < additionalBlocksNeeded; i++) {
    KVBlock* block = createNewBlock();
    if (block) {
      freeBlocks.push_back(block);
    } else {
      // Failed to allocate memory, stop preallocating
      break;
    }
  }
  
  // Update metrics
  if (collectMetrics) {
    updateMetrics();
  }
}

int64_t BlockAllocator::coalesceBlocks(double fragmentationThreshold, 
                                int64_t maxBlocksToCoalesce,
                                bool preserveOrder) {
  // Find blocks with high fragmentation that we could potentially merge
  std::vector<std::pair<int32_t, double>> fragmentedBlocks;
  
  for (size_t i = 0; i < allocatedBlocks.size(); i++) {
    KVBlock* block = allocatedBlocks[i];
    if (block && block->getRefCount() == 0) { // Only consider unreferenced blocks
      // Update fragmentation metric
      block->updateFragmentation();
      
      if (block->getFragmentation() > fragmentationThreshold) {
        fragmentedBlocks.push_back(std::make_pair(
            static_cast<int32_t>(i), block->getFragmentation()));
      }
    }
  }
  
  // Sort by fragmentation (most fragmented first)
  std::sort(fragmentedBlocks.begin(), fragmentedBlocks.end(),
           [](const auto& a, const auto& b) {
             return a.second > b.second; // Higher fragmentation first
           });
  
  // Limit the number of blocks we'll try to coalesce
  if (fragmentedBlocks.size() > static_cast<size_t>(maxBlocksToCoalesce)) {
    fragmentedBlocks.resize(maxBlocksToCoalesce);
  }
  
  // If we don't have any fragmented blocks, return early
  if (fragmentedBlocks.empty()) {
    return 0;
  }
  
  int64_t blocksCoalesced = 0;
  std::set<int32_t> processedBlocks; // Keep track of blocks we've already processed
  
  // Try to coalesce each fragmented block
  for (const auto& [blockIdx, fragmentation] : fragmentedBlocks) {
    // Skip if we already processed this block
    if (processedBlocks.count(blockIdx) > 0) {
      continue;
    }
    
    KVBlock* sourceBlock = allocatedBlocks[blockIdx];
    
    // Find a potential target block to merge with
    int32_t targetIdx = findCoalescingTarget(sourceBlock);
    if (targetIdx >= 0 && targetIdx != blockIdx && 
        processedBlocks.count(targetIdx) == 0) {
      
      KVBlock* targetBlock = allocatedBlocks[targetIdx];
      
      // Calculate how many tokens we can move
      int64_t sourceUsed = sourceBlock->getUsedSlots();
      int64_t targetUsed = targetBlock->getUsedSlots();
      int64_t targetFree = blockSize - targetUsed;
      int64_t tokensToMove = std::min(sourceUsed, targetFree);
      
      if (tokensToMove > 0) {
        // Copy data from source to target
        if (succeeded(copyBetweenBlocks(sourceBlock, targetBlock, 
                                      0, targetUsed, tokensToMove))) {
          // Update slot counts
          targetBlock->incrementUsedSlots(tokensToMove);
          
          // If we moved all tokens from source, we can free it
          if (tokensToMove == sourceUsed) {
            freeBlock(sourceBlock);
            blocksCoalesced++;
          } else if (preserveOrder) {
            // If we need to preserve order, we're done with this block
            // since we can't safely move the remaining tokens without
            // potentially breaking sequence continuity
          } else {
            // Otherwise, we could try to move the remaining tokens to another block
            // This is left as a potential enhancement
          }
          
          // Mark both blocks as processed
          processedBlocks.insert(blockIdx);
          processedBlocks.insert(targetIdx);
          
          // Update the target's fragmentation metric
          targetBlock->updateFragmentation();
        }
      }
    }
  }
  
  // Update metrics
  if (collectMetrics && blocksCoalesced > 0) {
    metrics.numCoalesces += blocksCoalesced;
    updateMetrics();
  }
  
  return blocksCoalesced;
}

int32_t BlockAllocator::findCoalescingTarget(KVBlock* sourceBlock) const {
  if (!sourceBlock || sourceBlock->getRefCount() > 0) {
    return -1;
  }
  
  int64_t sourceUsed = sourceBlock->getUsedSlots();
  if (sourceUsed == 0) {
    return -1; // Nothing to move
  }
  
  int32_t bestTargetIdx = -1;
  int64_t bestScore = 0;
  
  // Find the best target for coalescing
  for (size_t i = 0; i < allocatedBlocks.size(); i++) {
    KVBlock* targetBlock = allocatedBlocks[i];
    
    // Skip if it's the same block or has references
    if (targetBlock == sourceBlock || targetBlock->getRefCount() > 0) {
      continue;
    }
    
    int64_t targetUsed = targetBlock->getUsedSlots();
    int64_t targetFree = blockSize - targetUsed;
    
    // Skip if target is full
    if (targetFree <= 0) {
      continue;
    }
    
    // Calculate how many tokens we could move
    int64_t tokensToMove = std::min(sourceUsed, targetFree);
    
    // Skip if we can't move any tokens
    if (tokensToMove <= 0) {
      continue;
    }
    
    // Calculate how full the target would be after coalescing
    double finalUtilization = static_cast<double>(targetUsed + tokensToMove) / blockSize;
    
    // Score is a combination of:
    // 1. How many tokens we can move (more is better)
    // 2. How full the target would be after (closer to full is better)
    int64_t score = static_cast<int64_t>(tokensToMove * 100 + finalUtilization * 50);
    
    // Pick the target with the highest score
    if (score > bestScore) {
      bestScore = score;
      bestTargetIdx = static_cast<int32_t>(i);
    }
  }
  
  return bestTargetIdx;
}

LogicalResult BlockAllocator::copyBetweenBlocks(KVBlock* sourceBlock, KVBlock* targetBlock,
                                             int64_t sourceStartPos, int64_t targetStartPos,
                                             int64_t numTokens) {
  if (!sourceBlock || !targetBlock || 
      sourceStartPos + numTokens > sourceBlock->getUsedSlots() ||
      targetStartPos + numTokens > blockSize) {
    return failure();
  }
  
  int64_t typeSizeInBytes = getElementTypeSize();
  int64_t tokensPerHead = numHeads * headDim;
  int64_t singleTokenSize = tokensPerHead * typeSizeInBytes;
  
  // Calculate offsets
  int64_t sourceOffset = sourceStartPos * singleTokenSize;
  int64_t targetOffset = targetStartPos * singleTokenSize;
  int64_t copySize = numTokens * singleTokenSize;
  
  // Copy key data
  copyMemory(
      static_cast<char*>(targetBlock->getKeyPtr()) + targetOffset,
      static_cast<const char*>(sourceBlock->getKeyPtr()) + sourceOffset,
      copySize, useGPU);
  
  // Copy value data
  copyMemory(
      static_cast<char*>(targetBlock->getValuePtr()) + targetOffset,
      static_cast<const char*>(sourceBlock->getValuePtr()) + sourceOffset,
      copySize, useGPU);
  
  return success();
}

void BlockAllocator::evictBlocksIfNeeded(int64_t numBlocksNeeded) {
  // If we have enough free blocks, no need to evict
  if (static_cast<int64_t>(freeBlocks.size()) >= numBlocksNeeded) {
    return;
  }
  
  // Use the eviction policy to select blocks for eviction
  if (evictionPolicy) {
    std::vector<int32_t> blocksToEvict = evictionPolicy->selectBlocksForEviction(
        *this, numBlocksNeeded - freeBlocks.size());
    
    // Free the selected blocks
    for (int32_t blockIdx : blocksToEvict) {
      if (blockIdx >= 0 && blockIdx < static_cast<int32_t>(allocatedBlocks.size())) {
        KVBlock* block = allocatedBlocks[blockIdx];
        
        // Verify the block is evictable
        if (block && block->isEvictable()) {
          freeBlock(block);
          
          if (collectMetrics) {
            metrics.numEvictions++;
          }
        }
      }
    }
  }
  
  // Update metrics
  if (collectMetrics) {
    updateMetrics();
  }
}

BlockMetrics BlockAllocator::getMetrics() const {
  // Refresh metrics before returning
  if (collectMetrics) {
    metrics.totalBlocks = allocatedBlocks.size() + freeBlocks.size();
    metrics.freeBlocks = freeBlocks.size();
    metrics.usedBlocks = allocatedBlocks.size();
    metrics.avgFragmentation = calculateFragmentation();
  }
  
  return metrics;
}

void BlockAllocator::updateMetrics() {
  metrics.totalBlocks = allocatedBlocks.size() + freeBlocks.size();
  metrics.freeBlocks = freeBlocks.size();
  metrics.usedBlocks = allocatedBlocks.size();
  metrics.avgFragmentation = calculateFragmentation();
}

double BlockAllocator::calculateFragmentation() const {
  if (allocatedBlocks.empty()) {
    return 0.0;
  }
  
  double totalFragmentation = 0.0;
  for (auto* block : allocatedBlocks) {
    block->updateFragmentation();
    totalFragmentation += block->getFragmentation();
  }
  
  return totalFragmentation / allocatedBlocks.size();
}

//===----------------------------------------------------------------------===//
// LRUEvictionPolicy Implementation
//===----------------------------------------------------------------------===//

std::vector<int32_t> LRUEvictionPolicy::selectBlocksForEviction(
    const BlockAllocator& allocator, int64_t numBlocksNeeded) const {
  std::vector<int32_t> blocksToEvict;
  
  // If no blocks needed, return empty list
  if (numBlocksNeeded <= 0) {
    return blocksToEvict;
  }
  
  // Create a vector of (blockIdx, lastAccessTime) pairs for sorting
  std::vector<std::pair<int32_t, int64_t>> blockTimes;
  
  for (size_t i = 0; i < allocator.allocatedBlocks.size(); i++) {
    KVBlock* block = allocator.allocatedBlocks[i];
    
    // Only consider evictable blocks (refCount == 0 and isEvictable)
    if (block && block->isEvictable()) {
      blockTimes.push_back(std::make_pair(
          static_cast<int32_t>(i), block->getLastAccessTime()));
    }
  }
  
  // Sort by last access time (older blocks first)
  std::sort(blockTimes.begin(), blockTimes.end(), 
           [](const auto& a, const auto& b) {
             return a.second < b.second;
           });
  
  // Take up to numBlocksNeeded blocks
  for (size_t i = 0; i < std::min(static_cast<size_t>(numBlocksNeeded), blockTimes.size()); i++) {
    blocksToEvict.push_back(blockTimes[i].first);
  }
  
  return blocksToEvict;
}

std::unique_ptr<EvictionPolicy> LRUEvictionPolicy::clone() const {
  return std::make_unique<LRUEvictionPolicy>(*this);
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
  
  // Create one block allocator per layer
  blockAllocators.reserve(numLayers);
  for (int64_t i = 0; i < numLayers; i++) {
    blockAllocators.push_back(
        std::make_unique<BlockAllocator>(blockSize, numHeads, headDim, elementType, useGPU));
  }
  
  // Initialize the layer sequence info vectors
  layerSeqInfo.resize(numLayers);
  
  // Preallocate blocks based on expected usage
  // A heuristic: allocate blocks to handle at least 4 sequences of half maxSeqLen
  int64_t estimatedTokens = 4 * (maxSeqLen / 2);
  int64_t blocksPerLayer = (estimatedTokens + blockSize - 1) / blockSize; // Ceiling division
  
  for (auto& allocator : blockAllocators) {
    allocator->preallocateBlocks(blocksPerLayer);
  }
}

PagedKVCache::~PagedKVCache() {
  // BlockAllocators will be automatically freed by their destructors
  blockAllocators.clear();
  
  // Clear all sequence information
  for (auto& layerInfo : layerSeqInfo) {
    layerInfo.clear();
  }
  contentHashToSeqIds.clear();
}

LogicalResult PagedKVCache::appendKV(const void* keyPtr, const void* valuePtr,
                                   int64_t batchSize, int64_t seqLen, 
                                   const int32_t* seqIds, int32_t* blockIndices) {
  // Check for duplicate sequences before appending
  for (int64_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    int32_t seqId = seqIds[batchIdx];
    
    // Compute a content hash for this batch item
    std::size_t contentHash = generateContentHash(
        static_cast<const char*>(keyPtr) + batchIdx * seqLen * numHeads * headDim * getTypeSizeInBytes(elementType),
        static_cast<const char*>(valuePtr) + batchIdx * seqLen * numHeads * headDim * getTypeSizeInBytes(elementType),
        seqLen);
    
    // Check if we have identical content already cached
    int32_t sourceSeqId = findIdenticalSequence(
        static_cast<const char*>(keyPtr) + batchIdx * seqLen * numHeads * headDim * getTypeSizeInBytes(elementType),
        static_cast<const char*>(valuePtr) + batchIdx * seqLen * numHeads * headDim * getTypeSizeInBytes(elementType),
        seqLen);
    
    if (sourceSeqId >= 0 && sourceSeqId != seqId) {
      // We found a duplicate sequence, share blocks instead of creating new ones
      if (succeeded(shareSequenceBlocks(sourceSeqId, seqId, seqLen))) {
        // Update the content hash mapping
        updateContentHashMapping(seqId, contentHash);
        
        // Set block indices for the caller
        for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
          auto& seqInfo = layerSeqInfo[layerIdx][seqId];
          blockIndices[batchIdx * numLayers + layerIdx] = seqInfo.lastBlockIdx;
        }
        
        // Skip processing this sequence since we're sharing blocks
        continue;
      }
    }
    
    // Regular processing for non-duplicate sequences
    for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      auto& layerInfo = layerSeqInfo[layerIdx];
      auto& allocator = blockAllocators[layerIdx];
      
      // Create entry for this sequence if it doesn't exist
      if (layerInfo.find(seqId) == layerInfo.end()) {
        layerInfo[seqId] = SequenceInfo();
        layerInfo[seqId].contentHash = contentHash;
      }
      
      auto& info = layerInfo[seqId];
      
      // Check if we need a new block or can use the existing one
      if (info.lastBlockIdx < 0 || 
          allocator->getBlock(info.lastBlockIdx)->getUsedSlots() + seqLen > blockSize) {
        
        // Try to coalesce blocks if memory is constrained
        if (allocator->getNumFreeBlocks() == 0) {
          allocator->coalesceBlocks(0.5, 1, false);
          allocator->evictBlocksIfNeeded(1);
        }
        
        // Need a new block
        int32_t newBlockIdx;
        if (failed(allocateBlockForSequence(seqId, layerIdx, newBlockIdx))) {
          return failure();
        }
        
        // Update block info
        info.lastBlockIdx = newBlockIdx;
        info.posInLastBlock = 0;
        
        // Add to block positions
        info.blockPositions.push_back(std::make_pair(newBlockIdx, 0));
      }
      
      // Get the block
      KVBlock* block = allocator->getBlock(info.lastBlockIdx);
      if (!block) {
        return failure();
      }
      
      // Update the block's access time for LRU tracking
      block->updateAccessTime(allocator->getCurrentTimestamp());
      
      // Copy data to the block
      if (failed(copyToBlock(block, info.posInLastBlock,
                            keyPtr, valuePtr,
                            batchIdx * seqLen, seqLen))) {
        return failure();
      }
      
      // Update pointers
      info.posInLastBlock += seqLen;
      info.currentPos += seqLen;
      
      // Increment the block's used slots
      block->incrementUsedSlots(seqLen);
      
      // Update fragmentation metric
      block->updateFragmentation();
      
      // Store block index for the caller
      blockIndices[batchIdx * numLayers + layerIdx] = info.lastBlockIdx;
    }
    
    // Update the content hash mapping for future sharing
    updateContentHashMapping(seqId, contentHash);
  }
  
  // Try auto-coalescing if enabled and memory is constrained
  if (autoCoalescingEnabled && allocator->getNumFreeBlocks() == 0) {
    // Check if average fragmentation exceeds threshold
    auto metrics = allocator->getMetrics();
    if (metrics.avgFragmentation > autoCoalescingThreshold) {
      // Run coalescing to free up some blocks
      allocator->coalesceBlocks(autoCoalescingThreshold, 5, true);
    }
  }
  
  // Update sequence length statistics to help with future preallocation
  allocator->updateSeqLengthStats(info.currentPos + seqLen);
  
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
        
        // Update the block's access time for LRU tracking
        block->updateAccessTime(blockAllocators[layer]->getCurrentTimestamp());
        
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
  bool hadSequence = false;
  
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    auto& layerInfo = layerSeqInfo[layerIdx];
    auto& allocator = blockAllocators[layerIdx];
    
    auto it = layerInfo.find(seqId);
    if (it != layerInfo.end()) {
      hadSequence = true;
      
      // If this sequence shares blocks with another, don't free them,
      // just decrement reference counts
      if (it->second.sharesBlocks) {
        // Remove from hash mapping
        std::size_t contentHash = it->second.contentHash;
        auto& seqIds = contentHashToSeqIds[contentHash];
        seqIds.erase(std::remove(seqIds.begin(), seqIds.end(), seqId), seqIds.end());
        
        // Decrement reference counts for shared blocks
        for (const auto& [blockIdx, posInBlock] : it->second.blockPositions) {
          KVBlock* block = allocator->getBlock(blockIdx);
          if (block) {
            block->decrementRefCount();
          }
        }
      } else {
        // Remove from hash mapping
        std::size_t contentHash = it->second.contentHash;
        auto& seqIds = contentHashToSeqIds[contentHash];
        seqIds.erase(std::remove(seqIds.begin(), seqIds.end(), seqId), seqIds.end());
        
        // Free blocks only if they're not shared with other sequences
        for (const auto& [blockIdx, posInBlock] : it->second.blockPositions) {
          KVBlock* block = allocator->getBlock(blockIdx);
          if (block && block->getRefCount() <= 1) {
            allocator->freeBlock(block);
          } else if (block) {
            block->decrementRefCount();
          }
        }
      }
      
      // Remove sequence info
      layerInfo.erase(it);
    }
  }
  
  return hadSequence ? success() : failure();
}

void PagedKVCache::reset() {
  // Reset the content hash to sequence ID mapping
  contentHashToSeqIds.clear();
  
  // Clear all sequence information
  for (auto& layerInfo : layerSeqInfo) {
    layerInfo.clear();
  }
  
  // Reset all block allocators
  for (auto& allocator : blockAllocators) {
    // Move all allocated blocks to free blocks
    for (auto* block : allocator->allocatedBlocks) {
      // Reset block state
      block->resetUsedSlots();
      block->setEvictable(true);
      
      // Add to free blocks
      allocator->freeBlocks.push_back(block);
    }
    
    // Clear allocated blocks
    allocator->allocatedBlocks.clear();
    
    // Update metrics
    if (auto metrics = allocator->getMetrics(); metrics.numEvictions > 0 || metrics.numCoalesces > 0) {
      allocator->enableMetrics(true); // Ensure metrics are updated
    }
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
  
  // Collect unique sequence IDs from all layers
  for (const auto& layerInfo : layerSeqInfo) {
    for (const auto& entry : layerInfo) {
      uniqueSeqIds.insert(entry.first); // Insert sequence ID
    }
  }
  
  return uniqueSeqIds.size();
}

int64_t PagedKVCache::getSequenceLength(int32_t seqId) const {
  // Find the sequence info for the first layer, if it exists
  if (layerSeqInfo.empty()) {
    return 0;
  }
  
  const auto& firstLayerInfo = layerSeqInfo[0];
  auto it = firstLayerInfo.find(seqId);
  
  if (it != firstLayerInfo.end()) {
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

// Implementation of cross-sequence cache sharing methods

bool PagedKVCache::hasIdenticalPrefix(int32_t seqId1, int32_t seqId2, int64_t prefixLen) const {
  // Check if both sequences exist
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const auto& layerInfo = layerSeqInfo[layerIdx];
    
    auto it1 = layerInfo.find(seqId1);
    auto it2 = layerInfo.find(seqId2);
    
    if (it1 == layerInfo.end() || it2 == layerInfo.end()) {
      return false;
    }
    
    const auto& info1 = it1->second;
    const auto& info2 = it2->second;
    
    // Check if sequences are long enough
    if (info1.currentPos < prefixLen || info2.currentPos < prefixLen) {
      return false;
    }
  }
  
  // Compare sequence contents
  return compareSequenceContent(seqId1, seqId2, prefixLen);
}

LogicalResult PagedKVCache::shareSequenceBlocks(int32_t sourceSeqId, int32_t targetSeqId, int64_t numTokens) {
  // Check if source sequence exists and has enough tokens
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    auto& layerInfo = layerSeqInfo[layerIdx];
    
    auto it = layerInfo.find(sourceSeqId);
    if (it == layerInfo.end() || it->second.currentPos < numTokens) {
      return failure();
    }
  }
  
  // Share blocks from source to target
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    auto& layerInfo = layerSeqInfo[layerIdx];
    auto& allocator = blockAllocators[layerIdx];
    
    // Get source sequence info
    const auto& sourceInfo = layerInfo[sourceSeqId];
    
    // Create or get target sequence info
    if (layerInfo.find(targetSeqId) == layerInfo.end()) {
      layerInfo[targetSeqId] = SequenceInfo();
    }
    
    auto& targetInfo = layerInfo[targetSeqId];
    
    // Mark as sharing blocks
    targetInfo.sharesBlocks = true;
    targetInfo.sourceSeqId = sourceSeqId;
    
    // Calculate how many blocks we need to share
    int64_t tokensRemaining = numTokens;
    int64_t blockPos = 0;
    
    while (tokensRemaining > 0 && blockPos < sourceInfo.blockPositions.size()) {
      auto [blockIdx, posInBlock] = sourceInfo.blockPositions[blockPos];
      
      // Get the block
      KVBlock* block = allocator->getBlock(blockIdx);
      if (!block) {
        return failure();
      }
      
      // Calculate tokens in this block
      int64_t tokensInBlock = std::min(blockSize - posInBlock, tokensRemaining);
      
      // Increment reference count for the shared block
      block->incrementRefCount();
      
      // Add the block position mapping
      targetInfo.blockPositions.push_back(std::make_pair(blockIdx, posInBlock));
      targetInfo.sharedBlockMapping[blockIdx] = posInBlock;
      
      // Update tokens remaining
      tokensRemaining -= tokensInBlock;
      blockPos++;
    }
    
    // Update target sequence info
    targetInfo.lastBlockIdx = sourceInfo.blockPositions[blockPos - 1].first;
    targetInfo.posInLastBlock = sourceInfo.blockPositions[blockPos - 1].second + 
                               (blockSize - tokensRemaining);
    targetInfo.currentPos = numTokens;
  }
  
  return success();
}

std::size_t PagedKVCache::getSequenceContentHash(int32_t seqId) const {
  // Return the content hash for the first layer (should be the same for all layers)
  if (numLayers == 0) {
    return 0;
  }
  
  const auto& layerInfo = layerSeqInfo[0];
  auto it = layerInfo.find(seqId);
  
  if (it != layerInfo.end()) {
    return it->second.contentHash;
  }
  
  return 0;
}

std::size_t PagedKVCache::generateContentHash(const void* keyPtr, const void* valuePtr, 
                                            int64_t seqLen) const {
  // Generate a hash based on the first few tokens (or all if sequence is short)
  int64_t tokensToHash = std::min(seqLen, static_cast<int64_t>(16));
  
  // Get element size
  int64_t elementSize = getTypeSizeInBytes(elementType);
  
  // Calculate total bytes to hash
  int64_t bytesToHash = tokensToHash * numHeads * headDim * elementSize;
  
  // Create a simple hash
  std::size_t hash = 0;
  
  // Hash key data
  const char* keyData = static_cast<const char*>(keyPtr);
  for (int64_t i = 0; i < bytesToHash; i++) {
    hash ^= std::hash<char>{}(keyData[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  
  // Hash value data
  const char* valueData = static_cast<const char*>(valuePtr);
  for (int64_t i = 0; i < bytesToHash; i++) {
    hash ^= std::hash<char>{}(valueData[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  
  return hash;
}

int32_t PagedKVCache::findIdenticalSequence(const void* keyPtr, const void* valuePtr, 
                                          int64_t seqLen) const {
  // Generate content hash
  std::size_t contentHash = generateContentHash(keyPtr, valuePtr, seqLen);
  
  // Look up sequences with the same hash
  auto it = contentHashToSeqIds.find(contentHash);
  if (it == contentHashToSeqIds.end()) {
    return -1;
  }
  
  // Check each sequence with the same hash for actual content equality
  for (int32_t candidateSeqId : it->second) {
    bool identical = true;
    
    // Verify sequence length
    for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const auto& layerInfo = layerSeqInfo[layerIdx];
      auto seqIt = layerInfo.find(candidateSeqId);
      
      if (seqIt == layerInfo.end() || seqIt->second.currentPos != seqLen) {
        identical = false;
        break;
      }
    }
    
    if (!identical) {
      continue;
    }
    
    // For detailed comparison, we'd need to extract the actual data
    // This would be more complex, so for simplicity we'll trust the hash
    // In a real implementation, you might want to do a detailed comparison
    
    return candidateSeqId;
  }
  
  return -1;
}

bool PagedKVCache::compareSequenceContent(int32_t seqId1, int32_t seqId2, int64_t length) const {
  // Compare content hash as a quick check
  std::size_t hash1 = getSequenceContentHash(seqId1);
  std::size_t hash2 = getSequenceContentHash(seqId2);
  
  if (hash1 != hash2) {
    return false;
  }
  
  // For simplicity, we'll just compare the content hash
  // In a more detailed implementation, we would need to extract and compare
  // the actual KV data from the cache
  
  return true;
}

void PagedKVCache::updateContentHashMapping(int32_t seqId, std::size_t contentHash) {
  // Update the content hash to sequence ID mapping
  contentHashToSeqIds[contentHash].push_back(seqId);
  
  // Update the hash in all layers
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    auto& layerInfo = layerSeqInfo[layerIdx];
    if (layerInfo.find(seqId) != layerInfo.end()) {
      layerInfo[seqId].contentHash = contentHash;
    }
  }
}

// Add new methods to access and configure the block allocation optimization features

void PagedKVCache::configureBlockAllocators(int64_t initialBlocksPerLayer, bool enableMetrics) {
  for (auto& allocator : blockAllocators) {
    allocator->enableMetrics(enableMetrics);
    
    // If requested blocks are more than current free blocks, preallocate more
    int64_t additionalBlocks = initialBlocksPerLayer - allocator->getNumFreeBlocks();
    if (additionalBlocks > 0) {
      allocator->preallocateBlocks(additionalBlocks);
    }
  }
}

void PagedKVCache::setEvictionPolicy(std::unique_ptr<EvictionPolicy> policy) {
  for (auto& allocator : blockAllocators) {
    if (policy) {
      // Clone the policy for each allocator
      // Use the clone method directly instead of dynamic_cast
      void* clonedPtr = policy->clone();
      if (clonedPtr) {
        allocator->setEvictionPolicy(
            std::unique_ptr<EvictionPolicy>(static_cast<EvictionPolicy*>(clonedPtr)));
      } else {
        // Fallback to LRU if clone fails
        allocator->setEvictionPolicy(std::make_unique<LRUEvictionPolicy>());
      }
    } else {
      // Default to LRU if no policy provided
      allocator->setEvictionPolicy(std::make_unique<LRUEvictionPolicy>());
    }
  }
}

void PagedKVCache::runBlockCoalescing() {
  for (auto& allocator : blockAllocators) {
    allocator->coalesceBlocks(0.5, 1, false);
  }
}

std::vector<BlockMetrics> PagedKVCache::getAllBlockMetrics() const {
  std::vector<BlockMetrics> allMetrics;
  allMetrics.reserve(blockAllocators.size());
  
  for (const auto& allocator : blockAllocators) {
    allMetrics.push_back(allocator->getMetrics());
  }
  
  return allMetrics;
}

// We need to add the clone method to EvictionPolicy interface
void* EvictionPolicy::clone() const {
  return nullptr; // Base implementation returns nullptr
}

// LRUEvictionPolicy implementation of clone
std::unique_ptr<EvictionPolicy> LRUEvictionPolicy::clone() const {
  return std::make_unique<LRUEvictionPolicy>(*this);
}

std::vector<int32_t> FragmentationAwareLRUPolicy::selectBlocksForEviction(
    const BlockAllocator& allocator, int64_t numBlocksNeeded) const {
  std::vector<int32_t> blocksToEvict;
  
  // If no blocks needed, return empty list
  if (numBlocksNeeded <= 0) {
    return blocksToEvict;
  }
  
  // Create a vector of (blockIdx, score) pairs for sorting
  // Score is a weighted combination of last access time and fragmentation
  std::vector<std::pair<int32_t, double>> blockScores;
  
  for (size_t i = 0; i < allocator.allocatedBlocks.size(); i++) {
    KVBlock* block = allocator.allocatedBlocks[i];
    
    // Only consider evictable blocks (refCount == 0 and isEvictable)
    if (block && block->isEvictable()) {
      // Normalize last access time to 0.0-1.0 scale
      // Since higher timestamp means more recent, we invert it
      double ageScore = 0.0;
      // Since we don't have the max timestamp, we use a simpler approach:
      // For LRU part, we'll use the raw access time which will give
      // older blocks (lower timestamps) preference
      int64_t lastAccessTime = block->getLastAccessTime();
      
      // Fragmentation score is already in 0.0-1.0 range
      double fragScore = block->getFragmentation();
      
      // Combine scores based on weight
      // For LRU part, we'll use negative lastAccessTime so that 
      // older blocks have higher priority
      double combinedScore = (1.0 - fragmentationWeight) * (-lastAccessTime) + 
                            fragmentationWeight * fragScore * 1000000;
                            
      blockScores.push_back(std::make_pair(static_cast<int32_t>(i), combinedScore));
    }
  }
  
  // Sort by combined score (higher score first)
  std::sort(blockScores.begin(), blockScores.end(), 
           [](const auto& a, const auto& b) {
             return a.second > b.second;
           });
  
  // Take up to numBlocksNeeded blocks
  for (size_t i = 0; i < std::min(static_cast<size_t>(numBlocksNeeded), blockScores.size()); i++) {
    blocksToEvict.push_back(blockScores[i].first);
  }
  
  return blocksToEvict;
}

void BlockAllocator::updateSeqLengthStats(int64_t seqLen) {
  if (seqLen <= 0) {
    return;
  }
  
  // First observation
  if (totalObservations == 0) {
    minObservedSeqLen = seqLen;
    maxObservedSeqLen = seqLen;
    avgObservedSeqLen = seqLen;
    totalObservations = 1;
    return;
  }
  
  // Update min and max
  minObservedSeqLen = std::min(minObservedSeqLen, seqLen);
  maxObservedSeqLen = std::max(maxObservedSeqLen, seqLen);
  
  // Update running average
  // Use a weighted average to give more weight to recent observations
  // but still keep historical data somewhat relevant
  const double alpha = 0.25; // Weight for new observation
  avgObservedSeqLen = (1 - alpha) * avgObservedSeqLen + alpha * seqLen;
  
  // Increment observation count
  totalObservations++;
}

int64_t PagedKVCache::runAdvancedBlockCoalescing(double fragmentationThreshold,
                                         int64_t maxBlocksToCoalesce,
                                         bool preserveOrder) {
  int64_t totalCoalesced = 0;
  
  for (auto& allocator : blockAllocators) {
    totalCoalesced += allocator->coalesceBlocks(fragmentationThreshold, 
                                              maxBlocksToCoalesce, 
                                              preserveOrder);
  }
  
  return totalCoalesced;
}

void PagedKVCache::enableAutoCoalescing(double fragmentationThreshold) {
  autoCoalescingEnabled = true;
  autoCoalescingThreshold = fragmentationThreshold;
}

void PagedKVCache::disableAutoCoalescing() {
  autoCoalescingEnabled = false;
}

void PagedKVCache::configureBlockAllocatorsAdvanced(int64_t avgSeqLen, int64_t maxConcurrentSeqs,
                                                 bool enableMetrics, int preallocationStrategy) {
  // Determine base number of blocks to preallocate per layer based on strategy
  int64_t baseBlocksPerLayer;
  switch (preallocationStrategy) {
    case 0: // Minimal
      baseBlocksPerLayer = 4;
      break;
    case 1: // Balanced
      baseBlocksPerLayer = 8;
      break;
    case 2: // Aggressive
      baseBlocksPerLayer = 16;
      break;
    default:
      baseBlocksPerLayer = 8; // Default to balanced
      break;
  }
  
  // Configure each allocator
  for (auto& allocator : blockAllocators) {
    // Enable/disable metrics
    allocator->enableMetrics(enableMetrics);
    
    // Set eviction policy based on strategy
    if (preallocationStrategy == 0) {
      // Minimal strategy prioritizes memory usage over performance
      // Use fragmentation-aware policy with high weight on fragmentation
      allocator->setEvictionPolicy(std::make_unique<FragmentationAwareLRUPolicy>(0.7));
    } else {
      // Other strategies can use regular LRU
      allocator->setEvictionPolicy(std::make_unique<LRUEvictionPolicy>());
    }
    
    // Preallocate blocks with advanced parameters
    allocator->preallocateBlocks(baseBlocksPerLayer, avgSeqLen, maxConcurrentSeqs);
  }
}

int64_t PagedKVCache::calculateOptimalBlockSize(int64_t minBlockSize, int64_t maxBlockSize) const {
  // Get sequence length statistics
  int64_t minSeqLen, maxSeqLen;
  double avgSeqLen;
  getSequenceLengthStats(minSeqLen, maxSeqLen, avgSeqLen);
  
  // If we don't have enough data, use default block size
  if (minSeqLen <= 0 || maxSeqLen <= 0) {
    return blockSize;
  }
  
  // Calculate a block size that balances:
  // 1. Not too small (would require too many blocks for long sequences)
  // 2. Not too large (would waste memory for short sequences)
  // A good heuristic is to aim for blocks that can hold about 20-25% of the average sequence
  int64_t optimalSize = static_cast<int64_t>(avgSeqLen * 0.25);
  
  // Make sure it's a power of 2 for memory alignment
  int64_t powerOf2 = 1;
  while (powerOf2 < optimalSize) {
    powerOf2 *= 2;
  }
  
  // Clamp to min/max range
  return std::max(minBlockSize, std::min(maxBlockSize, powerOf2));
}

void PagedKVCache::getSequenceLengthStats(int64_t& minSeqLen, int64_t& maxSeqLen, double& avgSeqLen) const {
  minSeqLen = 0;
  maxSeqLen = 0;
  avgSeqLen = 0.0;
  
  // Count all sequences
  int64_t totalSequences = 0;
  int64_t totalTokens = 0;
  
  // We can use any layer's sequence info since they all have the same sequences
  if (!layerSeqInfo.empty()) {
    const auto& seqInfo = layerSeqInfo[0];
    
    if (seqInfo.empty()) {
      return;
    }
    
    minSeqLen = std::numeric_limits<int64_t>::max();
    
    for (const auto& [seqId, info] : seqInfo) {
      int64_t seqLen = info.currentPos;
      
      if (seqLen > 0) {
        minSeqLen = std::min(minSeqLen, seqLen);
        maxSeqLen = std::max(maxSeqLen, seqLen);
        totalTokens += seqLen;
        totalSequences++;
      }
    }
    
    // Calculate average
    if (totalSequences > 0) {
      avgSeqLen = static_cast<double>(totalTokens) / totalSequences;
    }
    
    // If we didn't find any non-zero lengths, reset minSeqLen
    if (minSeqLen == std::numeric_limits<int64_t>::max()) {
      minSeqLen = 0;
    }
  }
}

void PagedKVCache::configureAttentionOpt(const AttentionConfig& config) {
  // Create a copy of the config and ensure it has the correct parameters
  AttentionConfig attConfig = config;
  attConfig.numHeads = numHeads;
  attConfig.headDim = headDim;
  attConfig.setDefaultsFromHeadDim();
  
  // Configure Flash Attention parameters for optimal performance with paged KV cache
  if (attConfig.useFlashAttention) {
    // Adjust block sizes based on the paged KV cache block size for optimal tiling
    // The query block size (M) should be related to a multiple of query size in tokens
    // The key block size (N) should be related to the KV cache block size
    
    // A good default block size for M is 64
    if (attConfig.blockSizeM <= 0) {
      attConfig.blockSizeM = 64;
    }
    
    // For N, we want a value that's a multiple of the cache block size
    // But capped to a reasonable maximum for memory efficiency
    if (attConfig.blockSizeN <= 0) {
      // Use the block size of the KV cache as a basis, but ensure at least 32
      // and no more than 128 for good performance
      attConfig.blockSizeN = std::min(128, std::max(32, static_cast<int>(blockSize)));
    }
    
    // Enable prefetching by default for better memory access patterns
    attConfig.usePrefetching = true;
  }
  
  // Create an appropriate attention implementation
  attentionImpl = createAttentionImpl(attConfig, elementType, useGPU);
}

LogicalResult PagedKVCache::computeAttention(
    void* output,
    const void* queries,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int64_t batchSize,
    int64_t seqLen) {
  
  // Check if attention implementation is configured
  if (!attentionImpl) {
    // Create a default attention configuration
    AttentionConfig defaultConfig;
    defaultConfig.numHeads = numHeads;
    defaultConfig.headDim = headDim;
    defaultConfig.maskType = AttentionMaskType::CAUSAL;
    defaultConfig.optLevel = AttentionOptLevel::BASIC;
    defaultConfig.setDefaultsFromHeadDim();
    
    // Create attention implementation
    attentionImpl = createAttentionImpl(defaultConfig, elementType, useGPU);
  }
  
  // Compute attention using the configured implementation
  attentionImpl->computePaged(
      output, queries, this, blockIndices, seqLens, batchSize, seqLen);
  
  return success();
}

// Efficiently gather keys and values from KV cache for attention computation
LogicalResult PagedKVCache::gatherKVForAttention(
    void* outputKeys,
    void* outputValues,
    int32_t seqId,
    int64_t startPos,
    int64_t numTokens) {
  
  // Check if sequence exists
  if (layerSeqInfo.empty() || layerSeqInfo[0].count(seqId) == 0) {
    return failure();
  }
  
  // Check if start position and length are valid
  int64_t seqLen = layerSeqInfo[0].at(seqId).currentPos;
  if (startPos < 0 || startPos + numTokens > seqLen) {
    return failure();
  }
  
  // Gather keys and values from each layer and concatenate them
  // This is more efficient than making separate lookupKV calls
  int64_t elementTypeSize = getElementTypeSize();
  int64_t tokensPerKey = numHeads * headDim;
  int64_t tokenSizeInBytes = tokensPerKey * elementTypeSize;
  
  for (int64_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const auto& seqInfo = layerSeqInfo[layerIdx].at(seqId);
    int64_t currentTokenOffset = 0;
    
    // Find the starting block position for this sequence
    int64_t blockPos = 0;
    while (blockPos < seqInfo.blockPositions.size() && currentTokenOffset + blockSize <= startPos) {
      // Skip whole blocks that come before our start position
      currentTokenOffset += blockSize;
      blockPos++;
    }
    
    // Now we're at the first block we need to read from
    int64_t remainingTokens = numTokens;
    int64_t outputOffset = 0;
    
    // Calculate layer offsets in the output
    char* layerOutputKeys = static_cast<char*>(outputKeys) + layerIdx * numTokens * tokenSizeInBytes;
    char* layerOutputValues = static_cast<char*>(outputValues) + layerIdx * numTokens * tokenSizeInBytes;
    
    // Keep copying until we've gathered all requested tokens
    while (remainingTokens > 0 && blockPos < seqInfo.blockPositions.size()) {
      auto [blockIdx, posInBlock] = seqInfo.blockPositions[blockPos];
      
      // If this sequence shares blocks, map to the actual block index
      if (seqInfo.sharesBlocks && seqInfo.sharedBlockMapping.count(blockIdx) > 0) {
        blockIdx = seqInfo.sharedBlockMapping.at(blockIdx);
      }
      
      // Get block from allocator
      KVBlock* block = blockAllocators[layerIdx]->getBlock(blockIdx);
      if (!block) {
        return failure();
      }
      
      // Calculate position within block to start reading
      int64_t inBlockOffset = startPos - currentTokenOffset + posInBlock;
      
      // Calculate how many tokens to read from this block
      int64_t tokensInBlock = std::min(remainingTokens, blockSize - (inBlockOffset - posInBlock));
      
      // Copy keys and values from this block
      if (failed(copyFromBlock(block, inBlockOffset, 
                              layerOutputKeys + outputOffset * tokenSizeInBytes,
                              layerOutputValues + outputOffset * tokenSizeInBytes,
                              0, tokensInBlock))) {
        return failure();
      }
      
      // Update counters
      remainingTokens -= tokensInBlock;
      outputOffset += tokensInBlock;
      currentTokenOffset += blockSize - posInBlock;
      blockPos++;
    }
    
    // Check if we gathered all tokens
    if (remainingTokens > 0) {
      return failure();
    }
  }
  
  return success();
}

// Add implementation for the GPU memory configuration method
void PagedKVCache::configureGPUMemoryOptions(bool enablePool, 
                                          size_t unifiedMemThreshold,
                                          size_t initialPoolSize) {
  if (!useGPU) {
    return;  // No-op for CPU-only mode
  }

  // Configure memory pool
  GPUMemoryPool::getInstance().enable(enablePool);
  
  // Set unified memory threshold
  UnifiedMemoryManager::getInstance().setThreshold(unifiedMemThreshold);
  
  // Set initial pool size if enabled
  if (enablePool && initialPoolSize > 0) {
    GPUMemoryPool::getInstance().setInitialCapacity(initialPoolSize);
  }
}

// Add implementation for the GPU memory stats method
std::string PagedKVCache::getGPUMemoryStats() const {
  if (!useGPU) {
    return "GPU not enabled";
  }
  
  std::ostringstream oss;
  
  // Get pool stats
  GPUMemoryPool::PoolStats poolStats = GPUMemoryPool::getInstance().getStats();
  
  oss << "GPU Memory Pool Stats:\n"
      << "  Total memory: " << (poolStats.totalMemory / (1024.0 * 1024.0)) << " MB\n"
      << "  Used memory: " << (poolStats.usedMemory / (1024.0 * 1024.0)) << " MB\n"
      << "  Free memory: " << (poolStats.freeMemory / (1024.0 * 1024.0)) << " MB\n"
      << "  Block count: " << poolStats.blockCount << "\n"
      << "  Hit rate: " << (poolStats.hitCount + poolStats.missCount > 0 ? 
                          100.0 * poolStats.hitCount / (poolStats.hitCount + poolStats.missCount) : 0.0)
      << "%\n";
  
  // Try to get device properties
  GPUMemoryUtils::GPUDeviceProperties props;
  if (succeeded(GPUMemoryUtils::getDeviceProperties(props))) {
    oss << "GPU Device Stats:\n"
        << "  Device: " << props.name << " (ID: " << props.deviceId << ")\n"
        << "  Total device memory: " << (props.totalMemory / (1024.0 * 1024.0)) << " MB\n"
        << "  Free device memory: " << (props.freeMemory / (1024.0 * 1024.0)) << " MB\n"
        << "  Compute capability: " << props.computeCapabilityMajor 
        << "." << props.computeCapabilityMinor << "\n";
  }
  
  return oss.str();
}

// Add implementation for the shrink GPU memory method
void PagedKVCache::shrinkGPUMemory(float keepRatio) {
  if (!useGPU) {
    return;  // No-op for CPU-only mode
  }
  
  // Shrink the memory pool
  GPUMemoryPool::getInstance().shrink(keepRatio);
}

} // namespace runtime
} // namespace llm
} // namespace mlir 