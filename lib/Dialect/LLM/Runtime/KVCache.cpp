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

namespace mlir {
namespace llm {
namespace runtime {

namespace {
// Helper function to get the size in bytes for a type
// This is a simplified version for testing without actual MLIR Type definitions
int64_t getTypeSizeInBytes(Type type) {
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

} // anonymous namespace

//===----------------------------------------------------------------------===//
// BlockAllocator Implementation
//===----------------------------------------------------------------------===//

BlockAllocator::BlockAllocator(int64_t blockSize, int64_t numHeads, 
                              int64_t headDim, Type elementType, bool useGPU)
    : blockSize(blockSize), numHeads(numHeads), headDim(headDim),
      elementType(elementType), useGPU(useGPU) {
  // Pre-allocate some blocks to avoid frequent allocation
  for (int i = 0; i < 8; i++) {
    freeBlocks.push_back(createNewBlock());
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
    allocatedBlocks.push_back(block);
    return block;
  }
  
  // Reuse an existing free block
  KVBlock* block = freeBlocks.back();
  freeBlocks.pop_back();
  allocatedBlocks.push_back(block);
  return block;
}

void BlockAllocator::freeBlock(KVBlock* block) {
  // Find the block in the allocated blocks
  auto it = std::find(allocatedBlocks.begin(), allocatedBlocks.end(), block);
  if (it != allocatedBlocks.end()) {
    allocatedBlocks.erase(it);
    freeBlocks.push_back(block);
  }
}

KVBlock* BlockAllocator::createNewBlock() {
  int64_t typeSizeInBytes = getTypeSizeInBytes(elementType);
  
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
  seqToBlocks.clear();
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
      // Get or create the block indices for this sequence and layer
      auto key = std::make_pair(seqId, layer);
      auto& blocks = seqToBlocks[key];
      
      // For each token in the sequence
      for (int64_t t = 0; t < seqLen; t++) {
        // Check if we need a new block
        if (blocks.empty() || (blocks.size() * blockSize) % maxSeqLen == 0) {
          // Allocate a new block
          KVBlock* block = blockAllocators[layer]->allocateBlock();
          if (!block) {
            return failure();
          }
          
          // Add the block index
          blocks.push_back(blockAllocators[layer]->getNumAllocatedBlocks() - 1);
        }
        
        // Get the current block index
        int32_t blockIdx = blocks.back();
        
        // Calculate position within the block
        int64_t posInBlock = (blocks.size() * blockSize - 1) % blockSize;
        
        // Get the KV block
        KVBlock* block = blockAllocators[layer]->allocatedBlocks[blockIdx];
        
        // Calculate offsets for the token in the source and destination
        int64_t srcOffset = (b * seqLen + t) * singleTokenSize;
        int64_t dstOffset = posInBlock * singleTokenSize;
        
        // Copy key and value data to the block
        char* blockKeyPtr = static_cast<char*>(block->getKeyPtr()) + dstOffset;
        char* blockValuePtr = static_cast<char*>(block->getValuePtr()) + dstOffset;
        const char* srcKeyPtr = static_cast<const char*>(keyPtr) + srcOffset;
        const char* srcValuePtr = static_cast<const char*>(valuePtr) + srcOffset;
        
        copyMemory(blockKeyPtr, srcKeyPtr, singleTokenSize, useGPU);
        copyMemory(blockValuePtr, srcValuePtr, singleTokenSize, useGPU);
        
        // Update block indices output
        blockIndices[b * seqLen + t] = blockIdx;
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
  
  // For each sequence in the batch
  for (int64_t b = 0; b < batchSize; b++) {
    int32_t seqLen = seqLens[b];
    
    // For each layer
    for (int64_t layer = 0; layer < numLayers; layer++) {
      // For each token position
      for (int64_t t = 0; t < seqLen; t++) {
        // Get the block index for this token
        int32_t blockIdx = blockIndices[b * maxSeqLen + t];
        
        // Check if the block index is valid
        if (blockIdx < 0 || blockIdx >= (int32_t)blockAllocators[layer]->getNumAllocatedBlocks()) {
          return failure();
        }
        
        // Get the block
        KVBlock* block = blockAllocators[layer]->allocatedBlocks[blockIdx];
        
        // Calculate position within the block
        int64_t posInBlock = t % blockSize;
        
        // Calculate offsets
        int64_t srcOffset = posInBlock * singleTokenSize;
        int64_t dstOffset = (b * seqLen + t) * singleTokenSize;
        
        // Copy key and value data from the block to the output
        char* dstKeyPtr = static_cast<char*>(outputKeys) + dstOffset;
        char* dstValuePtr = static_cast<char*>(outputValues) + dstOffset;
        const char* srcKeyPtr = static_cast<const char*>(block->getKeyPtr()) + srcOffset;
        const char* srcValuePtr = static_cast<const char*>(block->getValuePtr()) + srcOffset;
        
        copyMemory(dstKeyPtr, srcKeyPtr, singleTokenSize, useGPU);
        copyMemory(dstValuePtr, srcValuePtr, singleTokenSize, useGPU);
      }
    }
  }
  
  return success();
}

} // namespace runtime
} // namespace llm
} // namespace mlir 