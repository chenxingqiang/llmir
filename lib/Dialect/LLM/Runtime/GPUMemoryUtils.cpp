//===- GPUMemoryUtils.cpp - GPU memory utilities for LLM KV cache ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements GPU memory utilities for LLM KV cache.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/GPUMemoryUtils.h"
#include <chrono>
#include <iostream>
#include <algorithm>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// PinnedMemoryManager Implementation
//===----------------------------------------------------------------------===//

void* PinnedMemoryManager::allocate(size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mutex);
  void* ptr = nullptr;

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMallocHost(&ptr, sizeBytes);
  if (error != cudaSuccess) {
    std::cerr << "CUDA pinned memory allocation failed: " 
              << cudaGetErrorString(error) << std::endl;
    return nullptr;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipHostMalloc(&ptr, sizeBytes, hipHostMallocPortable);
  if (error != hipSuccess) {
    std::cerr << "HIP pinned memory allocation failed: " 
              << hipGetErrorString(error) << std::endl;
    return nullptr;
  }
#else
  // Fallback to regular malloc on non-GPU systems
  ptr = std::malloc(sizeBytes);
#endif

  if (ptr) {
    allocations[ptr] = sizeBytes;
  }
  
  return ptr;
}

void PinnedMemoryManager::free(void* ptr) {
  if (!ptr) return;
  
  std::lock_guard<std::mutex> lock(mutex);
  
  auto it = allocations.find(ptr);
  if (it == allocations.end()) {
    // Not managed by us, ignore
    return;
  }
  
#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaFreeHost(ptr);
  if (error != cudaSuccess) {
    std::cerr << "CUDA pinned memory free failed: " 
              << cudaGetErrorString(error) << std::endl;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipHostFree(ptr);
  if (error != hipSuccess) {
    std::cerr << "HIP pinned memory free failed: " 
              << hipGetErrorString(error) << std::endl;
  }
#else
  // Fallback to regular free on non-GPU systems
  std::free(ptr);
#endif

  allocations.erase(it);
}

LogicalResult PinnedMemoryManager::registerHostMemory(void* ptr, size_t sizeBytes) {
  if (!ptr) return failure();
  
#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaHostRegister(ptr, sizeBytes, cudaHostRegisterPortable);
  if (error != cudaSuccess) {
    std::cerr << "CUDA host memory registration failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipHostRegister(ptr, sizeBytes, hipHostRegisterPortable);
  if (error != hipSuccess) {
    std::cerr << "HIP host memory registration failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
#else
  // No-op on non-GPU systems
  return success();
#endif

  std::lock_guard<std::mutex> lock(mutex);
  allocations[ptr] = sizeBytes;
  return success();
}

LogicalResult PinnedMemoryManager::unregisterHostMemory(void* ptr) {
  if (!ptr) return failure();
  
#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaHostUnregister(ptr);
  if (error != cudaSuccess) {
    std::cerr << "CUDA host memory unregistration failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipHostUnregister(ptr);
  if (error != hipSuccess) {
    std::cerr << "HIP host memory unregistration failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
#else
  // No-op on non-GPU systems
  return success();
#endif

  std::lock_guard<std::mutex> lock(mutex);
  allocations.erase(ptr);
  return success();
}

PinnedMemoryManager::~PinnedMemoryManager() {
  // Make a copy of the allocations to avoid modification during iteration
  auto allocationsCopy = allocations;
  
  for (const auto& [ptr, size] : allocationsCopy) {
    free(ptr);
  }
  
  allocations.clear();
}

//===----------------------------------------------------------------------===//
// UnifiedMemoryManager Implementation
//===----------------------------------------------------------------------===//

void* UnifiedMemoryManager::allocate(size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mutex);
  void* ptr = nullptr;

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMallocManaged(&ptr, sizeBytes, cudaMemAttachGlobal);
  if (error != cudaSuccess) {
    std::cerr << "CUDA unified memory allocation failed: " 
              << cudaGetErrorString(error) << std::endl;
    return nullptr;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipMallocManaged(&ptr, sizeBytes);
  if (error != hipSuccess) {
    std::cerr << "HIP unified memory allocation failed: " 
              << hipGetErrorString(error) << std::endl;
    return nullptr;
  }
#else
  // Fallback to regular malloc on non-GPU systems
  ptr = std::malloc(sizeBytes);
#endif

  if (ptr) {
    allocations[ptr] = sizeBytes;
  }
  
  return ptr;
}

void UnifiedMemoryManager::free(void* ptr) {
  if (!ptr) return;
  
  std::lock_guard<std::mutex> lock(mutex);
  
  auto it = allocations.find(ptr);
  if (it == allocations.end()) {
    // Not managed by us, ignore
    return;
  }
  
#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaFree(ptr);
  if (error != cudaSuccess) {
    std::cerr << "CUDA unified memory free failed: " 
              << cudaGetErrorString(error) << std::endl;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipFree(ptr);
  if (error != hipSuccess) {
    std::cerr << "HIP unified memory free failed: " 
              << hipGetErrorString(error) << std::endl;
  }
#else
  // Fallback to regular free on non-GPU systems
  std::free(ptr);
#endif

  allocations.erase(it);
}

UnifiedMemoryManager::~UnifiedMemoryManager() {
  // Make a copy of the allocations to avoid modification during iteration
  auto allocationsCopy = allocations;
  
  for (const auto& [ptr, size] : allocationsCopy) {
    free(ptr);
  }
  
  allocations.clear();
}

//===----------------------------------------------------------------------===//
// GPUMemoryPool Implementation
//===----------------------------------------------------------------------===//

void* GPUMemoryPool::allocate(size_t sizeBytes) {
  if (!enabled) {
    // If pool is disabled, allocate directly
    missCount++;
    return GPUMemoryUtils::allocateDevice(sizeBytes);
  }
  
  std::lock_guard<std::mutex> lock(mutex);
  
  // Try to find a suitable free block
  MemoryBlockDescriptor* block = findFreeBlock(sizeBytes);
  
  if (block) {
    // Use existing block
    block->inUse = true;
    block->lastAccessTime = std::chrono::steady_clock::now().time_since_epoch().count();
    hitCount++;
    return block->ptr;
  }
  
  // No suitable block found, create a new one
  block = createBlock(sizeBytes);
  if (!block) {
    // Failed to create a new block
    return nullptr;
  }
  
  block->inUse = true;
  block->lastAccessTime = std::chrono::steady_clock::now().time_since_epoch().count();
  missCount++;
  return block->ptr;
}

void GPUMemoryPool::free(void* ptr) {
  if (!ptr) return;
  
  if (!enabled) {
    // If pool is disabled, free directly
    GPUMemoryUtils::freeDevice(ptr);
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex);
  
  // Find the block containing this pointer
  auto it = std::find_if(blocks.begin(), blocks.end(),
                        [ptr](const std::unique_ptr<MemoryBlockDescriptor>& block) {
                          return block->ptr == ptr;
                        });
                        
  if (it == blocks.end()) {
    // Not managed by this pool, free directly
    GPUMemoryUtils::freeDevice(ptr);
    return;
  }
  
  // Mark the block as free for reuse
  (*it)->inUse = false;
}

void GPUMemoryPool::setInitialCapacity(size_t bytes) {
  if (!enabled || bytes == 0) return;
  
  std::lock_guard<std::mutex> lock(mutex);
  
  // Create a block with the specified capacity
  // Use 1MB chunks for better reuse
  const size_t chunkSize = 1024 * 1024;
  size_t remaining = bytes;
  
  while (remaining >= chunkSize) {
    size_t blockSize = std::min(remaining, chunkSize * 16); // Max 16MB per block
    createBlock(blockSize);
    remaining -= blockSize;
  }
  
  // Create a final block with any remaining bytes
  if (remaining > 0) {
    createBlock(remaining);
  }
}

GPUMemoryPool::PoolStats GPUMemoryPool::getStats() const {
  std::lock_guard<std::mutex> lock(mutex);
  
  PoolStats stats;
  stats.blockCount = blocks.size();
  stats.hitCount = hitCount;
  stats.missCount = missCount;
  
  for (const auto& block : blocks) {
    stats.totalMemory += block->size;
    if (block->inUse) {
      stats.usedMemory += block->size;
    } else {
      stats.freeMemory += block->size;
    }
  }
  
  return stats;
}

void GPUMemoryPool::shrink(float keepRatio) {
  if (!enabled || keepRatio >= 1.0) return;
  
  std::lock_guard<std::mutex> lock(mutex);
  
  // Count free blocks
  std::vector<std::pair<int, int64_t>> freeBlocks; // (index, lastAccessTime)
  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i]->inUse) {
      freeBlocks.emplace_back(i, blocks[i]->lastAccessTime);
    }
  }
  
  if (freeBlocks.empty()) return;
  
  // Sort by last access time (older first)
  std::sort(freeBlocks.begin(), freeBlocks.end(),
           [](const auto& a, const auto& b) {
             return a.second < b.second;
           });
           
  // Calculate how many blocks to keep
  size_t keepCount = std::max(size_t(1), static_cast<size_t>(freeBlocks.size() * keepRatio));
  size_t releaseCount = freeBlocks.size() - keepCount;
  
  // Release the oldest blocks
  std::vector<int> indicesToRemove;
  for (size_t i = 0; i < releaseCount; i++) {
    int blockIndex = freeBlocks[i].first;
    MemoryBlockDescriptor* block = blocks[blockIndex].get();
    GPUMemoryUtils::freeDevice(block->ptr);
    indicesToRemove.push_back(blockIndex);
  }
  
  // Sort indices in descending order to safely remove elements
  std::sort(indicesToRemove.begin(), indicesToRemove.end(), std::greater<int>());
  
  for (int index : indicesToRemove) {
    blocks.erase(blocks.begin() + index);
  }
}

MemoryBlockDescriptor* GPUMemoryPool::createBlock(size_t sizeBytes) {
  void* ptr = GPUMemoryUtils::allocateDevice(sizeBytes);
  if (!ptr) return nullptr;
  
  auto block = std::make_unique<MemoryBlockDescriptor>(ptr, sizeBytes);
  blocks.push_back(std::move(block));
  return blocks.back().get();
}

MemoryBlockDescriptor* GPUMemoryPool::findFreeBlock(size_t sizeBytes) {
  // First-fit approach with a 10% overhead allowance
  size_t maxAllowableSize = sizeBytes * 1.1;
  
  MemoryBlockDescriptor* bestFit = nullptr;
  size_t bestFitSize = SIZE_MAX;
  
  for (auto& block : blocks) {
    if (!block->inUse && block->size >= sizeBytes) {
      // This block is big enough
      if (block->size <= maxAllowableSize) {
        // This block is within our overhead allowance, use it immediately
        return block.get();
      }
      
      // Keep track of the smallest block that's still big enough
      if (block->size < bestFitSize) {
        bestFit = block.get();
        bestFitSize = block->size;
      }
    }
  }
  
  return bestFit;
}

GPUMemoryPool::~GPUMemoryPool() {
  // Free all blocks
  for (auto& block : blocks) {
    if (block->ptr) {
      GPUMemoryUtils::freeDevice(block->ptr);
    }
  }
  
  blocks.clear();
}

//===----------------------------------------------------------------------===//
// GPUMemoryUtils Implementation
//===----------------------------------------------------------------------===//

bool GPUMemoryUtils::isGPUAvailable() {
#if defined(LLMIR_ENABLE_CUDA)
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  return (error == cudaSuccess && deviceCount > 0);
#elif defined(LLMIR_ENABLE_HIP)
  int deviceCount = 0;
  hipError_t error = hipGetDeviceCount(&deviceCount);
  return (error == hipSuccess && deviceCount > 0);
#elif defined(LLMIR_ENABLE_METAL) && defined(__APPLE__)
  // Check if Metal is available on this device
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  bool available = (device != nil);
  if (device) {
    [device release];
  }
  return available;
#else
  return false;
#endif
}

void* GPUMemoryUtils::allocateDevice(size_t sizeBytes) {
  void* ptr = nullptr;

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMalloc(&ptr, sizeBytes);
  if (error != cudaSuccess) {
    std::cerr << "CUDA device memory allocation failed: " 
              << cudaGetErrorString(error) << std::endl;
    return nullptr;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipMalloc(&ptr, sizeBytes);
  if (error != hipSuccess) {
    std::cerr << "HIP device memory allocation failed: " 
              << hipGetErrorString(error) << std::endl;
    return nullptr;
  }
#elif defined(LLMIR_ENABLE_METAL) && defined(__APPLE__)
  // Metal buffer allocation
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "Metal device not available" << std::endl;
    return nullptr;
  }
  
  id<MTLBuffer> buffer = [device newBufferWithLength:sizeBytes options:MTLResourceStorageModeShared];
  if (!buffer) {
    std::cerr << "Metal buffer allocation failed" << std::endl;
    [device release];
    return nullptr;
  }
  
  // Store the Metal buffer pointer
  // Note: we're returning the buffer contents pointer and storing the Metal buffer elsewhere
  // to be able to release it properly
  ptr = [buffer contents];
  
  // Store buffer reference for cleanup (implementation would need a map to track these)
  // For simplicity, we're assuming this will be managed elsewhere
  
  [device release];
#else
  // No GPU support
  std::cerr << "GPU support not enabled" << std::endl;
  return nullptr;
#endif

  return ptr;
}

void GPUMemoryUtils::freeDevice(void* ptr) {
  if (!ptr) return;

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaFree(ptr);
  if (error != cudaSuccess) {
    std::cerr << "CUDA device memory free failed: " 
              << cudaGetErrorString(error) << std::endl;
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipFree(ptr);
  if (error != hipSuccess) {
    std::cerr << "HIP device memory free failed: " 
              << hipGetErrorString(error) << std::endl;
  }
#endif
}

void* GPUMemoryUtils::allocateHostPinned(size_t sizeBytes) {
  return PinnedMemoryManager::getInstance().allocate(sizeBytes);
}

void GPUMemoryUtils::freeHostPinned(void* ptr) {
  PinnedMemoryManager::getInstance().free(ptr);
}

void* GPUMemoryUtils::allocateUnified(size_t sizeBytes) {
  return UnifiedMemoryManager::getInstance().allocate(sizeBytes);
}

void GPUMemoryUtils::freeUnified(void* ptr) {
  UnifiedMemoryManager::getInstance().free(ptr);
}

LogicalResult GPUMemoryUtils::copyHostToDevice(void* deviceDst, const void* hostSrc, size_t sizeBytes) {
  if (!deviceDst || !hostSrc) return failure();

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMemcpy(deviceDst, hostSrc, sizeBytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    std::cerr << "CUDA host to device copy failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipMemcpy(deviceDst, hostSrc, sizeBytes, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    std::cerr << "HIP host to device copy failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
#else
  std::cerr << "GPU support not enabled" << std::endl;
  return failure();
#endif

  return success();
}

LogicalResult GPUMemoryUtils::copyDeviceToHost(void* hostDst, const void* deviceSrc, size_t sizeBytes) {
  if (!hostDst || !deviceSrc) return failure();

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMemcpy(hostDst, deviceSrc, sizeBytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    std::cerr << "CUDA device to host copy failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipMemcpy(hostDst, deviceSrc, sizeBytes, hipMemcpyDeviceToHost);
  if (error != hipSuccess) {
    std::cerr << "HIP device to host copy failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
#else
  std::cerr << "GPU support not enabled" << std::endl;
  return failure();
#endif

  return success();
}

LogicalResult GPUMemoryUtils::copyDeviceToDevice(void* deviceDst, const void* deviceSrc, size_t sizeBytes) {
  if (!deviceDst || !deviceSrc) return failure();

#if defined(LLMIR_ENABLE_CUDA)
  cudaError_t error = cudaMemcpy(deviceDst, deviceSrc, sizeBytes, cudaMemcpyDeviceToDevice);
  if (error != cudaSuccess) {
    std::cerr << "CUDA device to device copy failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
#elif defined(LLMIR_ENABLE_HIP)
  hipError_t error = hipMemcpy(deviceDst, deviceSrc, sizeBytes, hipMemcpyDeviceToDevice);
  if (error != hipSuccess) {
    std::cerr << "HIP device to device copy failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
#else
  std::cerr << "GPU support not enabled" << std::endl;
  return failure();
#endif

  return success();
}

LogicalResult GPUMemoryUtils::getDeviceProperties(GPUDeviceProperties& props) {
#if defined(LLMIR_ENABLE_CUDA)
  int device;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    std::cerr << "CUDA get device failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
  
  cudaDeviceProp deviceProp;
  error = cudaGetDeviceProperties(&deviceProp, device);
  if (error != cudaSuccess) {
    std::cerr << "CUDA get device properties failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
  
  size_t free, total;
  error = cudaMemGetInfo(&free, &total);
  if (error != cudaSuccess) {
    std::cerr << "CUDA get memory info failed: " 
              << cudaGetErrorString(error) << std::endl;
    return failure();
  }
  
  props.deviceId = device;
  props.totalMemory = total;
  props.freeMemory = free;
  props.computeCapabilityMajor = deviceProp.major;
  props.computeCapabilityMinor = deviceProp.minor;
  
  // Copy device name safely
  std::strncpy(props.name, deviceProp.name, sizeof(props.name) - 1);
  props.name[sizeof(props.name) - 1] = '\0';
#elif defined(LLMIR_ENABLE_HIP)
  int device;
  hipError_t error = hipGetDevice(&device);
  if (error != hipSuccess) {
    std::cerr << "HIP get device failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
  
  hipDeviceProp_t deviceProp;
  error = hipGetDeviceProperties(&deviceProp, device);
  if (error != hipSuccess) {
    std::cerr << "HIP get device properties failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
  
  size_t free, total;
  error = hipMemGetInfo(&free, &total);
  if (error != hipSuccess) {
    std::cerr << "HIP get memory info failed: " 
              << hipGetErrorString(error) << std::endl;
    return failure();
  }
  
  props.deviceId = device;
  props.totalMemory = total;
  props.freeMemory = free;
  props.computeCapabilityMajor = deviceProp.major;
  props.computeCapabilityMinor = deviceProp.minor;
  
  // Copy device name safely
  std::strncpy(props.name, deviceProp.name, sizeof(props.name) - 1);
  props.name[sizeof(props.name) - 1] = '\0';
#else
  // No GPU support
  return failure();
#endif

  return success();
}

} // namespace runtime
} // namespace llm
} // namespace mlir 