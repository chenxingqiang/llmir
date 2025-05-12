//===- GPUMemoryUtils.h - GPU memory utilities for LLM KV cache ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines GPU memory utilities for LLM KV cache implementation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_GPUMEMORYUTILS_H_
#define MLIR_DIALECT_LLM_RUNTIME_GPUMEMORYUTILS_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "mlir/Support/LogicalResult.h"

// Include CUDA headers when GPU support is enabled
#if defined(LLMIR_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(LLMIR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(LLMIR_ENABLE_METAL) && defined(__APPLE__)
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

namespace mlir {
namespace llm {
namespace runtime {

/// Class to manage pinned memory for efficient host-device transfers
class PinnedMemoryManager {
public:
  static PinnedMemoryManager& getInstance() {
    static PinnedMemoryManager instance;
    return instance;
  }

  /// Allocate pinned memory of given size
  void* allocate(size_t sizeBytes);
  
  /// Free pinned memory
  void free(void* ptr);
  
  /// Register existing host memory as pinned
  LogicalResult registerHostMemory(void* ptr, size_t sizeBytes);
  
  /// Unregister previously registered host memory
  LogicalResult unregisterHostMemory(void* ptr);

private:
  PinnedMemoryManager() = default;
  ~PinnedMemoryManager();
  
  // Disable copy/move
  PinnedMemoryManager(const PinnedMemoryManager&) = delete;
  PinnedMemoryManager& operator=(const PinnedMemoryManager&) = delete;
  
  std::mutex mutex;
  std::unordered_map<void*, size_t> allocations;
};

/// Class to manage unified memory for small blocks
class UnifiedMemoryManager {
public:
  static UnifiedMemoryManager& getInstance() {
    static UnifiedMemoryManager instance;
    return instance;
  }

  /// Allocate unified memory of given size
  void* allocate(size_t sizeBytes);
  
  /// Free unified memory
  void free(void* ptr);
  
  /// Set the size threshold for using unified memory (default: 1MB)
  void setThreshold(size_t thresholdBytes) { sizeThreshold = thresholdBytes; }
  
  /// Get the current threshold for using unified memory
  size_t getThreshold() const { return sizeThreshold; }
  
  /// Check if a block should use unified memory based on its size
  bool shouldUseUnifiedMemory(size_t sizeBytes) const {
    return sizeBytes <= sizeThreshold;
  }

private:
  UnifiedMemoryManager() : sizeThreshold(1 * 1024 * 1024) {} // Default: 1MB
  ~UnifiedMemoryManager();
  
  // Disable copy/move
  UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
  UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
  
  std::mutex mutex;
  std::unordered_map<void*, size_t> allocations;
  size_t sizeThreshold;
};

/// Memory block descriptor for GPU memory pool
struct MemoryBlockDescriptor {
  void* ptr = nullptr;
  size_t size = 0;
  bool inUse = false;
  int64_t lastAccessTime = 0;
  
  MemoryBlockDescriptor(void* ptr, size_t size)
    : ptr(ptr), size(size), inUse(false), lastAccessTime(0) {}
};

/// Class to manage a memory pool for GPU allocations
class GPUMemoryPool {
public:
  static GPUMemoryPool& getInstance() {
    static GPUMemoryPool instance;
    return instance;
  }

  /// Allocate memory from the pool
  void* allocate(size_t sizeBytes);
  
  /// Return memory to the pool
  void free(void* ptr);
  
  /// Enable/disable the memory pool
  void enable(bool enable) { enabled = enable; }
  
  /// Check if the memory pool is enabled
  bool isEnabled() const { return enabled; }
  
  /// Set the initial capacity of the pool in bytes
  void setInitialCapacity(size_t bytes);
  
  /// Get pool usage statistics
  struct PoolStats {
    size_t totalMemory = 0;
    size_t usedMemory = 0;
    size_t freeMemory = 0;
    size_t blockCount = 0;
    size_t hitCount = 0;
    size_t missCount = 0;
  };
  
  PoolStats getStats() const;
  
  /// Clean up unused blocks to reduce memory footprint
  void shrink(float keepRatio = 0.5);

private:
  GPUMemoryPool() : enabled(false), hitCount(0), missCount(0) {}
  ~GPUMemoryPool();
  
  // Disable copy/move
  GPUMemoryPool(const GPUMemoryPool&) = delete;
  GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
  
  /// Create a new memory block
  MemoryBlockDescriptor* createBlock(size_t sizeBytes);
  
  /// Find a suitable free block
  MemoryBlockDescriptor* findFreeBlock(size_t sizeBytes);
  
  std::mutex mutex;
  std::vector<std::unique_ptr<MemoryBlockDescriptor>> blocks;
  bool enabled;
  size_t hitCount;
  size_t missCount;
};

/// Helper functions for cross-platform GPU memory operations
namespace GPUMemoryUtils {
  /// Allocate device memory
  void* allocateDevice(size_t sizeBytes);
  
  /// Free device memory
  void freeDevice(void* ptr);
  
  /// Allocate pinned host memory
  void* allocateHostPinned(size_t sizeBytes);
  
  /// Free pinned host memory
  void freeHostPinned(void* ptr);
  
  /// Allocate unified memory
  void* allocateUnified(size_t sizeBytes);
  
  /// Free unified memory
  void freeUnified(void* ptr);
  
  /// Copy memory from host to device
  LogicalResult copyHostToDevice(void* deviceDst, const void* hostSrc, size_t sizeBytes);
  
  /// Copy memory from device to host
  LogicalResult copyDeviceToHost(void* hostDst, const void* deviceSrc, size_t sizeBytes);
  
  /// Copy memory within device
  LogicalResult copyDeviceToDevice(void* deviceDst, const void* deviceSrc, size_t sizeBytes);
  
  /// Check if CUDA/HIP is available
  bool isGPUAvailable();
  
  /// Get current device properties
  struct GPUDeviceProperties {
    int deviceId = -1;
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    char name[256] = {0};
  };
  
  LogicalResult getDeviceProperties(GPUDeviceProperties& props);
}

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_GPUMEMORYUTILS_H_ 