# LLMIR KV Cache Implementation - Phase 2

This document describes the implementation of the PagedKVCache runtime support library for the LLMIR (Large Language Model Intermediate Representation) project, which is part of Phase 2 of the development roadmap.

## Overview

The KV cache (Key-Value cache) is a critical optimization technique in large language model inference that significantly improves performance by storing previously computed key and value tensors across generation steps. This implementation follows the PagedAttention mechanism originally introduced in the vLLM project, which organizes memory in fixed-size blocks for efficient management.

## Components

The implementation consists of three main components:

### 1. KVBlock

`KVBlock` represents a fixed-size block of memory storing key-value pairs for a number of tokens.

**Key features:**
- Fixed-size memory allocation for keys and values
- Reference counting for memory management
- Usage tracking to monitor block utilization
- Support for both CPU and GPU memory

```cpp
class KVBlock {
public:
  KVBlock(void* keyPtr, void* valuePtr, int64_t blockSize, int64_t headDim);
  
  void* getKeyPtr() const;
  void* getValuePtr() const;
  int64_t getBlockSize() const;
  int64_t getHeadDim() const;
  
  int64_t getUsedSlots() const;
  void incrementUsedSlots(int64_t count = 1);
  void resetUsedSlots();
  bool isFull() const;
  
  int32_t getRefCount() const;
  void incrementRefCount();
  bool decrementRefCount();
};
```

### 2. BlockAllocator

`BlockAllocator` manages a pool of KV blocks for efficient memory allocation and reuse.

**Key features:**
- Pre-allocation of blocks to reduce allocation overhead
- Block pooling for memory reuse
- Efficient tracking of allocated and free blocks
- Support for retrieving blocks by index

```cpp
class BlockAllocator {
public:
  BlockAllocator(int64_t blockSize, int64_t numHeads, int64_t headDim, 
                Type elementType, bool useGPU = true);
  ~BlockAllocator();
  
  KVBlock* allocateBlock();
  void freeBlock(KVBlock* block);
  KVBlock* getBlock(int32_t blockIdx) const;
  
  int64_t getNumAllocatedBlocks() const;
  int64_t getNumFreeBlocks() const;
};
```

### 3. PagedKVCache

`PagedKVCache` is the main class providing the runtime API for interacting with the KV cache.

**Key features:**
- Multi-layer KV cache management
- Efficient mapping of sequence tokens to block positions
- Support for batched sequence processing
- Memory-efficient appending and lookup operations

```cpp
class PagedKVCache {
public:
  PagedKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
              int64_t blockSize, int64_t maxSeqLen, Type elementType,
              bool useGPU = true);
  ~PagedKVCache();
  
  LogicalResult appendKV(const void* keyPtr, const void* valuePtr,
                        int64_t batchSize, int64_t seqLen, 
                        const int32_t* seqIds, int32_t* blockIndices);
                        
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int64_t batchSize, void* outputKeys, void* outputValues);
                        
  LogicalResult clearSequence(int32_t seqId);
  void reset();
  
  // Various getter methods...
};
```

## Memory Management Strategy

The key innovation in this implementation is the block-based memory management approach:

1. **Fixed-Size Blocks**: Memory is allocated in fixed-size blocks (typically 16-128 tokens per block), enabling efficient memory reuse.

2. **Block Pooling**: The `BlockAllocator` maintains a pool of blocks that can be reused, minimizing allocation overhead.

3. **Reference Counting**: Each block tracks how many sequences are referencing it, allowing for memory to be freed only when no longer needed.

4. **Sequence Mapping**: The `PagedKVCache` maintains a mapping from sequence IDs to block positions, enabling efficient lookups.

## Integration with MLIR

The runtime library is integrated with MLIR through the following components:

1. **LLM Dialect Operations**:
   - `llm.append_kv`: Appends key-value pairs to the cache
   - `llm.lookup_kv`: Retrieves key-value pairs from the cache
   - `llm.paged_attention`: Performs attention computation using the paged cache

2. **Runtime Interfaces**:
   - `KVCacheInterface`: Interface for operations that interact with the KV cache
   - `AttentionInterface`: Interface for operations that perform attention computation

3. **Optimization Passes**:
   - `KVCacheOptimizationPass`: Optimizes block sizes and ensures compatibility between operations

## Performance Considerations

The implementation is designed with the following performance aspects in mind:

1. **Memory Efficiency**:
   - Reduced memory fragmentation through block-based allocation
   - Minimal memory waste by tracking block utilization
   - Proper reference counting to ensure timely memory release

2. **Computational Efficiency**:
   - Efficient batch processing of sequences
   - Fast lookups through direct block indexing
   - Support for cross-block sequences

3. **Scalability**:
   - Handles variable-length sequences efficiently
   - Supports multiple concurrent sequences
   - Scales to large transformer models with many layers

## GPU Integration

The implementation includes support for GPU memory through the following mechanisms:

1. **Memory Allocation**: The allocator handles both CPU and GPU memory allocation.
2. **Memory Copy**: Efficient memory copying between CPU and GPU when needed.
3. **Parameter Configuration**: GPU optimization can be toggled with the `useGPU` parameter.

## Testing

The implementation includes comprehensive unit tests that verify:

1. **Basic Operations**: Creation, appending, and lookup of KV pairs
2. **Memory Management**: Block allocation, reference counting, and memory cleanup
3. **Edge Cases**: Cross-block boundary handling, sequence clearing, and cache resetting

## Usage Example

```cpp
// Create a PagedKVCache
PagedKVCache cache(
    numLayers,    // Number of transformer layers
    numHeads,     // Number of attention heads per layer
    headDim,      // Dimension of each attention head
    blockSize,    // Size of each memory block (in tokens)
    maxSeqLen,    // Maximum sequence length
    elementType,  // Data type (e.g., f16, f32)
    useGPU        // Whether to use GPU memory
);

// Append KV pairs
cache.appendKV(
    keyPtr,        // Pointer to key tensor data
    valuePtr,      // Pointer to value tensor data
    batchSize,     // Number of sequences in the batch
    seqLen,        // Number of tokens per sequence
    seqIds,        // Array of sequence IDs
    blockIndices   // Output array for block indices
);

// Lookup KV pairs
cache.lookupKV(
    blockIndices,  // Array of block indices for each token
    seqLens,       // Array of sequence lengths
    batchSize,     // Number of sequences in the batch
    outputKeys,    // Output buffer for keys
    outputValues   // Output buffer for values
);
```

## Future Work

1. **Full GPU Integration**: Implement CUDA/HIP-specific memory operations
2. **Quantization Support**: Add support for quantized KV caches (INT8/INT4)
3. **Advanced Memory Strategies**: Implement eviction policies for constrained memory environments
4. **Multi-GPU Support**: Extend the implementation for distributed inference

## Conclusion

The PagedKVCache implementation provides a robust foundation for efficient LLM inference in the LLMIR project. By adopting a block-based memory management approach, it significantly improves memory efficiency and enables high-performance autoregressive generation for large language models. 