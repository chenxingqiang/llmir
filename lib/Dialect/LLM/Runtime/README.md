# LLM Runtime Support Library - KV Cache Implementation

## Overview

This directory contains the runtime support library for the LLM dialect's KV cache functionality. 
The key-value (KV) cache is a critical optimization for efficient LLM inference, enabling the reuse
of previously computed key and value tensors for transformer attention layers across multiple decoding steps.

The implementation is inspired by the PagedAttention mechanism from vLLM, organizing memory in fixed-size
blocks to efficiently handle variable sequence lengths and optimize memory usage.

## Components

### Core Runtime Classes

#### `KVBlock`

Represents a single memory block for storing key-value tensors:
- Contains pointers to memory regions for keys and values
- Tracks block size and dimensions
- Manages memory layout for efficient access

```cpp
KVBlock block(keyPtr, valuePtr, blockSize, headDim);
void* keys = block.getKeyPtr();
void* values = block.getValuePtr();
```

#### `BlockAllocator`

Manages a pool of KV blocks for efficient memory allocation:
- Pre-allocates blocks to reduce allocation overhead
- Provides block allocation and deallocation with efficient reuse
- Handles one memory layer (typically one transformer layer)

```cpp
BlockAllocator allocator(blockSize, numHeads, headDim, elementType);
KVBlock* block = allocator.allocateBlock();
allocator.freeBlock(block);
```

#### `PagedKVCache`

Main interface for the KV cache system:
- Manages multiple transformer layers with separate block allocators
- Maps sequence IDs to their corresponding blocks
- Provides operations for appending and retrieving KV pairs
- Optimizes memory usage with block sharing

```cpp
PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType);
cache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);
cache.lookupKV(blockIndices, seqLens, batchSize, outputKeys, outputValues);
```

## Memory Management Strategy

The implementation uses a block-based memory management approach:
1. Memory is allocated in fixed-size blocks (e.g., 16 tokens per block)
2. Blocks can be dynamically allocated and freed
3. Sequences can span multiple blocks
4. Block allocation is tracked per sequence and layer

Benefits:
- **Memory Efficiency**: Only allocate what's needed
- **Variable Sequence Support**: Efficiently handle varying sequence lengths
- **Reduced Fragmentation**: Block pooling minimizes memory fragmentation
- **Memory Reuse**: Efficient reuse of freed blocks

## Using the KV Cache in MLIR

### MLIR Operations

The following operations are provided for interacting with the KV cache:

#### `llm.append_kv`

Appends new key-value pairs to a KV cache:

```mlir
%new_kv, %block_indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!llm.paged_kv_cache, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
    -> (!llm.paged_kv_cache, tensor<2x1xi32>)
```

#### `llm.lookup_kv`

Retrieves key-value pairs from a KV cache:

```mlir
%keys, %values = llm.lookup_kv %kv_cache, %block_indices, %seq_lens {
  num_heads = 16 : i32,
  head_dim = 64 : i32
} : (!llm.paged_kv_cache, tensor<2x128xi32>, tensor<2xi32>) 
    -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>)
```

#### `llm.paged_attention`

Performs attention computation with a KV cache:

```mlir
%output = llm.paged_attention %query, %kv_cache, %block_indices, %seq_lens {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<2x1x16x64xf16>, !llm.paged_kv_cache, tensor<2x128xi32>, tensor<2xi32>) 
    -> tensor<2x1x16x64xf16>
```

### C++ Runtime API

```cpp
// Create a KV cache for a model
PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType);

// Append new KV pairs to the cache
cache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);

// Lookup existing KV pairs
cache.lookupKV(blockIndices, seqLens, batchSize, outputKeys, outputValues);
```

## Key Features

1. **Block-Based Allocation**: Efficient memory management with fixed-size blocks
2. **Sequence Tracking**: Track multiple sequences with different lengths
3. **Multi-Layer Support**: Separate block allocators for each transformer layer
4. **Batch Processing**: Support for batched operations with multiple sequences
5. **GPU Support**: Designed with GPU memory operations in mind (TODO: complete implementation)

## Future Enhancements

1. **GPU Support**: Complete GPU memory allocator implementation with CUDA/HIP
2. **Quantization**: Support for quantized KV caches (INT8/INT4)
3. **Memory Optimization**: Advanced block size tuning and memory layout optimization
4. **Pruning/Eviction**: Implement policies for managing limited memory scenarios
5. **Multi-GPU Distribution**: Extend for distributed inference across multiple devices 