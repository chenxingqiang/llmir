# LLMIR KV Cache Implementation

## Overview

This repository contains a high-performance Key-Value (KV) cache implementation for Large Language Model inference, integrated with the MLIR compiler infrastructure. The KV cache is a critical optimization for efficient LLM inference, enabling the reuse of previously computed key and value tensors across multiple decoding steps.

The implementation follows the PagedAttention mechanism (inspired by vLLM), organizing memory in fixed-size blocks to efficiently handle variable sequence lengths and optimize memory usage.

## Architecture

The implementation consists of three main components:

1. **Runtime Support Library**: Provides efficient memory management for KV caches
2. **MLIR Dialect Operations**: Defines operations for interacting with the KV cache
3. **Runtime Interfaces**: Connects the MLIR operations to the runtime implementation

### Runtime Components

- `KVBlock`: Represents a single memory block for storing key-value tensors
- `BlockAllocator`: Manages a pool of KV blocks for efficient memory management
- `PagedKVCache`: Main interface for the KV cache system

### MLIR Operations

- `llm.append_kv`: Appends new key-value pairs to the cache
- `llm.lookup_kv`: Retrieves key-value pairs from the cache
- `llm.paged_attention`: Performs attention computation with the KV cache

### Interface Layers

- `KVCacheInterface`: Interface for operations that interact with KV cache
- `AttentionInterface`: Interface for operations that perform attention computation

## Key Features

1. **Memory Efficiency**: Block-based allocation optimizes memory usage
2. **Variable Sequence Support**: Efficiently handles sequences of different lengths
3. **Reduced Fragmentation**: Block pooling minimizes memory fragmentation
4. **Batch Processing**: Support for multiple sequences in a single batch
5. **Multi-Layer Support**: Manages KV caches for multiple transformer layers

## Usage Examples

### MLIR Integration

```mlir
// Create a paged KV cache type
!kv_cache_t = !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>

// Append key-value pairs to the cache
%new_kv, %block_indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!kv_cache_t, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
    -> (!kv_cache_t, tensor<2x1xi32>)

// Perform paged attention with the KV cache
%output = llm.paged_attention %query, %new_kv, %block_indices, %seq_lens {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<2x1x16x64xf16>, !kv_cache_t, tensor<2x128xi32>, tensor<2xi32>) 
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

## Directory Structure

```
include/mlir/Dialect/LLM/
  ├── IR/                      # MLIR dialect definitions
  │   ├── LLMKVCacheOps.td     # KV cache operations TableGen
  │   └── LLMTypes.td          # Type definitions including PagedKVCache
  └── Runtime/                 # Runtime support
      ├── KVCache.h            # Runtime header
      ├── KVCache.td           # TableGen definitions
      ├── RuntimeInterfaces.h  # Interface declarations
      └── RuntimeInterfaces.td # Interface TableGen definitions

lib/Dialect/LLM/
  ├── IR/                      # MLIR operation implementations
  │   ├── LLMKVCacheOps.cpp    # KV cache operations
  │   └── LLMKVCacheOpsInterface.cpp # Interface implementation
  └── Runtime/                 # Runtime implementation
      └── KVCache.cpp          # Runtime support

test/Dialect/LLM/
  ├── kv_cache_ops.mlir        # MLIR operation tests
  └── Runtime/                 # Runtime tests
      └── kv_cache_unit_test.cpp # Unit tests

examples/
  └── kv_cache_example.cpp    # Example using the KV cache runtime
```

## Testing

The KV cache implementation includes comprehensive tests:

1. **MLIR operation tests**: Verifies that the dialect operations work correctly
2. **Runtime unit tests**: Tests for the runtime components (KVBlock, BlockAllocator, PagedKVCache)
3. **Example application**: Demonstrates how to use the KV cache in practice

## Future Enhancements

1. **GPU Support**: Complete GPU memory allocator implementation with CUDA/HIP
2. **Quantization**: Support for quantized KV caches (INT8/INT4)
3. **Memory Optimization**: Advanced block size tuning and memory layout
4. **Pruning/Eviction**: Strategies for limited memory scenarios
5. **Multi-GPU Distribution**: Support for distributed inference 