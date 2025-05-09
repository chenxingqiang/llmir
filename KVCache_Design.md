# KV Cache Runtime Design Document (Phase 2)

## 1. Overview

This document outlines the design for the Key-Value Cache runtime support library for the LLMIR project, following the Phase 2 development plan. The KV cache is a critical component for efficient LLM inference, enabling reuse of previously computed key and value tensors across multiple decoding steps.

The implementation is inspired by the PagedAttention mechanism from vLLM, which organizes memory in fixed-size blocks to efficiently handle variable sequence lengths and optimize memory usage.

## 2. Components

### 2.1 Core Runtime Classes

#### KVBlock
- Represents a single memory block for storing key-value tensors
- Contains pointers to memory for keys and values
- Tracks block size and dimensions

#### BlockAllocator
- Manages a pool of KV blocks for efficient memory management
- Pre-allocates blocks to reduce allocation overhead
- Provides block allocation and deallocation with efficient reuse

#### PagedKVCache
- Main interface for the KV cache system
- Manages multiple layers of transformer model KV caches
- Maps sequence IDs to their corresponding blocks
- Provides operations for appending and retrieving KV pairs

### 2.2 TableGen Definitions

#### KVCache.td
- Defines the KVCacheConfig struct attribute
- Defines KVCacheStrategy enum for different allocation strategies
- Defines KVCacheStats struct for runtime statistics

#### RuntimeInterfaces.td
- Defines KVCacheInterface for operations that interact with KV cache
- Defines AttentionInterface for operations that perform attention

### 2.3 MLIR Operations

#### LLMKVCacheOps.td
- `llm.append_kv`: Operation to add new KV pairs to the cache
- `llm.lookup_kv`: Operation to retrieve KV pairs from the cache
- `llm.paged_attention`: Operation that performs attention with the KV cache

#### CustomTypes.td
- `llm.paged_kv_cache`: Custom type to represent the KV cache in MLIR

## 3. Memory Management

### 3.1 Block-Based Memory Organization
- Fixed-size blocks (e.g., 16 tokens per block)
- Tokens from a sequence can span multiple blocks
- Blocks are reused from a pool when tokens are processed

### 3.2 Memory Efficiency
- Only allocate memory as needed
- Support for variable sequence lengths
- Reduced memory fragmentation
- Efficient memory reuse

## 4. Implementation Plan

### 4.1 Runtime Library
1. Complete the implementation of KVBlock class
2. Implement BlockAllocator with efficient block pooling
3. Implement PagedKVCache with multi-layer support
4. Add support for GPU memory allocation
5. Optimize memory operations for performance

### 4.2 MLIR Integration
1. Define operation interfaces in RuntimeInterfaces.td
2. Implement custom types for KV cache
3. Create operations for KV cache manipulation
4. Implement operation verifiers

### 4.3 Testing
1. Unit tests for each component
2. Integration tests for end-to-end workflows
3. Performance benchmarks

## 5. API Design

### 5.1 PagedKVCache API
```cpp
// Create a KV cache for a model
PagedKVCache cache(numLayers, numHeads, headDim, blockSize, maxSeqLen, elementType);

// Append new KV pairs to the cache
cache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);

// Lookup existing KV pairs
cache.lookupKV(blockIndices, seqLens, batchSize, outputKeys, outputValues);
```

### 5.2 MLIR Operation Examples
```mlir
// Append key-value pairs to the cache
%new_kv = "llm.append_kv"(%kv_cache, %key, %value) {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!llm.paged_kv_cache, tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>) -> !llm.paged_kv_cache

// Lookup from the KV cache
%keys, %values = "llm.lookup_kv"(%kv_cache, %block_indices) {
  num_heads = 16 : i32,
  head_dim = 64 : i32
} : (!llm.paged_kv_cache, tensor<1x16xi32>) -> (tensor<1x16x16x64xf16>, tensor<1x16x16x64xf16>)

// Paged attention with KV cache
%output = "llm.paged_attention"(%query, %kv_cache, %block_indices) {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<1x1x16x64xf16>, !llm.paged_kv_cache, tensor<1x16xi32>) -> tensor<1x1x16x64xf16>
```

## 6. Optimization Opportunities

1. **Block Size Tuning**: Analyze optimal block sizes for different models and hardware
2. **Pre-allocation Strategies**: Different strategies for pre-allocating blocks
3. **Memory Layout**: Optimize memory layout for cache locality
4. **Quantization Support**: Add INT8/INT4 quantization for KV caches
5. **Attention Fusion**: Fuse attention computation with KV cache operations

## 7. Extension Points

1. **Alternative Strategies**: Support for continuous KV caches
2. **Multi-GPU Distribution**: Extend for multi-GPU and multi-node setups
3. **Memory-Efficient Attention**: Implement memory-efficient attention algorithms
4. **Eviction Policies**: Add eviction policies for constrained memory environments

## 8. Dependencies and Interoperability

The KV cache implementation will work with:
1. The LLM dialect operations
2. Existing MLIR tensor operations
3. Memory allocation on both CPU and GPU
4. Integration with attention computation modules 