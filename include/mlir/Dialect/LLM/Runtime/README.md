# LLM Runtime Support Library

This directory contains the runtime support library for the LLM dialect. The runtime library provides efficient implementations of key components needed for LLM inference, with a focus on optimizing memory usage and computation performance.

## Key-Value Cache Implementation

The KV cache is a critical component for efficient LLM inference, enabling reuse of previously computed key and value tensors across multiple decoding steps. This implementation is inspired by the PagedAttention mechanism from vLLM, which uses a block-based memory management approach to efficiently handle varying sequence lengths.

### Core Components

#### `KVBlock`

Represents a single block of memory containing key-value pairs for a fixed number of tokens. Each block has:
- Pointers to key and value memory regions
- Information about block size and dimensions
- Support for both CPU and GPU memory

#### `BlockAllocator`

Manages a pool of KV blocks for efficient memory allocation:
- Pre-allocates blocks to avoid frequent memory allocations
- Tracks used and free blocks
- Provides efficient block allocation and deallocation

#### `PagedKVCache`

Provides the main API for interacting with the paged KV cache:
- Manages multiple layers of KV caches for transformer models
- Maps sequence IDs to their respective blocks
- Handles appending new KV pairs and looking up existing ones
- Supports efficient batch processing

### Memory Management Strategy

The library implements a block-based memory management approach:
- Memory is allocated in fixed-size blocks (e.g., 16 tokens per block)
- Blocks can be dynamically allocated and freed
- Sequences can span multiple blocks

This strategy provides several advantages:
1. Memory efficiency - only allocate what's needed
2. Support for variable sequence lengths
3. Reduced memory fragmentation
4. Efficient memory reuse

### API and Operations

The KV cache library supports the following operations:

1. **Append KV Pairs**: Add new key-value pairs to the cache
   - Input: New key-value tensors, batch information
   - Output: Block indices for the new tokens

2. **Lookup KV Pairs**: Retrieve key-value pairs from the cache
   - Input: Block indices, sequence lengths
   - Output: Retrieved key-value tensors

### Future Enhancements

1. **GPU Support**: Full implementation of GPU memory allocation and operations
2. **Quantization**: Support for quantized KV caches (INT8/INT4)
3. **Advanced Memory Strategies**: Implement eviction policies for constrained memory environments
4. **Continuous KV Cache**: Alternative implementation for scenarios where block-based approach is suboptimal
5. **Distributed KV Cache**: Support for multi-GPU and multi-node setups

## Usage Example

In MLIR, operations can use these runtime components through the defined interfaces:

```mlir
// Append key-value pairs to the cache
%new_kv = "llm.append_kv"(%kv_cache, %key, %value) {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!llm.paged_kv_cache, tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>) -> !llm.paged_kv_cache

// Perform paged attention with KV cache
%output = "llm.paged_attention"(%query, %new_kv, %block_indices) {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<1x1x16x64xf16>, !llm.paged_kv_cache, tensor<1x16xi32>) -> tensor<1x1x16x64xf16>
``` 