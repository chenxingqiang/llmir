# LLMIR Runtime Support Library

This directory contains runtime support libraries for the LLMIR compiler infrastructure,
including implementations for the PagedKVCache data structure used for optimizing
LLM inference.

## PagedKVCache

The `PagedKVCache` is a high-performance, memory-efficient implementation of the key-value
cache mechanism used in transformer-based language models. It is designed based on the
PagedAttention algorithm from vLLM, organizing memory in fixed-size blocks for efficient
memory management during autoregressive generation.

### Core Components

1. **KVBlock**
   - Represents a fixed-size memory block for storing key-value pairs
   - Manages memory for keys and values with reference counting
   - Tracks usage statistics for efficient memory management

2. **BlockAllocator**
   - Manages memory allocation and pooling of KV blocks
   - Provides efficient block reuse to minimize allocation overhead
   - Maintains separate allocators per transformer layer

3. **PagedKVCache**
   - Main API for interacting with the KV cache
   - Supports multi-layer, multi-head attention models
   - Efficiently tracks and manages sequences through block-based mapping

### Memory Management

The KV cache implementation uses a block-based approach to minimize memory fragmentation and
optimize cache usage. Instead of allocating memory per sequence, which can lead to inefficient
memory usage, it:

- Allocates fixed-size blocks of memory that can be shared across sequences
- Maintains a pool of blocks for reuse
- Tracks reference counts to efficiently free blocks when no longer needed
- Maps sequence tokens to block positions for efficient lookup

### Block Size Considerations

The block size is a critical parameter that affects memory efficiency and performance:

- Small block sizes (e.g., 16 tokens) reduce memory waste but may increase overhead
- Larger block sizes reduce allocation overhead but might waste memory
- Optimal block size depends on the distribution of sequence lengths
- The optimization pass `llm-optimize-kv-cache` can automatically adjust block sizes

### API Usage

#### Creating a PagedKVCache

```cpp
PagedKVCache cache(
    numLayers,    // Number of transformer layers
    numHeads,     // Number of attention heads per layer
    headDim,      // Dimension of each attention head
    blockSize,    // Size of each memory block (in tokens)
    maxSeqLen,    // Maximum sequence length
    elementType,  // Data type (e.g., f16, f32)
    useGPU        // Whether to use GPU memory
);
```

#### Appending Key-Value Pairs

```cpp
// For a batch of sequences
LogicalResult result = cache.appendKV(
    keyPtr,        // Pointer to key tensor data
    valuePtr,      // Pointer to value tensor data
    batchSize,     // Number of sequences in the batch
    seqLen,        // Number of tokens per sequence
    seqIds,        // Array of sequence IDs
    blockIndices   // Output array for block indices
);
```

#### Looking Up Key-Value Pairs

```cpp
// Retrieve KV pairs for a batch of sequences
LogicalResult result = cache.lookupKV(
    blockIndices,  // Array of block indices for each token
    seqLens,       // Array of sequence lengths
    batchSize,     // Number of sequences in the batch
    outputKeys,    // Output buffer for keys
    outputValues   // Output buffer for values
);
```

#### Managing Sequences

```cpp
// Clear a sequence when no longer needed
cache.clearSequence(seqId);

// Reset the entire cache
cache.reset();
```

### Performance Considerations

1. **Memory Efficiency**
   - Block-based allocation reduces fragmentation
   - Reference counting ensures timely release of memory
   - Pre-allocation of blocks reduces allocation overhead
   - Block pooling improves memory reuse

2. **GPU Optimization**
   - Memory operations are optimized for both CPU and GPU usage
   - Compatible with CUDA/HIP for GPU-accelerated LLM inference
   - Minimizes data transfers between CPU and GPU

3. **Scaling**
   - Efficiently handles batched inference across multiple sequences
   - Scales to support long context models
   - Supports variable-length sequences efficiently

### Integration with MLIR

The PagedKVCache runtime library is integrated with MLIR through operations defined
in the LLM dialect:

- `llm.append_kv`: Appends key-value pairs to the cache
- `llm.lookup_kv`: Retrieves key-value pairs from the cache
- `llm.paged_attention`: Performs attention computation using the paged cache

See the full example in `examples/kv_cache_example.cpp` for a complete usage demonstration. 