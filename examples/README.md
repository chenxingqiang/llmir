# LLMIR Examples

This directory contains example code demonstrating how to use various components of the LLMIR (Large Language Model Intermediate Representation) system.

## KV Cache Example

The file `kv_cache_example.cpp` demonstrates how to use the PagedKVCache runtime support library, which is a key component for optimizing LLM inference. This example shows how to:

1. Create and configure a PagedKVCache
2. Append key-value pairs for multiple sequences
3. Perform lookups for efficient autoregressive generation
4. Manage memory with block-based allocation

### Building the Example

```bash
# From the build directory
cmake -G Ninja ..
ninja examples/kv_cache_example

# Run the example
./bin/kv_cache_example
```

### Key Concepts Demonstrated

1. **Block-based Memory Management**
   - Efficient memory allocation using fixed-size blocks
   - Memory pooling and reuse

2. **Multi-sequence Management**
   - Support for batched inference
   - Independent sequence tracking

3. **Autoregressive Generation**
   - Simulates token-by-token generation
   - Shows pattern of append, lookup, and update

### Expected Output

The example demonstrates:
- Creation of a PagedKVCache with configurable parameters
- Generation of 10 tokens for 2 sequences
- Lookup of the entire generated sequences
- Memory usage statistics
- Sequence cleanup and cache management

### Integration with MLIR

This example shows the runtime implementation of features exposed through MLIR operations:
- `llm.append_kv`
- `llm.lookup_kv`
- `llm.paged_attention`

For examples of these operations in MLIR, see the tests in `test/Dialect/LLM/`.

## C++ Runtime Algorithm Test

The files `test_llmir_cpp.cpp` and `test_llmir_runtime.exp` provide standalone C++ tests for LLMIR runtime algorithms (attention, KV cache) without MLIR dependencies. Use with your build system or compile manually.

## Other Examples

More examples demonstrating other aspects of LLMIR will be added in the future, including:

- Attention mechanism optimization
- Integration with MLIR compilation pipelines
- Quantization support
- Custom hardware acceleration 