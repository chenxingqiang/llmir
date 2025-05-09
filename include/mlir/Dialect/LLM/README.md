# LLM Dialect

The LLM (Large Language Model) dialect is an MLIR dialect designed to represent and optimize operations specific to large language model inference. It provides optimized operations, custom types, and abstractions tailored to efficient LLM deployment, focusing on memory usage and execution speed.

## Overview

Large language models have specific computational patterns and requirements that existing dialects don't directly address in an optimal way. This dialect provides:

1. Memory-optimized abstractions like paged KV caches
2. LLM-specific operations like optimized attention implementations
3. Support for quantization and tensor parallelism
4. Building blocks for efficient LLM serving

## Types

The LLM dialect defines the following custom types:

### PagedKVCache

```mlir
!llm.paged_kv_cache<elementType, numLayers, numHeads, headDim, blockSize, maxSeqLen>
```

A memory-efficient key-value cache storage for transformer models. It uses a paging mechanism similar to virtual memory systems to efficiently manage KV tensors in GPU memory.

Parameters:
- `elementType`: Storage type for the cached tensors (typically f16 or bf16)
- `numLayers`: Number of transformer layers in the model
- `numHeads`: Number of attention heads per layer
- `headDim`: Head dimension size 
- `blockSize`: Block size used for paging
- `maxSeqLen`: Maximum supported sequence length

### ShardedTensor

```mlir
!llm.sharded_tensor<originalType, shardDim, numShards, shardIndex>
```

Represents a tensor that is partitioned across multiple devices for tensor parallelism.

Parameters:
- `originalType`: The original tensor type before sharding
- `shardDim`: The dimension along which the tensor is sharded
- `numShards`: The total number of shards
- `shardIndex`: The shard index of this particular tensor

### QuantizedTensor 

```mlir
!llm.quantized_tensor<elementType, shape, isSymmetric, isPerChannel, quantAxis, groupSize, numBits>
```

Represents a tensor that has been quantized to a lower precision format with associated quantization parameters.

Parameters:
- `elementType`: Element type of the quantized tensor (e.g., i8, i4)
- `shape`: Original tensor shape
- `isSymmetric`: Whether the quantization is symmetric or asymmetric
- `isPerChannel`: Whether quantization is per-tensor or per-channel
- `quantAxis`: Axis for per-channel quantization
- `groupSize`: Group size for block-wise quantization
- `numBits`: Number of bits used for quantization

## Operations

### Attention Operations

- `llm.attention`: Standard scaled dot-product attention
- `llm.paged_attention`: Optimized attention for autoregressive decoding with KV cache

### KV Cache Management

- `llm.append_kv`: Append key-value pairs to a paged KV cache
- `llm.lookup_kv`: Retrieve key-value pairs from a paged KV cache

### Quantization Operations

- `llm.quantize`: Convert a floating-point tensor to a quantized representation
- `llm.dequantize`: Convert a quantized tensor back to floating-point
- `llm.quantized_matmul`: Optimized matrix multiplication with quantized weights

### Parallel Computation

- `llm.sharded_linear`: Linear layer with sharded weights for tensor parallelism
- `llm.all_gather`: Gather values from all shards
- `llm.reduce_scatter`: Reduce values across shards and scatter the results

## Example Usage

Here's a simple example showing the use of the paged attention operation:

```mlir
func.func @inference_step(%query: tensor<1x1x16x64xf16>, 
                     %kv_cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>,
                     %block_idxs: tensor<1xi32>, 
                     %seq_lens: tensor<1xi32>) -> tensor<1x1x16x64xf16> {
  %result = llm.paged_attention %query, %kv_cache, %block_idxs, %seq_lens { 
    scale = 0.125 : f32, causal = true 
  } : tensor<1x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, 
     tensor<1xi32>, tensor<1xi32> -> tensor<1x1x16x64xf16>
  return %result : tensor<1x1x16x64xf16>
}
```

Check the `examples/LLM` directory for more examples.

## Related Tools and Extensions

- **Optimization passes**: The dialect includes optimization passes specifically designed for LLM inference
- **Lowering and conversion patterns**: For integrating with other MLIR dialects and backends
- **Runtime support**: For efficiently executing LLM operations on various hardware targets

## References

- [PagedAttention: Memory-Efficient Inference with Trillion-Parameter LLMs](https://arxiv.org/abs/2309.06180)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [The paged attention paper](https://arxiv.org/abs/2309.06180) 