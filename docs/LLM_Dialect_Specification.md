# LLMIR Dialect Formal Specification

## 1. Overview

The **LLM Dialect** is a specialized MLIR dialect designed to represent and optimize Large Language Model (LLM) inference workloads. This document provides a formal specification of the dialect's types, operations, and semantic invariants.

### 1.1 Design Rationale

The LLM dialect provides high-level abstractions that capture LLM-specific computation patterns, enabling:

1. **IR-Level PagedAttention**: First compiler-level representation of PagedAttention, enabling static analysis of dynamic memory patterns
2. **KV Cache Management**: Explicit representation of key-value cache operations with block-based memory management
3. **Quantization Support**: Native support for mixed-precision computation with quantized tensors
4. **Parallel Computation**: Built-in support for tensor and pipeline parallelism

### 1.2 Dialect Namespace

All types and operations in this dialect use the `llm` namespace prefix:
- Types: `!llm.paged_kv_cache`, `!llm.sharded_tensor`, `!llm.quantized_tensor`
- Operations: `llm.attention`, `llm.paged_attention`, `llm.append_kv`, etc.

---

## 2. Type System

### 2.1 PagedKVCacheType

The `PagedKVCacheType` represents a paged key-value cache for efficient attention computation during autoregressive decoding.

#### Syntax

```mlir
!llm.paged_kv_cache<element_type, num_layers, num_heads, head_dim, block_size, max_seq_len>
```

#### Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `element_type` | `mlir::Type` | Element type for cached tensors | Must be `f16`, `bf16`, or `f32` |
| `num_layers` | `int64_t` | Number of transformer layers | > 0 |
| `num_heads` | `int64_t` | Number of attention heads | > 0 |
| `head_dim` | `int64_t` | Dimension of each attention head | > 0 |
| `block_size` | `int64_t` | Number of tokens per memory block | > 0, typically 16, 32, 64, 128, or 256 |
| `max_seq_len` | `int64_t` | Maximum supported sequence length | > 0 |

#### Semantic Invariants

1. **Memory Layout**: Each block stores `block_size` tokens, organized as:
   ```
   Block[layer][head] = tensor<block_size × head_dim × element_type>
   ```

2. **Block Count**: Maximum blocks per sequence = ⌈max_seq_len / block_size⌉

3. **Memory Bound**: Total KV cache memory ≤ 
   ```
   2 × num_layers × num_heads × max_blocks × block_size × head_dim × sizeof(element_type)
   ```

#### Example

```mlir
// KV cache for LLaMA-7B: 32 layers, 32 heads, 128 head_dim, block size 16, max 8192 tokens
!llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>
```

---

### 2.2 ShardedTensorType

The `ShardedTensorType` represents a tensor partitioned across multiple devices for tensor parallelism.

#### Syntax

```mlir
!llm.sharded_tensor<original_type, shard_dim, num_shards, shard_index>
```

#### Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `original_type` | `mlir::TensorType` | Original tensor type before sharding | Valid tensor type |
| `shard_dim` | `int64_t` | Dimension along which tensor is sharded | 0 ≤ shard_dim < rank |
| `num_shards` | `int64_t` | Total number of shards | > 0 |
| `shard_index` | `int64_t` | Index of this particular shard | 0 ≤ shard_index < num_shards |

#### Semantic Invariants

1. **Dimension Divisibility**: `original_shape[shard_dim] % num_shards == 0`

2. **Shard Shape**: The actual shard shape is computed as:
   ```
   shard_shape[i] = original_shape[i]                          if i ≠ shard_dim
   shard_shape[i] = original_shape[i] / num_shards             if i == shard_dim
   ```

3. **Reconstruction**: Concatenating all shards along `shard_dim` reconstructs the original tensor.

#### Example

```mlir
// 4096x4096 weight matrix sharded column-wise across 4 GPUs, this is shard 2
!llm.sharded_tensor<tensor<4096x4096xf16>, 1, 4, 2>
// Actual shard shape: tensor<4096x1024xf16>
```

---

### 2.3 QuantizedTensorType

The `QuantizedTensorType` represents a quantized tensor with associated quantization parameters.

#### Syntax

```mlir
!llm.quantized_tensor<element_type, shape, is_symmetric, is_per_channel, quant_axis, group_size, num_bits>
```

#### Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `element_type` | `mlir::Type` | Quantized element type | `i4`, `i8`, `i16` |
| `shape` | `ArrayRef<int64_t>` | Tensor shape | Valid shape |
| `is_symmetric` | `bool` | Whether quantization is symmetric | - |
| `is_per_channel` | `bool` | Whether scales are per-channel | - |
| `quant_axis` | `int64_t` | Axis for per-channel quantization | -1 for per-tensor, else 0 ≤ axis < rank |
| `group_size` | `int64_t` | Group size for block-wise quantization | -1 for no grouping, else > 0 |
| `num_bits` | `int64_t` | Number of bits for quantization | 4, 8, or 16 |

#### Quantization Formula

For **symmetric quantization**:
```
quantized_value = round(real_value / scale)
real_value = quantized_value × scale
```

For **asymmetric quantization**:
```
quantized_value = round(real_value / scale) + zero_point
real_value = (quantized_value - zero_point) × scale
```

#### Example

```mlir
// INT8 symmetric per-channel quantized weight matrix
!llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>

// INT4 group-wise quantized with group size 128
!llm.quantized_tensor<i4, [4096, 4096], true, false, -1, 128, 4>
```

---

## 3. Operations

### 3.1 Attention Operations

#### 3.1.1 `llm.attention`

Standard scaled dot-product attention operation.

**Syntax:**
```mlir
%result = llm.attention %query, %key, %value [, %mask] 
    { scale = <f32>, causal = <bool> } 
    : tensor_type, tensor_type, tensor_type [, mask_type] -> tensor_type
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `query` | `tensor<batch × seq_q × heads × head_dim × element>` | Query tensor |
| `key` | `tensor<batch × seq_kv × heads × head_dim × element>` | Key tensor |
| `value` | `tensor<batch × seq_kv × heads × head_dim × element>` | Value tensor |
| `mask` | `tensor<batch × heads × seq_q × seq_kv × i1>` (optional) | Attention mask |

**Attributes:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scale` | `f32` | `1/√head_dim` | Scaling factor for attention scores |
| `causal` | `bool` | `false` | Apply causal masking |

**Semantics:**

```
scores = (query @ key^T) × scale
if mask is provided:
    scores = scores + (mask ? 0 : -∞)
if causal:
    scores[i,j] = -∞ where j > i
attention_weights = softmax(scores, axis=-1)
result = attention_weights @ value
```

**Example:**
```mlir
%output = llm.attention %query, %key, %value { scale = 0.125 : f32, causal = true } 
    : tensor<2x512x32x128xf16>, tensor<2x512x32x128xf16>, tensor<2x512x32x128xf16> 
    -> tensor<2x512x32x128xf16>
```

---

#### 3.1.2 `llm.paged_attention`

Memory-efficient attention using paged KV cache for autoregressive decoding.

**Syntax:**
```mlir
%result = llm.paged_attention %query, %kv_cache, %block_indices, %seq_lens 
    { num_heads = <i32>, head_dim = <i32>, scale = <f32> } 
    : (tensor_type, kv_cache_type, index_type, len_type) -> tensor_type
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `query` | `tensor<batch × seq_q × heads × head_dim × element>` | Query tensor (typically seq_q=1 for decoding) |
| `kv_cache` | `!llm.paged_kv_cache<...>` | Paged KV cache |
| `block_indices` | `tensor<batch × max_blocks × i32>` | Block indices for each sequence |
| `seq_lens` | `tensor<batch × i32>` | Current sequence lengths |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `num_heads` | `i32` | Number of attention heads |
| `head_dim` | `i32` | Head dimension |
| `scale` | `f32` | Attention scale factor |

**Semantics:**

1. For each sequence in the batch:
   - Retrieve K,V from `kv_cache` using `block_indices`
   - Compute attention only up to `seq_lens[i]` tokens
   - Return attention output

**Compile-Time Optimizations:**

The IR representation enables:
- **Block Prefetching**: Compiler can analyze access patterns and insert prefetch instructions
- **Memory Planning**: Static analysis of maximum memory requirements
- **Kernel Selection**: Choose optimal kernel based on sequence length bounds

**Example:**
```mlir
%output = llm.paged_attention %query, %kv_cache, %block_indices, %seq_lens {
    num_heads = 32 : i32,
    head_dim = 128 : i32,
    scale = 0.0883883476 : f32
} : (tensor<4x1x32x128xf16>, !llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>, 
     tensor<4x512xi32>, tensor<4xi32>) -> tensor<4x1x32x128xf16>
```

---

### 3.2 KV Cache Operations

#### 3.2.1 `llm.append_kv`

Append new key-value pairs to the paged KV cache.

**Syntax:**
```mlir
%updated_cache, %block_indices = llm.append_kv(%keys, %values, %seq_ids [, %kv_cache])
    : (tensor_type, tensor_type, tensor_type [, kv_cache_type]) -> (kv_cache_type, tensor_type)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `keys` | `tensor<batch × seq × heads × head_dim × element>` | Keys to append |
| `values` | `tensor<batch × seq × heads × head_dim × element>` | Values to append |
| `seq_ids` | `tensor<batch × i32>` | Sequence identifiers |
| `kv_cache` | `!llm.paged_kv_cache<...>` (optional) | Existing cache (creates new if absent) |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| `updated_cache` | `!llm.paged_kv_cache<...>` | Updated KV cache |
| `block_indices` | `tensor<batch × seq × i32>` | Block indices for appended tokens |

**Semantics:**

1. Allocate new blocks if necessary
2. Copy K,V data into allocated blocks
3. Update block mapping for each sequence
4. Return updated cache and block indices

**Example:**
```mlir
%new_cache, %indices = llm.append_kv(%keys, %values, %seq_ids, %cache)
    : (tensor<4x1x32x128xf16>, tensor<4x1x32x128xf16>, tensor<4xi32>, 
       !llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>) 
    -> (!llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>, tensor<4x1xi32>)
```

---

#### 3.2.2 `llm.lookup_kv`

Retrieve key-value pairs from the paged KV cache.

**Syntax:**
```mlir
%keys, %values = llm.lookup_kv(%block_indices, %seq_lens, %kv_cache)
    : (tensor_type, tensor_type, kv_cache_type) -> (tensor_type, tensor_type)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `block_indices` | `tensor<batch × max_blocks × i32>` | Block indices to retrieve |
| `seq_lens` | `tensor<batch × i32>` | Actual sequence lengths |
| `kv_cache` | `!llm.paged_kv_cache<...>` | KV cache to read from |

**Example:**
```mlir
%keys, %values = llm.lookup_kv(%indices, %seq_lens, %cache)
    : (tensor<4x128xi32>, tensor<4xi32>, !llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>) 
    -> (tensor<4x128x32x128xf16>, tensor<4x128x32x128xf16>)
```

---

### 3.3 Quantization Operations

#### 3.3.1 `llm.quantize`

Quantize a floating-point tensor to lower precision.

**Syntax:**
```mlir
%quantized = llm.quantize %input, %scales [, %zero_points] 
    { bits = <i32>, symmetric = <bool>, axis = <i64>, group_size = <i64> }
    : tensor_type, scale_type [, zp_type] -> quantized_tensor_type
```

**Quantization Safety Analysis:**

The compiler performs static analysis to determine safe quantization:

1. **Range Analysis**: Compute min/max bounds of tensor values
2. **Sensitivity Analysis**: Identify operations sensitive to quantization error
3. **Error Propagation**: Track quantization error through computation graph

Safe quantization conditions:
- Weight matrices in linear layers: Generally safe for INT8/INT4
- Attention logits before softmax: Requires higher precision (FP16/FP32)
- Normalization layers: Requires FP16 or higher
- Residual connections: Accumulated quantization error consideration

**Example:**
```mlir
// Symmetric INT8 per-channel quantization
%q_weights = llm.quantize %weights, %scales { bits = 8, symmetric = true, axis = 0, group_size = -1 }
    : tensor<4096x4096xf16>, tensor<4096xf32> -> !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>
```

---

#### 3.3.2 `llm.dequantize`

Convert quantized tensor back to floating-point.

**Syntax:**
```mlir
%result = llm.dequantize %input, %scales [, %zero_points]
    : quantized_tensor_type, scale_type [, zp_type] -> tensor_type
```

---

#### 3.3.3 `llm.quantized_matmul`

Matrix multiplication with on-the-fly dequantization.

**Syntax:**
```mlir
%result = llm.quantized_matmul %lhs, %rhs, %scales [, %zero_points]
    : tensor_type, quantized_tensor_type, scale_type [, zp_type] -> tensor_type
```

**Optimization:**

This fused operation avoids materializing the full dequantized weight matrix:
- Memory bandwidth: O(M×N×bits/8) instead of O(M×N×sizeof(float))
- On-the-fly dequantization during matmul computation

**Example:**
```mlir
%output = llm.quantized_matmul %activations, %q_weights, %scales
    : tensor<4x2048xf16>, !llm.quantized_tensor<i8, [2048, 4096], true, true, 0, -1, 8>, 
      tensor<4096xf32> -> tensor<4x4096xf16>
```

---

### 3.4 Parallel Computation Operations

#### 3.4.1 `llm.sharded_linear`

Linear layer with tensor-parallel sharded weights.

**Syntax:**
```mlir
%output = llm.sharded_linear %input, %weight [, %bias] 
    { shard_dim = <i64>, num_shards = <i64>, shard_id = <i64> }
    : tensor_type, tensor_type [, tensor_type] -> tensor_type
```

**Sharding Strategies:**

1. **Column Parallelism** (`shard_dim = 1`):
   - Each device computes a portion of output features
   - Followed by `all_gather` to collect full output

2. **Row Parallelism** (`shard_dim = 0`):
   - Each device holds portion of input features
   - Followed by `reduce_scatter` for aggregation

**Example:**
```mlir
// Column-parallel MLP: each of 4 GPUs computes 1/4 of output
%partial = llm.sharded_linear %input, %weight { shard_dim = 1, num_shards = 4, shard_id = 0 }
    : tensor<4x4096xf16>, tensor<4096x4096xf16> -> tensor<4x1024xf16>
```

---

#### 3.4.2 `llm.all_gather`

Gather tensor values from all devices.

**Syntax:**
```mlir
%full = llm.all_gather %partial { dim = <i64>, group_size = <i64> }
    : tensor_type -> tensor_type
```

---

#### 3.4.3 `llm.reduce_scatter`

Reduce across devices and scatter results.

**Syntax:**
```mlir
%scattered = llm.reduce_scatter %input { dim = <i64>, group_size = <i64>, reduce_op = "<op>" }
    : tensor_type -> tensor_type
```

**Supported Reduction Operations:** `"sum"`, `"mean"`, `"max"`, `"min"`

---

## 4. Verifier Rules

### 4.1 Type Compatibility

1. **Element Type Consistency**: Query, Key, Value must have matching element types
2. **Shape Compatibility**: Batch and head dimensions must match across attention inputs
3. **KV Cache Type Matching**: Block indices must be compatible with KV cache dimensions

### 4.2 Attribute Constraints

1. **Block Size**: Must be power of 2 for efficient memory alignment
2. **Scale Factor**: Must be positive for attention operations
3. **Quantization Bits**: Must be 4, 8, or 16

### 4.3 Static Analysis Requirements

Operations must satisfy these invariants for static analysis:
1. **Bounded Memory**: Maximum memory usage must be statically determinable
2. **Deterministic Execution**: No data-dependent control flow in core operations
3. **Type Safety**: All type conversions must be explicit

---

## 5. Canonicalization Patterns

### 5.1 Attention Fusion

```mlir
// Pattern: Fuse append_kv followed by paged_attention
%cache2, %indices = llm.append_kv(%keys, %values, %seq_ids, %cache1)
%output = llm.paged_attention %query, %cache2, %indices, %seq_lens

// Fuses to:
%output, %cache2, %indices = llm.fused_attention(%query, %keys, %values, %seq_ids, %cache1, %seq_lens)
```

### 5.2 Quantization Fusion

```mlir
// Pattern: Fuse dequantize into matmul
%deq = llm.dequantize %q_weights, %scales
%output = linalg.matmul %input, %deq

// Fuses to:
%output = llm.quantized_matmul %input, %q_weights, %scales
```

### 5.3 Dead Code Elimination

```mlir
// Remove unused KV cache lookups
%keys, %values = llm.lookup_kv(%indices, %seq_lens, %cache)
// %keys, %values have no uses -> eliminated
```

---

## 6. Lowering Specifications

### 6.1 Lowering to Runtime Calls

LLM dialect operations lower to runtime library calls:

| Operation | Runtime Function |
|-----------|-----------------|
| `llm.append_kv` | `mlir_llm_append_kv()` |
| `llm.lookup_kv` | `mlir_llm_lookup_kv()` |
| `llm.paged_attention` | `mlir_llm_paged_attention()` |

### 6.2 Lowering to GPU Kernels

For CUDA targets, operations lower through:
1. **LLM dialect** → **Linalg/Tensor dialects** → **GPU dialect** → **NVVM/PTX**

Key kernel variants:
- `flash_paged_attention`: FlashAttention-style fused kernel for long sequences
- `standard_paged_attention`: Standard attention for short sequences
- `chunked_paged_attention`: Chunked processing for memory-constrained scenarios

---

## 7. Interface Definitions

### 7.1 KVCacheInterface

```cpp
interface LLM_KVCacheInterface {
  // Returns true if this operation uses a KV cache
  bool usesKVCache();
  
  // Returns the number of new KV tokens this operation adds
  int64_t getNumKVTokens();
  
  // Returns the KV cache input value (if any)
  mlir::Value getKVCacheInput();
  
  // Returns the KV cache output value (if any)
  mlir::Value getKVCacheOutput();
}
```

### 7.2 AttentionInterface

```cpp
interface LLM_AttentionInterface {
  // Returns true if this is an attention operation
  bool isAttentionOp();
  
  // Returns batch size
  int64_t getBatchSize();
  
  // Returns sequence length
  int64_t getSeqLength();
  
  // Returns number of attention heads
  int64_t getNumHeads();
  
  // Returns head dimension
  int64_t getHeadDim();
}
```

---

## 8. Examples

### 8.1 Complete Transformer Layer

```mlir
func.func @transformer_layer(%input: tensor<4x512x4096xf16>, 
                              %kv_cache: !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
                              %block_indices: tensor<4x32xi32>,
                              %seq_lens: tensor<4xi32>,
                              %q_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
                              %k_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
                              %v_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
                              %o_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
                              %scales: tensor<4096xf32>) 
    -> (tensor<4x512x4096xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>) {
  
  // Compute Q, K, V projections with quantized weights
  %query = llm.quantized_matmul %input, %q_proj, %scales : ... -> tensor<4x512x4096xf16>
  %keys = llm.quantized_matmul %input, %k_proj, %scales : ... -> tensor<4x512x4096xf16>
  %values = llm.quantized_matmul %input, %v_proj, %scales : ... -> tensor<4x512x4096xf16>
  
  // Reshape for multi-head attention: [batch, seq, heads, head_dim]
  %query_mha = tensor.reshape %query ... : tensor<4x512x32x128xf16>
  %keys_mha = tensor.reshape %keys ... : tensor<4x512x32x128xf16>
  %values_mha = tensor.reshape %values ... : tensor<4x512x32x128xf16>
  
  // Append K, V to cache
  %new_cache, %new_indices = llm.append_kv(%keys_mha, %values_mha, %seq_ids, %kv_cache) : ...
  
  // Compute attention using paged KV cache
  %attn_out = llm.paged_attention %query_mha, %new_cache, %new_indices, %seq_lens {
    num_heads = 32, head_dim = 128, scale = 0.0883883476
  } : ...
  
  // Output projection
  %attn_flat = tensor.reshape %attn_out ... : tensor<4x512x4096xf16>
  %output = llm.quantized_matmul %attn_flat, %o_proj, %scales : ... -> tensor<4x512x4096xf16>
  
  return %output, %new_cache : tensor<4x512x4096xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>
}
```

---

## 9. References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
2. MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
3. FlashAttention: Fast and Memory-Efficient Exact Attention, Dao et al., NeurIPS 2022
4. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, Frantar et al., ICLR 2023
