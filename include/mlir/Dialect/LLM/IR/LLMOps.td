//===- LLMOps.td - LLM dialect operation definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLM_OPS
#define LLM_OPS

include "mlir/Dialect/LLM/IR/LLMBase.td"
include "mlir/Dialect/LLM/IR/LLMTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def LLM_AttentionOp : LLM_Op<"attention", [
    Pure,
    TypesMatchWith<"result element type matches input element type",
                   "query", "result",
                   "$_self.cast<TensorType>().getElementType()">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Standard attention operation";
  let description = [{
    The `llm.attention` operation implements the standard scaled dot-product attention
    mechanism used in transformer models. It takes query, key, value tensors and an
    optional attention mask, and produces the attention output.
    
    The operation is defined as:
      output = softmax((Q * K^T) / sqrt(d_k)) * V
      
    Where Q is the query tensor, K is the key tensor, V is the value tensor,
    and d_k is the head dimension size.
  }];
  
  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$query,
    TensorOf<[F32, F16, BF16]>:$key,
    TensorOf<[F32, F16, BF16]>:$value,
    Optional<TensorOf<[I1]>>:$mask,
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<BoolAttr, "false">:$causal
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );
  
  let assemblyFormat = [{
    $query `,` $key `,` $value (`,` $mask^)? attr-dict `:` type($query) `,` type($key) `,` type($value) (`,` type($mask)^)? `->` type($result)
  }];
}

def LLM_PagedAttentionOp : LLM_Op<"paged_attention", [
    Pure,
    TypesMatchWith<"result element type matches query element type",
                   "query", "result",
                   "$_self.cast<TensorType>().getElementType()">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Memory-efficient attention with paged KV cache";
  let description = [{
    The `llm.paged_attention` operation implements an optimized version of the attention
    mechanism for autoregressive decoding. It uses a paged KV cache to efficiently
    handle variable-length sequences without wasting memory.
    
    Instead of processing the entire key and value tensors, it retrieves only the
    required blocks from the KV cache based on the current sequence position.
  }];
  
  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$query,
    LLM_PagedKVCacheType:$kvCache,
    TensorOf<[I32]>:$blockIndexes,
    TensorOf<[I32]>:$seqLens,
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<BoolAttr, "true">:$causal
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );
  
  let assemblyFormat = [{
    $query `,` $kvCache `,` $blockIndexes `,` $seqLens attr-dict `:` 
    type($query) `,` type($kvCache) `,` type($blockIndexes) `,` type($seqLens) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// KV Cache Management Operations
//===----------------------------------------------------------------------===//

def LLM_AppendKVOp : LLM_Op<"append_kv", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Append key-value pairs to the KV cache";
  let description = [{
    The `llm.append_kv` operation adds new key-value pairs to the paged KV cache.
    It allocates new blocks if necessary and updates the block index mapping.
    
    This operation is typically used during autoregressive decoding, when new tokens
    are generated and their key-value representations need to be stored.
  }];
  
  let arguments = (ins
    LLM_PagedKVCacheType:$kvCache,
    TensorOf<[F32, F16, BF16]>:$key,
    TensorOf<[F32, F16, BF16]>:$value,
    TensorOf<[I32]>:$seqIds
  );
  
  let results = (outs
    LLM_PagedKVCacheType:$updatedCache,
    TensorOf<[I32]>:$blockIndexes
  );
  
  let assemblyFormat = [{
    $kvCache `,` $key `,` $value `,` $seqIds attr-dict `:` 
    type($kvCache) `,` type($key) `,` type($value) `,` type($seqIds) `->` 
    type($updatedCache) `,` type($blockIndexes)
  }];
}

def LLM_LookupKVOp : LLM_Op<"lookup_kv", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Lookup key-value pairs from the KV cache";
  let description = [{
    The `llm.lookup_kv` operation retrieves key-value tensors from the paged KV cache
    based on the provided block indexes. It's used to access previously computed
    key-value pairs during attention computation.
  }];
  
  let arguments = (ins
    LLM_PagedKVCacheType:$kvCache,
    TensorOf<[I32]>:$blockIndexes,
    TensorOf<[I32]>:$seqLens
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$keys,
    TensorOf<[F32, F16, BF16]>:$values
  );
  
  let assemblyFormat = [{
    $kvCache `,` $blockIndexes `,` $seqLens attr-dict `:` 
    type($kvCache) `,` type($blockIndexes) `,` type($seqLens) `->` 
    type($keys) `,` type($values)
  }];
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def LLM_QuantizeOp : LLM_Op<"quantize", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Quantize a tensor to a lower-precision format";
  let description = [{
    The `llm.quantize` operation converts a floating-point tensor to a lower-precision
    quantized representation. It applies scaling and optional zero-point adjustment
    based on the quantization parameters.
    
    This operation supports various quantization schemes including symmetric and
    asymmetric quantization, per-tensor and per-channel quantization, and group-wise
    quantization.
  }];
  
  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$input,
    TensorOf<[F32]>:$scales,
    Optional<TensorOf<[I8, I16, I32]>>:$zeroPoints,
    DefaultValuedAttr<I32Attr, "8">:$bits,
    DefaultValuedAttr<BoolAttr, "true">:$symmetric,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<I64Attr, "128">:$groupSize
  );
  
  let results = (outs
    LLM_QuantizedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input `,` $scales (`,` $zeroPoints^)? attr-dict `:` 
    type($input) `,` type($scales) (`,` type($zeroPoints)^)? `->` type($result)
  }];
}

def LLM_DequantizeOp : LLM_Op<"dequantize", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Dequantize a tensor back to floating-point format";
  let description = [{
    The `llm.dequantize` operation converts a quantized tensor back to floating-point
    format by applying the inverse of the quantization transformation. It uses the
    quantization parameters stored in the quantized tensor.
  }];
  
  let arguments = (ins
    LLM_QuantizedTensorType:$input,
    TensorOf<[F32]>:$scales,
    Optional<TensorOf<[I8, I16, I32]>>:$zeroPoints
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );
  
  let assemblyFormat = [{
    $input `,` $scales (`,` $zeroPoints^)? attr-dict `:` 
    type($input) `,` type($scales) (`,` type($zeroPoints)^)? `->` type($result)
  }];
}

def LLM_QuantizedMatMulOp : LLM_Op<"quantized_matmul", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Matrix multiplication with quantized weights";
  let description = [{
    The `llm.quantized_matmul` operation performs matrix multiplication where one of
    the inputs (typically the weights) is in a quantized format. It dequantizes on-the-fly
    during computation, which is more efficient than explicitly dequantizing the entire
    weight matrix.
    
    This operation is commonly used for optimized linear layers in LLM inference.
  }];
  
  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$lhs,
    LLM_QuantizedTensorType:$rhs,
    TensorOf<[F32]>:$scales,
    Optional<TensorOf<[I8, I16, I32]>>:$zeroPoints
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$result
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $scales (`,` $zeroPoints^)? attr-dict `:` 
    type($lhs) `,` type($rhs) `,` type($scales) (`,` type($zeroPoints)^)? `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Parallel Computation Operations
//===----------------------------------------------------------------------===//

def LLM_ShardedLinearOp : LLM_Op<"sharded_linear", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Linear layer with sharded weights";
  let description = [{
    The `llm.sharded_linear` operation represents a linear (fully-connected) layer
    where the weights are sharded across multiple devices for tensor parallelism.
    Each device computes only a portion of the output based on its shard of the weights.
    
    This operation is typically followed by an all-gather or reduce-scatter operation
    to combine the partial results.
  }];
  
  let arguments = (ins
    TensorOf<[F32, F16, BF16]>:$input,
    TensorOf<[F32, F16, BF16]>:$weight,
    Optional<TensorOf<[F32, F16, BF16]>>:$bias,
    I64Attr:$shardDim,
    I64Attr:$numShards,
    I64Attr:$shardId
  );
  
  let results = (outs
    TensorOf<[F32, F16, BF16]>:$output
  );
  
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:` 
    type($input) `,` type($weight) (`,` type($bias)^)? `->` type($output)
  }];
}

def LLM_AllGatherOp : LLM_Op<"all_gather", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Gather values from all shards";
  let description = [{
    The `llm.all_gather` operation collects tensor values from all shards and
    concatenates them along the specified dimension. It's a collective operation
    that requires synchronization across devices.
    
    This is commonly used after sharded computations to reconstruct the full tensor.
  }];
  
  let arguments = (ins
    TensorOf<[AnyType]>:$input,
    I64Attr:$dim,
    I64Attr:$groupSize
  );
  
  let results = (outs
    TensorOf<[AnyType]>:$output
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def LLM_ReduceScatterOp : LLM_Op<"reduce_scatter", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Reduce values across shards and scatter results";
  let description = [{
    The `llm.reduce_scatter` operation performs a reduction (e.g., sum) across
    corresponding values from all shards, then distributes (scatters) the results
    so that each shard gets a portion of the reduced tensor.
    
    This is used in distributed computations to combine results while minimizing
    communication overhead.
  }];
  
  let arguments = (ins
    TensorOf<[AnyType]>:$input,
    I64Attr:$dim,
    I64Attr:$groupSize,
    StrAttr:$reduceOp
  );
  
  let results = (outs
    TensorOf<[AnyType]>:$output
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

#endif // LLM_OPS 