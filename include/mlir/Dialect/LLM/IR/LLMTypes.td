//===- LLMTypes.td - LLM dialect type definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLM_TYPES
#define LLM_TYPES

include "mlir/Dialect/LLM/IR/LLMBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// LLM Type Definitions
//===----------------------------------------------------------------------===//

class LLM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PagedKVCache Type
//===----------------------------------------------------------------------===//

def LLM_PagedKVCacheType : LLM_Type<"PagedKVCache", "paged_kv_cache"> {
  let summary = "Type representing a paged KV cache for efficient attention";
  let description = [{
    The `llm.paged_kv_cache` type represents a memory-efficient key-value cache
    used in Transformer model inference. It uses a paging mechanism similar to OS
    virtual memory systems, where KV tensors are divided into blocks that can be
    efficiently managed in GPU memory.
    
    This type stores the following information:
    - Element type of the cached tensors (typically f16 or bf16)
    - Number of layers in the model
    - Number of attention heads
    - Head dimension size
    - Block size used for paging
    - Maximum supported sequence length
  }];
  
  let parameters = (ins
    "mlir::Type":$elementType,
    "int64_t":$numLayers,
    "int64_t":$numHeads,
    "int64_t":$headDim,
    "int64_t":$blockSize,
    "int64_t":$maxSeqLen
  );
  
  let assemblyFormat = "`<` $elementType `,` $numLayers `,` $numHeads `,` "
                       "$headDim `,` $blockSize `,` $maxSeqLen `>`";
}

//===----------------------------------------------------------------------===//
// ShardedTensor Type
//===----------------------------------------------------------------------===//

def LLM_ShardedTensorType : LLM_Type<"ShardedTensor", "sharded_tensor"> {
  let summary = "Type representing a tensor sharded across devices";
  let description = [{
    The `llm.sharded_tensor` type represents a tensor that is partitioned across
    multiple devices for parallel computation. This is commonly used in tensor
    parallelism strategies for large language models.
    
    This type stores the following information:
    - Original tensor type before sharding
    - Sharding dimension
    - Number of shards
    - Shard index (which portion this particular tensor represents)
  }];
  
  let parameters = (ins
    "mlir::TensorType":$originalType,
    "int64_t":$shardDim,
    "int64_t":$numShards,
    "int64_t":$shardIndex
  );
  
  let assemblyFormat = "`<` $originalType `,` $shardDim `,` "
                       "$numShards `,` $shardIndex `>`";
}

//===----------------------------------------------------------------------===//
// QuantizedTensor Type
//===----------------------------------------------------------------------===//

def LLM_QuantizedTensorType : LLM_Type<"QuantizedTensor", "quantized_tensor"> {
  let summary = "Type representing a quantized tensor with associated quantization parameters";
  let description = [{
    The `llm.quantized_tensor` type represents a tensor that has been quantized
    to a lower precision format (e.g., int8, int4) with associated quantization
    parameters. This is used for efficient storage and computation in LLM inference.
    
    This type stores the following information:
    - Element type of the quantized tensor (e.g., i8, i4)
    - Original tensor shape
    - Quantization scheme (symmetric vs asymmetric)
    - Scale type (per-tensor vs per-channel)
    - Quantization axis (for per-channel quantization)
    - Group size (for block-wise quantization)
    - Number of bits used for quantization
  }];
  
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::ArrayRef<int64_t>":$shape,
    "bool":$isSymmetric,
    "bool":$isPerChannel,
    "int64_t":$quantAxis,
    "int64_t":$groupSize,
    "int64_t":$numBits
  );
  
  let assemblyFormat = "`<` $elementType `,` custom<DimensionList>($shape) `,` "
                       "$isSymmetric `,` $isPerChannel `,` $quantAxis `,` "
                       "$groupSize `,` $numBits `>`";
}

#endif // LLM_TYPES 