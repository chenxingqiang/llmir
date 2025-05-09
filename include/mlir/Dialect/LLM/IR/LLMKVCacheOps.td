//===- LLMKVCacheOps.td - LLM dialect KV cache ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines KV cache related operations in the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_IR_LLMKVCACHEOPS
#define MLIR_DIALECT_LLM_IR_LLMKVCACHEOPS

include "mlir/Dialect/LLM/IR/LLMBase.td"
include "mlir/Dialect/LLM/IR/LLMTypes.td"
include "mlir/Dialect/LLM/Runtime/RuntimeInterfaces.td"
include "mlir/Dialect/LLM/Runtime/KVCache.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AppendKV Op
//===----------------------------------------------------------------------===//

def LLM_AppendKVOp : LLM_Op<"append_kv", [
    DeclareOpInterfaceMethods<LLM_KVCacheInterface>,
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "Append key-value pairs to a KV cache";
  let description = [{
    This operation appends new key-value pairs to a PagedKVCache.
    
    The operation takes the current KV cache, a batch of new key tensors,
    a batch of new value tensors, and sequence IDs. It returns the updated
    KV cache and the block indices for the appended tokens.
    
    Example:
    ```mlir
    %new_kv, %block_indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
      block_size = 16 : i32,
      max_seq_len = 4096 : i32
    } : (!llm.paged_kv_cache, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
        -> (!llm.paged_kv_cache, tensor<2x1xi32>)
    ```
  }];
  
  let arguments = (ins
    LLM_PagedKVCacheType:$kv_cache,
    AnyTensor:$keys,
    AnyTensor:$values,
    I32Tensor:$seq_ids,
    
    DefaultValuedAttr<I32Attr, "16">:$block_size,
    DefaultValuedAttr<I32Attr, "4096">:$max_seq_len
  );
  
  let results = (outs
    LLM_PagedKVCacheType:$new_kv_cache,
    I32Tensor:$block_indices
  );
  
  let hasVerifier = 1;
  
  let assemblyFormat = [{
    $kv_cache `,` $keys `,` $values `,` $seq_ids attr-dict `:` 
    `(` type($kv_cache) `,` type($keys) `,` type($values) `,` type($seq_ids) `)`
    `->` `(` type($new_kv_cache) `,` type($block_indices) `)`
  }];
  
  let extraClassDeclaration = [{
    // LLM_KVCacheInterface methods
    bool usesKVCache() { return true; }
    int64_t getNumKVTokens();
    mlir::Value getKVCacheInput() { return getKvCache(); }
    mlir::Value getKVCacheOutput() { return getNewKvCache(); }
  }];
}

//===----------------------------------------------------------------------===//
// LookupKV Op
//===----------------------------------------------------------------------===//

def LLM_LookupKVOp : LLM_Op<"lookup_kv", [
    DeclareOpInterfaceMethods<LLM_KVCacheInterface>,
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "Look up key-value pairs from a KV cache";
  let description = [{
    This operation retrieves key-value pairs from a PagedKVCache.
    
    The operation takes the KV cache, block indices for the tokens to retrieve,
    and sequence lengths. It returns the retrieved key and value tensors.
    
    Example:
    ```mlir
    %keys, %values = llm.lookup_kv %kv_cache, %block_indices, %seq_lens {
      num_heads = 16 : i32,
      head_dim = 64 : i32
    } : (!llm.paged_kv_cache, tensor<2x128xi32>, tensor<2xi32>) 
        -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>)
    ```
  }];
  
  let arguments = (ins
    LLM_PagedKVCacheType:$kv_cache,
    I32Tensor:$block_indices,
    I32Tensor:$seq_lens,
    
    I32Attr:$num_heads,
    I32Attr:$head_dim
  );
  
  let results = (outs
    AnyTensor:$keys,
    AnyTensor:$values
  );
  
  let hasVerifier = 1;
  
  let assemblyFormat = [{
    $kv_cache `,` $block_indices `,` $seq_lens attr-dict `:` 
    `(` type($kv_cache) `,` type($block_indices) `,` type($seq_lens) `)`
    `->` `(` type($keys) `,` type($values) `)`
  }];
  
  let extraClassDeclaration = [{
    // LLM_KVCacheInterface methods
    bool usesKVCache() { return true; }
    int64_t getNumKVTokens() { return 0; } // No new tokens, just lookup
    mlir::Value getKVCacheInput() { return getKvCache(); }
    mlir::Value getKVCacheOutput() { return nullptr; }
  }];
}

//===----------------------------------------------------------------------===//
// PagedAttention Op
//===----------------------------------------------------------------------===//

def LLM_PagedAttentionOp : LLM_Op<"paged_attention", [
    DeclareOpInterfaceMethods<LLM_KVCacheInterface>,
    DeclareOpInterfaceMethods<LLM_AttentionInterface>,
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "Perform attention with paged KV cache";
  let description = [{
    This operation performs attention computation using a paged KV cache.
    
    It takes query tensors, a KV cache, and block indices, and performs
    an efficient attention operation. It is particularly optimized for
    autoregressive generation where the query is much shorter than the
    cached key-value pairs.
    
    Example:
    ```mlir
    %output = llm.paged_attention %query, %kv_cache, %block_indices, %seq_lens {
      num_heads = 16 : i32,
      head_dim = 64 : i32,
      scale = 0.125 : f32
    } : (tensor<2x1x16x64xf16>, !llm.paged_kv_cache, tensor<2x128xi32>, tensor<2xi32>) 
        -> tensor<2x1x16x64xf16>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$query,
    LLM_PagedKVCacheType:$kv_cache,
    I32Tensor:$block_indices,
    I32Tensor:$seq_lens,
    
    OptionalAttr<AnyAttr>:$attention_mask,
    I32Attr:$num_heads,
    I32Attr:$head_dim,
    F32Attr:$scale
  );
  
  let results = (outs
    AnyTensor:$output
  );
  
  let hasVerifier = 1;
  
  let assemblyFormat = [{
    $query `,` $kv_cache `,` $block_indices `,` $seq_lens attr-dict `:` 
    `(` type($query) `,` type($kv_cache) `,` type($block_indices) `,` type($seq_lens) `)`
    `->` type($output)
  }];
  
  let extraClassDeclaration = [{
    // LLM_KVCacheInterface methods
    bool usesKVCache() { return true; }
    int64_t getNumKVTokens() { return 0; } // No new tokens, just reads
    mlir::Value getKVCacheInput() { return getKvCache(); }
    mlir::Value getKVCacheOutput() { return nullptr; }
    
    // LLM_AttentionInterface methods
    bool isAttentionOp() { return true; }
    int64_t getBatchSize();
    int64_t getSeqLength();
    int64_t getNumHeads() { return getNumHeadsAttr().getInt(); }
    int64_t getHeadDim() { return getHeadDimAttr().getInt(); }
  }];
}

#endif // MLIR_DIALECT_LLM_IR_LLMKVCACHEOPS 