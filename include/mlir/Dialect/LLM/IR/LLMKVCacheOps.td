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
// Note: KVCache.td (StructAttr) excluded for MLIR 18 compatibility.
// LLMKVCacheOps do not use KVCacheConfig/KVCacheStats attributes.
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// KVCache Type definitions
//===----------------------------------------------------------------------===//

def LLM_KVCacheType : DialectType<LLM_Dialect,
    "KVCache", "!llm.kvcache">,
    BuildableType<"$_builder.getType<::mlir::llm::KVCacheType>()">;

//===----------------------------------------------------------------------===//
// AppendKV Operation
//===----------------------------------------------------------------------===//

def LLM_AppendKVOp : LLM_Op<"append_kv", [
    AttrSizedResultSegments,
    DeclareOpInterfaceMethods<LLM_KVCacheInterface>
]> {
  let summary = "Append key-value pairs to a KV cache";
  let description = [{
    The `llm.append_kv` operation adds new key-value pairs to a paged KV cache.
    It takes key and value tensors, sequence IDs, and an optional existing KV cache.
    It returns an updated KV cache and block indices for the newly added key-value pairs.

    Example:
    ```mlir
    %kv_cache_updated, %block_indices = llm.append_kv(%keys, %values, %seq_ids, %kv_cache) : 
      (tensor<2x8x16xf16>, tensor<2x8x16xf16>, tensor<2xi32>, !llm.kvcache) -> 
      (!llm.kvcache, tensor<2x1xi32>)
    ```
  }];

  let arguments = (ins
    Arg<AnyTensor, "Key tensor to append to the cache", [MemRead]>:$keys,
    Arg<AnyTensor, "Value tensor to append to the cache", [MemRead]>:$values,
    Arg<AnyTensor, "Sequence IDs for the batch", [MemRead]>:$seq_ids,
    Optional<LLM_KVCacheType>:$kv_cache
  );

  let results = (outs
    LLM_KVCacheType:$updated_kv_cache,
    Res<AnyTensor, "Block indices for the appended KV pairs", [MemWrite]>:$block_indices
  );

  let extraClassDeclaration = [{
    // KVCacheInterface methods
    bool usesKVCache() { return true; }
    int64_t getNumKVTokens();
    mlir::Value getKVCacheInput() { return getKVCache(); }
    mlir::Value getKVCacheOutput() { return getUpdatedKVCache(); }
  }];

  let assemblyFormat = [{
    `(` $keys `,` $values `,` $seq_ids (`,` $kv_cache^)? `)`
    attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LookupKV Operation
//===----------------------------------------------------------------------===//

def LLM_LookupKVOp : LLM_Op<"lookup_kv", [
    DeclareOpInterfaceMethods<LLM_KVCacheInterface>
]> {
  let summary = "Look up key-value pairs from a KV cache";
  let description = [{
    The `llm.lookup_kv` operation retrieves key-value pairs from a paged KV cache 
    based on block indices. It returns the relevant key and value tensors.

    Example:
    ```mlir
    %keys, %values = llm.lookup_kv(%block_indices, %seq_lens, %kv_cache) : 
      (tensor<2x8xi32>, tensor<2xi32>, !llm.kvcache) -> 
      (tensor<2x8x16xf16>, tensor<2x8x16xf16>)
    ```
  }];

  let arguments = (ins
    Arg<AnyTensor, "Block indices for tokens to look up", [MemRead]>:$block_indices,
    Arg<AnyTensor, "Sequence lengths for each sequence in the batch", [MemRead]>:$seq_lens,
    LLM_KVCacheType:$kv_cache
  );

  let results = (outs
    Res<AnyTensor, "Retrieved key tensors", [MemWrite]>:$keys,
    Res<AnyTensor, "Retrieved value tensors", [MemWrite]>:$values
  );

  let extraClassDeclaration = [{
    // KVCacheInterface methods
    bool usesKVCache() { return true; }
    int64_t getNumKVTokens();
    mlir::Value getKVCacheInput() { return getKVCache(); }
    mlir::Value getKVCacheOutput() { return Value(); } // No output cache
  }];

  let assemblyFormat = [{
    `(` $block_indices `,` $seq_lens `,` $kv_cache `)`
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
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