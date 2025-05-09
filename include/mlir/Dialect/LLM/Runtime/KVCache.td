//===- KVCache.td - KV Cache Runtime Support TableGen defs ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines TableGen structures for LLM KV cache runtime support.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_KVCACHE_TD
#define MLIR_DIALECT_LLM_RUNTIME_KVCACHE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLM/IR/LLMBase.td"

//===----------------------------------------------------------------------===//
// KV Cache Configuration
//===----------------------------------------------------------------------===//

def LLM_KVCacheConfig : StructAttr<"KVCacheConfig", LLM_Dialect, [
                        StructFieldAttr<"blockSize", I64Attr>,
                        StructFieldAttr<"numHeads", I64Attr>,
                        StructFieldAttr<"headDim", I64Attr>,
                        StructFieldAttr<"maxSeqLen", I64Attr>,
                        StructFieldAttr<"dtype", TypeAttr>,
                        StructFieldAttr<"useGPU", BoolAttr, "true">
                       ]> {
  let description = [{
    Configuration for the KV cache.
    
    blockSize: The number of tokens in each memory block.
    numHeads: The number of attention heads.
    headDim: The dimension of each attention head.
    maxSeqLen: The maximum sequence length.
    dtype: The data type for the KV cache (usually f16).
    useGPU: Whether to use GPU memory (default: true).
  }];
}

//===----------------------------------------------------------------------===//
// KV Cache Allocation Strategy
//===----------------------------------------------------------------------===//

def LLM_KVCacheStrategy : I32EnumAttr<"KVCacheStrategy", "KV cache allocation strategy", [
    I32EnumAttrCase<"BlockBased", 0, "block_based">,
    I32EnumAttrCase<"Continuous", 1, "continuous">,
    I32EnumAttrCase<"PagedAttention", 2, "paged_attention">
]> {
  let cppNamespace = "::mlir::llm::runtime";
  let description = [{
    Strategy for allocating and managing KV cache memory.
    
    BlockBased: Allocate memory in fixed-size blocks.
    Continuous: Allocate a continuous memory region for each sequence.
    PagedAttention: Use the paged attention mechanism (similar to vLLM).
  }];
}

//===----------------------------------------------------------------------===//
// KV Cache Block Statistics
//===----------------------------------------------------------------------===//

def LLM_KVCacheStats : StructAttr<"KVCacheStats", LLM_Dialect, [
                       StructFieldAttr<"totalBlocks", I64Attr>,
                       StructFieldAttr<"allocatedBlocks", I64Attr>,
                       StructFieldAttr<"freeBlocks", I64Attr>,
                       StructFieldAttr<"totalMemoryBytes", I64Attr>,
                       StructFieldAttr<"avgUtilizationPercent", F32Attr>
                      ]> {
  let description = [{
    Statistics about the KV cache usage.
    
    totalBlocks: Total number of memory blocks.
    allocatedBlocks: Number of currently allocated blocks.
    freeBlocks: Number of free blocks in the pool.
    totalMemoryBytes: Total memory used in bytes.
    avgUtilizationPercent: Average utilization percentage of blocks.
  }];
}

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHE_TD 