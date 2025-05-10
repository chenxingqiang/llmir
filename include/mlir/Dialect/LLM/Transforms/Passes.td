//===- Passes.td - LLM dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes specific to the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLM_PASSES
#define LLM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// KVCacheOptimizationPass
//===----------------------------------------------------------------------===//

def KVCacheOptimization : Pass<"llm-optimize-kv-cache", "func::FuncOp"> {
  let summary = "Optimize KV cache operations";
  let description = [{
    This pass performs various optimizations on KV cache operations to improve
    performance, including:
    
    - Block size optimization for better memory utilization
    - KV cache operation compatibility checking
    - Parameter tuning for improved performance
    
    Example:
    
    ```mlir
    // Before optimization
    %new_kv, %indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
      block_size = 1024 : i32,  // Too large for the sequence length
      max_seq_len = 2048 : i32
    }
    
    // After optimization
    %new_kv, %indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
      block_size = 128 : i32,   // Optimized block size
      max_seq_len = 2048 : i32
    }
    ```
  }];
  
  let constructor = "mlir::llm::createKVCacheOptimizationPass()";
  let dependentDialects = ["mlir::llm::LLMDialect"];
}

def LLMLowering : Pass<"llm-lowering", "func::FuncOp"> {
  let summary = "Lower high-level LLM operations to lower-level ones";
  
  let description = [{
    This pass lowers high-level LLM operations to lower-level ones, such as
    lowering attention operations to tensor and linalg operations.
  }];
  
  let constructor = "mlir::llm::createLLMLoweringPass()";
  let dependentDialects = [
      "func::FuncDialect",
      "tensor::TensorDialect",
      "linalg::LinalgDialect"
  ];
}

def LowerKVCacheOps : Pass<"llm-lower-kv-cache-ops", "func::FuncOp"> {
  let summary = "Lower KV cache operations to runtime function calls";
  
  let description = [{
    This pass lowers KV cache operations (`llm.append_kv`, `llm.lookup_kv`, 
    and `llm.paged_attention`) to runtime function calls. The runtime functions
    are implemented in the LLM runtime library and provide efficient implementations
    of the KV cache operations.
    
    Example:
    ```mlir
    %kv_cache_updated, %block_indices = llm.append_kv(%keys, %values, %seq_ids, %kv_cache)
    ```
    
    becomes:
    
    ```mlir
    %batchSize = tensor.dim %keys, %c0
    %seqLen = tensor.dim %keys, %c1
    %kv_cache_updated, %block_indices = call @mlir_llm_append_kv(%kv_cache, %keys, %values, 
                                          %seq_ids, %batchSize, %seqLen)
    ```
  }];
  
  let constructor = "mlir::llm::createLowerKVCacheOpsPass()";
  let dependentDialects = [
      "func::FuncDialect",
      "arith::ArithDialect",
      "tensor::TensorDialect",
      "math::MathDialect"
  ];
}

#endif // LLM_PASSES 