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

#endif // LLM_PASSES 