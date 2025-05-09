//===- RuntimeInterfaces.td - Runtime interfaces -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the LLM runtime components.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_INTERFACES_TD
#define MLIR_DIALECT_LLM_RUNTIME_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// KVCache Interface
//===----------------------------------------------------------------------===//

def LLM_KVCacheInterface : OpInterface<"KVCacheInterface"> {
  let description = [{
    Interface for operations that interact with KV Cache.
  }];
  
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation accesses the KV cache.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"usesKVCache",
      /*args=*/(ins),
      /*methodBody=*/[{ return true; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of key-value tokens this operation will add to the cache.
        Returns -1 if the number is not statically known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumKVTokens",
      /*args=*/(ins),
      /*methodBody=*/[{ return -1; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand that represents the KV cache input.
        Returns nullptr if no cache input exists.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getKVCacheInput",
      /*args=*/(ins),
      /*methodBody=*/[{ return nullptr; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the result that represents the KV cache output.
        Returns nullptr if no cache output exists.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getKVCacheOutput",
      /*args=*/(ins),
      /*methodBody=*/[{ return nullptr; }]
    >
  ];
}

//===----------------------------------------------------------------------===//
// Attention Interface
//===----------------------------------------------------------------------===//

def LLM_AttentionInterface : OpInterface<"AttentionInterface"> {
  let description = [{
    Interface for operations that perform attention computation.
  }];
  
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation performs attention computation.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isAttentionOp",
      /*args=*/(ins),
      /*methodBody=*/[{ return true; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the batch size for this attention operation.
        Returns -1 if not statically known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getBatchSize",
      /*args=*/(ins),
      /*methodBody=*/[{ return -1; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the sequence length for this attention operation.
        Returns -1 if not statically known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getSeqLength",
      /*args=*/(ins),
      /*methodBody=*/[{ return -1; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of attention heads.
        Returns -1 if not statically known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumHeads",
      /*args=*/(ins),
      /*methodBody=*/[{ return -1; }]
    >,
    
    InterfaceMethod<
      /*desc=*/[{
        Returns the head dimension for this attention operation.
        Returns -1 if not statically known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getHeadDim",
      /*args=*/(ins),
      /*methodBody=*/[{ return -1; }]
    >
  ];
}

#endif // MLIR_DIALECT_LLM_RUNTIME_INTERFACES_TD 