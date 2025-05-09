//===- LLMKVCacheOps.cpp - LLM dialect KV cache ops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the KV cache operations in the LLM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// AppendKVOp Implementation
//===----------------------------------------------------------------------===//

int64_t AppendKVOp::getNumKVTokens() {
  // Try to determine the number of tokens from the input tensor shape
  if (auto keysType = getKeys().getType().dyn_cast<RankedTensorType>()) {
    auto shape = keysType.getShape();
    // Assuming shape format: [batch_size, seq_len, num_heads, head_dim]
    if (shape.size() >= 2 && shape[0] != ShapedType::kDynamic && 
        shape[1] != ShapedType::kDynamic) {
      return shape[0] * shape[1]; // batch_size * seq_len
    }
  }
  return -1; // Not statically known
}

//===----------------------------------------------------------------------===//
// PagedAttentionOp Implementation
//===----------------------------------------------------------------------===//

int64_t PagedAttentionOp::getBatchSize() {
  // Get batch size from the query tensor
  if (auto queryType = getQuery().getType().dyn_cast<RankedTensorType>()) {
    auto shape = queryType.getShape();
    // Assuming shape format: [batch_size, seq_len, num_heads, head_dim]
    if (shape.size() >= 1 && shape[0] != ShapedType::kDynamic) {
      return shape[0];
    }
  }
  return -1; // Not statically known
}

int64_t PagedAttentionOp::getSeqLength() {
  // Get sequence length from the query tensor
  if (auto queryType = getQuery().getType().dyn_cast<RankedTensorType>()) {
    auto shape = queryType.getShape();
    // Assuming shape format: [batch_size, seq_len, num_heads, head_dim]
    if (shape.size() >= 2 && shape[1] != ShapedType::kDynamic) {
      return shape[1];
    }
  }
  return -1; // Not statically known
}

//===----------------------------------------------------------------------===//
// Op Verifiers
//===----------------------------------------------------------------------===//

LogicalResult AppendKVOp::verify() {
  // Verify input/output KV cache type matches
  auto inputType = getKvCache().getType().cast<PagedKVCacheType>();
  auto outputType = getNewKvCache().getType().cast<PagedKVCacheType>();
  
  // The input and output KV cache must have the same type
  if (inputType != outputType) {
    return emitOpError("input and output KV cache types must match");
  }
  
  // Verify key and value tensors have compatible shapes
  auto keysType = getKeys().getType().dyn_cast<RankedTensorType>();
  auto valuesType = getValues().getType().dyn_cast<RankedTensorType>();
  
  if (keysType && valuesType) {
    // Key and value should have the same shape
    if (keysType.getShape() != valuesType.getShape()) {
      return emitOpError("key and value tensors must have the same shape");
    }
    
    // If we can determine the number of heads and head dimension, verify they match the KV cache
    auto shape = keysType.getShape();
    if (shape.size() >= 4) {
      if (shape[2] != ShapedType::kDynamic && 
          static_cast<int64_t>(shape[2]) != inputType.getNumHeads()) {
        return emitOpError("number of heads in input tensor (")
               << shape[2] << ") doesn't match KV cache ("
               << inputType.getNumHeads() << ")";
      }
      
      if (shape[3] != ShapedType::kDynamic && 
          static_cast<int64_t>(shape[3]) != inputType.getHeadDim()) {
        return emitOpError("head dimension in input tensor (")
               << shape[3] << ") doesn't match KV cache ("
               << inputType.getHeadDim() << ")";
      }
    }
  }
  
  // Verify sequence IDs tensor
  auto seqIdsType = getSeqIds().getType().dyn_cast<RankedTensorType>();
  if (seqIdsType) {
    // Sequence IDs should be a 1D tensor
    if (seqIdsType.getRank() != 1) {
      return emitOpError("sequence IDs tensor must be 1D");
    }
    
    // If we can determine the batch size from keys, verify it matches seqIds
    if (keysType && keysType.getRank() >= 1 && 
        keysType.getDimSize(0) != ShapedType::kDynamic && 
        seqIdsType.getDimSize(0) != ShapedType::kDynamic &&
        keysType.getDimSize(0) != seqIdsType.getDimSize(0)) {
      return emitOpError("batch size in key tensor (")
             << keysType.getDimSize(0) << ") doesn't match "
             << "sequence IDs tensor size (" << seqIdsType.getDimSize(0) << ")";
    }
  }
  
  return success();
}

LogicalResult LookupKVOp::verify() {
  // Verify KV cache type
  auto kvCacheType = getKvCache().getType().cast<PagedKVCacheType>();
  
  // Verify block indices tensor
  auto blockIndicesType = getBlockIndices().getType().dyn_cast<RankedTensorType>();
  if (!blockIndicesType) {
    return emitOpError("block indices must have a ranked tensor type");
  }
  
  // Verify sequence lengths tensor
  auto seqLensType = getSeqLens().getType().dyn_cast<RankedTensorType>();
  if (!seqLensType) {
    return emitOpError("sequence lengths must have a ranked tensor type");
  }
  
  // Sequence lengths should be a 1D tensor
  if (seqLensType.getRank() != 1) {
    return emitOpError("sequence lengths tensor must be 1D");
  }
  
  // Verify output types
  auto keysType = getKeys().getType().dyn_cast<RankedTensorType>();
  auto valuesType = getValues().getType().dyn_cast<RankedTensorType>();
  
  if (keysType && valuesType) {
    // Keys and values should have the same shape
    if (keysType.getShape() != valuesType.getShape()) {
      return emitOpError("output key and value tensors must have the same shape");
    }
    
    // If we have num_heads and head_dim attributes, verify they match the output tensors
    if (keysType.getRank() >= 4) {
      int64_t numHeads = getNumHeadsAttr().getInt();
      int64_t headDim = getHeadDimAttr().getInt();
      
      if (keysType.getDimSize(2) != ShapedType::kDynamic && 
          keysType.getDimSize(2) != numHeads) {
        return emitOpError("number of heads in output tensor (")
               << keysType.getDimSize(2) << ") doesn't match num_heads attribute ("
               << numHeads << ")";
      }
      
      if (keysType.getDimSize(3) != ShapedType::kDynamic && 
          keysType.getDimSize(3) != headDim) {
        return emitOpError("head dimension in output tensor (")
               << keysType.getDimSize(3) << ") doesn't match head_dim attribute ("
               << headDim << ")";
      }
    }
  }
  
  return success();
}

LogicalResult PagedAttentionOp::verify() {
  // Verify query tensor
  auto queryType = getQuery().getType().dyn_cast<RankedTensorType>();
  if (!queryType) {
    return emitOpError("query must have a ranked tensor type");
  }
  
  // Verify KV cache type
  auto kvCacheType = getKvCache().getType().cast<PagedKVCacheType>();
  
  // Verify block indices tensor
  auto blockIndicesType = getBlockIndices().getType().dyn_cast<RankedTensorType>();
  if (!blockIndicesType) {
    return emitOpError("block indices must have a ranked tensor type");
  }
  
  // Verify sequence lengths tensor
  auto seqLensType = getSeqLens().getType().dyn_cast<RankedTensorType>();
  if (!seqLensType) {
    return emitOpError("sequence lengths must have a ranked tensor type");
  }
  
  // Sequence lengths should be a 1D tensor
  if (seqLensType.getRank() != 1) {
    return emitOpError("sequence lengths tensor must be 1D");
  }
  
  // If query has known rank, it should be 4D: [batch_size, seq_len, num_heads, head_dim]
  if (queryType.getRank() != 4) {
    return emitOpError("query tensor must have rank 4");
  }
  
  // Output shape should match query shape
  auto outputType = getOutput().getType().dyn_cast<RankedTensorType>();
  if (outputType && queryType) {
    if (outputType.getShape() != queryType.getShape()) {
      return emitOpError("output shape must match query shape");
    }
  }
  
  // Verify num_heads and head_dim attributes match query tensor if possible
  int64_t numHeads = getNumHeadsAttr().getInt();
  int64_t headDim = getHeadDimAttr().getInt();
  
  if (queryType.getDimSize(2) != ShapedType::kDynamic && 
      queryType.getDimSize(2) != numHeads) {
    return emitOpError("number of heads in query tensor (")
           << queryType.getDimSize(2) << ") doesn't match num_heads attribute ("
           << numHeads << ")";
  }
  
  if (queryType.getDimSize(3) != ShapedType::kDynamic && 
      queryType.getDimSize(3) != headDim) {
    return emitOpError("head dimension in query tensor (")
           << queryType.getDimSize(3) << ") doesn't match head_dim attribute ("
           << headDim << ")";
  }
  
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMKVCacheOps.cpp.inc" 