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

LogicalResult AppendKVOp::verify() {
  // Get batch size from keys tensor
  if (!getKeys().getType().isa<RankedTensorType>())
    return emitOpError("keys must have ranked tensor type");

  RankedTensorType keysTy = getKeys().getType().cast<RankedTensorType>();
  if (keysTy.getRank() < 2)
    return emitOpError("keys must have rank >= 2, got: ") << keysTy.getRank();

  // Verify values tensor shape matches keys
  if (!getValues().getType().isa<RankedTensorType>())
    return emitOpError("values must have ranked tensor type");

  RankedTensorType valuesTy = getValues().getType().cast<RankedTensorType>();
  if (valuesTy.getRank() != keysTy.getRank())
    return emitOpError("values rank must match keys rank");

  for (int64_t i = 0; i < keysTy.getRank(); ++i) {
    if (keysTy.getDimSize(i) != valuesTy.getDimSize(i) && 
        keysTy.isDynamicDim(i) && valuesTy.isDynamicDim(i)) {
      return emitOpError("values shape must match keys shape, mismatch at dim ")
             << i << ": " << keysTy.getDimSize(i) << " vs " 
             << valuesTy.getDimSize(i);
    }
  }

  // Verify seq_ids is 1D tensor with batch_size elements
  if (!getSeqIds().getType().isa<RankedTensorType>())
    return emitOpError("seq_ids must have ranked tensor type");

  RankedTensorType seqIdsTy = getSeqIds().getType().cast<RankedTensorType>();
  if (seqIdsTy.getRank() != 1)
    return emitOpError("seq_ids must be 1D tensor, got rank: ") 
           << seqIdsTy.getRank();

  int64_t batchSize = keysTy.getDimSize(0);
  if (!seqIdsTy.isDynamicDim(0) && !keysTy.isDynamicDim(0) && 
      seqIdsTy.getDimSize(0) != batchSize) {
    return emitOpError("seq_ids size must match batch size (first dim of keys), got: ") 
           << seqIdsTy.getDimSize(0) << " vs " << batchSize;
  }

  // Verify block_indices output is 2D tensor with shape [batch_size, seq_len]
  if (!getBlockIndices().getType().isa<RankedTensorType>())
    return emitOpError("block_indices must have ranked tensor type");

  RankedTensorType blockIdxTy = getBlockIndices().getType().cast<RankedTensorType>();
  if (blockIdxTy.getRank() != 2)
    return emitOpError("block_indices must be 2D tensor, got rank: ") 
           << blockIdxTy.getRank();

  return success();
}

void AppendKVOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  // Add canonicalization patterns here when needed
}

int64_t AppendKVOp::getNumKVTokens() {
  if (auto keysTy = getKeys().getType().dyn_cast<RankedTensorType>()) {
    // If we have a ranked keys tensor, we can determine the number of tokens
    // For a batch of sequences, the total number of tokens is batch_size * seq_len
    if (keysTy.getRank() >= 2 && !keysTy.isDynamicDim(0) && !keysTy.isDynamicDim(1)) {
      int64_t batchSize = keysTy.getDimSize(0);
      int64_t seqLen = keysTy.getDimSize(1);
      return batchSize * seqLen;
    }
  }
  return -1; // unknown at compile time
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
// LookupKVOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult LookupKVOp::verify() {
  // Verify block_indices has rank 2 (batch_size x seq_len)
  if (!getBlockIndices().getType().isa<RankedTensorType>())
    return emitOpError("block_indices must have ranked tensor type");

  RankedTensorType blockIdxTy = getBlockIndices().getType().cast<RankedTensorType>();
  if (blockIdxTy.getRank() != 2)
    return emitOpError("block_indices must be 2D tensor, got rank: ") 
           << blockIdxTy.getRank();

  // Verify seq_lens has rank 1 (batch_size)
  if (!getSeqLens().getType().isa<RankedTensorType>())
    return emitOpError("seq_lens must have ranked tensor type");

  RankedTensorType seqLensTy = getSeqLens().getType().cast<RankedTensorType>();
  if (seqLensTy.getRank() != 1)
    return emitOpError("seq_lens must be 1D tensor, got rank: ") 
           << seqLensTy.getRank();

  // Verify batch size is consistent (first dimension of block_indices and size of seq_lens)
  if (!blockIdxTy.isDynamicDim(0) && !seqLensTy.isDynamicDim(0) &&
      blockIdxTy.getDimSize(0) != seqLensTy.getDimSize(0)) {
    return emitOpError("batch size in block_indices (")
           << blockIdxTy.getDimSize(0) << ") doesn't match "
           << "batch size in seq_lens (" << seqLensTy.getDimSize(0) << ")";
  }

  // Output keys and values should have the same shape
  if (auto keysTy = getKeys().getType().dyn_cast<RankedTensorType>()) {
    if (auto valuesTy = getValues().getType().dyn_cast<RankedTensorType>()) {
      if (keysTy.getRank() != valuesTy.getRank())
        return emitOpError("output keys and values must have the same rank");

      for (int64_t i = 0; i < keysTy.getRank(); ++i) {
        if (!keysTy.isDynamicDim(i) && !valuesTy.isDynamicDim(i) &&
            keysTy.getDimSize(i) != valuesTy.getDimSize(i)) {
          return emitOpError("output keys and values must have the same shape, "
                            "mismatch at dimension ") 
                 << i << ": " << keysTy.getDimSize(i) << " vs " 
                 << valuesTy.getDimSize(i);
        }
      }
    }
  }

  return success();
}

int64_t LookupKVOp::getNumKVTokens() {
  // LookupKVOp doesn't add new tokens to the cache, it only retrieves them
  return 0;
}

//===----------------------------------------------------------------------===//
// Op Verifiers
//===----------------------------------------------------------------------===//

LogicalResult PagedAttentionOp::verify() {
  // Verify query tensor has proper rank (batch_size x seq_len x num_heads x head_dim)
  if (!getQuery().getType().isa<RankedTensorType>())
    return emitOpError("query must have ranked tensor type");
    
  RankedTensorType queryTy = getQuery().getType().cast<RankedTensorType>();
  if (queryTy.getRank() < 3)
    return emitOpError("query must have rank >= 3, got: ") << queryTy.getRank();
    
  // Verify block_indices has rank 2 (batch_size x max_seq_len)
  if (!getBlockIndices().getType().isa<RankedTensorType>())
    return emitOpError("block_indices must have ranked tensor type");

  RankedTensorType blockIdxTy = getBlockIndices().getType().cast<RankedTensorType>();
  if (blockIdxTy.getRank() != 2)
    return emitOpError("block_indices must be 2D tensor, got rank: ") 
           << blockIdxTy.getRank();
           
  // Verify seq_lens has rank 1 (batch_size)
  if (!getSeqLens().getType().isa<RankedTensorType>())
    return emitOpError("seq_lens must have ranked tensor type");

  RankedTensorType seqLensTy = getSeqLens().getType().cast<RankedTensorType>();
  if (seqLensTy.getRank() != 1)
    return emitOpError("seq_lens must be 1D tensor, got rank: ") 
           << seqLensTy.getRank();
           
  // Verify batch size is consistent across all tensors
  int64_t queryBatchSize = queryTy.getDimSize(0);
  int64_t blockIdxBatchSize = blockIdxTy.getDimSize(0);
  int64_t seqLensBatchSize = seqLensTy.getDimSize(0);
  
  if (!queryTy.isDynamicDim(0) && !blockIdxTy.isDynamicDim(0) && 
      queryBatchSize != blockIdxBatchSize) {
    return emitOpError("batch size in query (")
           << queryBatchSize << ") doesn't match "
           << "batch size in block_indices (" << blockIdxBatchSize << ")";
  }
  
  if (!queryTy.isDynamicDim(0) && !seqLensTy.isDynamicDim(0) && 
      queryBatchSize != seqLensBatchSize) {
    return emitOpError("batch size in query (")
           << queryBatchSize << ") doesn't match "
           << "batch size in seq_lens (" << seqLensBatchSize << ")";
  }
  
  // Verify output tensor shape matches query shape
  if (!getAttentionOutput().getType().isa<RankedTensorType>())
    return emitOpError("attention_output must have ranked tensor type");

  RankedTensorType outputTy = getAttentionOutput().getType().cast<RankedTensorType>();
  if (outputTy.getRank() != queryTy.getRank())
    return emitOpError("attention_output rank must match query rank");

  for (int64_t i = 0; i < queryTy.getRank(); ++i) {
    if (!outputTy.isDynamicDim(i) && !queryTy.isDynamicDim(i) &&
        outputTy.getDimSize(i) != queryTy.getDimSize(i)) {
      return emitOpError("attention_output shape must match query shape, "
                        "mismatch at dimension ") 
             << i << ": " << outputTy.getDimSize(i) << " vs " 
             << queryTy.getDimSize(i);
    }
  }
  
  // Verify alibi_slopes if present
  if (getAlibiSlopes()) {
    auto alibiSlopes = getAlibiSlopes().value();
    if (queryTy.getRank() >= 3 && !queryTy.isDynamicDim(2)) {
      int64_t numHeads = queryTy.getDimSize(2);
      if (alibiSlopes.size() != numHeads) {
        return emitOpError("alibi_slopes size (")
               << alibiSlopes.size() << ") doesn't match "
               << "number of heads (" << numHeads << ")";
      }
    }
  }
  
  return success();
}

int64_t PagedAttentionOp::getNumKVTokens() {
  // PagedAttentionOp doesn't add new tokens to the cache
  return 0;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMKVCacheOps.cpp.inc" 