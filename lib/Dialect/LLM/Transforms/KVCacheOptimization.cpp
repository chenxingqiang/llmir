//===- KVCacheOptimization.cpp - Optimize KV cache operations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to optimize LLM KV cache operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLMKVCacheOps.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

#define GEN_PASS_DEF_KVCACHEOPTIMIZATION
#include "mlir/Dialect/LLM/Transforms/Passes.h.inc"

namespace {

// Helper function to check if a block size is optimal for a given sequence length
// Returns an optimal block size based on the sequence length and head dim
int64_t getOptimalBlockSize(int64_t seqLen, int64_t headDim) {
  // If sequence length is very small, use a smaller block size
  if (seqLen <= 32) {
    return 16;
  } 
  // For medium sequences
  else if (seqLen <= 256) {
    return 32;
  }
  // For large sequences
  else if (seqLen <= 1024) {
    return 64;
  }
  // For very large sequences
  else {
    return 128;
  }
}

// Pattern to optimize block size in AppendKVOp
struct OptimizeAppendKVBlockSize : public OpRewritePattern<AppendKVOp> {
  using OpRewritePattern<AppendKVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    // Only optimize if we have a block size attribute
    if (!op.getBlockSize())
      return failure();
    
    // Get the keys tensor type
    auto keysType = op.getKeys().getType().cast<ShapedType>();
    if (!keysType.hasStaticShape())
      return failure();
      
    // Extract the sequence length and head dimension
    int64_t seqLen = keysType.getDimSize(1);
    int64_t headDim = keysType.getDimSize(3);
    
    // Get the current block size
    int64_t currentBlockSize = op.getBlockSize().value();
    
    // Calculate optimal block size
    int64_t optimalBlockSize = getOptimalBlockSize(seqLen, headDim);
    
    // If current block size is already optimal, return failure
    if (currentBlockSize == optimalBlockSize)
      return failure();
    
    // Create a new operation with the optimal block size
    auto newOp = rewriter.create<AppendKVOp>(
        op.getLoc(),
        op.getKVCache() ? op.getKVCache() : nullptr,
        op.getKeys(),
        op.getValues(),
        op.getSeqIds(),
        optimalBlockSize,
        op.getMaxSeqLen());
    
    // Replace the old operation with the new one
    rewriter.replaceOp(op, newOp.getResults());
    
    return success();
  }
};

// Pattern to identify and fuse duplicate KV cache operations
struct FuseDuplicateKVCacheOps : public OpRewritePattern<AppendKVOp> {
  using OpRewritePattern<AppendKVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    // Get the parent block
    Block *parentBlock = op->getBlock();
    
    // Look for another AppendKVOp in the same block with the same KV cache
    AppendKVOp duplicateOp = nullptr;
    
    for (auto &nextOp : *parentBlock) {
      if (auto appendOp = dyn_cast<AppendKVOp>(&nextOp)) {
        // Skip the current operation
        if (appendOp == op)
          continue;
          
        // Check if they operate on the same KV cache
        if (appendOp.getKVCache() && op.getKVCache() && 
            appendOp.getKVCache() == op.getKVCache()) {
          duplicateOp = appendOp;
          break;
        }
      }
    }
    
    if (!duplicateOp)
      return failure();
      
    // For simplicity in this implementation, we'll just reuse the most optimal
    // of the two operations based on block size
    
    // Get block sizes
    int64_t blockSize1 = op.getBlockSize() ? op.getBlockSize().value() : 0;
    int64_t blockSize2 = duplicateOp.getBlockSize() ? duplicateOp.getBlockSize().value() : 0;
    
    // Use the operation with the larger block size or the first one if equal
    bool useFirstOp = blockSize1 >= blockSize2;
    
    // Replace with the better operation
    if (useFirstOp) {
      rewriter.replaceOp(duplicateOp, op.getResults());
    } else {
      rewriter.replaceOp(op, duplicateOp.getResults());
    }
    
    return success();
  }
};

// Pattern to detect cross-sequence sharing opportunities
struct OptimizeCrossSequenceSharing : public OpRewritePattern<AppendKVOp> {
  using OpRewritePattern<AppendKVOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    // We need a KV cache and sequence IDs to optimize
    if (!op.getKVCache() || !op.getSeqIds())
      return failure();
      
    // We're looking for multiple AppendKVOp operations with similar content
    // but different sequence IDs
    
    // Get the current function
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();
      
    // Find other AppendKVOp operations in the function
    SmallVector<AppendKVOp, 4> appendOps;
    func.walk([&](AppendKVOp otherOp) {
      // Skip self
      if (otherOp == op)
        return;
        
      // Only consider ops with the same KV cache
      if (!otherOp.getKVCache() || otherOp.getKVCache() != op.getKVCache())
        return;
        
      appendOps.push_back(otherOp);
    });
    
    // If no other ops found, nothing to optimize
    if (appendOps.empty())
      return failure();
      
    // For each operation found, check if the content is similar
    bool madeChanges = false;
    
    // Get the keys and values from the current operation
    Value keys = op.getKeys();
    Value values = op.getValues();
    
    // Get the shape of the keys tensor
    auto keysType = keys.getType().dyn_cast<ShapedType>();
    if (!keysType || !keysType.hasStaticShape())
      return failure();
      
    int64_t seqLen = keysType.getDimSize(1);
    
    // Add a runtime call attribute to trigger cross-sequence sharing
    // This will be a hint to the runtime that it should look for sharing
    // opportunities at runtime
    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr("enable_sharing", rewriter.getBoolAttr(true));
    });
    
    // For each other AppendKVOp, add the same attribute
    for (auto otherOp : appendOps) {
      // Get the shape of the other keys tensor
      auto otherKeysType = otherOp.getKeys().getType().dyn_cast<ShapedType>();
      if (!otherKeysType || !otherKeysType.hasStaticShape())
        continue;
        
      int64_t otherSeqLen = otherKeysType.getDimSize(1);
      
      // Only consider operations with same sequence length for simplicity
      if (seqLen != otherSeqLen)
        continue;
        
      // Add sharing attribute
      rewriter.modifyOpInPlace(otherOp, [&]() {
        otherOp->setAttr("enable_sharing", rewriter.getBoolAttr(true));
      });
      
      madeChanges = true;
    }
    
    return madeChanges ? success() : failure();
  }
};

// Pattern to optimize PagedAttentionOp 
struct OptimizePagedAttention : public OpRewritePattern<PagedAttentionOp> {
  using OpRewritePattern<PagedAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PagedAttentionOp op,
                               PatternRewriter &rewriter) const override {
    // Get query tensor type
    auto queryType = op.getQuery().getType().dyn_cast<ShapedType>();
    if (!queryType || !queryType.hasStaticShape())
      return failure();
    
    // Extract dimensions
    int64_t batchSize = queryType.getDimSize(0);
    int64_t seqLen = queryType.getDimSize(1);
    
    // If we have a small batch size and sequence length, we can optimize by
    // ensuring we have an optimal scale value for numerical stability
    
    // Check if we need to optimize the scale value
    if (!op.getScale()) {
      // No scale specified, add a scale attribute
      // Scale = 1.0 / sqrt(head_dim)
      int64_t headDim;
      if (queryType.getRank() >= 4) {
        headDim = queryType.getDimSize(3);
      } else {
        // If no head dimension in the type, use the attribute
        if (!op.getHeadDim())
          return failure();
        headDim = op.getHeadDim().value();
      }
      
      // Calculate optimal scale
      float scale = 1.0 / std::sqrt(static_cast<float>(headDim));
      
      // Create a new operation with the optimal scale
      auto newOp = rewriter.create<PagedAttentionOp>(
          op.getLoc(),
          op.getAttentionOutput().getType(),
          op.getQuery(),
          op.getBlockIndices(),
          op.getSeqLens(),
          op.getKVCache(),
          op.getNumHeads(),
          op.getHeadDim(),
          scale);
      
      // Replace the old operation with the new one
      rewriter.replaceOp(op, newOp.getResult());
      
      return success();
    }
    
    return failure();
  }
};

// Optimization pass that optimizes KV cache operations
struct KVCacheOptimizationPass
    : public impl::KVCacheOptimizationBase<KVCacheOptimizationPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = &getContext();

    // Set up patterns
    RewritePatternSet patterns(context);
    
    // Add optimization patterns
    patterns.add<OptimizeAppendKVBlockSize>(context);
    patterns.add<FuseDuplicateKVCacheOps>(context);
    patterns.add<OptimizeCrossSequenceSharing>(context);
    patterns.add<OptimizePagedAttention>(context);
    
    // Apply patterns using the greedy rewrite driver
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

/// Create a pass to optimize KV cache operations
std::unique_ptr<Pass> mlir::llm::createKVCacheOptimizationPass() {
  return std::make_unique<KVCacheOptimizationPass>();
} 