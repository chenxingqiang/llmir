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

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// Pattern Implementations
//===----------------------------------------------------------------------===//

namespace {

// Pattern to identify and fuse duplicate KV cache operations
struct FuseDuplicateKVCacheOps : public OpRewritePattern<AppendKVOp> {
  using OpRewritePattern<AppendKVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    Block *parentBlock = op->getBlock();
    AppendKVOp duplicateOp = nullptr;
    
    for (auto &nextOp : *parentBlock) {
      if (auto appendOp = dyn_cast<AppendKVOp>(&nextOp)) {
        if (appendOp == op)
          continue;
        if (appendOp.getKvCache() == op.getKvCache()) {
          duplicateOp = appendOp;
          break;
        }
      }
    }
    
    if (!duplicateOp)
      return failure();
      
    rewriter.replaceOp(duplicateOp, op.getResults());
    return success();
  }
};

// Pattern to detect cross-sequence sharing opportunities
struct OptimizeCrossSequenceSharing : public OpRewritePattern<AppendKVOp> {
  using OpRewritePattern<AppendKVOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();
      
    SmallVector<AppendKVOp, 4> appendOps;
    func.walk([&](AppendKVOp otherOp) {
      if (otherOp == op)
        return;
      if (otherOp.getKvCache() != op.getKvCache())
        return;
      appendOps.push_back(otherOp);
    });
    
    if (appendOps.empty())
      return failure();
      
    bool madeChanges = false;
    Value keys = op.getKey();
    
    auto keysType = dyn_cast<ShapedType>(keys.getType());
    if (!keysType || !keysType.hasStaticShape())
      return failure();
      
    int64_t seqLen = keysType.getDimSize(1);
    
    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr("enable_sharing", rewriter.getBoolAttr(true));
    });
    
    for (auto otherOp : appendOps) {
      auto otherKeysType = dyn_cast<ShapedType>(otherOp.getKey().getType());
      if (!otherKeysType || !otherKeysType.hasStaticShape())
        continue;
        
      int64_t otherSeqLen = otherKeysType.getDimSize(1);
      if (seqLen != otherSeqLen)
        continue;
        
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
    auto queryType = dyn_cast<ShapedType>(op.getQuery().getType());
    if (!queryType || !queryType.hasStaticShape())
      return failure();
    
    float scale = op.getScale().convertToFloat();
    if (scale != 0.0f)
      return failure();
    
    if (queryType.getRank() < 4)
      return failure();
    
    int64_t headDim = queryType.getDimSize(3);
    float optimalScale = 1.0f / std::sqrt(static_cast<float>(headDim));
    
    rewriter.modifyOpInPlace(op, [&]() {
      op.setScaleAttr(rewriter.getF32FloatAttr(optimalScale));
    });
    
    return success();
  }
};

// Pattern to optimize LookupKVOp by adding caching hints
struct OptimizeLookupKV : public OpRewritePattern<LookupKVOp> {
  using OpRewritePattern<LookupKVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LookupKVOp op,
                               PatternRewriter &rewriter) const override {
    if (op->hasAttr("optimized"))
      return failure();
    
    auto kvCacheType = cast<PagedKVCacheType>(op.getKvCache().getType());
    int64_t blockSize = kvCacheType.getBlockSize();
    
    if (blockSize >= 64) {
      rewriter.modifyOpInPlace(op, [&]() {
        op->setAttr("prefetch", rewriter.getBoolAttr(true));
        op->setAttr("optimized", rewriter.getUnitAttr());
      });
      return success();
    }
    
    return failure();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation (using deprecated GEN_PASS_CLASSES for compatibility)
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "mlir/Dialect/LLM/Transforms/Passes.h.inc"

namespace {

struct KVCacheOptimizationPass
    : public KVCacheOptimizationBase<KVCacheOptimizationPass> {

  void runOnOperation() override {
    auto func = getOperation();
    auto context = &getContext();

    RewritePatternSet patterns(context);
    
    patterns.add<FuseDuplicateKVCacheOps>(context);
    patterns.add<OptimizeCrossSequenceSharing>(context);
    patterns.add<OptimizePagedAttention>(context);
    patterns.add<OptimizeLookupKV>(context);
    
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
