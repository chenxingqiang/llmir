//===- KVCacheOptimization.cpp - KV cache optimization pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization passes for KV cache operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/LLM/Runtime/RuntimeInterfaces.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llm-kv-cache-opt"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// KVCacheBlockSizeOptimizationPattern
//===----------------------------------------------------------------------===//

namespace {
/// Optimizes the block size parameter in append_kv operations.
/// This pattern analyzes the context of append_kv operations and adjusts 
/// the block size parameter for better performance based on sequence lengths.
struct KVCacheBlockSizeOptimizationPattern : public OpRewritePattern<AppendKVOp> {
  explicit KVCacheBlockSizeOptimizationPattern(MLIRContext *context,
                                               PatternBenefit benefit = 1)
      : OpRewritePattern<AppendKVOp>(context, benefit) {}

  LogicalResult matchAndRewrite(AppendKVOp op,
                               PatternRewriter &rewriter) const override {
    // Get the current block size
    auto blockSize = op.getBlockSizeAttr().getInt();
    
    // Get the maximum sequence length
    auto maxSeqLen = op.getMaxSeqLenAttr().getInt();
    
    // If the block size is greater than 1/8 of the max sequence length,
    // consider reducing it for better memory efficiency
    if (blockSize > maxSeqLen / 8) {
      int32_t newBlockSize = std::max(8, static_cast<int>(maxSeqLen / 16));
      
      LLVM_DEBUG(llvm::dbgs() << "Optimizing block size from " << blockSize
                             << " to " << newBlockSize << "\n");
      
      // Create a new operation with the optimized block size
      auto newOp = rewriter.create<AppendKVOp>(
          op.getLoc(), op.getNewKvCache().getType(), op.getBlockIndicesType(),
          op.getKvCache(), op.getKeys(), op.getValues(), op.getSeqIds(),
          rewriter.getI32IntegerAttr(newBlockSize),
          op.getMaxSeqLenAttr());
      
      // Replace the old operation with the new one
      rewriter.replaceOp(op, newOp.getResults());
      
      return success();
    }
    
    // No optimization applied
    return failure();
  }
};

/// Optimizes the append and lookup operations to ensure they work well together.
/// This pattern finds append and lookup operations on the same KV cache and
/// ensures their parameters are compatible.
struct KVCacheOpCompatibilityPattern : public OpRewritePattern<LookupKVOp> {
  explicit KVCacheOpCompatibilityPattern(MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern<LookupKVOp>(context, benefit) {}

  LogicalResult matchAndRewrite(LookupKVOp lookupOp,
                               PatternRewriter &rewriter) const override {
    // Get the KV cache value
    Value kvCache = lookupOp.getKvCache();
    
    // Find the append operation that created this KV cache
    AppendKVOp appendOp = nullptr;
    for (Operation *user : kvCache.getDefiningOp()->getUsers()) {
      if (auto appendKVOp = dyn_cast<AppendKVOp>(user)) {
        if (appendKVOp.getNewKvCache() == kvCache) {
          appendOp = appendKVOp;
          break;
        }
      }
    }
    
    // If we can't find the corresponding append operation, no optimization
    if (!appendOp)
      return failure();
    
    // Verify and fix compatibility issues
    if (lookupOp.getNumHeadsAttr().getInt() != 
        dyn_cast_or_null<PagedKVCacheType>(appendOp.getNewKvCache().getType()).getNumHeads()) {
      // Create a new operation with the correct number of heads
      auto newOp = rewriter.create<LookupKVOp>(
          lookupOp.getLoc(), lookupOp.getKeys().getType(), lookupOp.getValues().getType(),
          lookupOp.getKvCache(), lookupOp.getBlockIndices(), lookupOp.getSeqLens(),
          rewriter.getI32IntegerAttr(dyn_cast<PagedKVCacheType>(appendOp.getNewKvCache().getType()).getNumHeads()),
          lookupOp.getHeadDimAttr());
      
      // Replace the old operation with the new one
      rewriter.replaceOp(lookupOp, newOp.getResults());
      
      return success();
    }
    
    // No optimization applied
    return failure();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// KVCacheOptimizationPass
//===----------------------------------------------------------------------===//

namespace {
class KVCacheOptimizationPass
    : public PassWrapper<KVCacheOptimizationPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KVCacheOptimizationPass)
  
  StringRef getArgument() const final { return "llm-optimize-kv-cache"; }
  
  StringRef getDescription() const final {
    return "Optimize KV cache operations for better performance";
  }
  
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    
    // Add optimization patterns
    RewritePatternSet patterns(context);
    patterns.add<KVCacheBlockSizeOptimizationPattern,
                 KVCacheOpCompatibilityPattern>(context);
    
    // Apply patterns
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llm {

std::unique_ptr<Pass> createKVCacheOptimizationPass() {
  return std::make_unique<KVCacheOptimizationPass>();
}

} // namespace llm
} // namespace mlir 