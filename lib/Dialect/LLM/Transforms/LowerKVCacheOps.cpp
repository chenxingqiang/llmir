//===- LowerKVCacheOps.cpp - Lower KV cache operations to runtime calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to lower LLM KV cache operations to runtime calls.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLMKVCacheOps.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

#define GEN_PASS_DEF_LOWERKVCACHEOPS
#include "mlir/Dialect/LLM/Transforms/Passes.h.inc"

namespace {

// Helper function to extract dimensions from a tensor
SmallVector<Value> extractTensorDims(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = tensor.getType().cast<ShapedType>();
  SmallVector<Value> dims;
  
  for (int i = 0; i < tensorType.getRank(); ++i) {
    dims.push_back(builder.create<tensor::DimOp>(loc, tensor, i));
  }
  
  return dims;
}

// Helper function to create a runtime function call
func::CallOp createRuntimeCall(OpBuilder &builder, Location loc, 
                               StringRef functionName,
                               TypeRange resultTypes,
                               ValueRange operands) {
  return builder.create<func::CallOp>(loc, functionName, resultTypes, operands);
}

// Pattern to lower AppendKVOp to runtime call
struct AppendKVOpLowering : public OpConversionPattern<AppendKVOp> {
  using OpConversionPattern<AppendKVOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AppendKVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Get dimensions from the keys tensor
    auto keysTensor = adaptor.getKeys();
    auto keysType = keysTensor.getType().cast<ShapedType>();
    auto dims = extractTensorDims(rewriter, loc, keysTensor);
    
    // Extract batch size and sequence length from the keys tensor
    Value batchSize = dims[0];
    Value seqLen = dims[1];
    
    // Create a call to the runtime appendKV function
    auto kvCache = adaptor.getKVCache();
    if (!kvCache) {
      // If no KV cache is provided, create a new one
      auto numLayers = rewriter.create<arith::ConstantIndexOp>(loc, 1); // Default to 1 layer
      auto numHeads = rewriter.create<arith::ConstantIndexOp>(loc, keysType.getDimSize(2));
      auto headDim = rewriter.create<arith::ConstantIndexOp>(loc, keysType.getDimSize(3));
      auto blockSize = rewriter.create<arith::ConstantIndexOp>(loc, 16); // Default block size
      auto maxSeqLen = rewriter.create<arith::ConstantIndexOp>(loc, 4096); // Default max sequence length
      
      // Create call to create a new KV cache
      kvCache = createRuntimeCall(
          rewriter, loc, "mlir_llm_create_paged_kv_cache",
          {rewriter.getType<KVCacheType>()},
          {numLayers, numHeads, headDim, blockSize, maxSeqLen, 
           rewriter.create<arith::ConstantIntOp>(loc, 0, 1) // useGPU = false by default
          }).getResult(0);
    }
    
    // Create the result types
    auto blockIndicesType = op.getBlockIndices().getType();
    
    // Check if enable_sharing attribute is set
    Value enableSharing = rewriter.create<arith::ConstantIntOp>(loc, 0, 1); // Default to false
    if (op->hasAttr("enable_sharing")) {
      auto sharingAttr = op->getAttr("enable_sharing").cast<BoolAttr>();
      enableSharing = rewriter.create<arith::ConstantIntOp>(
          loc, sharingAttr.getValue() ? 1 : 0, 1);
    }
    
    // Create call to appendKV runtime function with the additional enable_sharing parameter
    auto results = createRuntimeCall(
        rewriter, loc, "mlir_llm_append_kv",
        {rewriter.getType<KVCacheType>(), blockIndicesType},
        {kvCache, keysTensor, adaptor.getValues(), adaptor.getSeqIds(), 
         batchSize, seqLen, enableSharing}).getResults();
    
    // Replace the original op with the results of the runtime call
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Pattern to lower LookupKVOp to runtime call
struct LookupKVOpLowering : public OpConversionPattern<LookupKVOp> {
  using OpConversionPattern<LookupKVOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LookupKVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Get dimensions from the block indices tensor
    auto blockIndicesTensor = adaptor.getBlockIndices();
    auto dims = extractTensorDims(rewriter, loc, blockIndicesTensor);
    
    // Extract batch size and sequence length from the block indices tensor
    Value batchSize = dims[0];
    Value maxSeqLen = dims[1];
    
    // Create the result types
    auto keysType = op.getKeys().getType();
    auto valuesType = op.getValues().getType();
    
    // Create call to lookupKV runtime function
    auto results = createRuntimeCall(
        rewriter, loc, "mlir_llm_lookup_kv",
        {keysType, valuesType},
        {adaptor.getKVCache(), blockIndicesTensor, adaptor.getSeqLens(), 
         batchSize, maxSeqLen}).getResults();
    
    // Replace the original op with the results of the runtime call
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Pattern to lower PagedAttentionOp to runtime call
struct PagedAttentionOpLowering : public OpConversionPattern<PagedAttentionOp> {
  using OpConversionPattern<PagedAttentionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Get dimensions from the query tensor
    auto queryTensor = adaptor.getQuery();
    auto queryType = queryTensor.getType().cast<ShapedType>();
    auto dims = extractTensorDims(rewriter, loc, queryTensor);
    
    // Extract batch size and sequence length from the query tensor
    Value batchSize = dims[0];
    Value seqLen = dims[1];
    
    // Extract number of heads and head dimension if available
    Value numHeads, headDim;
    if (queryType.getRank() >= 4) {
      numHeads = dims[2];
      headDim = dims[3];
    } else {
      numHeads = rewriter.create<arith::ConstantIndexOp>(loc, op.getNumHeads());
      headDim = rewriter.create<arith::ConstantIndexOp>(loc, op.getHeadDim());
    }
    
    // Create scale parameter
    Value scale;
    if (op.getScale()) {
      scale = rewriter.create<arith::ConstantFloatOp>(
          loc, op.getScaleAttr().getValueAsDouble(), 
          rewriter.getF32Type());
    } else {
      // Default scale: 1.0 / sqrt(head_dim)
      auto headDimF = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getF32Type(), headDim);
      auto sqrtHeadDim = rewriter.create<math::SqrtOp>(loc, headDimF);
      scale = rewriter.create<arith::DivFOp>(
          loc, rewriter.create<arith::ConstantFloatOp>(
                  loc, 1.0, rewriter.getF32Type()),
          sqrtHeadDim);
    }
    
    // Create the result type
    auto outputType = op.getAttentionOutput().getType();
    
    // Create call to pagedAttention runtime function
    auto result = createRuntimeCall(
        rewriter, loc, "mlir_llm_paged_attention",
        {outputType},
        {adaptor.getKVCache(), queryTensor, adaptor.getBlockIndices(),
         adaptor.getSeqLens(), batchSize, seqLen, numHeads, headDim, scale}).getResult(0);
    
    // Replace the original op with the result of the runtime call
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Conversion pass that lowers KV cache operations to runtime calls
struct LowerKVCacheOpsPass
    : public impl::LowerKVCacheOpsBase<LowerKVCacheOpsPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // Add patterns to convert KV cache operations
    patterns.add<AppendKVOpLowering, LookupKVOpLowering, PagedAttentionOpLowering>(context);

    // Operations that should be legal after conversion
    target.addLegalDialect<func::FuncDialect, arith::ArithDialect, 
                           tensor::TensorDialect, math::MathDialect>();
    
    // Mark the KV cache operations as illegal
    target.addIllegalOp<AppendKVOp, LookupKVOp, PagedAttentionOp>();

    // Apply the conversion
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

/// Registers the pass with the pass registry
void mlir::llm::registerLowerKVCacheOpsPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<LowerKVCacheOpsPass>();
  });
} 