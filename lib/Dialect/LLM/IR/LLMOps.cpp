//===- LLMOps.cpp - LLM dialect ops implementation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operations for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// InferTypeOpInterface Implementations
//===----------------------------------------------------------------------===//

LogicalResult AppendKVOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // First return value is updated KV cache, same type as input
  inferredReturnTypes.push_back(operands[0].getType());
  
  // Second return value is block indexes
  auto seqIdsType = cast<TensorType>(operands[3].getType());
  auto seqIdsShape = seqIdsType.getShape();
  
  inferredReturnTypes.push_back(RankedTensorType::get(
      seqIdsShape, IntegerType::get(context, 32)));
  
  return success();
}

LogicalResult LookupKVOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto kvCacheType = cast<PagedKVCacheType>(operands[0].getType());
  auto seqLensType = cast<TensorType>(operands[2].getType());
  
  auto batchSize = seqLensType.getDimSize(0);
  auto elementType = kvCacheType.getElementType();
  auto numHeads = kvCacheType.getNumHeads();
  auto headDim = kvCacheType.getHeadDim();
  
  int64_t maxSeqLen = ShapedType::kDynamic;
  SmallVector<int64_t, 4> resultShape = {batchSize, maxSeqLen, numHeads, headDim};
  
  auto resultType = RankedTensorType::get(resultShape, elementType);
  inferredReturnTypes.push_back(resultType);  // keys
  inferredReturnTypes.push_back(resultType);  // values
  
  return success();
}

LogicalResult QuantizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = cast<TensorType>(operands[0].getType());
  auto inputShape = inputType.getShape();
  
  int64_t bits = 8;
  if (auto bitsAttr = attributes.getNamed("bits"))
    bits = cast<IntegerAttr>(bitsAttr->getValue()).getInt();
  
  bool symmetric = true;
  if (auto symmetricAttr = attributes.getNamed("symmetric"))
    symmetric = cast<BoolAttr>(symmetricAttr->getValue()).getValue();
  
  int64_t axis = -1;
  if (auto axisAttr = attributes.getNamed("axis"))
    axis = cast<IntegerAttr>(axisAttr->getValue()).getInt();
  
  int64_t groupSize = 128;
  if (auto groupSizeAttr = attributes.getNamed("groupSize"))
    groupSize = cast<IntegerAttr>(groupSizeAttr->getValue()).getInt();
  
  IntegerType elementType;
  if (bits <= 4)
    elementType = IntegerType::get(context, 4);
  else if (bits <= 8)
    elementType = IntegerType::get(context, 8);
  else
    elementType = IntegerType::get(context, 16);
  
  auto quantizedType = QuantizedTensorType::get(
      context, elementType, inputShape, symmetric,
      axis >= 0, axis, groupSize, bits);
  
  inferredReturnTypes.push_back(quantizedType);
  return success();
}

LogicalResult DequantizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto quantizedType = cast<QuantizedTensorType>(operands[0].getType());
  auto shape = quantizedType.getShape();
  
  inferredReturnTypes.push_back(RankedTensorType::get(
      shape, FloatType::getF32(context)));
  
  return success();
}

LogicalResult QuantizedMatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = cast<TensorType>(operands[0].getType());
  auto rhsType = cast<QuantizedTensorType>(operands[1].getType());
  
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  SmallVector<int64_t, 4> resultShape;
  if (lhsShape.size() == 2 && rhsShape.size() == 2) {
    resultShape = {lhsShape[0], rhsShape[1]};
  } else {
    resultShape = {ShapedType::kDynamic, ShapedType::kDynamic};
  }
  
  auto elementType = lhsType.getElementType();
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  
  return success();
}

LogicalResult ShardedLinearOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = cast<TensorType>(operands[0].getType());
  auto weightType = cast<TensorType>(operands[1].getType());
  
  auto inputShape = inputType.getShape();
  auto weightShape = weightType.getShape();
  
  SmallVector<int64_t, 2> resultShape = {inputShape[0], weightShape[1]};
  
  auto elementType = inputType.getElementType();
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  
  return success();
}

LogicalResult AllGatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = cast<TensorType>(operands[0].getType());
  auto inputShape = inputType.getShape();
  
  int64_t dim = 0;
  if (auto dimAttr = attributes.getNamed("dim"))
    dim = cast<IntegerAttr>(dimAttr->getValue()).getInt();
  
  int64_t groupSize = 1;
  if (auto groupSizeAttr = attributes.getNamed("groupSize"))
    groupSize = cast<IntegerAttr>(groupSizeAttr->getValue()).getInt();
  
  SmallVector<int64_t, 4> resultShape(inputShape.begin(), inputShape.end());
  if (dim >= 0 && dim < (int64_t)resultShape.size() && resultShape[dim] != ShapedType::kDynamic) {
    resultShape[dim] *= groupSize;
  }
  
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType()));
  
  return success();
}

LogicalResult ReduceScatterOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = cast<TensorType>(operands[0].getType());
  auto inputShape = inputType.getShape();
  
  int64_t dim = 0;
  if (auto dimAttr = attributes.getNamed("dim"))
    dim = cast<IntegerAttr>(dimAttr->getValue()).getInt();
  
  int64_t groupSize = 1;
  if (auto groupSizeAttr = attributes.getNamed("groupSize"))
    groupSize = cast<IntegerAttr>(groupSizeAttr->getValue()).getInt();
  
  SmallVector<int64_t, 4> resultShape(inputShape.begin(), inputShape.end());
  if (dim >= 0 && dim < (int64_t)resultShape.size() && resultShape[dim] != ShapedType::kDynamic) {
    resultShape[dim] = resultShape[dim] / groupSize;
  }
  
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType()));
  
  return success();
}

//===----------------------------------------------------------------------===//
// Op Definitions (TableGen generated)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMOps.cpp.inc"
