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
// InferTypeOpInterface Implementation
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the query tensor type
  auto queryType = operands[0].getType().cast<TensorType>();
  
  // Result type has the same shape and element type as the query
  inferredReturnTypes.push_back(queryType);
  return success();
}

LogicalResult PagedAttentionOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the query tensor type
  auto queryType = operands[0].getType().cast<TensorType>();
  
  // Result type has the same shape and element type as the query
  inferredReturnTypes.push_back(queryType);
  return success();
}

LogicalResult AppendKVOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // First return value is updated KV cache, same type as input
  inferredReturnTypes.push_back(operands[0].getType());
  
  // Second return value is block indexes
  auto seqIdsType = operands[3].getType().cast<TensorType>();
  auto seqIdsShape = seqIdsType.getShape();
  auto resultShape = seqIdsShape;
  
  // Infer block indexes tensor type with same shape as seq_ids but i32 elements
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, IntegerType::get(context, 32)));
  
  return success();
}

LogicalResult LookupKVOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the KV cache type
  auto kvCacheType = operands[0].getType().cast<PagedKVCacheType>();
  auto seqLensType = operands[2].getType().cast<TensorType>();
  
  // Infer the shape of the output keys and values
  auto batchSize = seqLensType.getDimSize(0);
  auto elementType = kvCacheType.getElementType();
  auto numHeads = kvCacheType.getNumHeads();
  auto headDim = kvCacheType.getHeadDim();
  
  // Get max sequence length from the seqLens tensor
  // For now, use ShapedType::kDynamic to represent variable length
  int64_t maxSeqLen = ShapedType::kDynamic;
  
  // For keys and values, shape is [batch_size, max_seq_len, num_heads, head_dim]
  SmallVector<int64_t, 4> resultShape = {batchSize, maxSeqLen, numHeads, headDim};
  
  // Keys and values have the same shape and element type
  auto resultType = RankedTensorType::get(resultShape, elementType);
  inferredReturnTypes.push_back(resultType);  // keys
  inferredReturnTypes.push_back(resultType);  // values
  
  return success();
}

LogicalResult QuantizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor type
  auto inputType = operands[0].getType().cast<TensorType>();
  auto inputShape = inputType.getShape();
  
  // Extract attribute values
  auto bitsAttr = attributes.get("bits").cast<IntegerAttr>();
  auto symmetricAttr = attributes.get("symmetric").cast<BoolAttr>();
  auto axisAttr = attributes.get("axis").cast<IntegerAttr>();
  auto groupSizeAttr = attributes.get("group_size").cast<IntegerAttr>();
  
  // Determine the element type based on bits
  IntegerType elementType;
  if (bitsAttr.getInt() <= 4)
    elementType = IntegerType::get(context, 4);
  else if (bitsAttr.getInt() <= 8)
    elementType = IntegerType::get(context, 8);
  else
    elementType = IntegerType::get(context, 16);
  
  // Create the quantized tensor type
  auto quantizedType = QuantizedTensorType::get(
      elementType, inputShape, symmetricAttr.getValue(),
      axisAttr.getInt() >= 0, axisAttr.getInt(), groupSizeAttr.getInt(),
      bitsAttr.getInt());
  
  inferredReturnTypes.push_back(quantizedType);
  return success();
}

LogicalResult DequantizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the quantized tensor type
  auto quantizedType = operands[0].getType().cast<QuantizedTensorType>();
  auto shape = quantizedType.getShape();
  
  // Result is float tensor with same shape
  inferredReturnTypes.push_back(RankedTensorType::get(
      shape, FloatType::getF32(context)));
  
  return success();
}

LogicalResult QuantizedMatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get input and weight types
  auto lhsType = operands[0].getType().cast<TensorType>();
  auto rhsType = operands[1].getType().cast<QuantizedTensorType>();
  
  // Get shapes
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  // Matrix multiplication: [I, K] * [K, J] -> [I, J]
  SmallVector<int64_t, 4> resultShape;
  if (lhsShape.size() == 2 && rhsShape.size() == 2) {
    resultShape = {lhsShape[0], rhsShape[1]};
  } else {
    // Handle more complex broadcasting cases
    // For simplicity, use dynamic shapes
    resultShape = {ShapedType::kDynamic, ShapedType::kDynamic};
  }
  
  // Result has same element type as lhs
  auto elementType = lhsType.getElementType();
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  
  return success();
}

LogicalResult ShardedLinearOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get input and weight types
  auto inputType = operands[0].getType().cast<TensorType>();
  auto weightType = operands[1].getType().cast<TensorType>();
  
  // Get shapes
  auto inputShape = inputType.getShape();
  auto weightShape = weightType.getShape();
  
  // Extract attribute values
  auto shardDimAttr = attributes.get("shardDim").cast<IntegerAttr>();
  auto numShardsAttr = attributes.get("numShards").cast<IntegerAttr>();
  
  // Linear: [batch_size, in_features] * [in_features, out_features] -> [batch_size, out_features]
  // For sharded linear, the output features dimension is divided by numShards
  SmallVector<int64_t, 2> resultShape = {inputShape[0], weightShape[1]};
  
  // Result has same element type as input
  auto elementType = inputType.getElementType();
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  
  return success();
}

LogicalResult AllGatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get input tensor type
  auto inputType = operands[0].getType().cast<TensorType>();
  auto inputShape = inputType.getShape();
  
  // Extract attribute values
  auto dimAttr = attributes.get("dim").cast<IntegerAttr>();
  auto groupSizeAttr = attributes.get("groupSize").cast<IntegerAttr>();
  auto dim = dimAttr.getInt();
  auto groupSize = groupSizeAttr.getInt();
  
  // Output shape is same as input, but with the specified dimension multiplied by groupSize
  SmallVector<int64_t, 4> resultShape(inputShape.begin(), inputShape.end());
  if (dim >= 0 && dim < (int64_t)resultShape.size() && resultShape[dim] != ShapedType::kDynamic) {
    resultShape[dim] *= groupSize;
  }
  
  // Result has same element type as input
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType()));
  
  return success();
}

LogicalResult ReduceScatterOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get input tensor type
  auto inputType = operands[0].getType().cast<TensorType>();
  auto inputShape = inputType.getShape();
  
  // Extract attribute values
  auto dimAttr = attributes.get("dim").cast<IntegerAttr>();
  auto groupSizeAttr = attributes.get("groupSize").cast<IntegerAttr>();
  auto dim = dimAttr.getInt();
  auto groupSize = groupSizeAttr.getInt();
  
  // Output shape is same as input, but with the specified dimension divided by groupSize
  SmallVector<int64_t, 4> resultShape(inputShape.begin(), inputShape.end());
  if (dim >= 0 && dim < (int64_t)resultShape.size() && resultShape[dim] != ShapedType::kDynamic) {
    resultShape[dim] = resultShape[dim] / groupSize;
  }
  
  // Result has same element type as input
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType()));
  
  return success();
}

//===----------------------------------------------------------------------===//
// Op Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMOps.cpp.inc" 