//===- LLM.h - LLM dialect declarations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_IR_LLM_H_
#define MLIR_DIALECT_LLM_IR_LLM_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated headers (in build directory)
#include "mlir/Dialect/LLM/IR/LLMOpsDialect.h.inc"
#include "mlir/Dialect/LLM/IR/LLMTypes.h.inc"

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// LLM Dialect
//===----------------------------------------------------------------------===//

// Forward declarations
class PagedKVCacheType;
class ShardedTensorType;
class QuantizedTensorType;

namespace detail {
struct PagedKVCacheTypeStorage;
struct ShardedTensorTypeStorage;
struct QuantizedTensorTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Type Declarations
//===----------------------------------------------------------------------===//

class PagedKVCacheType : public Type::TypeBase<PagedKVCacheType, Type,
                                           detail::PagedKVCacheTypeStorage> {
public:
  using Base::Base;
  
  // Type constructors and accessors will be generated
  static PagedKVCacheType get(Type elementType, int64_t numLayers,
                          int64_t numHeads, int64_t headDim,
                          int64_t blockSize, int64_t maxSeqLen);

  Type getElementType();
  int64_t getNumLayers();
  int64_t getNumHeads();
  int64_t getHeadDim();
  int64_t getBlockSize();
  int64_t getMaxSeqLen();
};

class ShardedTensorType : public Type::TypeBase<ShardedTensorType, Type,
                                            detail::ShardedTensorTypeStorage> {
public:
  using Base::Base;
  
  static ShardedTensorType get(TensorType originalType, int64_t shardDim,
                           int64_t numShards, int64_t shardIndex);

  TensorType getOriginalType();
  int64_t getShardDim();
  int64_t getNumShards();
  int64_t getShardIndex();
};

class QuantizedTensorType : public Type::TypeBase<QuantizedTensorType, Type,
                                              detail::QuantizedTensorTypeStorage> {
public:
  using Base::Base;
  
  static QuantizedTensorType get(Type elementType, ArrayRef<int64_t> shape,
                             bool isSymmetric, bool isPerChannel,
                             int64_t quantAxis, int64_t groupSize,
                             int64_t numBits);

  Type getElementType();
  ArrayRef<int64_t> getShape();
  bool isSymmetric();
  bool isPerChannel();
  int64_t getQuantAxis();
  int64_t getGroupSize();
  int64_t getNumBits();
};

} // namespace llm
} // namespace mlir

// Generated operation declarations
#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMOps.h.inc"

// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLM/IR/LLMTypes.h.inc"

#endif // MLIR_DIALECT_LLM_IR_LLM_H_ 