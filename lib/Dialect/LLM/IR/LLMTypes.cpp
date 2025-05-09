//===- LLMTypes.cpp - LLM dialect types implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the types for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// Type Storage Classes
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llm {
namespace detail {

struct PagedKVCacheTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, int64_t, int64_t, int64_t, int64_t, int64_t>;

  PagedKVCacheTypeStorage(Type elementType, int64_t numLayers, int64_t numHeads,
                      int64_t headDim, int64_t blockSize, int64_t maxSeqLen)
      : elementType(elementType), numLayers(numLayers), numHeads(numHeads),
        headDim(headDim), blockSize(blockSize), maxSeqLen(maxSeqLen) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, numLayers, numHeads, headDim, blockSize, maxSeqLen);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                              std::get<3>(key), std::get<4>(key), std::get<5>(key));
  }

  static PagedKVCacheTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PagedKVCacheTypeStorage>())
        PagedKVCacheTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                           std::get<3>(key), std::get<4>(key), std::get<5>(key));
  }

  Type elementType;
  int64_t numLayers;
  int64_t numHeads;
  int64_t headDim;
  int64_t blockSize;
  int64_t maxSeqLen;
};

struct ShardedTensorTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<TensorType, int64_t, int64_t, int64_t>;

  ShardedTensorTypeStorage(TensorType originalType, int64_t shardDim,
                       int64_t numShards, int64_t shardIndex)
      : originalType(originalType), shardDim(shardDim), numShards(numShards),
        shardIndex(shardIndex) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(originalType, shardDim, numShards, shardIndex);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                              std::get<3>(key));
  }

  static ShardedTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<ShardedTensorTypeStorage>())
        ShardedTensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                            std::get<3>(key));
  }

  TensorType originalType;
  int64_t shardDim;
  int64_t numShards;
  int64_t shardIndex;
};

struct QuantizedTensorTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, ArrayRef<int64_t>, bool, bool, int64_t, int64_t, int64_t>;

  QuantizedTensorTypeStorage(Type elementType, ArrayRef<int64_t> shape,
                         bool isSymmetric, bool isPerChannel,
                         int64_t quantAxis, int64_t groupSize, int64_t numBits)
      : elementType(elementType), shape(shape), isSymmetric(isSymmetric),
        isPerChannel(isPerChannel), quantAxis(quantAxis), groupSize(groupSize),
        numBits(numBits) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, shape, isSymmetric, isPerChannel,
                       quantAxis, groupSize, numBits);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(
        std::get<0>(key), llvm::hash_combine_range(std::get<1>(key).begin(), std::get<1>(key).end()),
        std::get<2>(key), std::get<3>(key), std::get<4>(key),
        std::get<5>(key), std::get<6>(key));
  }

  static QuantizedTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    auto shape = allocator.copyInto(std::get<1>(key));
    return new (allocator.allocate<QuantizedTensorTypeStorage>())
        QuantizedTensorTypeStorage(std::get<0>(key), shape, std::get<2>(key),
                              std::get<3>(key), std::get<4>(key),
                              std::get<5>(key), std::get<6>(key));
  }

  Type elementType;
  ArrayRef<int64_t> shape;
  bool isSymmetric;
  bool isPerChannel;
  int64_t quantAxis;
  int64_t groupSize;
  int64_t numBits;
};

} // namespace detail
} // namespace llm
} // namespace mlir

//===----------------------------------------------------------------------===//
// PagedKVCache Type
//===----------------------------------------------------------------------===//

PagedKVCacheType PagedKVCacheType::get(Type elementType, int64_t numLayers,
                                  int64_t numHeads, int64_t headDim,
                                  int64_t blockSize, int64_t maxSeqLen) {
  return Base::get(elementType.getContext(), elementType, numLayers, numHeads,
                  headDim, blockSize, maxSeqLen);
}

Type PagedKVCacheType::getElementType() {
  return getImpl()->elementType;
}

int64_t PagedKVCacheType::getNumLayers() {
  return getImpl()->numLayers;
}

int64_t PagedKVCacheType::getNumHeads() {
  return getImpl()->numHeads;
}

int64_t PagedKVCacheType::getHeadDim() {
  return getImpl()->headDim;
}

int64_t PagedKVCacheType::getBlockSize() {
  return getImpl()->blockSize;
}

int64_t PagedKVCacheType::getMaxSeqLen() {
  return getImpl()->maxSeqLen;
}

//===----------------------------------------------------------------------===//
// ShardedTensor Type
//===----------------------------------------------------------------------===//

ShardedTensorType ShardedTensorType::get(TensorType originalType, int64_t shardDim,
                                    int64_t numShards, int64_t shardIndex) {
  return Base::get(originalType.getContext(), originalType, shardDim, numShards,
                  shardIndex);
}

TensorType ShardedTensorType::getOriginalType() {
  return getImpl()->originalType;
}

int64_t ShardedTensorType::getShardDim() {
  return getImpl()->shardDim;
}

int64_t ShardedTensorType::getNumShards() {
  return getImpl()->numShards;
}

int64_t ShardedTensorType::getShardIndex() {
  return getImpl()->shardIndex;
}

//===----------------------------------------------------------------------===//
// QuantizedTensor Type
//===----------------------------------------------------------------------===//

QuantizedTensorType QuantizedTensorType::get(Type elementType, ArrayRef<int64_t> shape,
                                        bool isSymmetric, bool isPerChannel,
                                        int64_t quantAxis, int64_t groupSize,
                                        int64_t numBits) {
  return Base::get(elementType.getContext(), elementType, shape, isSymmetric,
                  isPerChannel, quantAxis, groupSize, numBits);
}

Type QuantizedTensorType::getElementType() {
  return getImpl()->elementType;
}

ArrayRef<int64_t> QuantizedTensorType::getShape() {
  return getImpl()->shape;
}

bool QuantizedTensorType::isSymmetric() {
  return getImpl()->isSymmetric;
}

bool QuantizedTensorType::isPerChannel() {
  return getImpl()->isPerChannel;
}

int64_t QuantizedTensorType::getQuantAxis() {
  return getImpl()->quantAxis;
}

int64_t QuantizedTensorType::getGroupSize() {
  return getImpl()->groupSize;
}

int64_t QuantizedTensorType::getNumBits() {
  return getImpl()->numBits;
}

//===----------------------------------------------------------------------===//
// Generated Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLM/IR/LLMTypes.cpp.inc" 