//===- LLMKVCacheOpsInterface.cpp - KV cache op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements interfaces for LLM KV cache operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/LLM/Runtime/RuntimeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// KVCacheInterface Implementation
//===----------------------------------------------------------------------===//

namespace {
// Default implementation for the KVCacheInterface
struct DefaultKVCacheInterface : public KVCacheInterface::ExternalModel<
                                   DefaultKVCacheInterface, Operation> {
  bool usesKVCache(Operation *op) { return false; }
  
  int64_t getNumKVTokens(Operation *op) { return -1; }
  
  Value getKVCacheInput(Operation *op) { return nullptr; }
  
  Value getKVCacheOutput(Operation *op) { return nullptr; }
};
} // end anonymous namespace

void mlir::llm::registerKVCacheInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLMDialect *dialect) {
    Operation::attachInterface<DefaultKVCacheInterface>(*ctx);
  });
}

//===----------------------------------------------------------------------===//
// AttentionInterface Implementation
//===----------------------------------------------------------------------===//

namespace {
// Default implementation for the AttentionInterface
struct DefaultAttentionInterface : public AttentionInterface::ExternalModel<
                                    DefaultAttentionInterface, Operation> {
  bool isAttentionOp(Operation *op) { return false; }
  
  int64_t getBatchSize(Operation *op) { return -1; }
  
  int64_t getSeqLength(Operation *op) { return -1; }
  
  int64_t getNumHeads(Operation *op) { return -1; }
  
  int64_t getHeadDim(Operation *op) { return -1; }
};
} // end anonymous namespace

void mlir::llm::registerAttentionInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLMDialect *dialect) {
    Operation::attachInterface<DefaultAttentionInterface>(*ctx);
  });
} 