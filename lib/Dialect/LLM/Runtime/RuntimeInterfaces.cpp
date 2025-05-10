//===- RuntimeInterfaces.cpp - MLIR LLM Runtime interfaces -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements runtime interfaces for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/RuntimeInterfaces.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// KVCacheInterface
//===----------------------------------------------------------------------===//

// Implementations of KVCacheInterface methods would go here

//===----------------------------------------------------------------------===//
// AttentionInterface
//===----------------------------------------------------------------------===//

// Implementations of AttentionInterface methods would go here

//===----------------------------------------------------------------------===//
// External Model Registration
//===----------------------------------------------------------------------===//

void registerKVCacheInterfaceExternalModels(DialectRegistry &registry) {
  // Register external models for operations implementing KVCacheInterface
}

void registerAttentionInterfaceExternalModels(DialectRegistry &registry) {
  // Register external models for operations implementing AttentionInterface
}

} // namespace llm
} // namespace mlir 