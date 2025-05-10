//===- LLMTypes.h - Types for the LLM dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_IR_LLMTYPES_H_
#define MLIR_DIALECT_LLM_IR_LLMTYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

// Forward declaration
namespace mlir {
namespace llm {
class LLMDialect;
}
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLM/IR/LLMTypes.h.inc"

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// KVCacheType
//===----------------------------------------------------------------------===//

/// KVCacheType represents a key-value cache for transformer models
class KVCacheType : public Type::TypeBase<KVCacheType, Type, TypeStorage> {
public:
  using Base::Base;

  /// Get an instance of the KVCacheType
  static KVCacheType get(MLIRContext *context);

  /// Support method to enable LLVM-style RTTI type casting
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

/// Check if a type is an LLM dialect type
bool isLLMDialectType(Type type);

} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_IR_LLMTYPES_H_ 