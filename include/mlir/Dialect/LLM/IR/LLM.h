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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect header
#include "mlir/Dialect/LLM/IR/LLMOpsDialect.h.inc"

namespace mlir {
namespace llm {

// Forward declarations for type storage
namespace detail {
struct PagedKVCacheTypeStorage;
struct ShardedTensorTypeStorage;
struct QuantizedTensorTypeStorage;
} // namespace detail

} // namespace llm
} // namespace mlir

// Generated type declarations - includes full type definitions
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLM/IR/LLMTypes.h.inc"

// Generated operation declarations
#define GET_OP_CLASSES
#include "mlir/Dialect/LLM/IR/LLMOps.h.inc"

#endif // MLIR_DIALECT_LLM_IR_LLM_H_
