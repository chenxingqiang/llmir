//===- LLMDialect.cpp - LLM dialect implementation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLM dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::llm;

//===----------------------------------------------------------------------===//
// Include type storage definitions (must come before dialect initialization)
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLM/IR/LLMTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// LLM Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLMOpsDialect.cpp.inc"

void LLMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/LLM/IR/LLMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLM/IR/LLMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute LLMDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Currently, no custom attributes defined for LLM dialect
  parser.emitError(parser.getCurrentLocation(), "unknown LLM attribute");
  return Attribute();
}

void LLMDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Currently, no custom attributes defined for LLM dialect
  llvm_unreachable("unexpected LLM attribute");
}
