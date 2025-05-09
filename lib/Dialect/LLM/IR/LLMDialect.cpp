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
// LLM Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLMOpsDialect.cpp.inc"

void LLMDialect::initialize() {
  addTypes<PagedKVCacheType, ShardedTensorType, QuantizedTensorType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLM/IR/LLMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Parse: !llm.paged_kv_cache<elementType, numLayers, numHeads, headDim, blockSize, maxSeqLen>
Type LLMDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  Type type;
  OptionalParseResult result =
      generatedTypeParser(parser, keyword, type);
  if (result.has_value())
    return result.value() ? type : Type();

  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return Type();
}

// Print: !llm.paged_kv_cache<elementType, numLayers, numHeads, headDim, blockSize, maxSeqLen>
void LLMDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (succeeded(generatedTypePrinter(type, os)))
    return;

  llvm_unreachable("unexpected LLM type");
} 