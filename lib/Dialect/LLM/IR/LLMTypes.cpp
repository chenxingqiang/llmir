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
// QuantizedTensorType Custom Parsing/Printing
//===----------------------------------------------------------------------===//

Type QuantizedTensorType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  
  // Parse element type
  Type elementType;
  if (parser.parseType(elementType))
    return {};
  
  if (parser.parseComma())
    return {};
  
  // Parse shape as [d0 x d1 x ... x dn]
  if (parser.parseLSquare())
    return {};
  
  SmallVector<int64_t, 4> shape;
  
  // Check for empty shape
  if (succeeded(parser.parseOptionalRSquare())) {
    // Empty shape, continue
  } else {
    // Parse dimensions
    int64_t dim;
    if (parser.parseInteger(dim))
      return {};
    shape.push_back(dim);
    
    while (succeeded(parser.parseOptionalKeyword("x"))) {
      if (parser.parseInteger(dim))
        return {};
      shape.push_back(dim);
    }
    
    if (parser.parseRSquare())
      return {};
  }
  
  if (parser.parseComma())
    return {};
  
  // Parse boolean for isSymmetric
  bool isSymmetric = false;
  if (succeeded(parser.parseOptionalKeyword("symmetric"))) {
    isSymmetric = true;
  } else if (failed(parser.parseKeyword("asymmetric"))) {
    return {};
  }
  
  if (parser.parseComma())
    return {};
  
  // Parse boolean for isPerChannel
  bool isPerChannel = false;
  if (succeeded(parser.parseOptionalKeyword("per_channel"))) {
    isPerChannel = true;
  } else if (failed(parser.parseKeyword("per_tensor"))) {
    return {};
  }
  
  if (parser.parseComma())
    return {};
  
  // Parse quantAxis
  int64_t quantAxis;
  if (parser.parseInteger(quantAxis))
    return {};
  
  if (parser.parseComma())
    return {};
  
  // Parse groupSize
  int64_t groupSize;
  if (parser.parseInteger(groupSize))
    return {};
  
  if (parser.parseComma())
    return {};
  
  // Parse numBits
  int64_t numBits;
  if (parser.parseInteger(numBits))
    return {};
  
  if (parser.parseGreater())
    return {};
  
  return QuantizedTensorType::get(parser.getContext(), elementType, shape,
                                  isSymmetric, isPerChannel, quantAxis,
                                  groupSize, numBits);
}

void QuantizedTensorType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getElementType());
  printer << ", [";
  
  auto shape = getShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      printer << " x ";
    printer << shape[i];
  }
  printer << "], ";
  
  printer << (getIsSymmetric() ? "symmetric" : "asymmetric");
  printer << ", ";
  printer << (getIsPerChannel() ? "per_channel" : "per_tensor");
  printer << ", ";
  printer << getQuantAxis() << ", ";
  printer << getGroupSize() << ", ";
  printer << getNumBits();
  
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llm {

bool isLLMDialectType(Type type) {
  return type.getDialect().getNamespace() == LLMDialect::getDialectNamespace();
}

} // namespace llm
} // namespace mlir
