//===- LLMBase.td - LLM dialect base definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains base definitions for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLM_BASE
#define LLM_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLM Dialect
//===----------------------------------------------------------------------===//

def LLM_Dialect : Dialect {
  let name = "llm";
  let cppNamespace = "::mlir::llm";
  let description = [{
    The LLM dialect provides operations and types for large language model inference.
    It focuses on efficient implementations of key LLM operations like attention
    and specialized types like paged KV caches for optimized memory usage.
  }];
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class LLM_Op<string mnemonic, list<Trait> traits = []> :
    Op<LLM_Dialect, mnemonic, traits>;

#endif // LLM_BASE 