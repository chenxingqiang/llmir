//===- Passes.h - LLM dialect passes ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the LLM Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_LLM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class MLIRContext;
class Pass;

namespace func {
class FuncOp;
} // namespace func

namespace llm {

/// Creates a pass to optimize KV cache operations.
/// This pass includes:
/// - Block size optimization for better memory utilization
/// - KV cache operation compatibility checking
/// - Parameter tuning for improved performance
std::unique_ptr<Pass> createKVCacheOptimizationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LLM/Transforms/Passes.h.inc"

} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_TRANSFORMS_PASSES_H_ 