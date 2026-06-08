//===- RegisterPasses.cpp - Register LLM dialect passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Transforms/BlockSizeAnalysis.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace llm {

void registerLLMOptimizationPipeline();

void registerLLMPasses() {
  registerLLMOptimizationPipeline();

  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createKVCacheOptimizationPass(); });
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createLowerKVCacheOpsPass(); });
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createLLMLoweringPass(); });
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createBlockSizeAnalysisPass(); });
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createBlockSizeOptimizationPass(); });
}

void registerLowerKVCacheOpsPasses() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createLowerKVCacheOpsPass(); });
}

} // namespace llm
} // namespace mlir
