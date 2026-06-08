//===- BlockSizeAnalysis.h - KV cache block size analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_TRANSFORMS_BLOCKSIZEANALYSIS_H_
#define MLIR_DIALECT_LLM_TRANSFORMS_BLOCKSIZEANALYSIS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace llm {

/// Result of block size analysis (Algorithm 1 in the LLMIR paper).
struct BlockSizeAnalysisResult {
  int64_t optimalBlockSize = 128;
  double fragmentationScore = 0.0;
  double gpuUtilization = 0.0;
  double memoryAlignmentScore = 0.0;
  double combinedScore = 0.0;
};

/// Analyze ``func`` and return the recommended KV cache block size.
BlockSizeAnalysisResult analyzeBlockSizeForFunc(func::FuncOp func);

/// Rewrite ``block_size`` attributes on ``llm.append_kv`` ops in ``func``.
void applyBlockSizeOptimizationToFunc(func::FuncOp func);

std::unique_ptr<Pass> createBlockSizeAnalysisPass();
std::unique_ptr<Pass> createBlockSizeOptimizationPass();

} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_TRANSFORMS_BLOCKSIZEANALYSIS_H_
