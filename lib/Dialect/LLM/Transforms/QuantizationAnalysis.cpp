//===- QuantizationAnalysis.cpp - Quantization safety analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements quantization safety analysis for LLM operations.
// It identifies which operations can be safely quantized and recommends
// appropriate precision levels.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

using namespace mlir;

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// Quantization Safety Levels
//===----------------------------------------------------------------------===//

/// Safety level for quantization.
enum class QuantSafetyLevel {
  High,    // Safe for INT4/INT8 quantization
  Medium,  // Safe for INT8 only
  Low,     // FP16 recommended
  None     // Must keep FP32
};

/// Result of quantization safety analysis for an operation.
struct QuantAnalysisResult {
  QuantSafetyLevel level;
  int recommendedBits;
  double confidenceScore;
  std::string reason;
  
  QuantAnalysisResult(QuantSafetyLevel l = QuantSafetyLevel::None,
                      int bits = 32, double conf = 0.5,
                      std::string r = "")
      : level(l), recommendedBits(bits), confidenceScore(conf), reason(std::move(r)) {}
};

//===----------------------------------------------------------------------===//
// Quantization Safety Analyzer
//===----------------------------------------------------------------------===//

/// Analyzer for determining quantization safety of operations.
class QuantizationSafetyAnalyzer {
public:
  explicit QuantizationSafetyAnalyzer(double errorThreshold = 0.01)
      : errorThreshold_(errorThreshold) {}
  
  /// Analyze a single operation for quantization safety.
  QuantAnalysisResult analyze(Operation *op) {
    StringRef opName = op->getName().getStringRef();
    
    // Attention operations
    if (opName.contains("attention")) {
      return analyzeAttention(op);
    }
    
    // Linear/MatMul operations
    if (opName.contains("matmul") || opName.contains("linear")) {
      return analyzeMatmul(op);
    }
    
    // Softmax operations
    if (opName.contains("softmax")) {
      return {QuantSafetyLevel::None, 32, 1.0,
              "Softmax requires high precision due to exponential computation"};
    }
    
    // Layer normalization
    if (opName.contains("layer_norm") || opName.contains("layernorm")) {
      return {QuantSafetyLevel::Low, 16, 0.9,
              "LayerNorm division is sensitive to precision"};
    }
    
    // RMSNorm
    if (opName.contains("rms_norm") || opName.contains("rmsnorm")) {
      return {QuantSafetyLevel::Low, 16, 0.9,
              "RMSNorm requires FP16 minimum for stability"};
    }
    
    // Embedding lookup
    if (opName.contains("embedding") || opName.contains("embed")) {
      return {QuantSafetyLevel::High, 8, 0.95,
              "Embedding lookup safe for quantization"};
    }
    
    // Residual add
    if (opName.contains("add")) {
      return analyzeResidualAdd(op);
    }
    
    // GELU/SiLU activation
    if (opName.contains("gelu") || opName.contains("silu") || opName.contains("swish")) {
      return {QuantSafetyLevel::Medium, 8, 0.8,
              "Activation functions bounded, INT8 acceptable"};
    }
    
    // Default: conservative
    return {QuantSafetyLevel::Low, 16, 0.5, "Unknown operation type, defaulting to FP16"};
  }
  
  /// Analyze all operations in a module and return quantization plan.
  llvm::DenseMap<Operation*, QuantAnalysisResult> analyzeModule(ModuleOp module) {
    llvm::DenseMap<Operation*, QuantAnalysisResult> results;
    
    module.walk([&](Operation *op) {
      // Skip operations that don't need analysis
      if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op)) {
        return;
      }
      
      results[op] = analyze(op);
    });
    
    // Post-process: propagate constraints
    propagateConstraints(results);
    
    return results;
  }

private:
  double errorThreshold_;
  
  /// Analyze attention operation.
  QuantAnalysisResult analyzeAttention(Operation *op) {
    // Attention score computation (QK^T) is sensitive
    // but the final output can use lower precision
    StringRef opName = op->getName().getStringRef();
    
    if (opName.contains("paged")) {
      // PagedAttention: the operation itself needs FP16 for scores,
      // but inputs/outputs can be lower precision
      return {QuantSafetyLevel::Low, 16, 0.9,
              "Attention scores require FP16, but KV cache can be quantized separately"};
    }
    
    return {QuantSafetyLevel::Low, 16, 0.85,
            "Standard attention needs FP16 for numerical stability"};
  }
  
  /// Analyze matmul/linear operation.
  QuantAnalysisResult analyzeMatmul(Operation *op) {
    // Check if any operand is a constant (weight matrix)
    bool hasConstantWeight = false;
    for (Value operand : op->getOperands()) {
      if (operand.getDefiningOp<arith::ConstantOp>()) {
        hasConstantWeight = true;
        break;
      }
    }
    
    if (hasConstantWeight) {
      // Weight-only quantization is very safe
      return {QuantSafetyLevel::High, 8, 0.95,
              "Constant weight matrix, safe for INT8/INT4 quantization"};
    }
    
    // Check if this feeds into softmax
    if (feedsIntoSoftmax(op)) {
      return {QuantSafetyLevel::Low, 16, 0.85,
              "Matrix multiply feeds into softmax, keep FP16"};
    }
    
    // Dynamic matmul
    return {QuantSafetyLevel::Medium, 8, 0.75,
            "Dynamic matmul, INT8 acceptable with care"};
  }
  
  /// Analyze residual add operation.
  QuantAnalysisResult analyzeResidualAdd(Operation *op) {
    // Count depth in residual chain
    int depth = countResidualDepth(op);
    
    // Error accumulates with depth: ~sqrt(depth) growth
    double estimatedError = 0.001 * std::sqrt(static_cast<double>(depth));
    
    if (estimatedError < errorThreshold_) {
      return {QuantSafetyLevel::Medium, 8, 0.8,
              "Residual depth " + std::to_string(depth) + ", INT8 acceptable"};
    } else if (estimatedError < errorThreshold_ * 2) {
      return {QuantSafetyLevel::Low, 16, 0.9,
              "Deep residual chain, recommend FP16"};
    }
    
    return {QuantSafetyLevel::None, 32, 0.95,
            "Very deep residual, accumulated error requires FP32"};
  }
  
  /// Check if an operation feeds into a softmax.
  bool feedsIntoSoftmax(Operation *op) {
    for (Operation *user : op->getUsers()) {
      StringRef userName = user->getName().getStringRef();
      if (userName.contains("softmax")) {
        return true;
      }
    }
    return false;
  }
  
  /// Count the depth of residual connections.
  int countResidualDepth(Operation *op) {
    // Simplified: count number of add operations in the chain
    int depth = 1;
    
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->getName().getStringRef().contains("add")) {
          depth = std::max(depth, countResidualDepth(defOp) + 1);
        }
      }
    }
    
    return depth;
  }
  
  /// Propagate quantization constraints through the graph.
  void propagateConstraints(llvm::DenseMap<Operation*, QuantAnalysisResult> &results) {
    // If an operation requires high precision, its immediate inputs/outputs
    // may also need adjustment
    bool changed = true;
    
    while (changed) {
      changed = false;
      
      for (auto &[op, result] : results) {
        // If this op requires FP32, inputs should be at least FP16
        if (result.level == QuantSafetyLevel::None) {
          for (Value operand : op->getOperands()) {
            if (Operation *defOp = operand.getDefiningOp()) {
              auto it = results.find(defOp);
              if (it != results.end()) {
                if (it->second.level == QuantSafetyLevel::High ||
                    it->second.level == QuantSafetyLevel::Medium) {
                  it->second.level = QuantSafetyLevel::Low;
                  it->second.recommendedBits = 16;
                  it->second.reason += " (constrained by downstream FP32 op)";
                  changed = true;
                }
              }
            }
          }
        }
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Quantization Analysis Pass
//===----------------------------------------------------------------------===//

namespace {

struct QuantizationAnalysisPass
    : public PassWrapper<QuantizationAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizationAnalysisPass)
  
  StringRef getArgument() const override { return "llm-analyze-quantization"; }
  StringRef getDescription() const override {
    return "Analyze operations for safe quantization";
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    QuantizationSafetyAnalyzer analyzer;
    
    func.walk([&](Operation *op) {
      // Skip control flow operations
      if (isa<func::ReturnOp>(op)) {
        return;
      }
      
      QuantAnalysisResult result = analyzer.analyze(op);
      
      // Attach analysis results as attributes
      op->setAttr("quant_safety",
                  StringAttr::get(&getContext(), 
                                 safetyLevelToString(result.level)));
      op->setAttr("quant_bits",
                  IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                  result.recommendedBits));
      op->setAttr("quant_confidence",
                  FloatAttr::get(Float64Type::get(&getContext()),
                                result.confidenceScore));
      
      // Debug output
      llvm::errs() << "  " << op->getName() << ": " 
                   << safetyLevelToString(result.level) 
                   << " (" << result.recommendedBits << " bits, "
                   << result.confidenceScore << " confidence)\n";
      llvm::errs() << "    Reason: " << result.reason << "\n";
    });
  }
  
private:
  static StringRef safetyLevelToString(QuantSafetyLevel level) {
    switch (level) {
      case QuantSafetyLevel::High: return "high";
      case QuantSafetyLevel::Medium: return "medium";
      case QuantSafetyLevel::Low: return "low";
      case QuantSafetyLevel::None: return "none";
    }
    return "unknown";
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createQuantizationAnalysisPass() {
  return std::make_unique<QuantizationAnalysisPass>();
}

} // namespace llm
} // namespace mlir
