//===- BlockSizeAnalysis.cpp - KV cache block size analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements block size analysis for KV cache optimization.
// It determines optimal block sizes based on sequence length patterns,
// hardware characteristics, and memory efficiency.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/LLM/Transforms/BlockSizeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

using namespace mlir;

namespace mlir {
namespace llm {

//===----------------------------------------------------------------------===//
// Hardware Configuration
//===----------------------------------------------------------------------===//

/// Hardware configuration for optimization decisions.
struct HardwareConfig {
  int64_t warpSize = 32;           // CUDA warp size
  int64_t cacheLineSize = 128;     // L2 cache line size in bytes
  int64_t sharedMemorySize = 49152;// Shared memory per SM in bytes
  int64_t maxBlocksPerSM = 32;     // Maximum thread blocks per SM
  int64_t computeCapability = 80;  // CUDA compute capability (A100 = 80)
  
  static HardwareConfig getA100() {
    return HardwareConfig{32, 128, 166912, 32, 80};
  }
  
  static HardwareConfig getH100() {
    return HardwareConfig{32, 128, 232448, 32, 90};
  }
  
  static HardwareConfig getCPU() {
    return HardwareConfig{1, 64, 0, 1, 0};
  }
};

//===----------------------------------------------------------------------===//
// Block Size Analysis Result
//===----------------------------------------------------------------------===//

namespace {

struct BlockSizeAnalysisDetail {
  int64_t optimalBlockSize;
  double fragmentationScore;
  double gpuUtilization;
  double memoryAlignmentScore;
  double combinedScore;
  std::string reasoning;

  BlockSizeAnalysisDetail()
      : optimalBlockSize(128), fragmentationScore(0.0), gpuUtilization(0.0),
        memoryAlignmentScore(0.0), combinedScore(0.0), reasoning("default") {}

  BlockSizeAnalysisResult toPublic() const {
    BlockSizeAnalysisResult out;
    out.optimalBlockSize = optimalBlockSize;
    out.fragmentationScore = fragmentationScore;
    out.gpuUtilization = gpuUtilization;
    out.memoryAlignmentScore = memoryAlignmentScore;
    out.combinedScore = combinedScore;
    return out;
  }
};

//===----------------------------------------------------------------------===//
// Block Size Analyzer
//===----------------------------------------------------------------------===//

/// Analyzer for determining optimal KV cache block sizes.
class BlockSizeAnalyzer {
public:
  explicit BlockSizeAnalyzer(const HardwareConfig &hw = HardwareConfig::getA100())
      : hw_(hw) {}
  
  /// Analyze operations and determine optimal block size.
  BlockSizeAnalysisDetail analyze(Operation *rootOp) {
    // Collect sequence length information
    llvm::SmallVector<int64_t> seqLengths;
    collectSequenceLengths(rootOp, seqLengths);
    
    // If no static sequence info, use workload-based heuristics
    if (seqLengths.empty()) {
      return getDefaultResult();
    }
    
    // Candidate block sizes (powers of 2 for alignment)
    llvm::SmallVector<int64_t> candidates = {16, 32, 64, 128, 256};
    
    BlockSizeAnalysisDetail best;
    best.combinedScore = -1.0;
    
    for (int64_t blockSize : candidates) {
      BlockSizeAnalysisDetail result;
      result.optimalBlockSize = blockSize;
      
      // Compute metrics
      result.fragmentationScore = computeFragmentation(seqLengths, blockSize);
      result.gpuUtilization = computeGPUUtilization(blockSize);
      result.memoryAlignmentScore = computeMemoryAlignment(blockSize);
      
      // Combined score: balance all factors
      result.combinedScore = 
          (1.0 - result.fragmentationScore) * 0.4 +
          result.gpuUtilization * 0.35 +
          result.memoryAlignmentScore * 0.25;
      
      result.reasoning = formatReasoning(result, blockSize, seqLengths);
      
      if (result.combinedScore > best.combinedScore) {
        best = result;
      }
    }
    
    return best;
  }
  
  /// Get recommended block size for a specific workload type.
  int64_t getRecommendedBlockSize(StringRef workloadType) {
    if (workloadType == "chat" || workloadType == "dialogue") {
      // Short to medium sequences, frequent context switches
      return 64;
    }
    if (workloadType == "document_qa" || workloadType == "long_context") {
      // Long sequences, large context windows
      return 256;
    }
    if (workloadType == "code_generation") {
      // Variable length, medium sequences typical
      return 128;
    }
    if (workloadType == "real_time" || workloadType == "streaming") {
      // Very short sequences, latency-critical
      return 32;
    }
    if (workloadType == "batch_inference") {
      // Throughput-optimized, larger blocks
      return 128;
    }
    
    // Default
    return 128;
  }

private:
  HardwareConfig hw_;
  
  /// Collect sequence lengths from operations.
  void collectSequenceLengths(Operation *op, llvm::SmallVector<int64_t> &lengths) {
    op->walk([&](Operation *inner) {
      // Check append_kv operations
      if (inner->getName().getStringRef().contains("append_kv")) {
        for (Value operand : inner->getOperands()) {
          if (auto tensorType = operand.getType().dyn_cast<RankedTensorType>()) {
            if (tensorType.hasStaticShape() && tensorType.getRank() >= 2) {
              // Sequence length is typically dimension 1
              int64_t seqLen = tensorType.getDimSize(1);
              if (seqLen > 0) {
                lengths.push_back(seqLen);
              }
            }
          }
        }
      }
      
      // Check paged_attention operations
      if (inner->getName().getStringRef().contains("paged_attention")) {
        for (Value operand : inner->getOperands()) {
          if (auto tensorType = operand.getType().dyn_cast<RankedTensorType>()) {
            if (tensorType.hasStaticShape() && tensorType.getRank() >= 2) {
              int64_t seqLen = tensorType.getDimSize(1);
              if (seqLen > 0) {
                lengths.push_back(seqLen);
              }
            }
          }
        }
      }
      
    });
  }
  
  /// Compute memory fragmentation score (0.0 = no waste, 1.0 = maximum waste).
  double computeFragmentation(llvm::ArrayRef<int64_t> seqLengths, int64_t blockSize) {
    if (seqLengths.empty()) return 0.0;
    
    double totalWaste = 0.0;
    for (int64_t seqLen : seqLengths) {
      int64_t numBlocks = (seqLen + blockSize - 1) / blockSize;
      int64_t allocated = numBlocks * blockSize;
      int64_t waste = allocated - seqLen;
      totalWaste += static_cast<double>(waste) / allocated;
    }
    
    return totalWaste / seqLengths.size();
  }
  
  /// Compute GPU utilization score (0.0 = poor, 1.0 = optimal).
  double computeGPUUtilization(int64_t blockSize) {
    if (hw_.warpSize == 1) {
      // CPU: block size doesn't affect warp utilization
      return 1.0;
    }
    
    // Warp efficiency: ideally block_size >= warp_size
    double warpUtil = std::min(1.0, static_cast<double>(blockSize) / hw_.warpSize);
    
    // Occupancy: avoid too many registers per thread
    // Larger blocks need more registers, reducing occupancy
    double occupancy = 1.0;
    if (blockSize > 128) {
      occupancy = 0.9;
    }
    if (blockSize > 256) {
      occupancy = 0.75;
    }
    
    // Shared memory bank conflicts
    // Best when block_size is multiple of 32 (for f16)
    double bankConflicts = (blockSize % 32 == 0) ? 1.0 : 0.8;
    
    return warpUtil * occupancy * bankConflicts;
  }
  
  /// Compute memory alignment score (0.0 = poor, 1.0 = optimal).
  double computeMemoryAlignment(int64_t blockSize) {
    // Element size for f16
    int64_t elementSize = 2;
    int64_t bytesPerBlock = blockSize * elementSize;
    
    // Check cache line alignment
    if (bytesPerBlock % hw_.cacheLineSize == 0) {
      return 1.0;
    }
    if ((hw_.cacheLineSize % bytesPerBlock == 0) || (bytesPerBlock % (hw_.cacheLineSize / 2) == 0)) {
      return 0.9;
    }
    
    return 0.7;
  }
  
  /// Get default result when no static information is available.
  BlockSizeAnalysisDetail getDefaultResult() {
    BlockSizeAnalysisDetail result;
    result.optimalBlockSize = 128;  // Conservative default
    result.fragmentationScore = 0.15;
    result.gpuUtilization = 0.95;
    result.memoryAlignmentScore = 1.0;
    result.combinedScore = 0.85;
    result.reasoning = "No static sequence information; using default block size 128 "
                       "which balances fragmentation, GPU utilization, and alignment "
                       "for typical LLM workloads";
    return result;
  }
  
  /// Format reasoning string.
  std::string formatReasoning(const BlockSizeAnalysisDetail &result,
                               int64_t blockSize,
                               llvm::ArrayRef<int64_t> seqLengths) {
    std::string reasoning;
    llvm::raw_string_ostream os(reasoning);
    
    os << "Block size " << blockSize << " selected with score " 
       << result.combinedScore << ":\n";
    os << "  - Fragmentation: " << (result.fragmentationScore * 100) << "% waste\n";
    os << "  - GPU Utilization: " << (result.gpuUtilization * 100) << "%\n";
    os << "  - Memory Alignment: " << (result.memoryAlignmentScore * 100) << "%\n";
    
    if (!seqLengths.empty()) {
      int64_t minLen = *std::min_element(seqLengths.begin(), seqLengths.end());
      int64_t maxLen = *std::max_element(seqLengths.begin(), seqLengths.end());
      os << "  - Analyzed sequence lengths: [" << minLen << ", " << maxLen << "]\n";
    }
    
    return os.str();
  }
};

//===----------------------------------------------------------------------===//
// Block Size Analysis Pass
//===----------------------------------------------------------------------===//

namespace {

struct BlockSizeAnalysisPass
    : public PassWrapper<BlockSizeAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockSizeAnalysisPass)
  
  StringRef getArgument() const override { return "llm-analyze-block-size"; }
  StringRef getDescription() const override {
    return "Analyze and optimize KV cache block size";
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    // Use A100 config as default
    HardwareConfig hw = HardwareConfig::getA100();
    BlockSizeAnalyzer analyzer(hw);
    
    BlockSizeAnalysisDetail result = analyzer.analyze(func);
    
    // Print analysis results
    llvm::errs() << "Block Size Analysis for function '" << func.getName() << "':\n";
    llvm::errs() << result.reasoning << "\n";
    
    // Attach optimal block size as function attribute
    func->setAttr("llm.optimal_block_size",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                  result.optimalBlockSize));
    
    // Update KV cache operations with optimal block size
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("paged_kv_cache") ||
          op->getName().getStringRef().contains("append_kv") ||
          op->getName().getStringRef().contains("paged_attention")) {
        
        // Attach analysis results
        op->setAttr("optimal_block_size",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                    result.optimalBlockSize));
        op->setAttr("fragmentation_score",
                    FloatAttr::get(Float64Type::get(&getContext()),
                                  result.fragmentationScore));
        op->setAttr("gpu_utilization",
                    FloatAttr::get(Float64Type::get(&getContext()),
                                  result.gpuUtilization));
      }
    });
  }
};

/// Pass that applies block size optimization based on analysis.
struct BlockSizeOptimizationPass
    : public PassWrapper<BlockSizeOptimizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockSizeOptimizationPass)
  
  StringRef getArgument() const override { return "llm-optimize-block-size"; }
  StringRef getDescription() const override {
    return "Apply block size optimization to KV cache operations";
  }
  
  void runOnOperation() override {
    applyBlockSizeOptimizationToFunc(getOperation());
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createBlockSizeAnalysisPass() {
  return std::make_unique<BlockSizeAnalysisPass>();
}

std::unique_ptr<Pass> createBlockSizeOptimizationPass() {
  return std::make_unique<BlockSizeOptimizationPass>();
}

BlockSizeAnalysisResult analyzeBlockSizeForFunc(func::FuncOp func) {
  BlockSizeAnalyzer analyzer(HardwareConfig::getA100());
  return analyzer.analyze(func).toPublic();
}

void applyBlockSizeOptimizationToFunc(func::FuncOp func) {
  BlockSizeAnalysisDetail result =
      BlockSizeAnalyzer(HardwareConfig::getA100()).analyze(func);
  func->setAttr(
      "llm.optimal_block_size",
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       result.optimalBlockSize));
  func.walk([&](Operation *op) {
    if (!op->hasAttr("block_size"))
      return;
    auto currentSize = op->getAttrOfType<IntegerAttr>("block_size").getInt();
    if (currentSize == result.optimalBlockSize)
      return;
    op->setAttr("block_size",
                IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                 result.optimalBlockSize));
  });
}

} // namespace llm
} // namespace mlir
