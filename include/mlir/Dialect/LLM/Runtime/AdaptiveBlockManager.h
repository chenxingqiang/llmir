//===- AdaptiveBlockManager.h - Adaptive Block Size Management --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines adaptive block size management for the KV cache,
// automatically adjusting block sizes based on workload patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_ADAPTIVEBLOCKMANAGER_H_
#define MLIR_DIALECT_LLM_RUNTIME_ADAPTIVEBLOCKMANAGER_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Workload Statistics
//===----------------------------------------------------------------------===//

struct WorkloadStats {
  // Sequence length distribution
  double averageSeqLen;
  double maxSeqLen;
  double minSeqLen;
  double seqLenStdDev;
  
  // Batch characteristics
  double averageBatchSize;
  double maxBatchSize;
  
  // Memory patterns
  double averageBlockUtilization;  // 0-1, how full are blocks on average
  double fragmentationRatio;       // Wasted space due to fragmentation
  
  // Access patterns
  double readWriteRatio;           // Reads per write
  double localityScore;            // 0-1, higher means more temporal locality
  
  // Throughput metrics
  double tokensPerSecond;
  double appendLatencyMs;
  double lookupLatencyMs;
  
  WorkloadStats()
      : averageSeqLen(0), maxSeqLen(0), minSeqLen(0), seqLenStdDev(0),
        averageBatchSize(0), maxBatchSize(0), averageBlockUtilization(0),
        fragmentationRatio(0), readWriteRatio(0), localityScore(0),
        tokensPerSecond(0), appendLatencyMs(0), lookupLatencyMs(0) {}
};

//===----------------------------------------------------------------------===//
// Block Size Configuration
//===----------------------------------------------------------------------===//

struct BlockSizeConfig {
  int64_t primaryBlockSize;        // Main block size
  int64_t smallBlockSize;          // For short sequences
  int64_t largeBlockSize;          // For long sequences
  
  // Thresholds for using different block sizes
  int64_t smallBlockThreshold;     // Use small blocks if seq < this
  int64_t largeBlockThreshold;     // Use large blocks if seq > this
  
  // Memory configuration
  size_t smallBlockPoolSize;       // Number of small blocks to preallocate
  size_t primaryBlockPoolSize;
  size_t largeBlockPoolSize;
  
  BlockSizeConfig()
      : primaryBlockSize(16), smallBlockSize(4), largeBlockSize(64),
        smallBlockThreshold(32), largeBlockThreshold(512),
        smallBlockPoolSize(256), primaryBlockPoolSize(512),
        largeBlockPoolSize(64) {}
};

//===----------------------------------------------------------------------===//
// Adaptation Policy
//===----------------------------------------------------------------------===//

enum class AdaptationPolicy {
  STATIC,           // No adaptation
  REACTIVE,         // React to current workload
  PREDICTIVE,       // Predict future workload
  HYBRID            // Combination of reactive and predictive
};

struct AdaptationConfig {
  AdaptationPolicy policy;
  int64_t windowSize;              // Window for statistics collection
  double adaptationThreshold;      // Minimum change to trigger adaptation
  double stabilityFactor;          // Higher = more stable, less frequent changes
  int64_t minAdaptationInterval;   // Minimum time between adaptations (ms)
  bool enableAutoTuning;           // Auto-tune block sizes
  
  AdaptationConfig()
      : policy(AdaptationPolicy::REACTIVE), windowSize(1000),
        adaptationThreshold(0.1), stabilityFactor(0.8),
        minAdaptationInterval(5000), enableAutoTuning(true) {}
};

//===----------------------------------------------------------------------===//
// Multi-Size Block Allocator
//===----------------------------------------------------------------------===//

class MultiSizeBlockAllocator {
public:
  MultiSizeBlockAllocator(int64_t headDim, int64_t numHeads,
                          const BlockSizeConfig& config,
                          bool enableGPU = false);
  ~MultiSizeBlockAllocator();
  
  // Allocate a block of appropriate size
  KVBlock* allocateBlock(int64_t expectedTokens);
  
  // Allocate specific size
  KVBlock* allocateSmallBlock();
  KVBlock* allocatePrimaryBlock();
  KVBlock* allocateLargeBlock();
  
  // Deallocate
  void deallocateBlock(KVBlock* block);
  void deallocateBlock(int32_t blockId);
  
  // Get block by ID
  KVBlock* getBlock(int32_t blockId);
  
  // Configuration
  void updateConfig(const BlockSizeConfig& config);
  const BlockSizeConfig& getConfig() const { return config_; }
  
  // Statistics
  size_t getNumFreeSmallBlocks() const;
  size_t getNumFreePrimaryBlocks() const;
  size_t getNumFreeLargeBlocks() const;
  size_t getTotalMemoryUsage() const;
  float getAverageUtilization() const;
  
private:
  int64_t headDim_;
  int64_t numHeads_;
  BlockSizeConfig config_;
  bool enableGPU_;
  
  // Separate allocators for each size
  std::unique_ptr<BlockAllocator> smallAllocator_;
  std::unique_ptr<BlockAllocator> primaryAllocator_;
  std::unique_ptr<BlockAllocator> largeAllocator_;
  
  // Block ID to allocator mapping
  std::unordered_map<int32_t, BlockAllocator*> blockOwnership_;
  
  // ID offset for each size category
  int32_t smallIdOffset_;
  int32_t primaryIdOffset_;
  int32_t largeIdOffset_;
};

//===----------------------------------------------------------------------===//
// Workload Analyzer
//===----------------------------------------------------------------------===//

class WorkloadAnalyzer {
public:
  explicit WorkloadAnalyzer(int64_t windowSize = 1000);
  ~WorkloadAnalyzer();
  
  // Record events
  void recordSequenceLength(int64_t length);
  void recordBatchSize(int64_t size);
  void recordBlockUtilization(float utilization);
  void recordAppend(int64_t numTokens, double latencyMs);
  void recordLookup(int64_t numTokens, double latencyMs);
  
  // Get statistics
  WorkloadStats getStats() const;
  
  // Trend analysis
  bool isSeqLengthIncreasing() const;
  bool isBatchSizeIncreasing() const;
  bool isFragmentationHigh() const;
  
  // Predictions
  double predictNextSeqLength() const;
  double predictNextBatchSize() const;
  
  // Reset
  void reset();
  
private:
  int64_t windowSize_;
  
  // Rolling windows
  std::deque<int64_t> seqLengths_;
  std::deque<int64_t> batchSizes_;
  std::deque<float> blockUtilizations_;
  std::deque<std::pair<int64_t, double>> appendEvents_;
  std::deque<std::pair<int64_t, double>> lookupEvents_;
  
  // Aggregate stats
  mutable WorkloadStats cachedStats_;
  mutable bool statsDirty_;
  
  // Update cached statistics
  void updateStats() const;
  
  // Calculate trend (positive = increasing)
  double calculateTrend(const std::deque<int64_t>& data) const;
};

//===----------------------------------------------------------------------===//
// Adaptive Block Manager
//===----------------------------------------------------------------------===//

class AdaptiveBlockManager {
public:
  AdaptiveBlockManager(int64_t numLayers, int64_t numHeads, int64_t headDim,
                       const BlockSizeConfig& blockConfig,
                       const AdaptationConfig& adaptConfig,
                       bool enableGPU = false);
  ~AdaptiveBlockManager();
  
  // Basic getters
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  
  // Get current configuration
  const BlockSizeConfig& getBlockConfig() const { return blockConfig_; }
  const AdaptationConfig& getAdaptConfig() const { return adaptConfig_; }
  
  //===--------------------------------------------------------------------===//
  // Block Allocation
  //===--------------------------------------------------------------------===//
  
  // Allocate blocks for a sequence
  LogicalResult allocateBlocksForSequence(int32_t sequenceId,
                                           int64_t expectedLength,
                                           std::vector<KVBlock*>& blocks);
  
  // Allocate single block
  KVBlock* allocateBlock(int64_t expectedTokens);
  
  // Deallocate
  void deallocateBlock(KVBlock* block);
  void deallocateSequence(int32_t sequenceId);
  
  //===--------------------------------------------------------------------===//
  // Workload Tracking
  //===--------------------------------------------------------------------===//
  
  // Record workload events
  void recordAppend(int32_t sequenceId, int64_t numTokens, double latencyMs);
  void recordLookup(int32_t sequenceId, int64_t numTokens, double latencyMs);
  void recordSequenceComplete(int32_t sequenceId, int64_t finalLength);
  
  // Get current workload statistics
  WorkloadStats getWorkloadStats() const;
  
  //===--------------------------------------------------------------------===//
  // Adaptation
  //===--------------------------------------------------------------------===//
  
  // Trigger adaptation check
  LogicalResult checkAndAdapt();
  
  // Force adaptation with new config
  LogicalResult forceAdapt(const BlockSizeConfig& newConfig);
  
  // Get recommended block size for current workload
  int64_t getRecommendedBlockSize() const;
  
  // Get recommended configuration
  BlockSizeConfig getRecommendedConfig() const;
  
  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//
  
  // Compact memory by coalescing blocks
  LogicalResult compactMemory();
  
  // Defragment by moving data between block sizes
  LogicalResult defragment();
  
  // Get memory statistics
  size_t getTotalMemoryUsage() const;
  float getFragmentationRatio() const;
  float getAverageBlockUtilization() const;
  
  //===--------------------------------------------------------------------===//
  // Metrics
  //===--------------------------------------------------------------------===//
  
  struct AdaptationMetrics {
    int64_t numAdaptations;
    int64_t numCompactions;
    int64_t numDefragmentations;
    double totalAdaptationTime;
    double averageUtilizationImprovement;
    std::vector<std::pair<int64_t, BlockSizeConfig>> adaptationHistory;
  };
  
  AdaptationMetrics getMetrics() const { return metrics_; }
  void resetMetrics();
  
private:
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  BlockSizeConfig blockConfig_;
  AdaptationConfig adaptConfig_;
  bool enableGPU_;
  
  // Multi-size allocators per layer
  std::vector<std::unique_ptr<MultiSizeBlockAllocator>> allocators_;
  
  // Workload analyzer
  std::unique_ptr<WorkloadAnalyzer> analyzer_;
  
  // Sequence to block mapping
  std::unordered_map<int32_t, std::vector<std::vector<KVBlock*>>> sequenceBlocks_;
  
  // Adaptation state
  int64_t lastAdaptationTime_;
  int64_t adaptationCount_;
  
  // Metrics
  mutable AdaptationMetrics metrics_;
  
  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//
  
  // Determine optimal block size based on expected length
  int64_t selectBlockSize(int64_t expectedTokens) const;
  
  // Calculate optimal configuration from workload
  BlockSizeConfig calculateOptimalConfig(const WorkloadStats& stats) const;
  
  // Perform adaptation
  LogicalResult performAdaptation(const BlockSizeConfig& newConfig);
  
  // Migrate blocks to new size
  LogicalResult migrateBlocks(int64_t oldSize, int64_t newSize);
};

//===----------------------------------------------------------------------===//
// Auto-Tuning Block Manager
//===----------------------------------------------------------------------===//

// Extension with ML-based tuning (placeholder for future enhancement)
class AutoTuningBlockManager : public AdaptiveBlockManager {
public:
  AutoTuningBlockManager(int64_t numLayers, int64_t numHeads, int64_t headDim,
                         const BlockSizeConfig& blockConfig,
                         const AdaptationConfig& adaptConfig,
                         bool enableGPU = false);
  ~AutoTuningBlockManager();
  
  // Enable/disable auto-tuning
  void enableAutoTuning(bool enable) { autoTuneEnabled_ = enable; }
  bool isAutoTuningEnabled() const { return autoTuneEnabled_; }
  
  // Set tuning objective
  enum class TuningObjective {
    MINIMIZE_MEMORY,
    MINIMIZE_LATENCY,
    MAXIMIZE_THROUGHPUT,
    BALANCED
  };
  
  void setTuningObjective(TuningObjective objective) { objective_ = objective; }
  TuningObjective getTuningObjective() const { return objective_; }
  
  // Run tuning experiment
  LogicalResult runTuningExperiment(BlockSizeConfig& bestConfig);
  
private:
  bool autoTuneEnabled_;
  TuningObjective objective_;
  
  // Tuning history for learning
  std::vector<std::pair<BlockSizeConfig, double>> tuningHistory_;
  
  // Evaluate a configuration
  double evaluateConfig(const BlockSizeConfig& config) const;
  
  // Generate candidate configurations
  std::vector<BlockSizeConfig> generateCandidates() const;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_ADAPTIVEBLOCKMANAGER_H_
