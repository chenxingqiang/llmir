//===- SpeculativeKVCache.h - Speculative Decoding Support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines KV cache support for speculative decoding, enabling
// efficient branching and rollback of cache state during draft verification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_SPECULATIVEKVCACHE_H_
#define MLIR_DIALECT_LLM_RUNTIME_SPECULATIVEKVCACHE_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Speculative Decoding Configuration
//===----------------------------------------------------------------------===//

struct SpeculativeConfig {
  int64_t maxDraftTokens;      // Maximum number of draft tokens per step
  int64_t maxBranches;         // Maximum concurrent branches per sequence
  bool enableTreeAttention;    // Use tree-based attention for verification
  bool reuseRejectedTokens;    // Reuse KV for partially accepted drafts
  float acceptanceThreshold;   // Threshold for accepting draft tokens
  
  SpeculativeConfig()
      : maxDraftTokens(8), maxBranches(4), enableTreeAttention(true),
        reuseRejectedTokens(true), acceptanceThreshold(0.9f) {}
};

//===----------------------------------------------------------------------===//
// Branch Point
//===----------------------------------------------------------------------===//

// Represents a point where speculation branches from verified state
struct BranchPoint {
  int32_t sequenceId;
  int64_t position;           // Token position where branch occurs
  int64_t timestamp;          // Creation timestamp for ordering
  
  // Block state at branch point for each layer
  std::vector<int32_t> blockIndices;
  std::vector<int64_t> positionsInBlocks;
};

//===----------------------------------------------------------------------===//
// Speculation State
//===----------------------------------------------------------------------===//

// Tracks the state of speculative tokens for a sequence
struct SpeculationState {
  int32_t sequenceId;
  int32_t branchId;
  
  // Token tracking
  int64_t verifiedLength;     // Number of verified tokens
  int64_t speculativeLength;  // Number of speculative tokens
  
  // Draft tokens pending verification
  std::vector<int32_t> draftTokenIds;
  std::vector<float> draftLogProbs;
  
  // Block references for speculative tokens
  std::vector<std::vector<int32_t>> speculativeBlockIndices;
  
  // Parent branch (for tree speculation)
  int32_t parentBranchId;
  int64_t branchPosition;
  
  bool isActive;
};

//===----------------------------------------------------------------------===//
// Verification Result
//===----------------------------------------------------------------------===//

struct VerificationResult {
  int64_t acceptedCount;      // Number of accepted draft tokens
  int64_t rejectedPosition;   // Position of first rejection (-1 if all accepted)
  
  std::vector<bool> acceptedMask;  // Per-token acceptance
  std::vector<float> targetLogProbs; // Target model log probabilities
  
  // Statistics
  float acceptanceRate;
  double verificationTime;
};

//===----------------------------------------------------------------------===//
// Speculative KV Cache
//===----------------------------------------------------------------------===//

class SpeculativeKVCache {
public:
  SpeculativeKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                     int64_t blockSize, int64_t maxSeqLen,
                     Type elementType, const SpeculativeConfig& config,
                     bool enableGPU = false);
  ~SpeculativeKVCache();
  
  // Basic getters
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  int64_t getBlockSize() const { return blockSize_; }
  int64_t getMaxSeqLen() const { return maxSeqLen_; }
  const SpeculativeConfig& getConfig() const { return config_; }
  
  //===--------------------------------------------------------------------===//
  // Core KV Cache Operations
  //===--------------------------------------------------------------------===//
  
  // Append verified KV pairs (standard operation)
  LogicalResult appendVerifiedKV(const void* keyData, const void* valueData,
                                  int32_t batchSize, int32_t seqLen,
                                  const int32_t* seqIds, int32_t* blockIndices);
  
  // Append speculative (draft) KV pairs
  LogicalResult appendSpeculativeKV(const void* keyData, const void* valueData,
                                     int32_t sequenceId, int32_t numDraftTokens,
                                     int32_t branchId = 0);
  
  // Lookup KV pairs (includes both verified and speculative)
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int32_t batchSize, void* outputKeys, void* outputValues,
                        bool includeSpeculative = true);
  
  //===--------------------------------------------------------------------===//
  // Branching Operations
  //===--------------------------------------------------------------------===//
  
  // Create a branch point for speculation
  LogicalResult createBranch(int32_t sequenceId, int32_t& branchId);
  
  // Create multiple branches from same point (tree speculation)
  LogicalResult createTreeBranches(int32_t sequenceId, 
                                   int64_t numBranches,
                                   std::vector<int32_t>& branchIds);
  
  // Get branch point for a sequence
  LogicalResult getBranchPoint(int32_t sequenceId, int32_t branchId,
                               BranchPoint& branchPoint) const;
  
  //===--------------------------------------------------------------------===//
  // Verification and Commit
  //===--------------------------------------------------------------------===//
  
  // Verify speculative tokens and get acceptance result
  LogicalResult verifySpeculation(int32_t sequenceId, int32_t branchId,
                                   const float* targetLogProbs,
                                   int64_t numTokens,
                                   VerificationResult& result);
  
  // Commit accepted speculative tokens (make them verified)
  LogicalResult commitSpeculation(int32_t sequenceId, int32_t branchId,
                                   int64_t numAccepted);
  
  // Rollback speculative tokens (discard them)
  LogicalResult rollbackSpeculation(int32_t sequenceId, int32_t branchId);
  
  // Rollback to a specific position
  LogicalResult rollbackToPosition(int32_t sequenceId, int64_t position);
  
  //===--------------------------------------------------------------------===//
  // Tree Attention Support
  //===--------------------------------------------------------------------===//
  
  // Get attention mask for tree-structured speculation
  LogicalResult getTreeAttentionMask(int32_t sequenceId,
                                      std::vector<int32_t>& branchIds,
                                      std::vector<std::vector<bool>>& mask);
  
  // Compute attention with tree structure
  LogicalResult computeTreeAttention(const void* queries,
                                      int32_t sequenceId,
                                      const std::vector<int32_t>& branchIds,
                                      void* output);
  
  //===--------------------------------------------------------------------===//
  // State Management
  //===--------------------------------------------------------------------===//
  
  // Get speculation state for a sequence
  const SpeculationState* getSpeculationState(int32_t sequenceId, 
                                               int32_t branchId = 0) const;
  
  // Get all active branches for a sequence
  std::vector<int32_t> getActiveBranches(int32_t sequenceId) const;
  
  // Clear all speculation state for a sequence
  LogicalResult clearSpeculation(int32_t sequenceId);
  
  // Clear entire cache
  void reset();
  
  //===--------------------------------------------------------------------===//
  // Statistics and Metrics
  //===--------------------------------------------------------------------===//
  
  struct SpeculativeMetrics {
    int64_t totalDraftTokens;
    int64_t acceptedDraftTokens;
    int64_t rejectedDraftTokens;
    int64_t numVerifications;
    int64_t numRollbacks;
    int64_t numCommits;
    double totalVerificationTime;
    double averageAcceptanceRate;
    int64_t maxConcurrentBranches;
  };
  
  SpeculativeMetrics getMetrics() const { return metrics_; }
  void resetMetrics();
  
  // Get verified and speculative lengths
  int64_t getVerifiedLength(int32_t sequenceId) const;
  int64_t getSpeculativeLength(int32_t sequenceId, int32_t branchId = 0) const;
  int64_t getTotalLength(int32_t sequenceId, int32_t branchId = 0) const;
  
private:
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t blockSize_;
  int64_t maxSeqLen_;
  Type elementType_;
  SpeculativeConfig config_;
  bool enableGPU_;
  
  // Underlying KV cache for verified tokens
  std::unique_ptr<PagedKVCache> verifiedCache_;
  
  // Speculation-specific block allocator (for draft tokens)
  std::unique_ptr<BlockAllocator> speculativeAllocator_;
  
  // Speculation state per sequence and branch
  std::unordered_map<int32_t, std::unordered_map<int32_t, SpeculationState>> 
      speculationStates_;
  
  // Branch points per sequence
  std::unordered_map<int32_t, std::vector<BranchPoint>> branchPoints_;
  
  // Next branch ID counter
  int32_t nextBranchId_;
  
  // Metrics
  mutable SpeculativeMetrics metrics_;
  
  // Timestamp counter
  int64_t timestampCounter_;
  
  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//
  
  // Initialize speculation state for a new sequence
  void initializeSpeculationState(int32_t sequenceId);
  
  // Allocate blocks for speculative tokens
  LogicalResult allocateSpeculativeBlocks(int32_t sequenceId, int32_t branchId,
                                           int64_t numTokens);
  
  // Free speculative blocks
  void freeSpeculativeBlocks(int32_t sequenceId, int32_t branchId);
  
  // Copy block state for branching
  LogicalResult copyBlockState(int32_t sequenceId, int32_t srcBranchId,
                               int32_t dstBranchId);
  
  // Acceptance sampling for verification
  bool shouldAcceptToken(float draftLogProb, float targetLogProb) const;
  
  // Build tree attention mask
  void buildTreeMask(const std::vector<SpeculationState>& branches,
                     std::vector<std::vector<bool>>& mask);
};

//===----------------------------------------------------------------------===//
// Draft Model Integration
//===----------------------------------------------------------------------===//

// Interface for draft model that generates speculative tokens
class DraftModelInterface {
public:
  virtual ~DraftModelInterface() = default;
  
  // Generate draft tokens
  virtual LogicalResult generateDrafts(const void* inputIds,
                                       int32_t sequenceId,
                                       int64_t numDrafts,
                                       std::vector<int32_t>& draftTokens,
                                       std::vector<float>& logProbs) = 0;
  
  // Get KV cache for draft model (if using separate cache)
  virtual PagedKVCache* getDraftCache() = 0;
};

//===----------------------------------------------------------------------===//
// Speculative Decoding Engine
//===----------------------------------------------------------------------===//

// High-level engine for speculative decoding
class SpeculativeDecodingEngine {
public:
  SpeculativeDecodingEngine(SpeculativeKVCache& cache,
                            DraftModelInterface* draftModel = nullptr);
  ~SpeculativeDecodingEngine();
  
  // Run one step of speculative decoding
  // Returns number of tokens generated
  LogicalResult step(int32_t sequenceId,
                     const void* targetLogits,
                     std::vector<int32_t>& outputTokens,
                     int64_t& numGenerated);
  
  // Set draft model
  void setDraftModel(DraftModelInterface* model) { draftModel_ = model; }
  
  // Configure speculation
  void setMaxDraftTokens(int64_t n) { maxDraftTokens_ = n; }
  void setAcceptanceThreshold(float t) { acceptanceThreshold_ = t; }
  
  // Statistics
  float getAverageAcceptanceRate() const;
  float getSpeedup() const;
  
private:
  SpeculativeKVCache& cache_;
  DraftModelInterface* draftModel_;
  
  int64_t maxDraftTokens_;
  float acceptanceThreshold_;
  
  // Statistics
  int64_t totalSteps_;
  int64_t totalTokensGenerated_;
  int64_t totalDraftTokens_;
  int64_t totalAccepted_;
};

//===----------------------------------------------------------------------===//
// Parallel Speculation
//===----------------------------------------------------------------------===//

// Support for running multiple speculative branches in parallel
class ParallelSpeculator {
public:
  ParallelSpeculator(SpeculativeKVCache& cache, int64_t numParallelBranches);
  ~ParallelSpeculator();
  
  // Generate multiple draft sequences in parallel
  LogicalResult generateParallelDrafts(int32_t sequenceId,
                                        const std::vector<std::vector<int32_t>>& drafts,
                                        std::vector<int32_t>& branchIds);
  
  // Verify all branches and select best
  LogicalResult verifyAndSelectBest(int32_t sequenceId,
                                     const std::vector<int32_t>& branchIds,
                                     const std::vector<std::vector<float>>& targetLogProbs,
                                     int32_t& bestBranchId,
                                     int64_t& acceptedCount);
  
  // Cleanup non-selected branches
  LogicalResult cleanupBranches(int32_t sequenceId,
                                 const std::vector<int32_t>& branchIds,
                                 int32_t keepBranchId);
  
private:
  SpeculativeKVCache& cache_;
  int64_t numParallelBranches_;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_SPECULATIVEKVCACHE_H_
