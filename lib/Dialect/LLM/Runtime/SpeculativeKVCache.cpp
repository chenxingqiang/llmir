//===- SpeculativeKVCache.cpp - Speculative Decoding Support ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements KV cache support for speculative decoding.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/SpeculativeKVCache.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

double getCurrentTimeMs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

// Generate random number for acceptance sampling
thread_local std::mt19937 rng(std::random_device{}());

float sampleUniform() {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SpeculativeKVCache Implementation
//===----------------------------------------------------------------------===//

SpeculativeKVCache::SpeculativeKVCache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    Type elementType, const SpeculativeConfig& config,
    bool enableGPU)
    : numLayers_(numLayers), numHeads_(numHeads), headDim_(headDim),
      blockSize_(blockSize), maxSeqLen_(maxSeqLen),
      elementType_(elementType), config_(config), enableGPU_(enableGPU),
      nextBranchId_(0), timestampCounter_(0) {
  
  // Create verified cache
  verifiedCache_ = std::make_unique<PagedKVCache>(
      numLayers, numHeads, headDim, blockSize, maxSeqLen,
      elementType, enableGPU);
  
  // Create speculative block allocator
  speculativeAllocator_ = std::make_unique<BlockAllocator>(
      blockSize, headDim * numHeads, enableGPU);
  
  // Preallocate blocks for speculation
  int64_t speculativeBlocks = config_.maxDraftTokens * config_.maxBranches * 2;
  speculativeAllocator_->preallocateBlocks(speculativeBlocks);
  
  metrics_ = SpeculativeMetrics{};
}

SpeculativeKVCache::~SpeculativeKVCache() {
  reset();
}

//===----------------------------------------------------------------------===//
// Core KV Cache Operations
//===----------------------------------------------------------------------===//

LogicalResult SpeculativeKVCache::appendVerifiedKV(
    const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  // Append to verified cache
  if (verifiedCache_->appendKV(keyData, valueData, batchSize, seqLen,
                                seqIds, blockIndices).failed()) {
    return failure();
  }
  
  // Initialize speculation state for any new sequences
  for (int32_t b = 0; b < batchSize; b++) {
    int32_t seqId = seqIds[b];
    if (speculationStates_.find(seqId) == speculationStates_.end()) {
      initializeSpeculationState(seqId);
    }
    
    // Update verified length
    auto& mainState = speculationStates_[seqId][0];
    mainState.verifiedLength += seqLen;
  }
  
  return success();
}

LogicalResult SpeculativeKVCache::appendSpeculativeKV(
    const void* keyData, const void* valueData,
    int32_t sequenceId, int32_t numDraftTokens,
    int32_t branchId) {
  
  if (!keyData || !valueData || numDraftTokens <= 0) {
    return failure();
  }
  
  // Get or create speculation state
  if (speculationStates_.find(sequenceId) == speculationStates_.end()) {
    initializeSpeculationState(sequenceId);
  }
  
  auto& branchStates = speculationStates_[sequenceId];
  if (branchStates.find(branchId) == branchStates.end()) {
    // Create new branch state
    SpeculationState state;
    state.sequenceId = sequenceId;
    state.branchId = branchId;
    state.verifiedLength = branchStates[0].verifiedLength;
    state.speculativeLength = 0;
    state.parentBranchId = 0;
    state.branchPosition = state.verifiedLength;
    state.isActive = true;
    branchStates[branchId] = state;
  }
  
  auto& state = branchStates[branchId];
  
  // Allocate speculative blocks
  if (allocateSpeculativeBlocks(sequenceId, branchId, numDraftTokens).failed()) {
    return failure();
  }
  
  // Store speculative KV data
  const float* keys = static_cast<const float*>(keyData);
  const float* values = static_cast<const float*>(valueData);
  
  // For each layer, store the speculative data
  size_t tokenDataSize = numHeads_ * headDim_ * sizeof(float);
  
  for (int64_t layer = 0; layer < numLayers_; layer++) {
    auto& blockIndices = state.speculativeBlockIndices[layer];
    
    for (int32_t t = 0; t < numDraftTokens; t++) {
      int64_t tokenIdx = state.speculativeLength + t;
      int64_t blockIdx = tokenIdx / blockSize_;
      int64_t posInBlock = tokenIdx % blockSize_;
      
      // Get block for this token
      if (blockIdx >= static_cast<int64_t>(blockIndices.size())) {
        KVBlock* block = speculativeAllocator_->allocateBlock();
        if (!block) return failure();
        blockIndices.push_back(block->getBlockId());
      }
      
      // Copy data (simplified - actual implementation would use block storage)
    }
  }
  
  state.speculativeLength += numDraftTokens;
  metrics_.totalDraftTokens += numDraftTokens;
  
  return success();
}

LogicalResult SpeculativeKVCache::lookupKV(
    const int32_t* blockIndices, const int32_t* seqLens,
    int32_t batchSize, void* outputKeys, void* outputValues,
    bool includeSpeculative) {
  
  // First lookup from verified cache
  if (verifiedCache_->lookupKV(blockIndices, seqLens, batchSize,
                                outputKeys, outputValues).failed()) {
    return failure();
  }
  
  if (!includeSpeculative) {
    return success();
  }
  
  // Append speculative tokens to output
  // This would need to gather from speculative blocks
  
  return success();
}

//===----------------------------------------------------------------------===//
// Branching Operations
//===----------------------------------------------------------------------===//

LogicalResult SpeculativeKVCache::createBranch(int32_t sequenceId, 
                                                int32_t& branchId) {
  // Check branch limit
  auto& branchStates = speculationStates_[sequenceId];
  if (static_cast<int64_t>(branchStates.size()) >= config_.maxBranches) {
    return failure();
  }
  
  branchId = nextBranchId_++;
  
  // Create branch point
  BranchPoint bp;
  bp.sequenceId = sequenceId;
  bp.position = getVerifiedLength(sequenceId);
  bp.timestamp = timestampCounter_++;
  
  // Save block state at branch point
  // Would need to capture current block indices for all layers
  
  branchPoints_[sequenceId].push_back(bp);
  
  // Create speculation state for new branch
  SpeculationState state;
  state.sequenceId = sequenceId;
  state.branchId = branchId;
  state.verifiedLength = bp.position;
  state.speculativeLength = 0;
  state.parentBranchId = 0;
  state.branchPosition = bp.position;
  state.isActive = true;
  state.speculativeBlockIndices.resize(numLayers_);
  
  branchStates[branchId] = state;
  
  // Update metrics
  int64_t numBranches = branchStates.size();
  if (numBranches > metrics_.maxConcurrentBranches) {
    metrics_.maxConcurrentBranches = numBranches;
  }
  
  return success();
}

LogicalResult SpeculativeKVCache::createTreeBranches(
    int32_t sequenceId, 
    int64_t numBranches,
    std::vector<int32_t>& branchIds) {
  
  branchIds.clear();
  branchIds.reserve(numBranches);
  
  for (int64_t i = 0; i < numBranches; i++) {
    int32_t branchId;
    if (createBranch(sequenceId, branchId).failed()) {
      // Cleanup already created branches
      for (int32_t id : branchIds) {
        rollbackSpeculation(sequenceId, id);
      }
      return failure();
    }
    branchIds.push_back(branchId);
  }
  
  return success();
}

LogicalResult SpeculativeKVCache::getBranchPoint(
    int32_t sequenceId, int32_t branchId,
    BranchPoint& branchPoint) const {
  
  auto it = branchPoints_.find(sequenceId);
  if (it == branchPoints_.end()) {
    return failure();
  }
  
  // Find branch point for this branch ID
  for (const auto& bp : it->second) {
    auto stateIt = speculationStates_.find(sequenceId);
    if (stateIt != speculationStates_.end()) {
      auto branchIt = stateIt->second.find(branchId);
      if (branchIt != stateIt->second.end() &&
          branchIt->second.branchPosition == bp.position) {
        branchPoint = bp;
        return success();
      }
    }
  }
  
  return failure();
}

//===----------------------------------------------------------------------===//
// Verification and Commit
//===----------------------------------------------------------------------===//

LogicalResult SpeculativeKVCache::verifySpeculation(
    int32_t sequenceId, int32_t branchId,
    const float* targetLogProbs,
    int64_t numTokens,
    VerificationResult& result) {
  
  double startTime = getCurrentTimeMs();
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return failure();
  }
  
  auto& state = branchIt->second;
  
  if (numTokens > state.speculativeLength) {
    numTokens = state.speculativeLength;
  }
  
  result.acceptedMask.resize(numTokens);
  result.targetLogProbs.assign(targetLogProbs, targetLogProbs + numTokens);
  result.acceptedCount = 0;
  result.rejectedPosition = -1;
  
  // Verify each token using acceptance sampling
  for (int64_t i = 0; i < numTokens; i++) {
    float draftLogProb = i < static_cast<int64_t>(state.draftLogProbs.size()) ?
                         state.draftLogProbs[i] : 0.0f;
    float targetLogProb = targetLogProbs[i];
    
    bool accepted = shouldAcceptToken(draftLogProb, targetLogProb);
    result.acceptedMask[i] = accepted;
    
    if (accepted) {
      result.acceptedCount++;
    } else {
      if (result.rejectedPosition < 0) {
        result.rejectedPosition = i;
      }
      // Once we reject, all following tokens are rejected
      for (int64_t j = i + 1; j < numTokens; j++) {
        result.acceptedMask[j] = false;
      }
      break;
    }
  }
  
  result.acceptanceRate = numTokens > 0 ? 
      static_cast<float>(result.acceptedCount) / numTokens : 0.0f;
  
  double elapsed = getCurrentTimeMs() - startTime;
  result.verificationTime = elapsed;
  
  // Update metrics
  metrics_.numVerifications++;
  metrics_.totalVerificationTime += elapsed;
  metrics_.acceptedDraftTokens += result.acceptedCount;
  metrics_.rejectedDraftTokens += (numTokens - result.acceptedCount);
  
  // Update running average
  double alpha = 1.0 / metrics_.numVerifications;
  metrics_.averageAcceptanceRate = (1.0 - alpha) * metrics_.averageAcceptanceRate +
                                    alpha * result.acceptanceRate;
  
  return success();
}

LogicalResult SpeculativeKVCache::commitSpeculation(
    int32_t sequenceId, int32_t branchId,
    int64_t numAccepted) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return failure();
  }
  
  auto& state = branchIt->second;
  
  if (numAccepted > state.speculativeLength) {
    numAccepted = state.speculativeLength;
  }
  
  if (numAccepted <= 0) {
    return success();
  }
  
  // Move accepted speculative tokens to verified cache
  // This involves:
  // 1. Copying KV data from speculative blocks to verified cache
  // 2. Updating verified length
  // 3. Cleaning up committed speculative blocks
  
  // For now, simplified: just update the length counters
  state.verifiedLength += numAccepted;
  state.speculativeLength -= numAccepted;
  
  // Remove committed tokens from draft list
  if (numAccepted <= static_cast<int64_t>(state.draftTokenIds.size())) {
    state.draftTokenIds.erase(state.draftTokenIds.begin(),
                               state.draftTokenIds.begin() + numAccepted);
    state.draftLogProbs.erase(state.draftLogProbs.begin(),
                               state.draftLogProbs.begin() + numAccepted);
  }
  
  // If this is not the main branch, merge into main
  if (branchId != 0) {
    auto& mainState = stateIt->second[0];
    mainState.verifiedLength = state.verifiedLength;
    
    // Remove this branch
    state.isActive = false;
  }
  
  metrics_.numCommits++;
  
  return success();
}

LogicalResult SpeculativeKVCache::rollbackSpeculation(
    int32_t sequenceId, int32_t branchId) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return failure();
  }
  
  auto& state = branchIt->second;
  
  // Free all speculative blocks
  freeSpeculativeBlocks(sequenceId, branchId);
  
  // Reset speculation state
  state.speculativeLength = 0;
  state.draftTokenIds.clear();
  state.draftLogProbs.clear();
  
  for (auto& blockIndices : state.speculativeBlockIndices) {
    blockIndices.clear();
  }
  
  // If not main branch, deactivate
  if (branchId != 0) {
    state.isActive = false;
    stateIt->second.erase(branchId);
  }
  
  metrics_.numRollbacks++;
  
  return success();
}

LogicalResult SpeculativeKVCache::rollbackToPosition(
    int32_t sequenceId, int64_t position) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  // Rollback all branches that extend beyond position
  std::vector<int32_t> branchesToRemove;
  
  for (auto& [branchId, state] : stateIt->second) {
    if (state.branchPosition > position) {
      branchesToRemove.push_back(branchId);
    } else if (state.verifiedLength + state.speculativeLength > position) {
      // Partial rollback
      int64_t speculativeToKeep = position - state.verifiedLength;
      if (speculativeToKeep < 0) {
        // Need to rollback verified tokens too - not supported
        return failure();
      }
      
      // Trim speculative tokens
      int64_t toRemove = state.speculativeLength - speculativeToKeep;
      state.speculativeLength = speculativeToKeep;
      
      if (toRemove > 0 && static_cast<int64_t>(state.draftTokenIds.size()) > speculativeToKeep) {
        state.draftTokenIds.resize(speculativeToKeep);
        state.draftLogProbs.resize(speculativeToKeep);
      }
    }
  }
  
  for (int32_t branchId : branchesToRemove) {
    rollbackSpeculation(sequenceId, branchId);
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Tree Attention Support
//===----------------------------------------------------------------------===//

LogicalResult SpeculativeKVCache::getTreeAttentionMask(
    int32_t sequenceId,
    std::vector<int32_t>& branchIds,
    std::vector<std::vector<bool>>& mask) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  // Collect all active branches
  std::vector<SpeculationState> branches;
  branchIds.clear();
  
  for (auto& [branchId, state] : stateIt->second) {
    if (state.isActive) {
      branches.push_back(state);
      branchIds.push_back(branchId);
    }
  }
  
  buildTreeMask(branches, mask);
  
  return success();
}

LogicalResult SpeculativeKVCache::computeTreeAttention(
    const void* queries,
    int32_t sequenceId,
    const std::vector<int32_t>& branchIds,
    void* output) {
  
  if (!config_.enableTreeAttention) {
    return failure();
  }
  
  // Get tree attention mask
  std::vector<int32_t> actualBranchIds;
  std::vector<std::vector<bool>> mask;
  if (getTreeAttentionMask(sequenceId, actualBranchIds, mask).failed()) {
    return failure();
  }
  
  // Compute attention with mask
  // This would involve:
  // 1. Gathering KV from all branches
  // 2. Computing attention with the tree mask
  // 3. Storing results
  
  return success();
}

//===----------------------------------------------------------------------===//
// State Management
//===----------------------------------------------------------------------===//

const SpeculationState* SpeculativeKVCache::getSpeculationState(
    int32_t sequenceId, int32_t branchId) const {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return nullptr;
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return nullptr;
  }
  
  return &branchIt->second;
}

std::vector<int32_t> SpeculativeKVCache::getActiveBranches(
    int32_t sequenceId) const {
  
  std::vector<int32_t> branches;
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt != speculationStates_.end()) {
    for (const auto& [branchId, state] : stateIt->second) {
      if (state.isActive) {
        branches.push_back(branchId);
      }
    }
  }
  
  return branches;
}

LogicalResult SpeculativeKVCache::clearSpeculation(int32_t sequenceId) {
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return success(); // Already clear
  }
  
  // Rollback all branches
  std::vector<int32_t> branchIds;
  for (const auto& [branchId, _] : stateIt->second) {
    branchIds.push_back(branchId);
  }
  
  for (int32_t branchId : branchIds) {
    freeSpeculativeBlocks(sequenceId, branchId);
  }
  
  stateIt->second.clear();
  branchPoints_.erase(sequenceId);
  
  // Reinitialize main branch
  initializeSpeculationState(sequenceId);
  
  return success();
}

void SpeculativeKVCache::reset() {
  // Clear all speculation states
  for (auto& [seqId, branches] : speculationStates_) {
    for (auto& [branchId, _] : branches) {
      freeSpeculativeBlocks(seqId, branchId);
    }
  }
  
  speculationStates_.clear();
  branchPoints_.clear();
  
  // Reset verified cache
  verifiedCache_->reset();
  
  nextBranchId_ = 0;
  timestampCounter_ = 0;
  resetMetrics();
}

void SpeculativeKVCache::resetMetrics() {
  metrics_ = SpeculativeMetrics{};
}

int64_t SpeculativeKVCache::getVerifiedLength(int32_t sequenceId) const {
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return 0;
  }
  
  auto branchIt = stateIt->second.find(0);
  if (branchIt == stateIt->second.end()) {
    return 0;
  }
  
  return branchIt->second.verifiedLength;
}

int64_t SpeculativeKVCache::getSpeculativeLength(
    int32_t sequenceId, int32_t branchId) const {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return 0;
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return 0;
  }
  
  return branchIt->second.speculativeLength;
}

int64_t SpeculativeKVCache::getTotalLength(
    int32_t sequenceId, int32_t branchId) const {
  
  const auto* state = getSpeculationState(sequenceId, branchId);
  if (!state) {
    return 0;
  }
  
  return state->verifiedLength + state->speculativeLength;
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

void SpeculativeKVCache::initializeSpeculationState(int32_t sequenceId) {
  SpeculationState state;
  state.sequenceId = sequenceId;
  state.branchId = 0;
  state.verifiedLength = 0;
  state.speculativeLength = 0;
  state.parentBranchId = -1;
  state.branchPosition = 0;
  state.isActive = true;
  state.speculativeBlockIndices.resize(numLayers_);
  
  speculationStates_[sequenceId][0] = state;
}

LogicalResult SpeculativeKVCache::allocateSpeculativeBlocks(
    int32_t sequenceId, int32_t branchId, int64_t numTokens) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return failure();
  }
  
  auto& state = branchIt->second;
  
  // Calculate how many new blocks we need
  int64_t currentTotal = state.speculativeLength;
  int64_t newTotal = currentTotal + numTokens;
  
  int64_t currentBlocks = (currentTotal + blockSize_ - 1) / blockSize_;
  int64_t neededBlocks = (newTotal + blockSize_ - 1) / blockSize_;
  int64_t newBlocks = neededBlocks - currentBlocks;
  
  if (newBlocks > 0) {
    for (int64_t layer = 0; layer < numLayers_; layer++) {
      for (int64_t b = 0; b < newBlocks; b++) {
        KVBlock* block = speculativeAllocator_->allocateBlock();
        if (!block) {
          return failure();
        }
        state.speculativeBlockIndices[layer].push_back(block->getBlockId());
      }
    }
  }
  
  return success();
}

void SpeculativeKVCache::freeSpeculativeBlocks(
    int32_t sequenceId, int32_t branchId) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return;
  }
  
  auto branchIt = stateIt->second.find(branchId);
  if (branchIt == stateIt->second.end()) {
    return;
  }
  
  auto& state = branchIt->second;
  
  for (auto& blockIndices : state.speculativeBlockIndices) {
    for (int32_t blockId : blockIndices) {
      speculativeAllocator_->deallocateBlock(blockId);
    }
    blockIndices.clear();
  }
}

LogicalResult SpeculativeKVCache::copyBlockState(
    int32_t sequenceId, int32_t srcBranchId, int32_t dstBranchId) {
  
  auto stateIt = speculationStates_.find(sequenceId);
  if (stateIt == speculationStates_.end()) {
    return failure();
  }
  
  auto srcIt = stateIt->second.find(srcBranchId);
  auto dstIt = stateIt->second.find(dstBranchId);
  
  if (srcIt == stateIt->second.end() || dstIt == stateIt->second.end()) {
    return failure();
  }
  
  // Copy block references (shallow copy - blocks are shared until modified)
  dstIt->second.speculativeBlockIndices = srcIt->second.speculativeBlockIndices;
  
  return success();
}

bool SpeculativeKVCache::shouldAcceptToken(
    float draftLogProb, float targetLogProb) const {
  
  // Acceptance-rejection sampling based on log probabilities
  // Accept if target prob >= draft prob
  // Otherwise, accept with probability target_prob / draft_prob
  
  float draftProb = std::exp(draftLogProb);
  float targetProb = std::exp(targetLogProb);
  
  if (targetProb >= draftProb * config_.acceptanceThreshold) {
    return true;
  }
  
  // Probabilistic acceptance
  float acceptProb = targetProb / (draftProb + 1e-10f);
  return sampleUniform() < acceptProb;
}

void SpeculativeKVCache::buildTreeMask(
    const std::vector<SpeculationState>& branches,
    std::vector<std::vector<bool>>& mask) {
  
  // Build attention mask for tree structure
  // Each branch can attend to:
  // 1. All verified tokens (shared prefix)
  // 2. Its own speculative tokens
  // 3. Parent branch's speculative tokens (if any)
  
  if (branches.empty()) {
    mask.clear();
    return;
  }
  
  // Calculate total sequence length across all branches
  int64_t verifiedLen = branches[0].verifiedLength;
  int64_t totalLen = verifiedLen;
  
  for (const auto& branch : branches) {
    totalLen += branch.speculativeLength;
  }
  
  mask.resize(branches.size());
  
  for (size_t i = 0; i < branches.size(); i++) {
    mask[i].resize(totalLen, false);
    
    // Can attend to all verified tokens
    for (int64_t j = 0; j < verifiedLen; j++) {
      mask[i][j] = true;
    }
    
    // Can attend to own speculative tokens
    int64_t offset = verifiedLen;
    for (size_t b = 0; b < i; b++) {
      offset += branches[b].speculativeLength;
    }
    
    for (int64_t j = 0; j < branches[i].speculativeLength; j++) {
      mask[i][offset + j] = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// SpeculativeDecodingEngine Implementation
//===----------------------------------------------------------------------===//

SpeculativeDecodingEngine::SpeculativeDecodingEngine(
    SpeculativeKVCache& cache,
    DraftModelInterface* draftModel)
    : cache_(cache), draftModel_(draftModel),
      maxDraftTokens_(cache.getConfig().maxDraftTokens),
      acceptanceThreshold_(cache.getConfig().acceptanceThreshold),
      totalSteps_(0), totalTokensGenerated_(0),
      totalDraftTokens_(0), totalAccepted_(0) {}

SpeculativeDecodingEngine::~SpeculativeDecodingEngine() = default;

LogicalResult SpeculativeDecodingEngine::step(
    int32_t sequenceId,
    const void* targetLogits,
    std::vector<int32_t>& outputTokens,
    int64_t& numGenerated) {
  
  outputTokens.clear();
  numGenerated = 0;
  
  if (!draftModel_) {
    // Without draft model, just generate one token
    outputTokens.push_back(0); // Placeholder
    numGenerated = 1;
    return success();
  }
  
  // Generate draft tokens
  std::vector<int32_t> draftTokens;
  std::vector<float> draftLogProbs;
  
  if (draftModel_->generateDrafts(nullptr, sequenceId, maxDraftTokens_,
                                   draftTokens, draftLogProbs).failed()) {
    return failure();
  }
  
  totalDraftTokens_ += draftTokens.size();
  
  // Verify drafts against target model
  VerificationResult result;
  const float* logProbs = static_cast<const float*>(targetLogits);
  
  if (cache_.verifySpeculation(sequenceId, 0, logProbs,
                                draftTokens.size(), result).failed()) {
    return failure();
  }
  
  // Commit accepted tokens
  if (result.acceptedCount > 0) {
    if (cache_.commitSpeculation(sequenceId, 0, result.acceptedCount).failed()) {
      return failure();
    }
    
    for (int64_t i = 0; i < result.acceptedCount; i++) {
      outputTokens.push_back(draftTokens[i]);
    }
  }
  
  // Always generate at least one new token (from target model)
  outputTokens.push_back(0); // Placeholder for sampled token
  
  numGenerated = outputTokens.size();
  totalTokensGenerated_ += numGenerated;
  totalAccepted_ += result.acceptedCount;
  totalSteps_++;
  
  return success();
}

float SpeculativeDecodingEngine::getAverageAcceptanceRate() const {
  if (totalDraftTokens_ == 0) return 0.0f;
  return static_cast<float>(totalAccepted_) / totalDraftTokens_;
}

float SpeculativeDecodingEngine::getSpeedup() const {
  if (totalSteps_ == 0) return 1.0f;
  // Speedup = average tokens generated per step
  return static_cast<float>(totalTokensGenerated_) / totalSteps_;
}

//===----------------------------------------------------------------------===//
// ParallelSpeculator Implementation
//===----------------------------------------------------------------------===//

ParallelSpeculator::ParallelSpeculator(
    SpeculativeKVCache& cache, int64_t numParallelBranches)
    : cache_(cache), numParallelBranches_(numParallelBranches) {}

ParallelSpeculator::~ParallelSpeculator() = default;

LogicalResult ParallelSpeculator::generateParallelDrafts(
    int32_t sequenceId,
    const std::vector<std::vector<int32_t>>& drafts,
    std::vector<int32_t>& branchIds) {
  
  int64_t numBranches = std::min(
      static_cast<int64_t>(drafts.size()), numParallelBranches_);
  
  if (cache_.createTreeBranches(sequenceId, numBranches, branchIds).failed()) {
    return failure();
  }
  
  // Store drafts in each branch
  for (size_t i = 0; i < branchIds.size(); i++) {
    // Would append draft KV to each branch
  }
  
  return success();
}

LogicalResult ParallelSpeculator::verifyAndSelectBest(
    int32_t sequenceId,
    const std::vector<int32_t>& branchIds,
    const std::vector<std::vector<float>>& targetLogProbs,
    int32_t& bestBranchId,
    int64_t& acceptedCount) {
  
  bestBranchId = -1;
  acceptedCount = 0;
  
  for (size_t i = 0; i < branchIds.size(); i++) {
    VerificationResult result;
    
    if (cache_.verifySpeculation(sequenceId, branchIds[i],
                                  targetLogProbs[i].data(),
                                  targetLogProbs[i].size(),
                                  result).succeeded()) {
      if (result.acceptedCount > acceptedCount) {
        bestBranchId = branchIds[i];
        acceptedCount = result.acceptedCount;
      }
    }
  }
  
  if (bestBranchId < 0) {
    return failure();
  }
  
  return success();
}

LogicalResult ParallelSpeculator::cleanupBranches(
    int32_t sequenceId,
    const std::vector<int32_t>& branchIds,
    int32_t keepBranchId) {
  
  for (int32_t branchId : branchIds) {
    if (branchId != keepBranchId) {
      cache_.rollbackSpeculation(sequenceId, branchId);
    }
  }
  
  return success();
}

} // namespace runtime
} // namespace llm
} // namespace mlir
