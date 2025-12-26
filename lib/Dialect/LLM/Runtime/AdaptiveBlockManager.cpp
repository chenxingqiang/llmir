//===- AdaptiveBlockManager.cpp - Adaptive Block Size Management --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements adaptive block size management for the KV cache.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/AdaptiveBlockManager.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

int64_t getCurrentTimeMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

double calculateStdDev(const std::deque<int64_t>& data, double mean) {
  if (data.empty()) return 0.0;
  
  double sumSq = 0.0;
  for (int64_t val : data) {
    double diff = static_cast<double>(val) - mean;
    sumSq += diff * diff;
  }
  
  return std::sqrt(sumSq / data.size());
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// MultiSizeBlockAllocator Implementation
//===----------------------------------------------------------------------===//

MultiSizeBlockAllocator::MultiSizeBlockAllocator(
    int64_t headDim, int64_t numHeads,
    const BlockSizeConfig& config,
    bool enableGPU)
    : headDim_(headDim), numHeads_(numHeads), config_(config),
      enableGPU_(enableGPU),
      smallIdOffset_(0), primaryIdOffset_(100000), largeIdOffset_(200000) {
  
  // Create allocators for each block size
  smallAllocator_ = std::make_unique<BlockAllocator>(
      config_.smallBlockSize, headDim * numHeads, enableGPU);
  primaryAllocator_ = std::make_unique<BlockAllocator>(
      config_.primaryBlockSize, headDim * numHeads, enableGPU);
  largeAllocator_ = std::make_unique<BlockAllocator>(
      config_.largeBlockSize, headDim * numHeads, enableGPU);
  
  // Preallocate blocks
  smallAllocator_->preallocateBlocks(config_.smallBlockPoolSize);
  primaryAllocator_->preallocateBlocks(config_.primaryBlockPoolSize);
  largeAllocator_->preallocateBlocks(config_.largeBlockPoolSize);
}

MultiSizeBlockAllocator::~MultiSizeBlockAllocator() = default;

KVBlock* MultiSizeBlockAllocator::allocateBlock(int64_t expectedTokens) {
  if (expectedTokens < config_.smallBlockThreshold) {
    return allocateSmallBlock();
  } else if (expectedTokens > config_.largeBlockThreshold) {
    return allocateLargeBlock();
  } else {
    return allocatePrimaryBlock();
  }
}

KVBlock* MultiSizeBlockAllocator::allocateSmallBlock() {
  KVBlock* block = smallAllocator_->allocateBlock();
  if (block) {
    blockOwnership_[block->getBlockId() + smallIdOffset_] = smallAllocator_.get();
  }
  return block;
}

KVBlock* MultiSizeBlockAllocator::allocatePrimaryBlock() {
  KVBlock* block = primaryAllocator_->allocateBlock();
  if (block) {
    blockOwnership_[block->getBlockId() + primaryIdOffset_] = primaryAllocator_.get();
  }
  return block;
}

KVBlock* MultiSizeBlockAllocator::allocateLargeBlock() {
  KVBlock* block = largeAllocator_->allocateBlock();
  if (block) {
    blockOwnership_[block->getBlockId() + largeIdOffset_] = largeAllocator_.get();
  }
  return block;
}

void MultiSizeBlockAllocator::deallocateBlock(KVBlock* block) {
  if (!block) return;
  deallocateBlock(block->getBlockId());
}

void MultiSizeBlockAllocator::deallocateBlock(int32_t blockId) {
  // Determine which allocator owns this block
  BlockAllocator* allocator = nullptr;
  int32_t localId = blockId;
  
  if (blockId >= largeIdOffset_) {
    allocator = largeAllocator_.get();
    localId = blockId - largeIdOffset_;
  } else if (blockId >= primaryIdOffset_) {
    allocator = primaryAllocator_.get();
    localId = blockId - primaryIdOffset_;
  } else {
    allocator = smallAllocator_.get();
    localId = blockId - smallIdOffset_;
  }
  
  if (allocator) {
    allocator->deallocateBlock(localId);
    blockOwnership_.erase(blockId);
  }
}

KVBlock* MultiSizeBlockAllocator::getBlock(int32_t blockId) {
  BlockAllocator* allocator = nullptr;
  int32_t localId = blockId;
  
  if (blockId >= largeIdOffset_) {
    allocator = largeAllocator_.get();
    localId = blockId - largeIdOffset_;
  } else if (blockId >= primaryIdOffset_) {
    allocator = primaryAllocator_.get();
    localId = blockId - primaryIdOffset_;
  } else {
    allocator = smallAllocator_.get();
    localId = blockId - smallIdOffset_;
  }
  
  if (allocator) {
    return allocator->getBlock(localId);
  }
  return nullptr;
}

void MultiSizeBlockAllocator::updateConfig(const BlockSizeConfig& config) {
  config_ = config;
  // Note: Changing block sizes at runtime requires migration
}

size_t MultiSizeBlockAllocator::getNumFreeSmallBlocks() const {
  return smallAllocator_->getNumFreeBlocks();
}

size_t MultiSizeBlockAllocator::getNumFreePrimaryBlocks() const {
  return primaryAllocator_->getNumFreeBlocks();
}

size_t MultiSizeBlockAllocator::getNumFreeLargeBlocks() const {
  return largeAllocator_->getNumFreeBlocks();
}

size_t MultiSizeBlockAllocator::getTotalMemoryUsage() const {
  return smallAllocator_->getTotalMemoryUsage() +
         primaryAllocator_->getTotalMemoryUsage() +
         largeAllocator_->getTotalMemoryUsage();
}

float MultiSizeBlockAllocator::getAverageUtilization() const {
  size_t totalBlocks = smallAllocator_->getTotalBlocks() +
                       primaryAllocator_->getTotalBlocks() +
                       largeAllocator_->getTotalBlocks();
  
  size_t freeBlocks = getNumFreeSmallBlocks() +
                      getNumFreePrimaryBlocks() +
                      getNumFreeLargeBlocks();
  
  if (totalBlocks == 0) return 0.0f;
  return 1.0f - static_cast<float>(freeBlocks) / totalBlocks;
}

//===----------------------------------------------------------------------===//
// WorkloadAnalyzer Implementation
//===----------------------------------------------------------------------===//

WorkloadAnalyzer::WorkloadAnalyzer(int64_t windowSize)
    : windowSize_(windowSize), statsDirty_(true) {}

WorkloadAnalyzer::~WorkloadAnalyzer() = default;

void WorkloadAnalyzer::recordSequenceLength(int64_t length) {
  seqLengths_.push_back(length);
  while (static_cast<int64_t>(seqLengths_.size()) > windowSize_) {
    seqLengths_.pop_front();
  }
  statsDirty_ = true;
}

void WorkloadAnalyzer::recordBatchSize(int64_t size) {
  batchSizes_.push_back(size);
  while (static_cast<int64_t>(batchSizes_.size()) > windowSize_) {
    batchSizes_.pop_front();
  }
  statsDirty_ = true;
}

void WorkloadAnalyzer::recordBlockUtilization(float utilization) {
  blockUtilizations_.push_back(utilization);
  while (static_cast<int64_t>(blockUtilizations_.size()) > windowSize_) {
    blockUtilizations_.pop_front();
  }
  statsDirty_ = true;
}

void WorkloadAnalyzer::recordAppend(int64_t numTokens, double latencyMs) {
  appendEvents_.push_back({numTokens, latencyMs});
  while (static_cast<int64_t>(appendEvents_.size()) > windowSize_) {
    appendEvents_.pop_front();
  }
  statsDirty_ = true;
}

void WorkloadAnalyzer::recordLookup(int64_t numTokens, double latencyMs) {
  lookupEvents_.push_back({numTokens, latencyMs});
  while (static_cast<int64_t>(lookupEvents_.size()) > windowSize_) {
    lookupEvents_.pop_front();
  }
  statsDirty_ = true;
}

WorkloadStats WorkloadAnalyzer::getStats() const {
  if (statsDirty_) {
    updateStats();
  }
  return cachedStats_;
}

bool WorkloadAnalyzer::isSeqLengthIncreasing() const {
  return calculateTrend(seqLengths_) > 0.1;
}

bool WorkloadAnalyzer::isBatchSizeIncreasing() const {
  return calculateTrend(batchSizes_) > 0.1;
}

bool WorkloadAnalyzer::isFragmentationHigh() const {
  auto stats = getStats();
  return stats.fragmentationRatio > 0.3;
}

double WorkloadAnalyzer::predictNextSeqLength() const {
  if (seqLengths_.empty()) return 0.0;
  
  // Simple exponential moving average prediction
  double alpha = 0.3;
  double prediction = seqLengths_.front();
  
  for (int64_t len : seqLengths_) {
    prediction = alpha * len + (1.0 - alpha) * prediction;
  }
  
  // Apply trend adjustment
  double trend = calculateTrend(seqLengths_);
  prediction += trend * prediction * 0.1;
  
  return prediction;
}

double WorkloadAnalyzer::predictNextBatchSize() const {
  if (batchSizes_.empty()) return 1.0;
  
  double alpha = 0.3;
  double prediction = batchSizes_.front();
  
  for (int64_t size : batchSizes_) {
    prediction = alpha * size + (1.0 - alpha) * prediction;
  }
  
  return prediction;
}

void WorkloadAnalyzer::reset() {
  seqLengths_.clear();
  batchSizes_.clear();
  blockUtilizations_.clear();
  appendEvents_.clear();
  lookupEvents_.clear();
  statsDirty_ = true;
}

void WorkloadAnalyzer::updateStats() const {
  WorkloadStats& stats = cachedStats_;
  
  // Sequence length stats
  if (!seqLengths_.empty()) {
    double sum = 0;
    stats.maxSeqLen = 0;
    stats.minSeqLen = std::numeric_limits<double>::max();
    
    for (int64_t len : seqLengths_) {
      sum += len;
      stats.maxSeqLen = std::max(stats.maxSeqLen, static_cast<double>(len));
      stats.minSeqLen = std::min(stats.minSeqLen, static_cast<double>(len));
    }
    
    stats.averageSeqLen = sum / seqLengths_.size();
    stats.seqLenStdDev = calculateStdDev(seqLengths_, stats.averageSeqLen);
  }
  
  // Batch size stats
  if (!batchSizes_.empty()) {
    double sum = 0;
    stats.maxBatchSize = 0;
    
    for (int64_t size : batchSizes_) {
      sum += size;
      stats.maxBatchSize = std::max(stats.maxBatchSize, static_cast<double>(size));
    }
    
    stats.averageBatchSize = sum / batchSizes_.size();
  }
  
  // Block utilization stats
  if (!blockUtilizations_.empty()) {
    double sum = 0;
    for (float util : blockUtilizations_) {
      sum += util;
    }
    stats.averageBlockUtilization = sum / blockUtilizations_.size();
    stats.fragmentationRatio = 1.0 - stats.averageBlockUtilization;
  }
  
  // Latency and throughput stats
  if (!appendEvents_.empty()) {
    double totalTokens = 0;
    double totalLatency = 0;
    
    for (const auto& [tokens, latency] : appendEvents_) {
      totalTokens += tokens;
      totalLatency += latency;
    }
    
    stats.appendLatencyMs = totalLatency / appendEvents_.size();
    if (totalLatency > 0) {
      stats.tokensPerSecond = totalTokens / (totalLatency / 1000.0);
    }
  }
  
  if (!lookupEvents_.empty()) {
    double totalLatency = 0;
    
    for (const auto& [_, latency] : lookupEvents_) {
      totalLatency += latency;
    }
    
    stats.lookupLatencyMs = totalLatency / lookupEvents_.size();
  }
  
  // Read/write ratio
  if (!appendEvents_.empty()) {
    stats.readWriteRatio = static_cast<double>(lookupEvents_.size()) /
                           appendEvents_.size();
  }
  
  statsDirty_ = false;
}

double WorkloadAnalyzer::calculateTrend(const std::deque<int64_t>& data) const {
  if (data.size() < 2) return 0.0;
  
  // Simple linear regression
  int64_t n = data.size();
  double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  
  for (int64_t i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i];
    sumXY += i * data[i];
    sumX2 += i * i;
  }
  
  double denominator = n * sumX2 - sumX * sumX;
  if (denominator == 0) return 0.0;
  
  double slope = (n * sumXY - sumX * sumY) / denominator;
  double meanY = sumY / n;
  
  // Normalize slope by mean
  if (meanY == 0) return 0.0;
  return slope / meanY;
}

//===----------------------------------------------------------------------===//
// AdaptiveBlockManager Implementation
//===----------------------------------------------------------------------===//

AdaptiveBlockManager::AdaptiveBlockManager(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    const BlockSizeConfig& blockConfig,
    const AdaptationConfig& adaptConfig,
    bool enableGPU)
    : numLayers_(numLayers), numHeads_(numHeads), headDim_(headDim),
      blockConfig_(blockConfig), adaptConfig_(adaptConfig),
      enableGPU_(enableGPU), lastAdaptationTime_(0), adaptationCount_(0) {
  
  // Create allocators for each layer
  allocators_.reserve(numLayers);
  for (int64_t i = 0; i < numLayers; i++) {
    allocators_.push_back(std::make_unique<MultiSizeBlockAllocator>(
        headDim, numHeads, blockConfig, enableGPU));
  }
  
  // Create workload analyzer
  analyzer_ = std::make_unique<WorkloadAnalyzer>(adaptConfig.windowSize);
  
  metrics_ = AdaptationMetrics{};
}

AdaptiveBlockManager::~AdaptiveBlockManager() = default;

LogicalResult AdaptiveBlockManager::allocateBlocksForSequence(
    int32_t sequenceId,
    int64_t expectedLength,
    std::vector<KVBlock*>& blocks) {
  
  // Calculate number of blocks needed
  int64_t blockSize = selectBlockSize(expectedLength);
  int64_t numBlocks = (expectedLength + blockSize - 1) / blockSize;
  
  blocks.clear();
  blocks.reserve(numBlocks);
  
  // Allocate blocks for first layer (others share same pattern)
  auto& allocator = allocators_[0];
  
  for (int64_t i = 0; i < numBlocks; i++) {
    KVBlock* block = allocator->allocateBlock(expectedLength);
    if (!block) {
      // Rollback
      for (KVBlock* b : blocks) {
        allocator->deallocateBlock(b);
      }
      blocks.clear();
      return failure();
    }
    blocks.push_back(block);
  }
  
  // Store mapping
  sequenceBlocks_[sequenceId].resize(numLayers_);
  sequenceBlocks_[sequenceId][0] = blocks;
  
  // Allocate for other layers
  for (int64_t layer = 1; layer < numLayers_; layer++) {
    auto& layerAllocator = allocators_[layer];
    std::vector<KVBlock*> layerBlocks;
    layerBlocks.reserve(numBlocks);
    
    for (int64_t i = 0; i < numBlocks; i++) {
      KVBlock* block = layerAllocator->allocateBlock(expectedLength);
      if (!block) {
        // Rollback this layer
        for (KVBlock* b : layerBlocks) {
          layerAllocator->deallocateBlock(b);
        }
        // Rollback previous layers
        deallocateSequence(sequenceId);
        return failure();
      }
      layerBlocks.push_back(block);
    }
    
    sequenceBlocks_[sequenceId][layer] = std::move(layerBlocks);
  }
  
  return success();
}

KVBlock* AdaptiveBlockManager::allocateBlock(int64_t expectedTokens) {
  return allocators_[0]->allocateBlock(expectedTokens);
}

void AdaptiveBlockManager::deallocateBlock(KVBlock* block) {
  if (!block) return;
  
  for (auto& allocator : allocators_) {
    // Try each allocator (in practice, would track ownership)
    allocator->deallocateBlock(block);
  }
}

void AdaptiveBlockManager::deallocateSequence(int32_t sequenceId) {
  auto it = sequenceBlocks_.find(sequenceId);
  if (it == sequenceBlocks_.end()) return;
  
  for (int64_t layer = 0; layer < numLayers_; layer++) {
    for (KVBlock* block : it->second[layer]) {
      allocators_[layer]->deallocateBlock(block);
    }
  }
  
  sequenceBlocks_.erase(it);
}

void AdaptiveBlockManager::recordAppend(int32_t sequenceId, 
                                         int64_t numTokens,
                                         double latencyMs) {
  analyzer_->recordAppend(numTokens, latencyMs);
  
  // Periodically check for adaptation
  if (adaptConfig_.policy != AdaptationPolicy::STATIC) {
    checkAndAdapt();
  }
}

void AdaptiveBlockManager::recordLookup(int32_t sequenceId,
                                         int64_t numTokens,
                                         double latencyMs) {
  analyzer_->recordLookup(numTokens, latencyMs);
}

void AdaptiveBlockManager::recordSequenceComplete(int32_t sequenceId,
                                                   int64_t finalLength) {
  analyzer_->recordSequenceLength(finalLength);
  
  // Record block utilization
  auto it = sequenceBlocks_.find(sequenceId);
  if (it != sequenceBlocks_.end() && !it->second.empty()) {
    int64_t totalSlots = 0;
    int64_t usedSlots = finalLength;
    
    for (KVBlock* block : it->second[0]) {
      totalSlots += block->getBlockSize();
    }
    
    if (totalSlots > 0) {
      float utilization = static_cast<float>(usedSlots) / totalSlots;
      analyzer_->recordBlockUtilization(utilization);
    }
  }
}

WorkloadStats AdaptiveBlockManager::getWorkloadStats() const {
  return analyzer_->getStats();
}

LogicalResult AdaptiveBlockManager::checkAndAdapt() {
  if (adaptConfig_.policy == AdaptationPolicy::STATIC) {
    return success();
  }
  
  int64_t currentTime = getCurrentTimeMs();
  
  // Check if enough time has passed since last adaptation
  if (currentTime - lastAdaptationTime_ < adaptConfig_.minAdaptationInterval) {
    return success();
  }
  
  WorkloadStats stats = analyzer_->getStats();
  BlockSizeConfig recommended = calculateOptimalConfig(stats);
  
  // Check if change is significant enough
  double blockSizeChange = std::abs(
      static_cast<double>(recommended.primaryBlockSize) - 
      static_cast<double>(blockConfig_.primaryBlockSize)) /
      blockConfig_.primaryBlockSize;
  
  if (blockSizeChange < adaptConfig_.adaptationThreshold) {
    return success(); // Change too small
  }
  
  return performAdaptation(recommended);
}

LogicalResult AdaptiveBlockManager::forceAdapt(const BlockSizeConfig& newConfig) {
  return performAdaptation(newConfig);
}

int64_t AdaptiveBlockManager::getRecommendedBlockSize() const {
  WorkloadStats stats = analyzer_->getStats();
  
  // Heuristic: block size should be roughly sqrt of average sequence length
  // for good balance between fragmentation and overhead
  if (stats.averageSeqLen <= 0) {
    return blockConfig_.primaryBlockSize;
  }
  
  double optimalSize = std::sqrt(stats.averageSeqLen);
  
  // Round to power of 2 for alignment
  int64_t blockSize = 1;
  while (blockSize < optimalSize) {
    blockSize *= 2;
  }
  
  // Clamp to reasonable range
  blockSize = std::max(int64_t(4), std::min(int64_t(256), blockSize));
  
  return blockSize;
}

BlockSizeConfig AdaptiveBlockManager::getRecommendedConfig() const {
  return calculateOptimalConfig(analyzer_->getStats());
}

LogicalResult AdaptiveBlockManager::compactMemory() {
  // Coalesce partially filled blocks within same sequence
  for (auto& [seqId, layers] : sequenceBlocks_) {
    for (int64_t layer = 0; layer < numLayers_; layer++) {
      auto& blocks = layers[layer];
      
      // Find partially filled blocks
      std::vector<KVBlock*> partialBlocks;
      for (KVBlock* block : blocks) {
        if (block->getUsedSlots() < block->getBlockSize()) {
          partialBlocks.push_back(block);
        }
      }
      
      // Try to coalesce pairs of partial blocks
      // (Simplified - actual implementation would copy data)
    }
  }
  
  metrics_.numCompactions++;
  return success();
}

LogicalResult AdaptiveBlockManager::defragment() {
  // Move data from small blocks to larger ones if beneficial
  metrics_.numDefragmentations++;
  return success();
}

size_t AdaptiveBlockManager::getTotalMemoryUsage() const {
  size_t total = 0;
  for (const auto& allocator : allocators_) {
    total += allocator->getTotalMemoryUsage();
  }
  return total;
}

float AdaptiveBlockManager::getFragmentationRatio() const {
  WorkloadStats stats = analyzer_->getStats();
  return static_cast<float>(stats.fragmentationRatio);
}

float AdaptiveBlockManager::getAverageBlockUtilization() const {
  float total = 0.0f;
  for (const auto& allocator : allocators_) {
    total += allocator->getAverageUtilization();
  }
  return total / numLayers_;
}

void AdaptiveBlockManager::resetMetrics() {
  metrics_ = AdaptationMetrics{};
}

int64_t AdaptiveBlockManager::selectBlockSize(int64_t expectedTokens) const {
  if (expectedTokens < blockConfig_.smallBlockThreshold) {
    return blockConfig_.smallBlockSize;
  } else if (expectedTokens > blockConfig_.largeBlockThreshold) {
    return blockConfig_.largeBlockSize;
  }
  return blockConfig_.primaryBlockSize;
}

BlockSizeConfig AdaptiveBlockManager::calculateOptimalConfig(
    const WorkloadStats& stats) const {
  
  BlockSizeConfig config = blockConfig_;
  
  // Calculate optimal primary block size
  int64_t optimalPrimary = getRecommendedBlockSize();
  config.primaryBlockSize = optimalPrimary;
  
  // Small block size: 1/4 of primary
  config.smallBlockSize = std::max(int64_t(4), optimalPrimary / 4);
  
  // Large block size: 4x primary
  config.largeBlockSize = std::min(int64_t(256), optimalPrimary * 4);
  
  // Thresholds based on sequence length distribution
  config.smallBlockThreshold = static_cast<int64_t>(
      stats.averageSeqLen - stats.seqLenStdDev);
  config.smallBlockThreshold = std::max(int64_t(16), config.smallBlockThreshold);
  
  config.largeBlockThreshold = static_cast<int64_t>(
      stats.averageSeqLen + 2 * stats.seqLenStdDev);
  config.largeBlockThreshold = std::min(int64_t(2048), config.largeBlockThreshold);
  
  // Pool sizes based on batch characteristics
  config.primaryBlockPoolSize = static_cast<size_t>(
      stats.maxBatchSize * stats.averageSeqLen / optimalPrimary * 2);
  config.primaryBlockPoolSize = std::max(size_t(64), config.primaryBlockPoolSize);
  
  return config;
}

LogicalResult AdaptiveBlockManager::performAdaptation(
    const BlockSizeConfig& newConfig) {
  
  double startTime = getCurrentTimeMs();
  
  // Record in history
  metrics_.adaptationHistory.push_back({adaptationCount_, newConfig});
  
  // Update configuration
  BlockSizeConfig oldConfig = blockConfig_;
  blockConfig_ = newConfig;
  
  // Update allocators
  for (auto& allocator : allocators_) {
    allocator->updateConfig(newConfig);
  }
  
  // Migrate existing blocks if sizes changed significantly
  if (newConfig.primaryBlockSize != oldConfig.primaryBlockSize) {
    migrateBlocks(oldConfig.primaryBlockSize, newConfig.primaryBlockSize);
  }
  
  double elapsed = getCurrentTimeMs() - startTime;
  
  metrics_.numAdaptations++;
  metrics_.totalAdaptationTime += elapsed;
  lastAdaptationTime_ = getCurrentTimeMs();
  adaptationCount_++;
  
  return success();
}

LogicalResult AdaptiveBlockManager::migrateBlocks(int64_t oldSize, 
                                                   int64_t newSize) {
  // Migration is complex - for now, just mark for gradual migration
  // New allocations will use new size, existing blocks remain
  return success();
}

//===----------------------------------------------------------------------===//
// AutoTuningBlockManager Implementation
//===----------------------------------------------------------------------===//

AutoTuningBlockManager::AutoTuningBlockManager(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    const BlockSizeConfig& blockConfig,
    const AdaptationConfig& adaptConfig,
    bool enableGPU)
    : AdaptiveBlockManager(numLayers, numHeads, headDim,
                           blockConfig, adaptConfig, enableGPU),
      autoTuneEnabled_(adaptConfig.enableAutoTuning),
      objective_(TuningObjective::BALANCED) {}

AutoTuningBlockManager::~AutoTuningBlockManager() = default;

LogicalResult AutoTuningBlockManager::runTuningExperiment(
    BlockSizeConfig& bestConfig) {
  
  if (!autoTuneEnabled_) {
    bestConfig = getBlockConfig();
    return success();
  }
  
  // Generate candidate configurations
  std::vector<BlockSizeConfig> candidates = generateCandidates();
  
  double bestScore = std::numeric_limits<double>::lowest();
  bestConfig = getBlockConfig();
  
  for (const auto& candidate : candidates) {
    double score = evaluateConfig(candidate);
    
    if (score > bestScore) {
      bestScore = score;
      bestConfig = candidate;
    }
    
    tuningHistory_.push_back({candidate, score});
  }
  
  return success();
}

double AutoTuningBlockManager::evaluateConfig(
    const BlockSizeConfig& config) const {
  
  WorkloadStats stats = getWorkloadStats();
  
  // Calculate expected metrics with this config
  double expectedUtilization = 0.0;
  double expectedLatency = 0.0;
  double expectedMemory = 0.0;
  
  // Utilization: how well blocks would be filled
  if (stats.averageSeqLen > 0) {
    double avgBlocksNeeded = stats.averageSeqLen / config.primaryBlockSize;
    double wastedSlots = std::ceil(avgBlocksNeeded) * config.primaryBlockSize - 
                         stats.averageSeqLen;
    expectedUtilization = 1.0 - wastedSlots / 
                          (std::ceil(avgBlocksNeeded) * config.primaryBlockSize);
  }
  
  // Latency: smaller blocks = more overhead
  expectedLatency = 1.0 + 0.01 * stats.averageSeqLen / config.primaryBlockSize;
  
  // Memory: larger blocks = more waste
  expectedMemory = stats.maxSeqLen * stats.maxBatchSize * 
                   config.primaryBlockSize / expectedUtilization;
  
  // Combine based on objective
  double score = 0.0;
  switch (objective_) {
    case TuningObjective::MINIMIZE_MEMORY:
      score = -expectedMemory + 0.3 * expectedUtilization;
      break;
    case TuningObjective::MINIMIZE_LATENCY:
      score = -expectedLatency + 0.2 * expectedUtilization;
      break;
    case TuningObjective::MAXIMIZE_THROUGHPUT:
      score = expectedUtilization - 0.1 * expectedLatency;
      break;
    case TuningObjective::BALANCED:
    default:
      score = expectedUtilization - 0.3 * expectedLatency - 0.0001 * expectedMemory;
      break;
  }
  
  return score;
}

std::vector<BlockSizeConfig> AutoTuningBlockManager::generateCandidates() const {
  std::vector<BlockSizeConfig> candidates;
  
  // Generate range of block sizes
  std::vector<int64_t> primarySizes = {8, 16, 32, 64, 128};
  
  for (int64_t primary : primarySizes) {
    BlockSizeConfig config;
    config.primaryBlockSize = primary;
    config.smallBlockSize = std::max(int64_t(4), primary / 4);
    config.largeBlockSize = std::min(int64_t(256), primary * 4);
    
    config.smallBlockThreshold = primary * 2;
    config.largeBlockThreshold = primary * 16;
    
    config.primaryBlockPoolSize = 512;
    config.smallBlockPoolSize = 256;
    config.largeBlockPoolSize = 64;
    
    candidates.push_back(config);
  }
  
  return candidates;
}

} // namespace runtime
} // namespace llm
} // namespace mlir
