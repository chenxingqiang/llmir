//===- PrefixCache.cpp - Prefix Caching Support ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements prefix caching support for efficient reuse of common
// prompt prefixes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/PrefixCache.h"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

int64_t getCurrentTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// PrefixHash Implementation
//===----------------------------------------------------------------------===//

size_t PrefixHash::operator()(const std::vector<int32_t>& tokens) const {
  return computeHash(tokens.data(), tokens.size());
}

size_t PrefixHash::computeHash(const int32_t* tokens, int64_t length) {
  // FNV-1a hash
  size_t hash = 14695981039346656037ULL;
  for (int64_t i = 0; i < length; i++) {
    hash ^= static_cast<size_t>(tokens[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

size_t PrefixHash::extendHash(size_t prevHash, int32_t token) {
  size_t hash = prevHash;
  hash ^= static_cast<size_t>(token);
  hash *= 1099511628211ULL;
  return hash;
}

//===----------------------------------------------------------------------===//
// RadixTreeNode Implementation
//===----------------------------------------------------------------------===//

RadixTreeNode::RadixTreeNode() : cachedPrefix_(nullptr) {}

RadixTreeNode::~RadixTreeNode() = default;

void RadixTreeNode::insert(const int32_t* tokens, int64_t length, 
                           CachedPrefix* prefix) {
  if (length == 0) {
    cachedPrefix_ = prefix;
    return;
  }
  
  int32_t firstToken = tokens[0];
  auto it = children_.find(firstToken);
  
  if (it == children_.end()) {
    // Create new child
    auto child = std::make_unique<RadixTreeNode>();
    child->edgeLabel_.assign(tokens, tokens + length);
    child->cachedPrefix_ = prefix;
    children_[firstToken] = std::move(child);
  } else {
    auto& child = it->second;
    
    // Find common prefix length
    int64_t commonLen = 0;
    int64_t edgeLen = child->edgeLabel_.size();
    while (commonLen < length && commonLen < edgeLen &&
           tokens[commonLen] == child->edgeLabel_[commonLen]) {
      commonLen++;
    }
    
    if (commonLen == edgeLen) {
      // Edge is a prefix of new key, recurse
      child->insert(tokens + commonLen, length - commonLen, prefix);
    } else {
      // Need to split the edge
      child->splitEdge(firstToken, commonLen);
      
      // Now insert the remaining part
      if (commonLen < length) {
        auto newChild = std::make_unique<RadixTreeNode>();
        newChild->edgeLabel_.assign(tokens + commonLen, tokens + length);
        newChild->cachedPrefix_ = prefix;
        child->children_[tokens[commonLen]] = std::move(newChild);
      } else {
        child->cachedPrefix_ = prefix;
      }
    }
  }
}

CachedPrefix* RadixTreeNode::findLongestMatch(const int32_t* tokens, 
                                               int64_t length,
                                               int64_t& matchLength) const {
  CachedPrefix* bestMatch = cachedPrefix_;
  matchLength = 0;
  
  if (length == 0) {
    return bestMatch;
  }
  
  int32_t firstToken = tokens[0];
  auto it = children_.find(firstToken);
  
  if (it != children_.end()) {
    const auto& child = it->second;
    
    // Check edge match
    int64_t edgeLen = child->edgeLabel_.size();
    int64_t i = 0;
    while (i < length && i < edgeLen && tokens[i] == child->edgeLabel_[i]) {
      i++;
    }
    
    if (i == edgeLen) {
      // Full edge match, recurse
      int64_t childMatchLen;
      CachedPrefix* childMatch = child->findLongestMatch(
          tokens + edgeLen, length - edgeLen, childMatchLen);
      
      if (childMatch) {
        bestMatch = childMatch;
        matchLength = edgeLen + childMatchLen;
      } else if (child->cachedPrefix_) {
        bestMatch = child->cachedPrefix_;
        matchLength = edgeLen;
      }
    } else if (i > 0 && child->cachedPrefix_) {
      // Partial edge match but has cached prefix at this point
      // This shouldn't happen in a proper radix tree
    }
  }
  
  return bestMatch;
}

bool RadixTreeNode::remove(const int32_t* tokens, int64_t length) {
  if (length == 0) {
    if (cachedPrefix_) {
      cachedPrefix_ = nullptr;
      return true;
    }
    return false;
  }
  
  int32_t firstToken = tokens[0];
  auto it = children_.find(firstToken);
  
  if (it == children_.end()) {
    return false;
  }
  
  auto& child = it->second;
  int64_t edgeLen = child->edgeLabel_.size();
  
  // Check if tokens match the edge
  if (length >= edgeLen) {
    bool matches = true;
    for (int64_t i = 0; i < edgeLen; i++) {
      if (tokens[i] != child->edgeLabel_[i]) {
        matches = false;
        break;
      }
    }
    
    if (matches) {
      if (child->remove(tokens + edgeLen, length - edgeLen)) {
        // If child is now empty, remove it
        if (!child->cachedPrefix_ && child->children_.empty()) {
          children_.erase(it);
        }
        return true;
      }
    }
  }
  
  return false;
}

void RadixTreeNode::collectPrefixes(std::vector<CachedPrefix*>& prefixes) const {
  if (cachedPrefix_) {
    prefixes.push_back(cachedPrefix_);
  }
  
  for (const auto& [_, child] : children_) {
    child->collectPrefixes(prefixes);
  }
}

RadixTreeNode* RadixTreeNode::findChild(int32_t token) const {
  auto it = children_.find(token);
  return it != children_.end() ? it->second.get() : nullptr;
}

void RadixTreeNode::splitEdge(int32_t token, int64_t position) {
  auto it = children_.find(token);
  if (it == children_.end() || position >= static_cast<int64_t>(it->second->edgeLabel_.size())) {
    return;
  }
  
  auto& child = it->second;
  
  // Create new intermediate node
  auto intermediate = std::make_unique<RadixTreeNode>();
  intermediate->edgeLabel_.assign(child->edgeLabel_.begin(),
                                   child->edgeLabel_.begin() + position);
  
  // Update child's edge label
  std::vector<int32_t> remainingLabel(child->edgeLabel_.begin() + position,
                                       child->edgeLabel_.end());
  child->edgeLabel_ = std::move(remainingLabel);
  
  // Move child under intermediate
  int32_t childFirstToken = child->edgeLabel_[0];
  intermediate->children_[childFirstToken] = std::move(child);
  
  // Replace child with intermediate
  children_[token] = std::move(intermediate);
}

//===----------------------------------------------------------------------===//
// PrefixCache Implementation
//===----------------------------------------------------------------------===//

PrefixCache::PrefixCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                         int64_t blockSize, Type elementType,
                         const PrefixCacheConfig& config)
    : numLayers_(numLayers), numHeads_(numHeads), headDim_(headDim),
      blockSize_(blockSize), elementType_(elementType), config_(config),
      currentMemoryUsage_(0), timestampCounter_(0) {
  
  if (config_.enableRadixTree) {
    radixTree_ = std::make_unique<RadixTreeNode>();
  }
  
  metrics_ = PrefixCacheMetrics{};
}

PrefixCache::~PrefixCache() {
  clear();
}

int64_t PrefixCache::lookupPrefix(const int32_t* tokens, int64_t length,
                                   std::vector<std::vector<int32_t>>& blockIndices) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  metrics_.totalLookups++;
  
  CachedPrefix* prefix = nullptr;
  int64_t matchLength = 0;
  
  if (config_.enableRadixTree && radixTree_) {
    prefix = radixTree_->findLongestMatch(tokens, length, matchLength);
  } else {
    // Hash-based lookup for exact matches
    size_t hash = PrefixHash::computeHash(tokens, length);
    auto it = hashIndex_.find(hash);
    if (it != hashIndex_.end()) {
      prefix = it->second.get();
      matchLength = prefix->length;
    }
  }
  
  if (prefix && matchLength >= config_.minPrefixLength) {
    // Cache hit
    prefix->hitCount++;
    prefix->lastAccessTime = timestampCounter_++;
    updateLRU(prefix);
    
    blockIndices = prefix->blockIndices;
    
    if (matchLength == length) {
      metrics_.cacheHits++;
    } else {
      metrics_.partialHits++;
    }
    
    return matchLength;
  }
  
  metrics_.cacheMisses++;
  return 0;
}

LogicalResult PrefixCache::insertPrefix(
    const int32_t* tokens, int64_t length,
    const std::vector<std::vector<int32_t>>& blockIndices) {
  
  if (length < config_.minPrefixLength || length > config_.maxPrefixLength) {
    return failure();
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Check if we need to evict
  size_t prefixMemory = calculatePrefixMemory(length, blockIndices.size());
  
  while (currentMemoryUsage_ + prefixMemory > config_.maxCacheMemory &&
         !lruList_.empty()) {
    evictSinglePrefix(lruList_.front());
  }
  
  if (static_cast<int64_t>(hashIndex_.size()) >= config_.maxCachedPrefixes) {
    if (!lruList_.empty()) {
      evictSinglePrefix(lruList_.front());
    }
  }
  
  // Create new prefix entry
  auto entry = createPrefixEntry(tokens, length, blockIndices);
  CachedPrefix* prefixPtr = entry.get();
  
  // Add to hash index
  hashIndex_[entry->hash] = std::move(entry);
  
  // Add to radix tree
  if (config_.enableRadixTree && radixTree_) {
    radixTree_->insert(tokens, length, prefixPtr);
  }
  
  // Add to LRU list
  lruList_.push_back(prefixPtr);
  lruIndex_[prefixPtr] = std::prev(lruList_.end());
  
  currentMemoryUsage_ += prefixMemory;
  metrics_.insertions++;
  metrics_.currentPrefixes = hashIndex_.size();
  metrics_.currentMemory = currentMemoryUsage_;
  
  return success();
}

bool PrefixCache::hasPrefix(const int32_t* tokens, int64_t length) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  return hashIndex_.find(hash) != hashIndex_.end();
}

const CachedPrefix* PrefixCache::getPrefix(const int32_t* tokens, 
                                            int64_t length) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  return it != hashIndex_.end() ? it->second.get() : nullptr;
}

LogicalResult PrefixCache::acquirePrefix(const int32_t* tokens, int64_t length) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  if (it == hashIndex_.end()) {
    return failure();
  }
  
  it->second->refCount++;
  return success();
}

LogicalResult PrefixCache::releasePrefix(const int32_t* tokens, int64_t length) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  if (it == hashIndex_.end()) {
    return failure();
  }
  
  if (it->second->refCount > 0) {
    it->second->refCount--;
  }
  
  return success();
}

LogicalResult PrefixCache::pinPrefix(const int32_t* tokens, int64_t length) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  if (it == hashIndex_.end()) {
    return failure();
  }
  
  it->second->isPinned = true;
  return success();
}

LogicalResult PrefixCache::unpinPrefix(const int32_t* tokens, int64_t length) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  if (it == hashIndex_.end()) {
    return failure();
  }
  
  it->second->isPinned = false;
  return success();
}

LogicalResult PrefixCache::evictLRU(size_t targetFreeMemory) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t freedMemory = 0;
  
  while (freedMemory < targetFreeMemory && !lruList_.empty()) {
    CachedPrefix* prefix = lruList_.front();
    
    // Skip pinned or in-use prefixes
    if (prefix->isPinned || prefix->refCount > 0) {
      lruList_.splice(lruList_.end(), lruList_, lruList_.begin());
      continue;
    }
    
    freedMemory += prefix->memoryUsage;
    evictSinglePrefix(prefix);
  }
  
  return freedMemory >= targetFreeMemory ? success() : failure();
}

LogicalResult PrefixCache::evictPrefix(const int32_t* tokens, int64_t length) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t hash = PrefixHash::computeHash(tokens, length);
  auto it = hashIndex_.find(hash);
  if (it == hashIndex_.end()) {
    return failure();
  }
  
  CachedPrefix* prefix = it->second.get();
  
  if (prefix->isPinned || prefix->refCount > 0) {
    return failure();
  }
  
  evictSinglePrefix(prefix);
  return success();
}

void PrefixCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  hashIndex_.clear();
  lruList_.clear();
  lruIndex_.clear();
  
  if (radixTree_) {
    radixTree_ = std::make_unique<RadixTreeNode>();
  }
  
  currentMemoryUsage_ = 0;
  metrics_.currentPrefixes = 0;
  metrics_.currentMemory = 0;
}

PrefixCache::PrefixCacheMetrics PrefixCache::getMetrics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  PrefixCacheMetrics m = metrics_;
  m.currentPrefixes = hashIndex_.size();
  m.currentMemory = currentMemoryUsage_;
  
  if (m.totalLookups > 0) {
    m.averageHitRatio = static_cast<double>(m.cacheHits + m.partialHits) / 
                        m.totalLookups;
  }
  
  if (!hashIndex_.empty()) {
    double totalLength = 0;
    for (const auto& [_, prefix] : hashIndex_) {
      totalLength += prefix->length;
    }
    m.averagePrefixLength = totalLength / hashIndex_.size();
  }
  
  return m;
}

void PrefixCache::resetMetrics() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_ = PrefixCacheMetrics{};
}

float PrefixCache::getHitRatio() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (metrics_.totalLookups == 0) return 0.0f;
  return static_cast<float>(metrics_.cacheHits + metrics_.partialHits) /
         metrics_.totalLookups;
}

float PrefixCache::getAverageMatchLength() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (hashIndex_.empty()) return 0.0f;
  
  double totalLength = 0;
  for (const auto& [_, prefix] : hashIndex_) {
    totalLength += prefix->length;
  }
  
  return static_cast<float>(totalLength / hashIndex_.size());
}

void PrefixCache::updateLRU(CachedPrefix* prefix) {
  auto it = lruIndex_.find(prefix);
  if (it != lruIndex_.end()) {
    lruList_.erase(it->second);
    lruList_.push_back(prefix);
    it->second = std::prev(lruList_.end());
  }
}

size_t PrefixCache::calculatePrefixMemory(int64_t length, int64_t numBlocks) const {
  // Memory for tokens
  size_t tokenMem = length * sizeof(int32_t);
  
  // Memory for block indices
  size_t blockMem = numBlocks * numLayers_ * sizeof(int32_t);
  
  // Memory for KV data (if stored)
  size_t kvMem = numBlocks * blockSize_ * numHeads_ * headDim_ * sizeof(float) * 2;
  
  return tokenMem + blockMem + kvMem + sizeof(CachedPrefix);
}

void PrefixCache::evictSinglePrefix(CachedPrefix* prefix) {
  // Remove from radix tree
  if (config_.enableRadixTree && radixTree_) {
    radixTree_->remove(prefix->tokens.data(), prefix->length);
  }
  
  // Remove from LRU
  auto lruIt = lruIndex_.find(prefix);
  if (lruIt != lruIndex_.end()) {
    lruList_.erase(lruIt->second);
    lruIndex_.erase(lruIt);
  }
  
  // Update memory
  currentMemoryUsage_ -= prefix->memoryUsage;
  
  // Remove from hash index
  hashIndex_.erase(prefix->hash);
  
  metrics_.evictions++;
}

std::unique_ptr<CachedPrefix> PrefixCache::createPrefixEntry(
    const int32_t* tokens, int64_t length,
    const std::vector<std::vector<int32_t>>& blockIndices) {
  
  auto entry = std::make_unique<CachedPrefix>();
  
  entry->hash = PrefixHash::computeHash(tokens, length);
  entry->tokens.assign(tokens, tokens + length);
  entry->length = length;
  entry->blockIndices = blockIndices;
  entry->numBlocks = blockIndices.empty() ? 0 : blockIndices[0].size();
  
  entry->hitCount = 0;
  entry->lastAccessTime = timestampCounter_++;
  entry->createdTime = getCurrentTimestamp();
  
  entry->refCount = 0;
  entry->isPinned = false;
  
  entry->memoryUsage = calculatePrefixMemory(length, entry->numBlocks);
  
  return entry;
}

//===----------------------------------------------------------------------===//
// PrefixAwareKVCache Implementation
//===----------------------------------------------------------------------===//

PrefixAwareKVCache::PrefixAwareKVCache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen,
    Type elementType, const PrefixCacheConfig& prefixConfig,
    bool enableGPU)
    : mainCache_(std::make_unique<PagedKVCache>(
          numLayers, numHeads, headDim, blockSize, maxSeqLen,
          elementType, enableGPU)),
      prefixCache_(std::make_unique<PrefixCache>(
          numLayers, numHeads, headDim, blockSize,
          elementType, prefixConfig)) {
  
  sharedBlockAllocator_ = std::make_unique<BlockAllocator>(
      blockSize, headDim * numHeads, enableGPU);
}

PrefixAwareKVCache::~PrefixAwareKVCache() = default;

int64_t PrefixAwareKVCache::initializeSequence(
    int32_t sequenceId,
    const int32_t* promptTokens,
    int64_t promptLength) {
  
  // Store tokens for this sequence
  sequenceTokens_[sequenceId].assign(promptTokens, promptTokens + promptLength);
  
  // Try to find matching prefix
  std::vector<std::vector<int32_t>> blockIndices;
  int64_t matchLength = prefixCache_->lookupPrefix(promptTokens, promptLength,
                                                    blockIndices);
  
  if (matchLength > 0) {
    // Found cached prefix
    sequencePrefixLengths_[sequenceId] = matchLength;
    
    // Acquire reference to prefix
    prefixCache_->acquirePrefix(promptTokens, matchLength);
    
    return matchLength;
  }
  
  sequencePrefixLengths_[sequenceId] = 0;
  return 0;
}

LogicalResult PrefixAwareKVCache::appendKV(
    const void* keyData, const void* valueData,
    int32_t batchSize, int32_t seqLen,
    const int32_t* seqIds, int32_t* blockIndices) {
  
  return mainCache_->appendKV(keyData, valueData, batchSize, seqLen,
                               seqIds, blockIndices);
}

LogicalResult PrefixAwareKVCache::lookupKV(
    const int32_t* blockIndices, const int32_t* seqLens,
    int32_t batchSize, void* outputKeys, void* outputValues) {
  
  return mainCache_->lookupKV(blockIndices, seqLens, batchSize,
                               outputKeys, outputValues);
}

LogicalResult PrefixAwareKVCache::completeSequence(int32_t sequenceId, 
                                                    bool cachePrefix) {
  auto tokenIt = sequenceTokens_.find(sequenceId);
  if (tokenIt == sequenceTokens_.end()) {
    return failure();
  }
  
  if (cachePrefix) {
    // Cache the prefix if it's long enough and not already cached
    const auto& tokens = tokenIt->second;
    int64_t prefixLength = sequencePrefixLengths_[sequenceId];
    
    if (prefixLength == 0 && 
        static_cast<int64_t>(tokens.size()) >= prefixCache_->getConfig().minPrefixLength) {
      // Get block indices from main cache
      // (simplified - would need to get actual block indices)
      std::vector<std::vector<int32_t>> blockIndices;
      
      prefixCache_->insertPrefix(tokens.data(), tokens.size(), blockIndices);
    }
  }
  
  // Release any acquired prefix
  auto prefixIt = sequencePrefixLengths_.find(sequenceId);
  if (prefixIt != sequencePrefixLengths_.end() && prefixIt->second > 0) {
    const auto& tokens = tokenIt->second;
    prefixCache_->releasePrefix(tokens.data(), prefixIt->second);
  }
  
  sequenceTokens_.erase(tokenIt);
  sequencePrefixLengths_.erase(sequenceId);
  
  return success();
}

LogicalResult PrefixAwareKVCache::clearSequence(int32_t sequenceId) {
  completeSequence(sequenceId, false);
  return mainCache_->clearSequence(sequenceId);
}

LogicalResult PrefixAwareKVCache::cacheSequencePrefix(int32_t sequenceId, 
                                                       int64_t prefixLength) {
  auto tokenIt = sequenceTokens_.find(sequenceId);
  if (tokenIt == sequenceTokens_.end()) {
    return failure();
  }
  
  const auto& tokens = tokenIt->second;
  if (prefixLength > static_cast<int64_t>(tokens.size())) {
    prefixLength = tokens.size();
  }
  
  // Get block indices for the prefix
  std::vector<std::vector<int32_t>> blockIndices;
  
  return prefixCache_->insertPrefix(tokens.data(), prefixLength, blockIndices);
}

int64_t PrefixAwareKVCache::getCachedPrefixLength(int32_t sequenceId) const {
  auto it = sequencePrefixLengths_.find(sequenceId);
  return it != sequencePrefixLengths_.end() ? it->second : 0;
}

bool PrefixAwareKVCache::isUsingCachedPrefix(int32_t sequenceId) const {
  return getCachedPrefixLength(sequenceId) > 0;
}

//===----------------------------------------------------------------------===//
// SystemPromptCache Implementation
//===----------------------------------------------------------------------===//

SystemPromptCache::SystemPromptCache(PrefixCache& prefixCache)
    : prefixCache_(prefixCache) {}

SystemPromptCache::~SystemPromptCache() = default;

LogicalResult SystemPromptCache::registerSystemPrompt(
    const std::string& promptId,
    const int32_t* tokens,
    int64_t length) {
  
  if (promptTokens_.find(promptId) != promptTokens_.end()) {
    return failure(); // Already registered
  }
  
  promptTokens_[promptId].assign(tokens, tokens + length);
  promptLengths_[promptId] = length;
  
  return success();
}

int64_t SystemPromptCache::getSystemPrompt(
    const std::string& promptId,
    std::vector<std::vector<int32_t>>& blockIndices) {
  
  auto it = promptTokens_.find(promptId);
  if (it == promptTokens_.end()) {
    return 0;
  }
  
  return prefixCache_.lookupPrefix(it->second.data(), it->second.size(),
                                    blockIndices);
}

bool SystemPromptCache::hasSystemPrompt(const std::string& promptId) const {
  return promptTokens_.find(promptId) != promptTokens_.end();
}

LogicalResult SystemPromptCache::precomputeSystemPrompt(
    const std::string& promptId,
    const void* keyData,
    const void* valueData) {
  
  auto it = promptTokens_.find(promptId);
  if (it == promptTokens_.end()) {
    return failure();
  }
  
  // Would compute and cache KV for system prompt
  // Simplified: just pin the prefix if it exists
  const auto& tokens = it->second;
  
  if (prefixCache_.hasPrefix(tokens.data(), tokens.size())) {
    return prefixCache_.pinPrefix(tokens.data(), tokens.size());
  }
  
  // Need to compute and insert
  std::vector<std::vector<int32_t>> blockIndices;
  // Would need to actually compute KV here
  
  return prefixCache_.insertPrefix(tokens.data(), tokens.size(), blockIndices);
}

std::vector<std::string> SystemPromptCache::listSystemPrompts() const {
  std::vector<std::string> prompts;
  prompts.reserve(promptTokens_.size());
  
  for (const auto& [id, _] : promptTokens_) {
    prompts.push_back(id);
  }
  
  return prompts;
}

LogicalResult SystemPromptCache::removeSystemPrompt(const std::string& promptId) {
  auto it = promptTokens_.find(promptId);
  if (it == promptTokens_.end()) {
    return failure();
  }
  
  // Unpin and potentially evict
  prefixCache_.unpinPrefix(it->second.data(), it->second.size());
  
  promptTokens_.erase(it);
  promptLengths_.erase(promptId);
  
  return success();
}

} // namespace runtime
} // namespace llm
} // namespace mlir
