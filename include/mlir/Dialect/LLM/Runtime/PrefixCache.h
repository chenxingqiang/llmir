//===- PrefixCache.h - Prefix Caching Support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines prefix caching support for efficient reuse of common
// prompt prefixes across multiple sequences.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_PREFIXCACHE_H_
#define MLIR_DIALECT_LLM_RUNTIME_PREFIXCACHE_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Prefix Configuration
//===----------------------------------------------------------------------===//

struct PrefixCacheConfig {
  int64_t maxPrefixLength;       // Maximum prefix length to cache
  int64_t minPrefixLength;       // Minimum prefix length to consider caching
  int64_t maxCachedPrefixes;     // Maximum number of cached prefixes
  size_t maxCacheMemory;         // Maximum memory for prefix cache
  bool enableRadixTree;          // Use radix tree for prefix matching
  bool enableAsyncEviction;      // Async eviction of cold prefixes
  float evictionThreshold;       // LRU eviction threshold (0-1)
  
  PrefixCacheConfig()
      : maxPrefixLength(2048), minPrefixLength(16),
        maxCachedPrefixes(1000), maxCacheMemory(4ULL * 1024 * 1024 * 1024),
        enableRadixTree(true), enableAsyncEviction(true),
        evictionThreshold(0.9f) {}
};

//===----------------------------------------------------------------------===//
// Prefix Hash
//===----------------------------------------------------------------------===//

// Hash function for token sequences
struct PrefixHash {
  size_t operator()(const std::vector<int32_t>& tokens) const;
  
  // Incremental hash computation
  static size_t computeHash(const int32_t* tokens, int64_t length);
  static size_t extendHash(size_t prevHash, int32_t token);
};

//===----------------------------------------------------------------------===//
// Cached Prefix Entry
//===----------------------------------------------------------------------===//

struct CachedPrefix {
  // Prefix identification
  size_t hash;
  std::vector<int32_t> tokens;
  int64_t length;
  
  // KV cache state
  std::vector<std::vector<int32_t>> blockIndices;  // Per-layer block indices
  int64_t numBlocks;
  
  // Usage statistics
  int64_t hitCount;
  int64_t lastAccessTime;
  int64_t createdTime;
  
  // Reference counting
  int32_t refCount;
  bool isPinned;  // Don't evict if pinned
  
  // Memory usage
  size_t memoryUsage;
};

//===----------------------------------------------------------------------===//
// Radix Tree Node (for efficient prefix matching)
//===----------------------------------------------------------------------===//

class RadixTreeNode {
public:
  RadixTreeNode();
  ~RadixTreeNode();
  
  // Insert a prefix
  void insert(const int32_t* tokens, int64_t length, CachedPrefix* prefix);
  
  // Find longest matching prefix
  CachedPrefix* findLongestMatch(const int32_t* tokens, int64_t length,
                                  int64_t& matchLength) const;
  
  // Remove a prefix
  bool remove(const int32_t* tokens, int64_t length);
  
  // Get all prefixes in subtree
  void collectPrefixes(std::vector<CachedPrefix*>& prefixes) const;
  
private:
  // Edge labels (token sequences)
  std::vector<int32_t> edgeLabel_;
  
  // Children indexed by first token of edge
  std::unordered_map<int32_t, std::unique_ptr<RadixTreeNode>> children_;
  
  // Cached prefix at this node (if any)
  CachedPrefix* cachedPrefix_;
  
  // Find child with matching edge start
  RadixTreeNode* findChild(int32_t token) const;
  
  // Split edge at given position
  void splitEdge(int32_t token, int64_t position);
};

//===----------------------------------------------------------------------===//
// Prefix Cache
//===----------------------------------------------------------------------===//

class PrefixCache {
public:
  PrefixCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
              int64_t blockSize, Type elementType,
              const PrefixCacheConfig& config);
  ~PrefixCache();
  
  // Basic getters
  int64_t getNumLayers() const { return numLayers_; }
  int64_t getNumHeads() const { return numHeads_; }
  int64_t getHeadDim() const { return headDim_; }
  int64_t getBlockSize() const { return blockSize_; }
  const PrefixCacheConfig& getConfig() const { return config_; }
  
  //===--------------------------------------------------------------------===//
  // Prefix Lookup and Insertion
  //===--------------------------------------------------------------------===//
  
  // Look up a prefix and return block indices if found
  // Returns the length of the matching prefix
  int64_t lookupPrefix(const int32_t* tokens, int64_t length,
                       std::vector<std::vector<int32_t>>& blockIndices);
  
  // Insert a new prefix into the cache
  LogicalResult insertPrefix(const int32_t* tokens, int64_t length,
                             const std::vector<std::vector<int32_t>>& blockIndices);
  
  // Check if a prefix exists
  bool hasPrefix(const int32_t* tokens, int64_t length) const;
  
  // Get cached prefix entry
  const CachedPrefix* getPrefix(const int32_t* tokens, int64_t length) const;
  
  //===--------------------------------------------------------------------===//
  // Reference Management
  //===--------------------------------------------------------------------===//
  
  // Increment reference count for a prefix
  LogicalResult acquirePrefix(const int32_t* tokens, int64_t length);
  
  // Decrement reference count
  LogicalResult releasePrefix(const int32_t* tokens, int64_t length);
  
  // Pin a prefix to prevent eviction
  LogicalResult pinPrefix(const int32_t* tokens, int64_t length);
  
  // Unpin a prefix
  LogicalResult unpinPrefix(const int32_t* tokens, int64_t length);
  
  //===--------------------------------------------------------------------===//
  // Eviction and Memory Management
  //===--------------------------------------------------------------------===//
  
  // Evict least recently used prefixes
  LogicalResult evictLRU(size_t targetFreeMemory);
  
  // Evict a specific prefix
  LogicalResult evictPrefix(const int32_t* tokens, int64_t length);
  
  // Clear all cached prefixes
  void clear();
  
  // Get memory usage
  size_t getMemoryUsage() const { return currentMemoryUsage_; }
  size_t getAvailableMemory() const { return config_.maxCacheMemory - currentMemoryUsage_; }
  
  //===--------------------------------------------------------------------===//
  // Statistics and Metrics
  //===--------------------------------------------------------------------===//
  
  struct PrefixCacheMetrics {
    int64_t totalLookups;
    int64_t cacheHits;
    int64_t cacheMisses;
    int64_t partialHits;
    int64_t insertions;
    int64_t evictions;
    int64_t currentPrefixes;
    size_t currentMemory;
    double averageHitRatio;
    double averagePrefixLength;
  };
  
  PrefixCacheMetrics getMetrics() const;
  void resetMetrics();
  
  // Get cache hit ratio
  float getHitRatio() const;
  
  // Get average matched prefix length
  float getAverageMatchLength() const;
  
private:
  int64_t numLayers_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t blockSize_;
  Type elementType_;
  PrefixCacheConfig config_;
  
  // Hash-based prefix lookup
  std::unordered_map<size_t, std::unique_ptr<CachedPrefix>> hashIndex_;
  
  // Radix tree for prefix matching
  std::unique_ptr<RadixTreeNode> radixTree_;
  
  // LRU tracking
  std::list<CachedPrefix*> lruList_;
  std::unordered_map<CachedPrefix*, std::list<CachedPrefix*>::iterator> lruIndex_;
  
  // Memory tracking
  size_t currentMemoryUsage_;
  
  // Timestamp counter
  int64_t timestampCounter_;
  
  // Metrics
  mutable PrefixCacheMetrics metrics_;
  
  // Thread safety
  mutable std::mutex mutex_;
  
  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//
  
  // Update LRU for accessed prefix
  void updateLRU(CachedPrefix* prefix);
  
  // Calculate memory usage for a prefix
  size_t calculatePrefixMemory(int64_t length, int64_t numBlocks) const;
  
  // Evict single prefix
  void evictSinglePrefix(CachedPrefix* prefix);
  
  // Create cached prefix entry
  std::unique_ptr<CachedPrefix> createPrefixEntry(
      const int32_t* tokens, int64_t length,
      const std::vector<std::vector<int32_t>>& blockIndices);
};

//===----------------------------------------------------------------------===//
// Prefix-Aware KV Cache
//===----------------------------------------------------------------------===//

// KV cache with integrated prefix caching support
class PrefixAwareKVCache {
public:
  PrefixAwareKVCache(int64_t numLayers, int64_t numHeads, int64_t headDim,
                     int64_t blockSize, int64_t maxSeqLen,
                     Type elementType, const PrefixCacheConfig& prefixConfig,
                     bool enableGPU = false);
  ~PrefixAwareKVCache();
  
  // Get underlying caches
  PagedKVCache& getMainCache() { return *mainCache_; }
  PrefixCache& getPrefixCache() { return *prefixCache_; }
  
  //===--------------------------------------------------------------------===//
  // Prefix-Aware Operations
  //===--------------------------------------------------------------------===//
  
  // Initialize a sequence, attempting to reuse cached prefix
  // Returns the number of tokens loaded from prefix cache
  int64_t initializeSequence(int32_t sequenceId,
                              const int32_t* promptTokens,
                              int64_t promptLength);
  
  // Append KV with automatic prefix detection
  LogicalResult appendKV(const void* keyData, const void* valueData,
                        int32_t batchSize, int32_t seqLen,
                        const int32_t* seqIds, int32_t* blockIndices);
  
  // Standard lookup
  LogicalResult lookupKV(const int32_t* blockIndices, const int32_t* seqLens,
                        int32_t batchSize, void* outputKeys, void* outputValues);
  
  // Complete a sequence and potentially cache its prefix
  LogicalResult completeSequence(int32_t sequenceId, bool cachePrefix = true);
  
  // Clear a sequence
  LogicalResult clearSequence(int32_t sequenceId);
  
  //===--------------------------------------------------------------------===//
  // Prefix Management
  //===--------------------------------------------------------------------===//
  
  // Explicitly cache a prefix for a sequence
  LogicalResult cacheSequencePrefix(int32_t sequenceId, int64_t prefixLength);
  
  // Get the cached prefix length for a sequence
  int64_t getCachedPrefixLength(int32_t sequenceId) const;
  
  // Check if sequence is using cached prefix
  bool isUsingCachedPrefix(int32_t sequenceId) const;
  
private:
  std::unique_ptr<PagedKVCache> mainCache_;
  std::unique_ptr<PrefixCache> prefixCache_;
  
  // Track which sequences are using cached prefixes
  std::unordered_map<int32_t, std::vector<int32_t>> sequenceTokens_;
  std::unordered_map<int32_t, int64_t> sequencePrefixLengths_;
  
  // Block allocator for prefix-shared blocks
  std::unique_ptr<BlockAllocator> sharedBlockAllocator_;
};

//===----------------------------------------------------------------------===//
// System Prompt Cache
//===----------------------------------------------------------------------===//

// Specialized cache for system prompts that are frequently reused
class SystemPromptCache {
public:
  SystemPromptCache(PrefixCache& prefixCache);
  ~SystemPromptCache();
  
  // Register a system prompt
  LogicalResult registerSystemPrompt(const std::string& promptId,
                                      const int32_t* tokens,
                                      int64_t length);
  
  // Get cached system prompt
  int64_t getSystemPrompt(const std::string& promptId,
                          std::vector<std::vector<int32_t>>& blockIndices);
  
  // Check if system prompt is cached
  bool hasSystemPrompt(const std::string& promptId) const;
  
  // Precompute KV for a system prompt
  LogicalResult precomputeSystemPrompt(const std::string& promptId,
                                        const void* keyData,
                                        const void* valueData);
  
  // List registered system prompts
  std::vector<std::string> listSystemPrompts() const;
  
  // Remove a system prompt
  LogicalResult removeSystemPrompt(const std::string& promptId);
  
private:
  PrefixCache& prefixCache_;
  
  // System prompt registry
  std::unordered_map<std::string, std::vector<int32_t>> promptTokens_;
  std::unordered_map<std::string, int64_t> promptLengths_;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_PREFIXCACHE_H_
