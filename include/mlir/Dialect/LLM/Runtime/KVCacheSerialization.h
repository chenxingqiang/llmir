//===- KVCacheSerialization.h - KV Cache Serialization --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines serialization and deserialization support for KV cache,
// enabling checkpointing and resumption of long-running LLM sessions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_KVCACHESERIALIZATION_H_
#define MLIR_DIALECT_LLM_RUNTIME_KVCACHESERIALIZATION_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Serialization Format Constants
//===----------------------------------------------------------------------===//

// Magic number to identify KV cache checkpoint files
constexpr uint32_t KVCACHE_MAGIC = 0x4B564348; // "KVCH"

// Version number for backward compatibility
constexpr uint32_t KVCACHE_VERSION = 1;

//===----------------------------------------------------------------------===//
// Serialization Options
//===----------------------------------------------------------------------===//

enum class CompressionType {
  NONE,      // No compression
  LZ4,       // LZ4 fast compression
  ZSTD       // Zstandard compression (better ratio)
};

struct SerializationOptions {
  CompressionType compression;
  bool includeMetadata;
  bool partialSave;          // Allow saving subset of sequences
  int64_t compressionLevel;  // Compression level (1-9 for ZSTD)
  
  SerializationOptions()
      : compression(CompressionType::NONE),
        includeMetadata(true),
        partialSave(false),
        compressionLevel(3) {}
};

//===----------------------------------------------------------------------===//
// Checkpoint Header
//===----------------------------------------------------------------------===//

struct CheckpointHeader {
  uint32_t magic;
  uint32_t version;
  
  // Cache configuration
  int64_t numLayers;
  int64_t numHeads;
  int64_t headDim;
  int64_t blockSize;
  int64_t maxSeqLen;
  
  // Content information
  int64_t numSequences;
  int64_t totalBlocks;
  int64_t totalTokens;
  
  // Serialization options used
  uint32_t compressionType;
  bool hasMetadata;
  
  // Timestamps
  int64_t createdAt;
  
  // Reserved for future use
  uint8_t reserved[32];
};

//===----------------------------------------------------------------------===//
// Sequence Metadata
//===----------------------------------------------------------------------===//

struct SequenceMetadata {
  int32_t sequenceId;
  int64_t length;
  int64_t numBlocks;
  int64_t createdAt;
  int64_t lastAccessTime;
  
  // Optional: user-defined tags
  std::string tag;
};

//===----------------------------------------------------------------------===//
// Block Data
//===----------------------------------------------------------------------===//

struct BlockData {
  int64_t layerIdx;
  int64_t blockIdx;
  int64_t usedSlots;
  int64_t refCount;
  
  // Serialized key and value data
  std::vector<uint8_t> keyData;
  std::vector<uint8_t> valueData;
};

//===----------------------------------------------------------------------===//
// KV Cache Serializer
//===----------------------------------------------------------------------===//

class KVCacheSerializer {
public:
  explicit KVCacheSerializer(const SerializationOptions& options = SerializationOptions());
  ~KVCacheSerializer();
  
  // Serialize to file
  LogicalResult saveToFile(const PagedKVCache& cache, 
                           const std::string& filepath);
  
  // Serialize to memory buffer
  LogicalResult saveToBuffer(const PagedKVCache& cache,
                             std::vector<uint8_t>& buffer);
  
  // Serialize specific sequences only
  LogicalResult saveSequences(const PagedKVCache& cache,
                              const std::vector<int32_t>& sequenceIds,
                              const std::string& filepath);
  
  // Get serialization info without saving
  size_t estimateSerializedSize(const PagedKVCache& cache) const;
  
private:
  SerializationOptions options_;
  
  // Header serialization
  void writeHeader(std::ostream& out, const PagedKVCache& cache) const;
  
  // Sequence metadata serialization
  void writeSequenceMetadata(std::ostream& out, 
                             const std::vector<SequenceMetadata>& metadata) const;
  
  // Block data serialization
  void writeBlockData(std::ostream& out, const BlockData& block) const;
  
  // Compression helpers
  std::vector<uint8_t> compress(const uint8_t* data, size_t size) const;
  
  // Collect sequence metadata from cache
  std::vector<SequenceMetadata> collectMetadata(const PagedKVCache& cache) const;
};

//===----------------------------------------------------------------------===//
// KV Cache Deserializer
//===----------------------------------------------------------------------===//

class KVCacheDeserializer {
public:
  explicit KVCacheDeserializer();
  ~KVCacheDeserializer();
  
  // Deserialize from file
  LogicalResult loadFromFile(PagedKVCache& cache,
                             const std::string& filepath);
  
  // Deserialize from memory buffer
  LogicalResult loadFromBuffer(PagedKVCache& cache,
                               const std::vector<uint8_t>& buffer);
  
  // Load specific sequences only
  LogicalResult loadSequences(PagedKVCache& cache,
                              const std::string& filepath,
                              const std::vector<int32_t>& sequenceIds);
  
  // Inspect checkpoint without loading
  LogicalResult inspectCheckpoint(const std::string& filepath,
                                  CheckpointHeader& header,
                                  std::vector<SequenceMetadata>& sequences);
  
  // Validation
  LogicalResult validateCheckpoint(const std::string& filepath) const;
  LogicalResult validateCompatibility(const std::string& filepath,
                                      const PagedKVCache& cache) const;
  
private:
  // Header deserialization
  LogicalResult readHeader(std::istream& in, CheckpointHeader& header);
  
  // Sequence metadata deserialization
  LogicalResult readSequenceMetadata(std::istream& in,
                                     std::vector<SequenceMetadata>& metadata,
                                     int64_t count);
  
  // Block data deserialization
  LogicalResult readBlockData(std::istream& in, BlockData& block);
  
  // Decompression helpers
  std::vector<uint8_t> decompress(const uint8_t* data, size_t compressedSize,
                                  size_t originalSize, CompressionType type) const;
};

//===----------------------------------------------------------------------===//
// Checkpoint Manager
//===----------------------------------------------------------------------===//

// High-level interface for managing KV cache checkpoints
class CheckpointManager {
public:
  CheckpointManager(const std::string& checkpointDir,
                    const SerializationOptions& options = SerializationOptions());
  ~CheckpointManager();
  
  // Create a new checkpoint
  LogicalResult createCheckpoint(const PagedKVCache& cache,
                                 const std::string& name = "");
  
  // Load the latest checkpoint
  LogicalResult loadLatestCheckpoint(PagedKVCache& cache);
  
  // Load a specific checkpoint by name
  LogicalResult loadCheckpoint(PagedKVCache& cache,
                               const std::string& name);
  
  // List available checkpoints
  std::vector<std::string> listCheckpoints() const;
  
  // Get checkpoint info
  LogicalResult getCheckpointInfo(const std::string& name,
                                  CheckpointHeader& header,
                                  std::vector<SequenceMetadata>& sequences);
  
  // Delete a checkpoint
  LogicalResult deleteCheckpoint(const std::string& name);
  
  // Cleanup old checkpoints (keep N most recent)
  LogicalResult cleanupCheckpoints(int64_t keepCount);
  
  // Auto-checkpoint support
  void enableAutoCheckpoint(int64_t intervalSeconds);
  void disableAutoCheckpoint();
  
private:
  std::string checkpointDir_;
  SerializationOptions options_;
  KVCacheSerializer serializer_;
  KVCacheDeserializer deserializer_;
  
  // Auto-checkpoint state
  bool autoCheckpointEnabled_;
  int64_t autoCheckpointInterval_;
  int64_t lastCheckpointTime_;
  
  // Helper methods
  std::string generateCheckpointName() const;
  std::string getCheckpointPath(const std::string& name) const;
  std::vector<std::pair<std::string, int64_t>> getCheckpointsByTime() const;
};

//===----------------------------------------------------------------------===//
// Incremental Checkpointing
//===----------------------------------------------------------------------===//

// Support for incremental checkpoints that only save changes
class IncrementalCheckpointer {
public:
  IncrementalCheckpointer(const std::string& baseCheckpointPath,
                          const SerializationOptions& options = SerializationOptions());
  ~IncrementalCheckpointer();
  
  // Initialize with base checkpoint
  LogicalResult initialize(const PagedKVCache& cache);
  
  // Save incremental changes since last checkpoint
  LogicalResult saveIncrement(const PagedKVCache& cache,
                              const std::string& incrementPath);
  
  // Compact increments into a new base checkpoint
  LogicalResult compact(const std::string& newBasePath);
  
  // Get list of increment files
  std::vector<std::string> getIncrements() const;
  
private:
  std::string basePath_;
  SerializationOptions options_;
  std::vector<std::string> incrementPaths_;
  
  // Track changes since last checkpoint
  std::unordered_map<int32_t, int64_t> sequenceLengthsAtCheckpoint_;
  
  // Detect changed blocks
  std::vector<std::pair<int32_t, int64_t>> detectChanges(const PagedKVCache& cache) const;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHESERIALIZATION_H_
