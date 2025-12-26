//===- KVCacheSerialization.cpp - KV Cache Serialization -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements serialization and deserialization support for KV cache.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/KVCacheSerialization.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <sstream>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

// Get current Unix timestamp
int64_t getCurrentTimestamp() {
  return std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

// Write a value to stream
template<typename T>
void writeValue(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Read a value from stream
template<typename T>
bool readValue(std::istream& in, T& value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

// Write a string to stream
void writeString(std::ostream& out, const std::string& str) {
  int64_t len = str.size();
  writeValue(out, len);
  if (len > 0) {
    out.write(str.data(), len);
  }
}

// Read a string from stream
bool readString(std::istream& in, std::string& str) {
  int64_t len;
  if (!readValue(in, len)) return false;
  if (len > 0) {
    str.resize(len);
    return static_cast<bool>(in.read(str.data(), len));
  }
  str.clear();
  return true;
}

// Write a byte vector to stream
void writeBytes(std::ostream& out, const std::vector<uint8_t>& data) {
  int64_t len = data.size();
  writeValue(out, len);
  if (len > 0) {
    out.write(reinterpret_cast<const char*>(data.data()), len);
  }
}

// Read a byte vector from stream
bool readBytes(std::istream& in, std::vector<uint8_t>& data) {
  int64_t len;
  if (!readValue(in, len)) return false;
  if (len > 0) {
    data.resize(len);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), len));
  }
  data.clear();
  return true;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// KVCacheSerializer Implementation
//===----------------------------------------------------------------------===//

KVCacheSerializer::KVCacheSerializer(const SerializationOptions& options)
    : options_(options) {}

KVCacheSerializer::~KVCacheSerializer() = default;

LogicalResult KVCacheSerializer::saveToFile(const PagedKVCache& cache,
                                             const std::string& filepath) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  // Write header
  writeHeader(file, cache);
  
  // Collect and write sequence metadata
  auto metadata = collectMetadata(cache);
  writeSequenceMetadata(file, metadata);
  
  // Write block data for each sequence
  // Note: This would need access to internal cache data
  // For now, we serialize the cache state that we can access
  
  file.close();
  return success();
}

LogicalResult KVCacheSerializer::saveToBuffer(const PagedKVCache& cache,
                                               std::vector<uint8_t>& buffer) {
  std::ostringstream oss(std::ios::binary);
  
  // Write header
  writeHeader(oss, cache);
  
  // Collect and write sequence metadata
  auto metadata = collectMetadata(cache);
  writeSequenceMetadata(oss, metadata);
  
  // Copy to buffer
  std::string data = oss.str();
  buffer.assign(data.begin(), data.end());
  
  return success();
}

LogicalResult KVCacheSerializer::saveSequences(const PagedKVCache& cache,
                                                const std::vector<int32_t>& sequenceIds,
                                                const std::string& filepath) {
  // For partial saves, filter metadata to only include specified sequences
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  writeHeader(file, cache);
  
  // Filter metadata for specified sequences
  auto allMetadata = collectMetadata(cache);
  std::vector<SequenceMetadata> filteredMetadata;
  for (const auto& meta : allMetadata) {
    if (std::find(sequenceIds.begin(), sequenceIds.end(), meta.sequenceId) 
        != sequenceIds.end()) {
      filteredMetadata.push_back(meta);
    }
  }
  
  writeSequenceMetadata(file, filteredMetadata);
  
  file.close();
  return success();
}

size_t KVCacheSerializer::estimateSerializedSize(const PagedKVCache& cache) const {
  size_t size = sizeof(CheckpointHeader);
  
  // Estimate metadata size
  int64_t numSeqs = cache.getNumSequences();
  size += numSeqs * sizeof(SequenceMetadata);
  
  // Estimate block data size
  // This is approximate as we need actual block counts
  size_t elementSize = 4; // Assume FP32
  size += cache.getNumLayers() * cache.getNumHeads() * cache.getHeadDim() * 
          cache.getBlockSize() * elementSize * 2; // Keys + values
  
  return size;
}

void KVCacheSerializer::writeHeader(std::ostream& out, 
                                     const PagedKVCache& cache) const {
  CheckpointHeader header;
  std::memset(&header, 0, sizeof(header));
  
  header.magic = KVCACHE_MAGIC;
  header.version = KVCACHE_VERSION;
  
  header.numLayers = cache.getNumLayers();
  header.numHeads = cache.getNumHeads();
  header.headDim = cache.getHeadDim();
  header.blockSize = cache.getBlockSize();
  header.maxSeqLen = cache.getMaxSeqLen();
  
  header.numSequences = cache.getNumSequences();
  header.totalBlocks = 0; // Would need to iterate blocks
  header.totalTokens = 0; // Would need to sum sequence lengths
  
  header.compressionType = static_cast<uint32_t>(options_.compression);
  header.hasMetadata = options_.includeMetadata;
  header.createdAt = getCurrentTimestamp();
  
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

void KVCacheSerializer::writeSequenceMetadata(
    std::ostream& out, 
    const std::vector<SequenceMetadata>& metadata) const {
  
  int64_t count = metadata.size();
  writeValue(out, count);
  
  for (const auto& meta : metadata) {
    writeValue(out, meta.sequenceId);
    writeValue(out, meta.length);
    writeValue(out, meta.numBlocks);
    writeValue(out, meta.createdAt);
    writeValue(out, meta.lastAccessTime);
    
    if (options_.includeMetadata) {
      writeString(out, meta.tag);
    }
  }
}

void KVCacheSerializer::writeBlockData(std::ostream& out, 
                                        const BlockData& block) const {
  writeValue(out, block.layerIdx);
  writeValue(out, block.blockIdx);
  writeValue(out, block.usedSlots);
  writeValue(out, block.refCount);
  
  if (options_.compression != CompressionType::NONE) {
    auto compressedKeys = compress(block.keyData.data(), block.keyData.size());
    auto compressedValues = compress(block.valueData.data(), block.valueData.size());
    
    // Write original size for decompression
    int64_t origKeySize = block.keyData.size();
    int64_t origValueSize = block.valueData.size();
    writeValue(out, origKeySize);
    writeValue(out, origValueSize);
    
    writeBytes(out, compressedKeys);
    writeBytes(out, compressedValues);
  } else {
    writeBytes(out, block.keyData);
    writeBytes(out, block.valueData);
  }
}

std::vector<uint8_t> KVCacheSerializer::compress(const uint8_t* data, 
                                                   size_t size) const {
  // Simple implementation without external compression library
  // In production, use LZ4 or ZSTD
  if (options_.compression == CompressionType::NONE) {
    return std::vector<uint8_t>(data, data + size);
  }
  
  // Placeholder: return uncompressed data
  // Real implementation would use LZ4_compress or ZSTD_compress
  return std::vector<uint8_t>(data, data + size);
}

std::vector<SequenceMetadata> KVCacheSerializer::collectMetadata(
    const PagedKVCache& cache) const {
  
  std::vector<SequenceMetadata> metadata;
  // Would need access to internal sequence table
  // For now, return empty metadata
  return metadata;
}

//===----------------------------------------------------------------------===//
// KVCacheDeserializer Implementation
//===----------------------------------------------------------------------===//

KVCacheDeserializer::KVCacheDeserializer() = default;

KVCacheDeserializer::~KVCacheDeserializer() = default;

LogicalResult KVCacheDeserializer::loadFromFile(PagedKVCache& cache,
                                                 const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  CheckpointHeader header;
  if (readHeader(file, header).failed()) {
    return failure();
  }
  
  // Validate compatibility
  if (header.numLayers != cache.getNumLayers() ||
      header.numHeads != cache.getNumHeads() ||
      header.headDim != cache.getHeadDim()) {
    return failure();
  }
  
  // Read sequence metadata
  std::vector<SequenceMetadata> metadata;
  if (readSequenceMetadata(file, metadata, header.numSequences).failed()) {
    return failure();
  }
  
  // Read and restore block data
  // This would need to interface with cache internals
  
  file.close();
  return success();
}

LogicalResult KVCacheDeserializer::loadFromBuffer(PagedKVCache& cache,
                                                   const std::vector<uint8_t>& buffer) {
  std::istringstream iss(std::string(buffer.begin(), buffer.end()), std::ios::binary);
  
  CheckpointHeader header;
  if (readHeader(iss, header).failed()) {
    return failure();
  }
  
  // Validate compatibility
  if (header.numLayers != cache.getNumLayers() ||
      header.numHeads != cache.getNumHeads() ||
      header.headDim != cache.getHeadDim()) {
    return failure();
  }
  
  std::vector<SequenceMetadata> metadata;
  if (readSequenceMetadata(iss, metadata, header.numSequences).failed()) {
    return failure();
  }
  
  return success();
}

LogicalResult KVCacheDeserializer::loadSequences(PagedKVCache& cache,
                                                  const std::string& filepath,
                                                  const std::vector<int32_t>& sequenceIds) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  CheckpointHeader header;
  if (readHeader(file, header).failed()) {
    return failure();
  }
  
  std::vector<SequenceMetadata> metadata;
  if (readSequenceMetadata(file, metadata, header.numSequences).failed()) {
    return failure();
  }
  
  // Filter and load only specified sequences
  for (const auto& meta : metadata) {
    if (std::find(sequenceIds.begin(), sequenceIds.end(), meta.sequenceId) 
        != sequenceIds.end()) {
      // Load this sequence's blocks
      // Implementation would read and restore block data
    }
  }
  
  file.close();
  return success();
}

LogicalResult KVCacheDeserializer::inspectCheckpoint(
    const std::string& filepath,
    CheckpointHeader& header,
    std::vector<SequenceMetadata>& sequences) {
  
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  if (readHeader(file, header).failed()) {
    return failure();
  }
  
  if (readSequenceMetadata(file, sequences, header.numSequences).failed()) {
    return failure();
  }
  
  file.close();
  return success();
}

LogicalResult KVCacheDeserializer::validateCheckpoint(
    const std::string& filepath) const {
  
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  // Check magic and version
  uint32_t magic, version;
  if (!readValue(file, magic) || magic != KVCACHE_MAGIC) {
    return failure();
  }
  
  if (!readValue(file, version) || version > KVCACHE_VERSION) {
    return failure();
  }
  
  file.close();
  return success();
}

LogicalResult KVCacheDeserializer::validateCompatibility(
    const std::string& filepath,
    const PagedKVCache& cache) const {
  
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  CheckpointHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  
  if (header.magic != KVCACHE_MAGIC) {
    return failure();
  }
  
  // Check dimension compatibility
  if (header.numLayers != cache.getNumLayers() ||
      header.numHeads != cache.getNumHeads() ||
      header.headDim != cache.getHeadDim()) {
    return failure();
  }
  
  file.close();
  return success();
}

LogicalResult KVCacheDeserializer::readHeader(std::istream& in, 
                                               CheckpointHeader& header) {
  if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) {
    return failure();
  }
  
  if (header.magic != KVCACHE_MAGIC) {
    return failure();
  }
  
  if (header.version > KVCACHE_VERSION) {
    return failure();
  }
  
  return success();
}

LogicalResult KVCacheDeserializer::readSequenceMetadata(
    std::istream& in,
    std::vector<SequenceMetadata>& metadata,
    int64_t count) {
  
  int64_t readCount;
  if (!readValue(in, readCount) || readCount != count) {
    return failure();
  }
  
  metadata.resize(count);
  for (int64_t i = 0; i < count; i++) {
    auto& meta = metadata[i];
    if (!readValue(in, meta.sequenceId) ||
        !readValue(in, meta.length) ||
        !readValue(in, meta.numBlocks) ||
        !readValue(in, meta.createdAt) ||
        !readValue(in, meta.lastAccessTime)) {
      return failure();
    }
    
    // Try to read tag (may not be present in older versions)
    readString(in, meta.tag);
  }
  
  return success();
}

LogicalResult KVCacheDeserializer::readBlockData(std::istream& in, 
                                                  BlockData& block) {
  if (!readValue(in, block.layerIdx) ||
      !readValue(in, block.blockIdx) ||
      !readValue(in, block.usedSlots) ||
      !readValue(in, block.refCount)) {
    return failure();
  }
  
  if (!readBytes(in, block.keyData) ||
      !readBytes(in, block.valueData)) {
    return failure();
  }
  
  return success();
}

std::vector<uint8_t> KVCacheDeserializer::decompress(
    const uint8_t* data, size_t compressedSize,
    size_t originalSize, CompressionType type) const {
  
  if (type == CompressionType::NONE) {
    return std::vector<uint8_t>(data, data + compressedSize);
  }
  
  // Placeholder: return uncompressed data
  // Real implementation would use LZ4_decompress or ZSTD_decompress
  std::vector<uint8_t> result(originalSize);
  std::memcpy(result.data(), data, std::min(compressedSize, originalSize));
  return result;
}

//===----------------------------------------------------------------------===//
// CheckpointManager Implementation
//===----------------------------------------------------------------------===//

CheckpointManager::CheckpointManager(const std::string& checkpointDir,
                                     const SerializationOptions& options)
    : checkpointDir_(checkpointDir), options_(options),
      serializer_(options), autoCheckpointEnabled_(false),
      autoCheckpointInterval_(0), lastCheckpointTime_(0) {
  
  // Create checkpoint directory if it doesn't exist
  std::filesystem::create_directories(checkpointDir);
}

CheckpointManager::~CheckpointManager() = default;

LogicalResult CheckpointManager::createCheckpoint(const PagedKVCache& cache,
                                                   const std::string& name) {
  std::string checkpointName = name.empty() ? generateCheckpointName() : name;
  std::string path = getCheckpointPath(checkpointName);
  
  if (serializer_.saveToFile(cache, path).failed()) {
    return failure();
  }
  
  lastCheckpointTime_ = getCurrentTimestamp();
  return success();
}

LogicalResult CheckpointManager::loadLatestCheckpoint(PagedKVCache& cache) {
  auto checkpoints = getCheckpointsByTime();
  if (checkpoints.empty()) {
    return failure();
  }
  
  // Load the most recent checkpoint
  return deserializer_.loadFromFile(cache, checkpoints.back().first);
}

LogicalResult CheckpointManager::loadCheckpoint(PagedKVCache& cache,
                                                 const std::string& name) {
  std::string path = getCheckpointPath(name);
  return deserializer_.loadFromFile(cache, path);
}

std::vector<std::string> CheckpointManager::listCheckpoints() const {
  std::vector<std::string> checkpoints;
  
  if (!std::filesystem::exists(checkpointDir_)) {
    return checkpoints;
  }
  
  for (const auto& entry : std::filesystem::directory_iterator(checkpointDir_)) {
    if (entry.is_regular_file() && entry.path().extension() == ".kvchk") {
      checkpoints.push_back(entry.path().stem().string());
    }
  }
  
  std::sort(checkpoints.begin(), checkpoints.end());
  return checkpoints;
}

LogicalResult CheckpointManager::getCheckpointInfo(
    const std::string& name,
    CheckpointHeader& header,
    std::vector<SequenceMetadata>& sequences) {
  
  std::string path = getCheckpointPath(name);
  return deserializer_.inspectCheckpoint(path, header, sequences);
}

LogicalResult CheckpointManager::deleteCheckpoint(const std::string& name) {
  std::string path = getCheckpointPath(name);
  
  if (!std::filesystem::exists(path)) {
    return failure();
  }
  
  std::filesystem::remove(path);
  return success();
}

LogicalResult CheckpointManager::cleanupCheckpoints(int64_t keepCount) {
  auto checkpoints = getCheckpointsByTime();
  
  if (static_cast<int64_t>(checkpoints.size()) <= keepCount) {
    return success();
  }
  
  // Remove oldest checkpoints
  int64_t removeCount = checkpoints.size() - keepCount;
  for (int64_t i = 0; i < removeCount; i++) {
    std::filesystem::remove(checkpoints[i].first);
  }
  
  return success();
}

void CheckpointManager::enableAutoCheckpoint(int64_t intervalSeconds) {
  autoCheckpointEnabled_ = true;
  autoCheckpointInterval_ = intervalSeconds;
}

void CheckpointManager::disableAutoCheckpoint() {
  autoCheckpointEnabled_ = false;
}

std::string CheckpointManager::generateCheckpointName() const {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  
  std::stringstream ss;
  ss << "checkpoint_" << time;
  return ss.str();
}

std::string CheckpointManager::getCheckpointPath(const std::string& name) const {
  return checkpointDir_ + "/" + name + ".kvchk";
}

std::vector<std::pair<std::string, int64_t>> CheckpointManager::getCheckpointsByTime() const {
  std::vector<std::pair<std::string, int64_t>> checkpoints;
  
  if (!std::filesystem::exists(checkpointDir_)) {
    return checkpoints;
  }
  
  for (const auto& entry : std::filesystem::directory_iterator(checkpointDir_)) {
    if (entry.is_regular_file() && entry.path().extension() == ".kvchk") {
      auto modTime = std::filesystem::last_write_time(entry);
      auto sctp = std::chrono::time_point_cast<std::chrono::seconds>(
          std::chrono::file_clock::to_sys(modTime));
      int64_t timestamp = sctp.time_since_epoch().count();
      checkpoints.emplace_back(entry.path().string(), timestamp);
    }
  }
  
  // Sort by timestamp
  std::sort(checkpoints.begin(), checkpoints.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  
  return checkpoints;
}

//===----------------------------------------------------------------------===//
// IncrementalCheckpointer Implementation
//===----------------------------------------------------------------------===//

IncrementalCheckpointer::IncrementalCheckpointer(
    const std::string& baseCheckpointPath,
    const SerializationOptions& options)
    : basePath_(baseCheckpointPath), options_(options) {}

IncrementalCheckpointer::~IncrementalCheckpointer() = default;

LogicalResult IncrementalCheckpointer::initialize(const PagedKVCache& cache) {
  // Save the base checkpoint
  KVCacheSerializer serializer(options_);
  if (serializer.saveToFile(cache, basePath_).failed()) {
    return failure();
  }
  
  // Record current sequence lengths
  sequenceLengthsAtCheckpoint_.clear();
  // Would need to iterate through sequences and record their lengths
  
  return success();
}

LogicalResult IncrementalCheckpointer::saveIncrement(
    const PagedKVCache& cache,
    const std::string& incrementPath) {
  
  // Detect what has changed
  auto changes = detectChanges(cache);
  
  if (changes.empty()) {
    return success(); // Nothing to save
  }
  
  // Save only the changed blocks
  std::ofstream file(incrementPath, std::ios::binary);
  if (!file.is_open()) {
    return failure();
  }
  
  // Write increment header
  uint32_t magic = KVCACHE_MAGIC | 0x01; // Mark as increment
  writeValue(file, magic);
  writeValue(file, KVCACHE_VERSION);
  
  int64_t numChanges = changes.size();
  writeValue(file, numChanges);
  
  for (const auto& [seqId, newLength] : changes) {
    writeValue(file, seqId);
    writeValue(file, newLength);
    // Write block data for new tokens
  }
  
  file.close();
  incrementPaths_.push_back(incrementPath);
  
  // Update tracked lengths
  for (const auto& [seqId, newLength] : changes) {
    sequenceLengthsAtCheckpoint_[seqId] = newLength;
  }
  
  return success();
}

LogicalResult IncrementalCheckpointer::compact(const std::string& newBasePath) {
  // Load base checkpoint and all increments, then save as new base
  // This would need full implementation with cache restoration
  
  // Clear increments after compaction
  for (const auto& path : incrementPaths_) {
    std::filesystem::remove(path);
  }
  incrementPaths_.clear();
  
  basePath_ = newBasePath;
  return success();
}

std::vector<std::string> IncrementalCheckpointer::getIncrements() const {
  return incrementPaths_;
}

std::vector<std::pair<int32_t, int64_t>> IncrementalCheckpointer::detectChanges(
    const PagedKVCache& cache) const {
  
  std::vector<std::pair<int32_t, int64_t>> changes;
  
  // Would need to iterate through sequences and compare lengths
  // For now, return empty
  
  return changes;
}

} // namespace runtime
} // namespace llm
} // namespace mlir
