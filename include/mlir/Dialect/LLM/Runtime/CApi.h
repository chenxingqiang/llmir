//===- CApi.h - LLMIR C API for Python Bindings -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the C API for LLMIR runtime components, enabling Python
// bindings via ctypes or cffi.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_CAPI_H
#define MLIR_DIALECT_LLM_RUNTIME_CAPI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Data Types
//===----------------------------------------------------------------------===//

/// Data type enumeration for KV cache
typedef enum {
  LLMIR_DTYPE_FLOAT16 = 0,
  LLMIR_DTYPE_FLOAT32 = 1,
  LLMIR_DTYPE_BFLOAT16 = 2,
  LLMIR_DTYPE_INT8 = 3,
  LLMIR_DTYPE_INT4 = 4,
  LLMIR_DTYPE_FP8 = 5,
} LlmirDtype;

/// Quantization type enumeration
typedef enum {
  LLMIR_QUANT_NONE = 0,
  LLMIR_QUANT_INT8 = 1,
  LLMIR_QUANT_INT4 = 2,
  LLMIR_QUANT_FP8 = 3,
} LlmirQuantType;

/// Quantization strategy
typedef enum {
  LLMIR_QUANT_STRATEGY_PER_TENSOR = 0,
  LLMIR_QUANT_STRATEGY_PER_CHANNEL = 1,
  LLMIR_QUANT_STRATEGY_PER_GROUP = 2,
} LlmirQuantStrategy;

/// Sharding strategy for distributed cache
typedef enum {
  LLMIR_SHARD_LAYER_WISE = 0,
  LLMIR_SHARD_HEAD_WISE = 1,
  LLMIR_SHARD_SEQUENCE_WISE = 2,
  LLMIR_SHARD_HYBRID = 3,
} LlmirShardingStrategy;

/// Scheduling policy for continuous batching
typedef enum {
  LLMIR_SCHED_FCFS = 0,
  LLMIR_SCHED_SHORTEST_FIRST = 1,
  LLMIR_SCHED_PRIORITY_BASED = 2,
  LLMIR_SCHED_FAIR_SHARE = 3,
  LLMIR_SCHED_ADAPTIVE = 4,
} LlmirSchedulingPolicy;

/// Error codes
typedef enum {
  LLMIR_SUCCESS = 0,
  LLMIR_ERROR_INVALID_ARGUMENT = 1,
  LLMIR_ERROR_OUT_OF_MEMORY = 2,
  LLMIR_ERROR_CUDA_ERROR = 3,
  LLMIR_ERROR_NOT_FOUND = 4,
  LLMIR_ERROR_INTERNAL = 5,
} LlmirError;

//===----------------------------------------------------------------------===//
// Opaque Handle Types
//===----------------------------------------------------------------------===//

typedef void* LlmirKVCacheHandle;
typedef void* LlmirQuantizedKVCacheHandle;
typedef void* LlmirDistributedKVCacheHandle;
typedef void* LlmirSpeculativeKVCacheHandle;
typedef void* LlmirPrefixCacheHandle;
typedef void* LlmirEngineHandle;
typedef void* LlmirProfilerHandle;

//===----------------------------------------------------------------------===//
// PagedKVCache API
//===----------------------------------------------------------------------===//

/// Create a new PagedKVCache
/// @param numLayers Number of transformer layers
/// @param numHeads Number of attention heads (KV heads for GQA)
/// @param headDim Dimension of each attention head
/// @param blockSize Number of tokens per cache block
/// @param maxSeqLen Maximum sequence length
/// @param dtype Data type for cache storage
/// @param enableGPU Whether to use GPU memory
/// @return Handle to the created cache, or NULL on failure
LlmirKVCacheHandle llmir_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t dtype,
    bool enableGPU);

/// Destroy a PagedKVCache
void llmir_kvcache_destroy(LlmirKVCacheHandle handle);

/// Append key-value pairs to the cache
/// @param handle KV cache handle
/// @param keyData Pointer to key data
/// @param valueData Pointer to value data
/// @param batchSize Number of sequences in batch
/// @param seqLen Sequence length
/// @param seqIds Sequence ID for each batch item
/// @param blockIndices Output block indices (preallocated)
/// @return 0 on success, error code on failure
int32_t llmir_kvcache_append(
    LlmirKVCacheHandle handle,
    const void* keyData,
    const void* valueData,
    int32_t batchSize,
    int32_t seqLen,
    const int32_t* seqIds,
    int32_t* blockIndices);

/// Lookup key-value pairs from the cache
/// @param handle KV cache handle
/// @param blockIndices Block indices to lookup
/// @param seqLens Sequence lengths
/// @param batchSize Number of sequences
/// @param outputKeys Output key buffer (preallocated)
/// @param outputValues Output value buffer (preallocated)
/// @return 0 on success, error code on failure
int32_t llmir_kvcache_lookup(
    LlmirKVCacheHandle handle,
    const int32_t* blockIndices,
    const int32_t* seqLens,
    int32_t batchSize,
    void* outputKeys,
    void* outputValues);

/// Clear cache for a specific sequence
int32_t llmir_kvcache_clear_sequence(LlmirKVCacheHandle handle, int32_t seqId);

/// Reset the entire cache
void llmir_kvcache_reset(LlmirKVCacheHandle handle);

/// Get current memory usage in bytes
int64_t llmir_kvcache_get_memory_usage(LlmirKVCacheHandle handle);

/// Get number of active sequences
int32_t llmir_kvcache_get_num_sequences(LlmirKVCacheHandle handle);

/// Get block size
int32_t llmir_kvcache_get_block_size(LlmirKVCacheHandle handle);

/// Get number of layers
int32_t llmir_kvcache_get_num_layers(LlmirKVCacheHandle handle);

//===----------------------------------------------------------------------===//
// QuantizedKVCache API
//===----------------------------------------------------------------------===//

/// Create a new QuantizedKVCache
LlmirQuantizedKVCacheHandle llmir_quantized_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t quantType,
    int32_t strategy,
    bool symmetric,
    int64_t groupSize,
    bool dynamicRange);

/// Destroy a QuantizedKVCache
void llmir_quantized_kvcache_destroy(LlmirQuantizedKVCacheHandle handle);

/// Get compression ratio
float llmir_quantized_kvcache_get_compression_ratio(
    LlmirQuantizedKVCacheHandle handle);

/// Get estimated accuracy loss
float llmir_quantized_kvcache_get_accuracy_loss(
    LlmirQuantizedKVCacheHandle handle);

//===----------------------------------------------------------------------===//
// DistributedKVCache API
//===----------------------------------------------------------------------===//

/// Create a distributed KV cache
LlmirDistributedKVCacheHandle llmir_distributed_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t numDevices,
    const int32_t* deviceIds,
    int32_t strategy,
    bool enableNCCL);

/// Destroy a distributed KV cache
void llmir_distributed_kvcache_destroy(LlmirDistributedKVCacheHandle handle);

/// Get memory usage per device
void llmir_distributed_kvcache_get_per_device_memory(
    LlmirDistributedKVCacheHandle handle,
    int64_t* memoryPerDevice);

/// Rebalance load across devices
void llmir_distributed_kvcache_rebalance(LlmirDistributedKVCacheHandle handle);

//===----------------------------------------------------------------------===//
// SpeculativeKVCache API
//===----------------------------------------------------------------------===//

/// Create a speculative KV cache
LlmirSpeculativeKVCacheHandle llmir_speculative_kvcache_create(
    int64_t numLayers,
    int64_t numHeads,
    int64_t headDim,
    int64_t blockSize,
    int64_t maxSeqLen,
    int32_t maxDraftTokens,
    int32_t maxBranches,
    bool enableTreeAttention);

/// Destroy a speculative KV cache
void llmir_speculative_kvcache_destroy(LlmirSpeculativeKVCacheHandle handle);

/// Create a speculation branch
int32_t llmir_speculative_kvcache_create_branch(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId);

/// Commit accepted tokens
void llmir_speculative_kvcache_commit(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId,
    int32_t branchId,
    int32_t numAccepted);

/// Rollback speculative tokens
void llmir_speculative_kvcache_rollback(
    LlmirSpeculativeKVCacheHandle handle,
    int32_t seqId,
    int32_t branchId);

//===----------------------------------------------------------------------===//
// PrefixCache API
//===----------------------------------------------------------------------===//

/// Create a prefix cache
LlmirPrefixCacheHandle llmir_prefix_cache_create(
    int32_t maxPrefixes,
    int64_t maxMemoryBytes,
    bool enableRadixTree,
    int32_t minPrefixLength);

/// Destroy a prefix cache
void llmir_prefix_cache_destroy(LlmirPrefixCacheHandle handle);

/// Cache a prefix
bool llmir_prefix_cache_insert(
    LlmirPrefixCacheHandle handle,
    const int32_t* tokens,
    int32_t numTokens,
    const int32_t* blockIndices,
    int32_t numBlocks);

/// Lookup a prefix
/// @return Length of matched prefix (0 if no match)
int32_t llmir_prefix_cache_lookup(
    LlmirPrefixCacheHandle handle,
    const int32_t* tokens,
    int32_t numTokens,
    int32_t* blockIndices,
    int32_t maxBlocks);

/// Get cache hit ratio
float llmir_prefix_cache_get_hit_ratio(LlmirPrefixCacheHandle handle);

/// Clear the prefix cache
void llmir_prefix_cache_clear(LlmirPrefixCacheHandle handle);

//===----------------------------------------------------------------------===//
// ContinuousBatchingEngine API
//===----------------------------------------------------------------------===//

/// Create a continuous batching engine
LlmirEngineHandle llmir_engine_create(
    LlmirKVCacheHandle cacheHandle,
    int32_t maxBatchSize,
    int32_t maxNumSeqs,
    int32_t chunkSize);

/// Destroy an engine
void llmir_engine_destroy(LlmirEngineHandle handle);

/// Start the engine
void llmir_engine_start(LlmirEngineHandle handle);

/// Stop the engine
void llmir_engine_stop(LlmirEngineHandle handle);

/// Check if engine is running
bool llmir_engine_is_running(LlmirEngineHandle handle);

/// Submit a generation request
/// @return Request ID (>= 0) or error code (< 0)
int32_t llmir_engine_submit(
    LlmirEngineHandle handle,
    const int32_t* promptTokens,
    int32_t numTokens,
    float temperature,
    float topP,
    int32_t topK,
    int32_t maxTokens);

/// Abort a request
bool llmir_engine_abort(LlmirEngineHandle handle, int32_t requestId);

/// Run one iteration step
/// @return Number of outputs available
int32_t llmir_engine_step(LlmirEngineHandle handle);

/// Get output tokens for a request
/// @return Number of tokens copied
int32_t llmir_engine_get_output(
    LlmirEngineHandle handle,
    int32_t requestId,
    int32_t* outputTokens,
    int32_t maxTokens,
    bool* finished);

/// Get engine statistics
void llmir_engine_get_stats(
    LlmirEngineHandle handle,
    int32_t* totalRequests,
    int32_t* completedRequests,
    int32_t* pendingRequests,
    int64_t* totalTokens);

//===----------------------------------------------------------------------===//
// Profiler API
//===----------------------------------------------------------------------===//

/// Create a profiler
LlmirProfilerHandle llmir_profiler_create(void);

/// Destroy a profiler
void llmir_profiler_destroy(LlmirProfilerHandle handle);

/// Start profiling
void llmir_profiler_start(LlmirProfilerHandle handle);

/// Stop profiling
void llmir_profiler_stop(LlmirProfilerHandle handle);

/// Record an event
void llmir_profiler_record_event(
    LlmirProfilerHandle handle,
    const char* name,
    int32_t eventType,
    float durationMs);

/// Export trace to file (Chrome trace format)
bool llmir_profiler_export_trace(
    LlmirProfilerHandle handle,
    const char* filepath);

/// Get total number of events
int32_t llmir_profiler_get_num_events(LlmirProfilerHandle handle);

//===----------------------------------------------------------------------===//
// Version and Build Info
//===----------------------------------------------------------------------===//

/// Get LLMIR version string
const char* llmir_get_version(void);

/// Check if CUDA support is enabled
bool llmir_has_cuda_support(void);

/// Check if NCCL support is enabled
bool llmir_has_nccl_support(void);

/// Check if Metal support is enabled
bool llmir_has_metal_support(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECT_LLM_RUNTIME_CAPI_H
