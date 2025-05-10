//===- KVCacheRuntime.h - Runtime API for KV cache operations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the C API for the KV cache runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_KVCACHERUNTIME_H_
#define MLIR_DIALECT_LLM_RUNTIME_KVCACHERUNTIME_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace llm {
namespace runtime {

// Forward declarations of runtime classes
class PagedKVCache;

extern "C" {

//===----------------------------------------------------------------------===//
// Runtime API for KV cache operations
//===----------------------------------------------------------------------===//

/// Create a new paged KV cache
/// 
/// @param[in] numLayers Number of transformer layers
/// @param[in] numHeads Number of attention heads per layer
/// @param[in] headDim Dimension of each attention head
/// @param[in] blockSize Size of each KV cache block
/// @param[in] maxSeqLen Maximum sequence length
/// @param[in] useGPU Whether to use GPU memory
/// @return A pointer to the created PagedKVCache object
PagedKVCache* mlir_llm_create_paged_kv_cache(
    int64_t numLayers, int64_t numHeads, int64_t headDim,
    int64_t blockSize, int64_t maxSeqLen, bool useGPU);

/// Free a paged KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object to free
void mlir_llm_free_paged_kv_cache(PagedKVCache* cache);

/// Append key-value pairs to a KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @param[in] keyPtr Pointer to the key tensor data
/// @param[in] valuePtr Pointer to the value tensor data
/// @param[in] seqIdsPtr Pointer to the sequence IDs
/// @param[in] batchSize Batch size
/// @param[in] seqLen Sequence length
/// @param[out] blockIndicesPtr Pointer to the output block indices
/// @return Success if the operation succeeds, failure otherwise
LogicalResult mlir_llm_append_kv(
    PagedKVCache* cache, void* keyPtr, void* valuePtr, int32_t* seqIdsPtr,
    int64_t batchSize, int64_t seqLen, int32_t* blockIndicesPtr);

/// Look up key-value pairs from a KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @param[in] blockIndicesPtr Pointer to the block indices
/// @param[in] seqLensPtr Pointer to the sequence lengths
/// @param[in] batchSize Batch size
/// @param[in] maxSeqLen Maximum sequence length
/// @param[out] outputKeysPtr Pointer to the output keys tensor
/// @param[out] outputValuesPtr Pointer to the output values tensor
/// @return Success if the operation succeeds, failure otherwise
LogicalResult mlir_llm_lookup_kv(
    PagedKVCache* cache, int32_t* blockIndicesPtr, int32_t* seqLensPtr,
    int64_t batchSize, int64_t maxSeqLen, void* outputKeysPtr, void* outputValuesPtr);

/// Perform attention computation with a paged KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @param[in] queryPtr Pointer to the query tensor data
/// @param[in] blockIndicesPtr Pointer to the block indices
/// @param[in] seqLensPtr Pointer to the sequence lengths
/// @param[in] batchSize Batch size
/// @param[in] seqLen Sequence length
/// @param[in] numHeads Number of attention heads
/// @param[in] headDim Dimension of each attention head
/// @param[in] scale Attention scale factor
/// @param[out] outputPtr Pointer to the output tensor
/// @return Success if the operation succeeds, failure otherwise
LogicalResult mlir_llm_paged_attention(
    PagedKVCache* cache, void* queryPtr, int32_t* blockIndicesPtr, int32_t* seqLensPtr,
    int64_t batchSize, int64_t seqLen, int64_t numHeads, int64_t headDim,
    float scale, void* outputPtr);

/// Clear a sequence from the KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @param[in] seqId Sequence ID to clear
/// @return Success if the operation succeeds, failure otherwise
LogicalResult mlir_llm_clear_sequence(PagedKVCache* cache, int32_t seqId);

/// Reset the KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
void mlir_llm_reset_kv_cache(PagedKVCache* cache);

/// Get the total memory usage of the KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @return The total memory usage in bytes
int64_t mlir_llm_get_total_memory_usage(PagedKVCache* cache);

/// Get the number of sequences in the KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @return The number of sequences
int64_t mlir_llm_get_num_sequences(PagedKVCache* cache);

/// Get the length of a sequence in the KV cache
///
/// @param[in] cache Pointer to the PagedKVCache object
/// @param[in] seqId Sequence ID
/// @return The length of the sequence, or 0 if not found
int64_t mlir_llm_get_sequence_length(PagedKVCache* cache, int32_t seqId);

} // extern "C"

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_KVCACHERUNTIME_H_ 