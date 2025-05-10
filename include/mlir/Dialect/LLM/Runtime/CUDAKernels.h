//===- CUDAKernels.h - CUDA kernels for attention optimizations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares CUDA kernels for optimized attention computations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_CUDAKERNELS_H_
#define MLIR_DIALECT_LLM_RUNTIME_CUDAKERNELS_H_

#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mlir {
namespace llm {
namespace runtime {
namespace cuda {

// Forward declarations of CUDA kernel launchers

// Flash Attention CUDA kernel launcher
// Implements the algorithm from "FlashAttention: Fast and Memory-Efficient Exact Attention"
cudaError_t launchFlashAttentionKernel(
    void* output,             // Output tensor [batchSize, seqLen, numHeads, headDim]
    const void* queries,      // Query tensor [batchSize, seqLen, numHeads, headDim]
    const void* keys,         // Key tensor [batchSize, contextLen, numHeads, headDim]
    const void* values,       // Value tensor [batchSize, contextLen, numHeads, headDim]
    int64_t batchSize,        // Batch size
    int64_t seqLen,           // Sequence length of queries
    int64_t contextLen,       // Context length of keys/values
    int64_t numHeads,         // Number of attention heads
    int64_t headDim,          // Dimension of each head
    float scale,              // Attention scale factor
    AttentionMaskType maskType, // Type of attention mask
    int64_t windowSize,       // Window size (for sliding window attention)
    int cudaBlockSize,        // CUDA block size
    bool useTensorCores,      // Whether to use Tensor Cores if available
    bool useHalfPrecision,    // Whether to use half precision
    cudaStream_t stream = 0); // CUDA stream (0 for default stream)

// Multi-Query Attention CUDA kernel launcher
cudaError_t launchMultiQueryAttentionKernel(
    void* output,             // Output tensor
    const void* queries,      // Query tensor [batchSize, seqLen, numHeads, headDim]
    const void* keys,         // Key tensor [batchSize, contextLen, 1, headDim]
    const void* values,       // Value tensor [batchSize, contextLen, 1, headDim]
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    int64_t numHeads,
    int64_t headDim,
    float scale,
    AttentionMaskType maskType,
    int64_t windowSize,
    int cudaBlockSize,
    bool useTensorCores,
    bool useHalfPrecision,
    cudaStream_t stream = 0);

// Grouped-Query Attention CUDA kernel launcher
cudaError_t launchGroupedQueryAttentionKernel(
    void* output,             // Output tensor
    const void* queries,      // Query tensor [batchSize, seqLen, numHeads, headDim]
    const void* keys,         // Key tensor [batchSize, contextLen, numKVHeads, headDim]
    const void* values,       // Value tensor [batchSize, contextLen, numKVHeads, headDim]
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    int64_t numHeads,         // Number of query heads
    int64_t numKVHeads,       // Number of key/value heads (smaller than numHeads)
    int64_t headDim,
    float scale,
    AttentionMaskType maskType,
    int64_t windowSize,
    int cudaBlockSize,
    bool useTensorCores,
    bool useHalfPrecision,
    cudaStream_t stream = 0);

// Pruned Attention CUDA kernel launcher
cudaError_t launchPrunedAttentionKernel(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    const void* pruningMask,  // Boolean mask for pruning [batchSize, seqLen, contextLen]
    float pruningThreshold,   // Threshold for dynamic pruning
    int64_t batchSize,
    int64_t seqLen,
    int64_t contextLen,
    int64_t numHeads,
    int64_t headDim,
    float scale,
    AttentionMaskType maskType,
    int64_t windowSize,
    int cudaBlockSize,
    bool useTensorCores,
    bool useHalfPrecision,
    cudaStream_t stream = 0);

// CUDA utility functions
bool isCUDAAvailable();
int getNumSMs();
bool tensorCoresAvailable();
cudaStream_t getCurrentStream();

} // namespace cuda
} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_CUDAKERNELS_H_
