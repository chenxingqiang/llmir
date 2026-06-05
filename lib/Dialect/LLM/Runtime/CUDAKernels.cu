//===- CUDAKernels.cu - CUDA kernels for attention optimizations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLM/Runtime/CUDAKernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace mlir {
namespace llm {
namespace runtime {
namespace cuda {

namespace {

__device__ float maskScore(float score, int maskType, int queryIdx, int keyIdx,
                          int windowSize, float customMask) {
  if (maskType == static_cast<int>(AttentionMaskType::CAUSAL) && keyIdx > queryIdx)
    return -CUDART_INF_F;
  if (maskType == static_cast<int>(AttentionMaskType::SLIDING_WINDOW) &&
      windowSize > 0) {
    int diff = keyIdx - queryIdx;
    if (diff < 0)
      diff = -diff;
    if (diff > windowSize)
      return -CUDART_INF_F;
  }
  if (maskType == static_cast<int>(AttentionMaskType::CUSTOM) && customMask == 0.0f)
    return -CUDART_INF_F;
  return score;
}

// Multi-query attention: Q [B,S,H,D], K/V [B,C,D] shared across heads.
__global__ void mqaAttentionKernel(
    float* output, const float* queries, const float* keys, const float* values,
    int batchSize, int seqLen, int contextLen, int numHeads, int headDim,
    float scale, int maskType, int windowSize, const float* customMask) {
  const int h = blockIdx.x;
  const int q = blockIdx.y;
  const int b = blockIdx.z;
  if (b >= batchSize || q >= seqLen || h >= numHeads)
    return;

  extern __shared__ float smem[];
  float* scores = smem;

  const float* queryVec =
      queries + ((b * seqLen + q) * numHeads + h) * headDim;
  float* outputVec = output + ((b * seqLen + q) * numHeads + h) * headDim;

  float maxVal = -CUDART_INF_F;
  for (int keyIdx = 0; keyIdx < contextLen; ++keyIdx) {
    const float* keyVec = keys + (b * contextLen + keyIdx) * headDim;
    float dot = 0.0f;
    for (int d = 0; d < headDim; ++d)
      dot += queryVec[d] * keyVec[d];
    dot *= scale;
    float custom = 1.0f;
    if (customMask != nullptr)
      custom = customMask[b * seqLen * contextLen + q * contextLen + keyIdx];
    scores[keyIdx] = maskScore(dot, maskType, q, keyIdx, windowSize, custom);
    maxVal = fmaxf(maxVal, scores[keyIdx]);
  }

  float expSum = 0.0f;
  for (int keyIdx = 0; keyIdx < contextLen; ++keyIdx) {
    if (scores[keyIdx] > -CUDART_INF_F / 2.0f) {
      scores[keyIdx] = expf(scores[keyIdx] - maxVal);
      expSum += scores[keyIdx];
    } else {
      scores[keyIdx] = 0.0f;
    }
  }

  for (int d = 0; d < headDim; ++d)
    outputVec[d] = 0.0f;

  if (expSum <= 0.0f)
    return;

  for (int keyIdx = 0; keyIdx < contextLen; ++keyIdx) {
    if (scores[keyIdx] == 0.0f)
      continue;
    const float weight = scores[keyIdx] / expSum;
    const float* valueVec = values + (b * contextLen + keyIdx) * headDim;
    for (int d = 0; d < headDim; ++d)
      outputVec[d] += weight * valueVec[d];
  }
}

cudaError_t launchMqaKernel(
    void* output, const void* queries, const void* keys, const void* values,
    int64_t batchSize, int64_t seqLen, int64_t contextLen, int64_t numHeads,
    int64_t headDim, float scale, AttentionMaskType maskType, int64_t windowSize,
    const void* attentionMask, cudaStream_t stream) {
  if (contextLen <= 0 || headDim <= 0 || numHeads <= 0)
    return cudaSuccess;

  const size_t smemBytes =
      static_cast<size_t>(contextLen) * sizeof(float);
  const dim3 grid(static_cast<unsigned>(numHeads),
                  static_cast<unsigned>(seqLen),
                  static_cast<unsigned>(batchSize));

  mqaAttentionKernel<<<grid, 1, smemBytes, stream>>>(
      static_cast<float*>(output), static_cast<const float*>(queries),
      static_cast<const float*>(keys), static_cast<const float*>(values),
      static_cast<int>(batchSize), static_cast<int>(seqLen),
      static_cast<int>(contextLen), static_cast<int>(numHeads),
      static_cast<int>(headDim), scale, static_cast<int>(maskType),
      static_cast<int>(windowSize),
      static_cast<const float*>(attentionMask));

  return cudaGetLastError();
}

} // namespace

cudaError_t launchFlashAttentionKernel(
    void* output, const void* queries, const void* keys, const void* values,
    int64_t batchSize, int64_t seqLen, int64_t contextLen, int64_t numHeads,
    int64_t headDim, float scale, AttentionMaskType maskType, int64_t windowSize,
    int, bool, bool, cudaStream_t stream) {
  // MVP: route standard MHA through MQA path when num_heads match layout.
  return launchMqaKernel(output, queries, keys, values, batchSize, seqLen,
                         contextLen, numHeads, headDim, scale, maskType,
                         windowSize, nullptr, stream);
}

cudaError_t launchMultiQueryAttentionKernel(
    void* output, const void* queries, const void* keys, const void* values,
    int64_t batchSize, int64_t seqLen, int64_t contextLen, int64_t numHeads,
    int64_t headDim, float scale, AttentionMaskType maskType, int64_t windowSize,
    int, bool, bool, cudaStream_t stream) {
  return launchMqaKernel(output, queries, keys, values, batchSize, seqLen,
                         contextLen, numHeads, headDim, scale, maskType,
                         windowSize, nullptr, stream);
}

cudaError_t launchGroupedQueryAttentionKernel(
    void* output, const void* queries, const void* keys, const void* values,
    int64_t batchSize, int64_t seqLen, int64_t contextLen, int64_t numHeads,
    int64_t, int64_t headDim, float scale, AttentionMaskType maskType,
    int64_t windowSize, int, bool, bool, cudaStream_t stream) {
  return launchMqaKernel(output, queries, keys, values, batchSize, seqLen,
                         contextLen, numHeads, headDim, scale, maskType,
                         windowSize, nullptr, stream);
}

cudaError_t launchPrunedAttentionKernel(
    void* output, const void* queries, const void* keys, const void* values,
    const void* pruningMask, float, int64_t batchSize, int64_t seqLen,
    int64_t contextLen, int64_t numHeads, int64_t headDim, float scale,
    AttentionMaskType maskType, int64_t windowSize, int, bool, bool,
    cudaStream_t stream) {
  return launchMqaKernel(output, queries, keys, values, batchSize, seqLen,
                         contextLen, numHeads, headDim, scale,
                         AttentionMaskType::CUSTOM, windowSize, pruningMask,
                         stream);
}

bool isCUDAAvailable() {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  return (error == cudaSuccess && deviceCount > 0);
}

int getNumSMs() {
  int deviceId = 0;
  if (cudaGetDevice(&deviceId) != cudaSuccess)
    return 0;
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess)
    return 0;
  return props.multiProcessorCount;
}

bool tensorCoresAvailable() {
  int deviceId = 0;
  if (cudaGetDevice(&deviceId) != cudaSuccess)
    return false;
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess)
    return false;
  return props.major >= 7;
}

cudaStream_t getCurrentStream() {
  return nullptr;
}

} // namespace cuda
} // namespace runtime
} // namespace llm
} // namespace mlir
