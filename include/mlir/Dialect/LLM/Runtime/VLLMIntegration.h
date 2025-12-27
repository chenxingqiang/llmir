//===- VLLMIntegration.h - vLLM Integration Layer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the integration layer for vLLM compatibility,
// providing adapters and wrappers to use LLMIR KV cache with vLLM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_VLLMINTEGRATION_H_
#define MLIR_DIALECT_LLM_RUNTIME_VLLMINTEGRATION_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Dialect/LLM/Runtime/ContinuousBatching.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {
namespace vllm {

//===----------------------------------------------------------------------===//
// vLLM-Compatible Types
//===----------------------------------------------------------------------===//

// Mirrors vLLM's SamplingParams
struct SamplingParams {
  int64_t n;                    // Number of sequences to return
  int64_t bestOf;               // Number of sequences to generate
  float presencePenalty;
  float frequencyPenalty;
  float repetitionPenalty;
  float temperature;
  float topP;
  int64_t topK;
  int64_t minP;
  bool useBeamSearch;
  float lengthPenalty;
  int64_t earlyStopping;
  std::vector<int32_t> stopTokenIds;
  std::vector<std::string> stop;
  bool ignoreEos;
  int64_t maxTokens;
  int64_t minTokens;
  std::optional<int64_t> logprobs;
  std::optional<int64_t> promptLogprobs;
  bool skipSpecialTokens;
  bool spacesBetweeSpecialTokens;
  
  SamplingParams()
      : n(1), bestOf(1), presencePenalty(0.0f), frequencyPenalty(0.0f),
        repetitionPenalty(1.0f), temperature(1.0f), topP(1.0f),
        topK(-1), minP(0), useBeamSearch(false), lengthPenalty(1.0f),
        earlyStopping(false), ignoreEos(false), maxTokens(16),
        minTokens(0), skipSpecialTokens(true), spacesBetweeSpecialTokens(true) {}
};

// Mirrors vLLM's CompletionOutput
struct CompletionOutput {
  int32_t index;
  std::string text;
  std::vector<int32_t> tokenIds;
  float cumulativeLogprob;
  std::vector<float> logprobs;
  std::string finishReason;
};

// Mirrors vLLM's RequestOutput
struct RequestOutput {
  std::string requestId;
  std::string prompt;
  std::vector<int32_t> promptTokenIds;
  std::vector<CompletionOutput> outputs;
  bool finished;
  RequestMetrics metrics;
  
  struct RequestMetrics {
    double arrivalTime;
    double firstTokenTime;
    double finishedTime;
    int64_t promptTokens;
    int64_t generatedTokens;
  };
};

//===----------------------------------------------------------------------===//
// Block Space Manager Adapter
//===----------------------------------------------------------------------===//

// Adapts LLMIR PagedKVCache to vLLM's block space manager interface
class BlockSpaceManagerAdapter {
public:
  BlockSpaceManagerAdapter(PagedKVCache& cache, int64_t numGpuBlocks,
                           int64_t numCpuBlocks = 0);
  ~BlockSpaceManagerAdapter();
  
  // vLLM-compatible interface
  bool canAllocate(int64_t numBlocks) const;
  std::vector<int32_t> allocate(int32_t seqId, int64_t numBlocks);
  void free(int32_t seqId);
  void forkSeq(int32_t parentSeqId, int32_t childSeqId);
  
  // Swap operations
  bool canSwapIn(int64_t numBlocks) const;
  bool canSwapOut(int64_t numBlocks) const;
  std::vector<std::pair<int32_t, int32_t>> swapIn(int32_t seqId);
  std::vector<std::pair<int32_t, int32_t>> swapOut(int32_t seqId);
  
  // Block tables
  std::vector<int32_t> getBlockTable(int32_t seqId) const;
  int64_t getNumFreeGpuBlocks() const;
  int64_t getNumFreeCpuBlocks() const;
  
  // Watermark for preemption
  float getWatermark() const { return watermark_; }
  void setWatermark(float watermark) { watermark_ = watermark; }
  
private:
  PagedKVCache& cache_;
  int64_t numGpuBlocks_;
  int64_t numCpuBlocks_;
  float watermark_;
  
  std::unordered_map<int32_t, std::vector<int32_t>> gpuBlockTables_;
  std::unordered_map<int32_t, std::vector<int32_t>> cpuBlockTables_;
  
  std::vector<int32_t> freeGpuBlocks_;
  std::vector<int32_t> freeCpuBlocks_;
  
  std::mutex mutex_;
};

//===----------------------------------------------------------------------===//
// Scheduler Adapter
//===----------------------------------------------------------------------===//

// Adapts LLMIR scheduler to vLLM's scheduler interface
class SchedulerAdapter {
public:
  SchedulerAdapter(Scheduler& scheduler, BlockSpaceManagerAdapter& blockManager);
  ~SchedulerAdapter();
  
  // vLLM-compatible scheduling output
  struct SchedulerOutputs {
    std::vector<int32_t> scheduledSeqIds;
    std::vector<int32_t> preemptedSeqIds;
    std::vector<int32_t> swappedInSeqIds;
    std::vector<int32_t> swappedOutSeqIds;
    int64_t numPrefillTokens;
    int64_t numDecodeTokens;
    int64_t numBatchedTokens;
    std::vector<std::vector<int32_t>> blockTables;
  };
  
  SchedulerOutputs schedule();
  
  void addSeq(int32_t seqId, int64_t promptLen);
  void abortSeq(int32_t seqId);
  void freeSeq(int32_t seqId);
  
  bool hasPendingRequests() const;
  int64_t getNumUnfinishedSeqs() const;
  
private:
  Scheduler& scheduler_;
  BlockSpaceManagerAdapter& blockManager_;
};

//===----------------------------------------------------------------------===//
// PagedAttention Wrapper
//===----------------------------------------------------------------------===//

// Provides vLLM-compatible paged attention interface
class PagedAttentionWrapper {
public:
  PagedAttentionWrapper(int64_t numHeads, int64_t headDim, 
                        int64_t numKvHeads, float scale);
  ~PagedAttentionWrapper();
  
  // Forward pass with vLLM-style arguments
  LogicalResult forward(
      // Input tensors
      const void* query,           // [num_seqs, num_heads, head_dim]
      const void* key,             // [num_seqs, num_kv_heads, head_dim]
      const void* value,           // [num_seqs, num_kv_heads, head_dim]
      // KV cache
      void* keyCache,              // [num_blocks, num_kv_heads, block_size, head_dim]
      void* valueCache,            // [num_blocks, num_kv_heads, block_size, head_dim]
      // Block tables
      const int32_t* blockTables,  // [num_seqs, max_blocks_per_seq]
      const int32_t* contextLens,  // [num_seqs]
      // Output
      void* output,                // [num_seqs, num_heads, head_dim]
      // Options
      int32_t blockSize,
      int64_t maxContextLen,
      const void* alibiSlopes = nullptr
  );
  
  // Prefill-specific
  LogicalResult forwardPrefill(
      const void* query,
      const void* key,
      const void* value,
      void* output,
      const int32_t* seqStartLocs,
      const int32_t* seqLens,
      int32_t numSeqs,
      int32_t maxSeqLen
  );
  
  // Decode-specific (single token per sequence)
  LogicalResult forwardDecode(
      const void* query,
      const void* keyCache,
      const void* valueCache,
      void* output,
      const int32_t* blockTables,
      const int32_t* contextLens,
      int32_t numSeqs,
      int32_t blockSize,
      int64_t maxContextLen
  );
  
private:
  int64_t numHeads_;
  int64_t headDim_;
  int64_t numKvHeads_;
  float scale_;
};

//===----------------------------------------------------------------------===//
// LLM Engine Adapter
//===----------------------------------------------------------------------===//

// High-level adapter providing vLLM LLMEngine-like interface
class LLMEngineAdapter {
public:
  struct EngineConfig {
    std::string modelPath;
    std::string tokenizerPath;
    int64_t tensorParallelSize;
    int64_t pipelineParallelSize;
    std::string dtype;
    int64_t maxNumSeqs;
    int64_t maxNumBatchedTokens;
    int64_t maxModelLen;
    int64_t blockSize;
    float gpuMemoryUtilization;
    float swapSpace;
    bool enableChunkedPrefill;
    bool enablePrefixCaching;
    
    EngineConfig()
        : tensorParallelSize(1), pipelineParallelSize(1),
          dtype("float16"), maxNumSeqs(256), maxNumBatchedTokens(8192),
          maxModelLen(4096), blockSize(16), gpuMemoryUtilization(0.9f),
          swapSpace(4.0f), enableChunkedPrefill(true),
          enablePrefixCaching(true) {}
  };
  
  explicit LLMEngineAdapter(const EngineConfig& config);
  ~LLMEngineAdapter();
  
  // Initialize the engine
  LogicalResult initialize();
  
  // Add request (returns request ID)
  std::string addRequest(const std::string& prompt,
                         const SamplingParams& params,
                         const std::string& requestId = "");
  
  // Abort request
  LogicalResult abortRequest(const std::string& requestId);
  
  // Step: run one iteration and get outputs
  std::vector<RequestOutput> step();
  
  // Get request outputs without stepping
  std::vector<RequestOutput> getOutputs();
  
  // Check if there are pending requests
  bool hasPendingRequests() const;
  
  // Statistics
  struct EngineMetrics {
    int64_t numRequests;
    int64_t numTokensGenerated;
    double avgPromptThroughput;
    double avgGenerationThroughput;
    double avgE2ELatency;
    double avgTTFT;  // Time to first token
    double avgTPOT;  // Time per output token
    int64_t numPreemptions;
    int64_t numSwaps;
    double gpuCacheUsage;
    double cpuCacheUsage;
  };
  
  EngineMetrics getMetrics() const;
  
private:
  EngineConfig config_;
  
  std::unique_ptr<PagedKVCache> cache_;
  std::unique_ptr<ContinuousBatchingEngine> engine_;
  std::unique_ptr<BlockSpaceManagerAdapter> blockManager_;
  std::unique_ptr<PagedAttentionWrapper> attention_;
  
  // Request tracking
  std::unordered_map<std::string, int32_t> requestIdMap_;
  std::unordered_map<int32_t, std::string> reverseIdMap_;
  std::unordered_map<std::string, RequestOutput> pendingOutputs_;
  
  int32_t nextInternalId_;
  mutable EngineMetrics metrics_;
  
  std::mutex mutex_;
};

//===----------------------------------------------------------------------===//
// Python Bindings Support
//===----------------------------------------------------------------------===//

// C API for Python bindings
extern "C" {

// Engine lifecycle
void* llmir_vllm_create_engine(const char* config_json);
void llmir_vllm_destroy_engine(void* engine);
int llmir_vllm_initialize(void* engine);

// Request management
const char* llmir_vllm_add_request(void* engine, const char* prompt,
                                    const char* sampling_params_json);
int llmir_vllm_abort_request(void* engine, const char* request_id);

// Execution
const char* llmir_vllm_step(void* engine);  // Returns JSON array of outputs
int llmir_vllm_has_pending_requests(void* engine);

// Metrics
const char* llmir_vllm_get_metrics(void* engine);

} // extern "C"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace utils {

// Convert between vLLM and LLMIR types
GenerationConfig toGenerationConfig(const SamplingParams& params);
SamplingParams toSamplingParams(const GenerationConfig& config);

// JSON serialization
std::string requestOutputToJson(const RequestOutput& output);
std::string metricsToJson(const LLMEngineAdapter::EngineMetrics& metrics);
SamplingParams samplingParamsFromJson(const std::string& json);

// Tokenization helpers (would integrate with actual tokenizer)
std::vector<int32_t> tokenize(const std::string& text, 
                               const std::string& tokenizerPath);
std::string detokenize(const std::vector<int32_t>& tokens,
                       const std::string& tokenizerPath);

} // namespace utils

} // namespace vllm
} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_VLLMINTEGRATION_H_
