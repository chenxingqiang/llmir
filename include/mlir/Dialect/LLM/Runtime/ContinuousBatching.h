//===- ContinuousBatching.h - Continuous Batching Support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines continuous batching support for efficient LLM serving,
// enabling dynamic batch management with varying sequence lengths.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_CONTINUOUSBATCHING_H_
#define MLIR_DIALECT_LLM_RUNTIME_CONTINUOUSBATCHING_H_

#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Support/LogicalResult.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace llm {
namespace runtime {

//===----------------------------------------------------------------------===//
// Request and Sequence Types
//===----------------------------------------------------------------------===//

enum class RequestStatus {
  PENDING,      // Waiting to be scheduled
  RUNNING,      // Currently being processed
  PREEMPTED,    // Paused for higher priority request
  COMPLETED,    // Successfully finished
  FAILED        // Failed with error
};

enum class RequestPriority {
  LOW = 0,
  NORMAL = 1,
  HIGH = 2,
  URGENT = 3
};

struct GenerationConfig {
  int64_t maxNewTokens;
  float temperature;
  float topP;
  int64_t topK;
  float repetitionPenalty;
  bool doSample;
  std::vector<int32_t> stopTokenIds;
  
  GenerationConfig()
      : maxNewTokens(256), temperature(1.0f), topP(1.0f),
        topK(50), repetitionPenalty(1.0f), doSample(true) {}
};

struct SequenceGroup {
  int32_t groupId;
  std::vector<int32_t> sequenceIds;
  std::vector<int32_t> promptTokens;
  GenerationConfig config;
  RequestPriority priority;
  RequestStatus status;
  
  // Timing
  int64_t arrivalTime;
  int64_t startTime;
  int64_t completionTime;
  
  // Progress
  int64_t promptLen;
  int64_t generatedLen;
  int64_t maxLen;
  
  // Output
  std::vector<std::vector<int32_t>> outputTokens;
  std::vector<std::vector<float>> outputLogProbs;
  
  bool isFinished() const { 
    return status == RequestStatus::COMPLETED || 
           status == RequestStatus::FAILED; 
  }
};

//===----------------------------------------------------------------------===//
// Scheduler Configuration
//===----------------------------------------------------------------------===//

enum class SchedulingPolicy {
  FCFS,              // First-come first-served
  SHORTEST_FIRST,    // Shortest remaining time first
  PRIORITY_BASED,    // Based on request priority
  FAIR_SHARE,        // Fair share among requests
  ADAPTIVE           // Adaptive based on load
};

struct SchedulerConfig {
  SchedulingPolicy policy;
  int64_t maxBatchSize;
  int64_t maxNumSeqs;
  int64_t maxBatchTokens;  // Token budget per batch
  int64_t chunkSize;       // Prefill chunk size for long prompts
  bool enablePreemption;
  float preemptionThreshold;  // Memory threshold for preemption
  bool enablePriorityScheduling;
  int64_t schedulingIntervalMs;
  
  SchedulerConfig()
      : policy(SchedulingPolicy::FCFS), maxBatchSize(256),
        maxNumSeqs(256), maxBatchTokens(8192), chunkSize(512),
        enablePreemption(true), preemptionThreshold(0.9f),
        enablePriorityScheduling(true), schedulingIntervalMs(1) {}
};

//===----------------------------------------------------------------------===//
// Batch State
//===----------------------------------------------------------------------===//

struct BatchState {
  // Sequences in this batch
  std::vector<int32_t> sequenceIds;
  std::vector<int32_t> contextLens;
  std::vector<int32_t> queryLens;
  
  // Block tables for paged attention
  std::vector<std::vector<int32_t>> blockTables;
  
  // Slot mapping for memory
  std::vector<int32_t> slotMapping;
  
  // Batch metadata
  int64_t numPrefillTokens;
  int64_t numDecodeTokens;
  int64_t numSeqs;
  
  // Split into prefill and decode
  std::vector<int32_t> prefillSeqIds;
  std::vector<int32_t> decodeSeqIds;
  
  bool isEmpty() const { return sequenceIds.empty(); }
  int64_t getTotalTokens() const { return numPrefillTokens + numDecodeTokens; }
};

//===----------------------------------------------------------------------===//
// Block Manager for Continuous Batching
//===----------------------------------------------------------------------===//

class ContinuousBatchingBlockManager {
public:
  ContinuousBatchingBlockManager(PagedKVCache& cache, int64_t maxNumBlocks);
  ~ContinuousBatchingBlockManager();
  
  // Block allocation for sequences
  LogicalResult allocateBlocks(int32_t seqId, int64_t numTokens,
                               std::vector<int32_t>& blockIds);
  
  // Free blocks
  void freeBlocks(int32_t seqId);
  
  // Preemption support
  LogicalResult swapOut(int32_t seqId);
  LogicalResult swapIn(int32_t seqId);
  
  // Query state
  int64_t getNumFreeBlocks() const;
  int64_t getNumUsedBlocks() const;
  bool canAllocate(int64_t numTokens) const;
  float getMemoryUsageRatio() const;
  
  // Get block table for a sequence
  const std::vector<int32_t>& getBlockTable(int32_t seqId) const;
  
private:
  PagedKVCache& cache_;
  int64_t maxNumBlocks_;
  
  // Per-sequence block tables
  std::unordered_map<int32_t, std::vector<int32_t>> blockTables_;
  
  // Swapped out sequences
  std::unordered_map<int32_t, std::vector<int32_t>> swappedBlocks_;
  
  // Free block list
  std::vector<int32_t> freeBlocks_;
  
  std::mutex mutex_;
};

//===----------------------------------------------------------------------===//
// Scheduler
//===----------------------------------------------------------------------===//

class Scheduler {
public:
  Scheduler(const SchedulerConfig& config,
            ContinuousBatchingBlockManager& blockManager);
  ~Scheduler();
  
  // Add new request
  LogicalResult addRequest(std::unique_ptr<SequenceGroup> request);
  
  // Schedule next batch
  BatchState schedule();
  
  // Update sequence status
  void updateSequence(int32_t seqId, int64_t newLen, bool isFinished);
  
  // Abort a request
  LogicalResult abortRequest(int32_t groupId);
  
  // Get request status
  RequestStatus getRequestStatus(int32_t groupId) const;
  
  // Statistics
  int64_t getNumPendingRequests() const;
  int64_t getNumRunningRequests() const;
  int64_t getNumCompletedRequests() const;
  
private:
  SchedulerConfig config_;
  ContinuousBatchingBlockManager& blockManager_;
  
  // Request queues
  std::priority_queue<SequenceGroup*, std::vector<SequenceGroup*>,
                     std::function<bool(SequenceGroup*, SequenceGroup*)>> 
      waitingQueue_;
  
  std::unordered_map<int32_t, std::unique_ptr<SequenceGroup>> runningRequests_;
  std::unordered_map<int32_t, std::unique_ptr<SequenceGroup>> swappedRequests_;
  std::unordered_map<int32_t, std::unique_ptr<SequenceGroup>> completedRequests_;
  
  int32_t nextGroupId_;
  std::atomic<int64_t> numPending_;
  std::atomic<int64_t> numRunning_;
  std::atomic<int64_t> numCompleted_;
  
  mutable std::mutex mutex_;
  
  // Scheduling helpers
  bool shouldPreempt() const;
  void preemptLowestPriority();
  std::vector<SequenceGroup*> selectForScheduling(int64_t tokenBudget);
  BatchState createBatch(const std::vector<SequenceGroup*>& groups);
};

//===----------------------------------------------------------------------===//
// Continuous Batching Engine
//===----------------------------------------------------------------------===//

// Callback types
using OutputCallback = std::function<void(int32_t groupId, 
                                          const std::vector<int32_t>& tokens,
                                          bool isFinished)>;

using ErrorCallback = std::function<void(int32_t groupId, 
                                         const std::string& error)>;

class ContinuousBatchingEngine {
public:
  ContinuousBatchingEngine(PagedKVCache& cache,
                           const SchedulerConfig& schedulerConfig);
  ~ContinuousBatchingEngine();
  
  // Start/stop the engine
  void start();
  void stop();
  bool isRunning() const { return running_; }
  
  // Submit requests
  int32_t submitRequest(const std::vector<int32_t>& promptTokens,
                        const GenerationConfig& config,
                        RequestPriority priority = RequestPriority::NORMAL);
  
  // Abort request
  LogicalResult abortRequest(int32_t groupId);
  
  // Wait for completion
  LogicalResult waitForCompletion(int32_t groupId, int64_t timeoutMs = -1);
  
  // Get output
  LogicalResult getOutput(int32_t groupId, 
                          std::vector<int32_t>& outputTokens);
  
  // Register callbacks
  void setOutputCallback(OutputCallback callback);
  void setErrorCallback(ErrorCallback callback);
  
  // Statistics
  struct EngineStats {
    int64_t totalRequests;
    int64_t completedRequests;
    int64_t failedRequests;
    int64_t totalTokensGenerated;
    double avgLatencyMs;
    double tokensPerSecond;
    double avgBatchSize;
    double avgBatchTokens;
  };
  
  EngineStats getStats() const;
  void resetStats();
  
  // Configuration
  void updateConfig(const SchedulerConfig& config);
  const SchedulerConfig& getConfig() const { return schedulerConfig_; }
  
private:
  PagedKVCache& cache_;
  SchedulerConfig schedulerConfig_;
  
  std::unique_ptr<ContinuousBatchingBlockManager> blockManager_;
  std::unique_ptr<Scheduler> scheduler_;
  
  // Callbacks
  OutputCallback outputCallback_;
  ErrorCallback errorCallback_;
  
  // Engine state
  std::atomic<bool> running_;
  std::thread engineThread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  
  // Request completion tracking
  std::unordered_map<int32_t, std::condition_variable> completionCvs_;
  std::unordered_map<int32_t, std::mutex> completionMutexes_;
  
  // Statistics
  mutable EngineStats stats_;
  
  // Main loop
  void engineLoop();
  
  // Process a batch
  void processBatch(const BatchState& batch);
  
  // Generate tokens for decode sequences
  void decodeStep(const std::vector<int32_t>& seqIds);
  
  // Process prefill for new sequences
  void prefillStep(const std::vector<int32_t>& seqIds,
                   const std::vector<int32_t>& tokenLens);
};

//===----------------------------------------------------------------------===//
// Request Builder
//===----------------------------------------------------------------------===//

class RequestBuilder {
public:
  RequestBuilder();
  
  RequestBuilder& setPrompt(const std::vector<int32_t>& tokens);
  RequestBuilder& setMaxNewTokens(int64_t maxTokens);
  RequestBuilder& setTemperature(float temp);
  RequestBuilder& setTopP(float topP);
  RequestBuilder& setTopK(int64_t topK);
  RequestBuilder& setPriority(RequestPriority priority);
  RequestBuilder& setStopTokens(const std::vector<int32_t>& stopTokens);
  RequestBuilder& enableSampling(bool enable);
  
  std::unique_ptr<SequenceGroup> build();
  
private:
  std::vector<int32_t> promptTokens_;
  GenerationConfig config_;
  RequestPriority priority_;
  static std::atomic<int32_t> nextId_;
};

//===----------------------------------------------------------------------===//
// Iteration-Level Scheduling
//===----------------------------------------------------------------------===//

// Fine-grained scheduling at the iteration level
class IterationScheduler {
public:
  IterationScheduler(Scheduler& scheduler, int64_t maxIterTokens);
  ~IterationScheduler();
  
  // Schedule one iteration
  struct IterationPlan {
    std::vector<int32_t> prefillSeqs;
    std::vector<int32_t> prefillLens;  // How many tokens to process
    std::vector<int32_t> decodeSeqs;
    int64_t totalTokens;
  };
  
  IterationPlan planIteration();
  
  // Chunked prefill support
  void setChunkSize(int64_t size) { chunkSize_ = size; }
  int64_t getChunkSize() const { return chunkSize_; }
  
private:
  Scheduler& scheduler_;
  int64_t maxIterTokens_;
  int64_t chunkSize_;
  
  // Track partial prefills
  std::unordered_map<int32_t, int64_t> prefillProgress_;
};

//===----------------------------------------------------------------------===//
// Memory Pressure Monitor
//===----------------------------------------------------------------------===//

class MemoryPressureMonitor {
public:
  MemoryPressureMonitor(ContinuousBatchingBlockManager& blockManager,
                        float highWatermark = 0.9f,
                        float lowWatermark = 0.7f);
  ~MemoryPressureMonitor();
  
  // Check current pressure level
  enum class PressureLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
  };
  
  PressureLevel getCurrentLevel() const;
  
  // Get recommended action
  enum class Action {
    NONE,
    SLOW_ADMISSION,
    PREEMPT_LOW_PRIORITY,
    PREEMPT_ANY,
    REJECT_NEW
  };
  
  Action getRecommendedAction() const;
  
  // Start background monitoring
  void startMonitoring(int64_t intervalMs = 100);
  void stopMonitoring();
  
  // Callbacks
  using PressureCallback = std::function<void(PressureLevel, Action)>;
  void setCallback(PressureCallback callback);
  
private:
  ContinuousBatchingBlockManager& blockManager_;
  float highWatermark_;
  float lowWatermark_;
  
  std::atomic<bool> monitoring_;
  std::thread monitorThread_;
  PressureCallback callback_;
};

} // namespace runtime
} // namespace llm
} // namespace mlir

#endif // MLIR_DIALECT_LLM_RUNTIME_CONTINUOUSBATCHING_H_
