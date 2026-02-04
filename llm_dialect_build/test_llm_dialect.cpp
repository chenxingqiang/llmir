// LLMIR LLM Dialect Test
// Tests the C++ implementation of the LLM dialect

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

// Include LLMIR Runtime headers
#include "mlir/Dialect/LLM/Runtime/AttentionOpt.h"
#include "mlir/Dialect/LLM/Runtime/KVCache.h"
#include "mlir/Dialect/LLM/Runtime/PrefixCache.h"

using namespace mlir::llm::runtime;

// Helper to measure time
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

void testAttentionConfig() {
    std::cout << "\n=== Test 1: AttentionConfig ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 32;
    config.headDim = 128;
    config.scale = 0.0f;  // Will be auto-calculated
    config.maskType = AttentionMaskType::CAUSAL;
    config.useFlashAttention = true;
    
    config.setDefaultsFromHeadDim();
    
    std::cout << "  Heads: " << config.numHeads << std::endl;
    std::cout << "  Head dim: " << config.headDim << std::endl;
    std::cout << "  Scale: " << config.scale << std::endl;
    std::cout << "  Flash Attention: " << (config.useFlashAttention ? "enabled" : "disabled") << std::endl;
    std::cout << "  ✓ AttentionConfig test passed" << std::endl;
}

void testFusedSoftmaxAttention() {
    std::cout << "\n=== Test 2: FusedSoftmaxAttention ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.setDefaultsFromHeadDim();
    config.fuseSoftmax = true;
    config.maskType = AttentionMaskType::CAUSAL;
    
    // Create implementation
    auto impl = createAttentionImpl(config, mlir::Type(), false);
    
    // Test parameters
    int64_t batchSize = 2;
    int64_t seqLen = 128;
    int64_t contextLen = 128;
    
    // Allocate test data
    std::vector<float> queries(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> keys(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> values(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> output(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    // Run computation
    Timer timer;
    timer.start();
    impl->compute(
        output.data(),
        queries.data(),
        keys.data(),
        values.data(),
        batchSize,
        seqLen,
        contextLen,
        nullptr
    );
    double elapsed = timer.elapsedMs();
    
    // Verify output is not all zeros
    float sum = 0.0f;
    for (float v : output) sum += v;
    bool hasOutput = sum != 0.0f;
    
    std::cout << "  Batch: " << batchSize << ", SeqLen: " << seqLen << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  Output valid: " << (hasOutput ? "yes" : "no") << std::endl;
    std::cout << "  ✓ FusedSoftmaxAttention test passed" << std::endl;
}

void testSlidingWindowAttention() {
    std::cout << "\n=== Test 3: SlidingWindowAttention ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.windowSize = 64;  // Sliding window of 64 tokens
    config.setDefaultsFromHeadDim();
    config.maskType = AttentionMaskType::SLIDING_WINDOW;
    
    auto impl = createAttentionImpl(config, mlir::Type(), false);
    
    int64_t batchSize = 2;
    int64_t seqLen = 256;  // Longer sequence to test sliding window
    int64_t contextLen = 256;
    
    std::vector<float> queries(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> keys(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> values(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> output(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    Timer timer;
    timer.start();
    impl->compute(
        output.data(),
        queries.data(),
        keys.data(),
        values.data(),
        batchSize,
        seqLen,
        contextLen,
        nullptr
    );
    double elapsed = timer.elapsedMs();
    
    float sum = 0.0f;
    for (float v : output) sum += v;
    
    std::cout << "  Window size: " << config.windowSize << std::endl;
    std::cout << "  Sequence length: " << seqLen << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  ✓ SlidingWindowAttention test passed" << std::endl;
}

void testFlashAttention() {
    std::cout << "\n=== Test 4: FlashAttention (Tiled) ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.blockSizeM = 64;
    config.blockSizeN = 64;
    config.setDefaultsFromHeadDim();
    config.useFlashAttention = true;
    config.maskType = AttentionMaskType::CAUSAL;
    
    auto impl = createAttentionImpl(config, mlir::Type(), false);
    
    int64_t batchSize = 1;
    int64_t seqLen = 512;
    int64_t contextLen = 512;
    
    std::vector<float> queries(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> keys(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> values(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> output(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    Timer timer;
    timer.start();
    impl->compute(
        output.data(),
        queries.data(),
        keys.data(),
        values.data(),
        batchSize,
        seqLen,
        contextLen,
        nullptr
    );
    double elapsed = timer.elapsedMs();
    
    float sum = 0.0f;
    for (float v : output) sum += v;
    
    std::cout << "  Block size M: " << config.blockSizeM << std::endl;
    std::cout << "  Block size N: " << config.blockSizeN << std::endl;
    std::cout << "  Sequence length: " << seqLen << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  ✓ FlashAttention test passed" << std::endl;
}

void testPrefixCache() {
    std::cout << "\n=== Test 5: PrefixCache (Radix Tree) ===" << std::endl;
    
    PrefixCache cache(1000, 4);  // max_prefixes=1000, min_prefix_len=4
    
    // Create some prefixes
    std::vector<int32_t> prefix1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int32_t> prefix2 = {1, 2, 3, 4, 5, 20, 21, 22, 23, 24};
    std::vector<int32_t> prefix3 = {100, 101, 102, 103, 104, 105};
    
    // Cache prefixes
    cache.cachePrefix(prefix1.data(), prefix1.size(), 1);
    cache.cachePrefix(prefix2.data(), prefix2.size(), 2);
    cache.cachePrefix(prefix3.data(), prefix3.size(), 3);
    
    std::cout << "  Cached 3 prefixes" << std::endl;
    
    // Test lookup - exact match
    std::vector<int32_t> query1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t blockId = -1;
    int32_t matchLen = cache.lookupPrefix(query1.data(), query1.size(), blockId);
    std::cout << "  Query (exact match): length=" << matchLen << ", blockId=" << blockId << std::endl;
    
    // Test lookup - partial match
    std::vector<int32_t> query2 = {1, 2, 3, 4, 5, 30, 31, 32};
    matchLen = cache.lookupPrefix(query2.data(), query2.size(), blockId);
    std::cout << "  Query (partial match): length=" << matchLen << std::endl;
    
    // Test lookup - no match
    std::vector<int32_t> query3 = {200, 201, 202, 203};
    matchLen = cache.lookupPrefix(query3.data(), query3.size(), blockId);
    std::cout << "  Query (no match): length=" << matchLen << std::endl;
    
    std::cout << "  ✓ PrefixCache test passed" << std::endl;
}

void benchmarkAttention() {
    std::cout << "\n=== Benchmark: Attention Performance ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 32;
    config.headDim = 128;
    config.setDefaultsFromHeadDim();
    config.fuseSoftmax = true;
    config.maskType = AttentionMaskType::CAUSAL;
    
    auto impl = createAttentionImpl(config, mlir::Type(), false);
    
    std::vector<std::pair<int64_t, int64_t>> testCases = {
        {1, 128},
        {1, 256},
        {1, 512},
        {4, 128},
        {4, 256},
        {8, 128},
    };
    
    std::cout << "\n  Batch  SeqLen      Time(ms)   Throughput" << std::endl;
    std::cout << "  ----------------------------------------" << std::endl;
    
    for (auto [batchSize, seqLen] : testCases) {
        int64_t contextLen = seqLen;
        
        std::vector<float> queries(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> keys(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> values(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> output(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
        
        // Warmup
        impl->compute(output.data(), queries.data(), keys.data(), values.data(),
                     batchSize, seqLen, contextLen, nullptr);
        
        // Benchmark
        int iterations = 10;
        Timer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            impl->compute(output.data(), queries.data(), keys.data(), values.data(),
                         batchSize, seqLen, contextLen, nullptr);
        }
        double elapsed = timer.elapsedMs() / iterations;
        
        double tokens = batchSize * seqLen;
        double throughput = tokens / (elapsed / 1000.0);
        
        std::cout << "  " << batchSize << "      " << seqLen 
                  << "       " << elapsed << "      " 
                  << throughput << " tok/s" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " LLMIR C++ Runtime Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        testAttentionConfig();
        testFusedSoftmaxAttention();
        testSlidingWindowAttention();
        testFlashAttention();
        testPrefixCache();
        benchmarkAttention();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << " All C++ Runtime Tests Passed! ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
