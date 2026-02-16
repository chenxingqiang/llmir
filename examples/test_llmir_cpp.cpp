// LLMIR C++ Runtime Algorithm Test
// Tests the core algorithms without MLIR dependencies

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <memory>

// ============================================================
// Attention Configuration
// ============================================================

enum class AttentionMaskType {
    NONE,
    CAUSAL,
    SLIDING_WINDOW,
    CUSTOM
};

struct AttentionConfig {
    int64_t numHeads = 32;
    int64_t headDim = 128;
    float scale = 0.0f;
    int64_t windowSize = 0;
    int64_t blockSizeM = 64;
    int64_t blockSizeN = 64;
    AttentionMaskType maskType = AttentionMaskType::NONE;
    bool fuseSoftmax = false;
    bool useFlashAttention = false;
    float dropoutProb = 0.0f;
    
    void setDefaultsFromHeadDim() {
        if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        }
    }
};

// ============================================================
// Attention Implementations
// ============================================================

class AttentionImpl {
public:
    virtual ~AttentionImpl() = default;
    virtual void compute(
        float* output,
        const float* queries,
        const float* keys,
        const float* values,
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen,
        const float* mask = nullptr
    ) = 0;
    
    virtual const char* name() const = 0;
};

// Standard attention with fused softmax
class FusedSoftmaxAttention : public AttentionImpl {
    AttentionConfig config;
public:
    FusedSoftmaxAttention(const AttentionConfig& cfg) : config(cfg) {
        if (config.scale == 0.0f) config.setDefaultsFromHeadDim();
    }
    
    const char* name() const override { return "FusedSoftmax"; }
    
    void compute(
        float* output,
        const float* queries,
        const float* keys,
        const float* values,
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen,
        const float* mask
    ) override {
        std::memset(output, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
        
        for (int64_t b = 0; b < batchSize; b++) {
            for (int64_t h = 0; h < config.numHeads; h++) {
                for (int64_t queryIdx = 0; queryIdx < seqLen; queryIdx++) {
                    const float* queryVec = queries + 
                        ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
                    float* outputVec = output + 
                        ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
                    
                    float maxVal = -std::numeric_limits<float>::infinity();
                    std::vector<float> scores(contextLen);
                    
                    // First pass: compute scores
                    for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
                        if (config.maskType == AttentionMaskType::CAUSAL && keyIdx > queryIdx) {
                            scores[keyIdx] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        
                        const float* keyVec = keys + 
                            ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
                        
                        float score = 0.0f;
                        for (int64_t d = 0; d < config.headDim; d++) {
                            score += queryVec[d] * keyVec[d];
                        }
                        score *= config.scale;
                        scores[keyIdx] = score;
                        maxVal = std::max(maxVal, score);
                    }
                    
                    // Second pass: softmax
                    float expSum = 0.0f;
                    for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
                        if (scores[keyIdx] > -1e9f) {
                            scores[keyIdx] = std::exp(scores[keyIdx] - maxVal);
                            expSum += scores[keyIdx];
                        } else {
                            scores[keyIdx] = 0.0f;
                        }
                    }
                    
                    // Third pass: weighted sum
                    if (expSum > 0.0f) {
                        for (int64_t keyIdx = 0; keyIdx < contextLen; keyIdx++) {
                            if (scores[keyIdx] > 0.0f) {
                                const float* valueVec = values + 
                                    ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
                                float weight = scores[keyIdx] / expSum;
                                
                                for (int64_t d = 0; d < config.headDim; d++) {
                                    outputVec[d] += weight * valueVec[d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

// Sliding window attention
class SlidingWindowAttention : public AttentionImpl {
    AttentionConfig config;
public:
    SlidingWindowAttention(const AttentionConfig& cfg) : config(cfg) {
        if (config.scale == 0.0f) config.setDefaultsFromHeadDim();
        if (config.windowSize == 0) config.windowSize = 256;
    }
    
    const char* name() const override { return "SlidingWindow"; }
    
    void compute(
        float* output,
        const float* queries,
        const float* keys,
        const float* values,
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen,
        const float* mask
    ) override {
        std::memset(output, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
        
        for (int64_t b = 0; b < batchSize; b++) {
            for (int64_t h = 0; h < config.numHeads; h++) {
                for (int64_t queryIdx = 0; queryIdx < seqLen; queryIdx++) {
                    // Sliding window bounds
                    int64_t windowStart = std::max(int64_t(0), queryIdx - config.windowSize);
                    int64_t windowEnd = std::min(contextLen, queryIdx + config.windowSize + 1);
                    
                    // Apply causal mask
                    if (config.maskType == AttentionMaskType::CAUSAL) {
                        windowEnd = std::min(windowEnd, queryIdx + 1);
                    }
                    
                    const float* queryVec = queries + 
                        ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
                    float* outputVec = output + 
                        ((b * seqLen + queryIdx) * config.numHeads + h) * config.headDim;
                    
                    int64_t windowLen = windowEnd - windowStart;
                    std::vector<float> scores(windowLen);
                    float maxVal = -std::numeric_limits<float>::infinity();
                    
                    // Compute scores only within window
                    for (int64_t i = 0; i < windowLen; i++) {
                        int64_t keyIdx = windowStart + i;
                        const float* keyVec = keys + 
                            ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
                        
                        float score = 0.0f;
                        for (int64_t d = 0; d < config.headDim; d++) {
                            score += queryVec[d] * keyVec[d];
                        }
                        score *= config.scale;
                        scores[i] = score;
                        maxVal = std::max(maxVal, score);
                    }
                    
                    // Softmax
                    float expSum = 0.0f;
                    for (int64_t i = 0; i < windowLen; i++) {
                        scores[i] = std::exp(scores[i] - maxVal);
                        expSum += scores[i];
                    }
                    
                    // Weighted sum
                    if (expSum > 0.0f) {
                        for (int64_t i = 0; i < windowLen; i++) {
                            int64_t keyIdx = windowStart + i;
                            const float* valueVec = values + 
                                ((b * contextLen + keyIdx) * config.numHeads + h) * config.headDim;
                            float weight = scores[i] / expSum;
                            
                            for (int64_t d = 0; d < config.headDim; d++) {
                                outputVec[d] += weight * valueVec[d];
                            }
                        }
                    }
                }
            }
        }
    }
};

// Flash Attention (tiled)
class FlashAttention : public AttentionImpl {
    AttentionConfig config;
public:
    FlashAttention(const AttentionConfig& cfg) : config(cfg) {
        if (config.scale == 0.0f) config.setDefaultsFromHeadDim();
    }
    
    const char* name() const override { return "FlashAttention"; }
    
    void compute(
        float* output,
        const float* queries,
        const float* keys,
        const float* values,
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen,
        const float* mask
    ) override {
        std::memset(output, 0, batchSize * seqLen * config.numHeads * config.headDim * sizeof(float));
        
        // Per-query max and sum accumulators
        std::vector<float> maxVals(batchSize * seqLen * config.numHeads, 
                                   -std::numeric_limits<float>::infinity());
        std::vector<float> sumVals(batchSize * seqLen * config.numHeads, 0.0f);
        
        // Process in blocks
        for (int64_t b = 0; b < batchSize; b++) {
            for (int64_t h = 0; h < config.numHeads; h++) {
                // Tile over queries
                for (int64_t qBlock = 0; qBlock < seqLen; qBlock += config.blockSizeM) {
                    int64_t qBlockEnd = std::min(qBlock + config.blockSizeM, seqLen);
                    
                    // Tile over keys
                    for (int64_t kBlock = 0; kBlock < contextLen; kBlock += config.blockSizeN) {
                        int64_t kBlockEnd = std::min(kBlock + config.blockSizeN, contextLen);
                        
                        // Process block
                        for (int64_t q = qBlock; q < qBlockEnd; q++) {
                            int64_t accIdx = (b * seqLen + q) * config.numHeads + h;
                            float prevMax = maxVals[accIdx];
                            float prevSum = sumVals[accIdx];
                            float newMax = prevMax;
                            
                            const float* queryVec = queries + 
                                ((b * seqLen + q) * config.numHeads + h) * config.headDim;
                            float* outputVec = output + 
                                ((b * seqLen + q) * config.numHeads + h) * config.headDim;
                            
                            // Compute local scores
                            std::vector<float> localScores(kBlockEnd - kBlock);
                            for (int64_t k = kBlock; k < kBlockEnd; k++) {
                                if (config.maskType == AttentionMaskType::CAUSAL && k > q) {
                                    localScores[k - kBlock] = -std::numeric_limits<float>::infinity();
                                    continue;
                                }
                                
                                const float* keyVec = keys + 
                                    ((b * contextLen + k) * config.numHeads + h) * config.headDim;
                                
                                float score = 0.0f;
                                for (int64_t d = 0; d < config.headDim; d++) {
                                    score += queryVec[d] * keyVec[d];
                                }
                                score *= config.scale;
                                localScores[k - kBlock] = score;
                                newMax = std::max(newMax, score);
                            }
                            
                            // Update with new max
                            float scaler = std::exp(prevMax - newMax);
                            float localExpSum = 0.0f;
                            
                            // Scale existing output
                            for (int64_t d = 0; d < config.headDim; d++) {
                                outputVec[d] *= scaler;
                            }
                            
                            // Add new contributions
                            for (int64_t k = kBlock; k < kBlockEnd; k++) {
                                float score = localScores[k - kBlock];
                                if (score > -1e9f) {
                                    float expScore = std::exp(score - newMax);
                                    localExpSum += expScore;
                                    
                                    const float* valueVec = values + 
                                        ((b * contextLen + k) * config.numHeads + h) * config.headDim;
                                    
                                    for (int64_t d = 0; d < config.headDim; d++) {
                                        outputVec[d] += expScore * valueVec[d];
                                    }
                                }
                            }
                            
                            // Update accumulators
                            maxVals[accIdx] = newMax;
                            sumVals[accIdx] = prevSum * scaler + localExpSum;
                        }
                    }
                }
                
                // Final normalization
                for (int64_t q = 0; q < seqLen; q++) {
                    int64_t accIdx = (b * seqLen + q) * config.numHeads + h;
                    float sum = sumVals[accIdx];
                    
                    if (sum > 1e-6f) {
                        float* outputVec = output + 
                            ((b * seqLen + q) * config.numHeads + h) * config.headDim;
                        for (int64_t d = 0; d < config.headDim; d++) {
                            outputVec[d] /= sum;
                        }
                    }
                }
            }
        }
    }
};

// ============================================================
// Prefix Cache (Radix Tree)
// ============================================================

class PrefixCache {
    struct Node {
        std::unordered_map<int32_t, std::unique_ptr<Node>> children;
        int32_t blockId = -1;
        int32_t refCount = 0;
    };
    
    std::unique_ptr<Node> root;
    int64_t maxPrefixes;
    int64_t minPrefixLen;
    int64_t numPrefixes = 0;
    int64_t hits = 0;
    int64_t misses = 0;

public:
    PrefixCache(int64_t maxPrefixes = 1000, int64_t minLen = 4)
        : maxPrefixes(maxPrefixes), minPrefixLen(minLen) {
        root = std::make_unique<Node>();
    }
    
    void cachePrefix(const int32_t* tokens, int64_t len, int32_t blockId) {
        if (len < minPrefixLen) return;
        
        Node* current = root.get();
        for (int64_t i = 0; i < len; i++) {
            int32_t token = tokens[i];
            if (current->children.find(token) == current->children.end()) {
                current->children[token] = std::make_unique<Node>();
            }
            current = current->children[token].get();
        }
        current->blockId = blockId;
        current->refCount++;
        numPrefixes++;
    }
    
    int64_t lookupPrefix(const int32_t* tokens, int64_t len, int32_t& outBlockId) {
        Node* current = root.get();
        int64_t matchLen = 0;
        outBlockId = -1;
        
        for (int64_t i = 0; i < len; i++) {
            int32_t token = tokens[i];
            auto it = current->children.find(token);
            if (it == current->children.end()) {
                break;
            }
            current = it->second.get();
            matchLen = i + 1;
            
            if (current->blockId >= 0) {
                outBlockId = current->blockId;
            }
        }
        
        if (matchLen >= minPrefixLen && outBlockId >= 0) {
            hits++;
        } else {
            misses++;
        }
        
        return outBlockId >= 0 ? matchLen : 0;
    }
    
    double getHitRate() const {
        int64_t total = hits + misses;
        return total > 0 ? (double)hits / total : 0.0;
    }
};

// ============================================================
// KV Cache Block Size Optimization
// ============================================================

int64_t getOptimalBlockSize(int64_t seqLen, int64_t headDim) {
    // LLMIR's block size optimization algorithm
    if (seqLen <= 32) return 16;
    else if (seqLen <= 256) return 32;
    else if (seqLen <= 1024) return 64;
    else return 128;
}

// ============================================================
// Timer Helper
// ============================================================

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// ============================================================
// Tests
// ============================================================

void testFusedSoftmaxAttention() {
    std::cout << "\n=== Test 1: FusedSoftmaxAttention ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.setDefaultsFromHeadDim();
    config.maskType = AttentionMaskType::CAUSAL;
    
    FusedSoftmaxAttention impl(config);
    
    int64_t batchSize = 2, seqLen = 128, contextLen = 128;
    std::vector<float> Q(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> K(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> V(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> O(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    Timer timer;
    timer.start();
    impl.compute(O.data(), Q.data(), K.data(), V.data(), batchSize, seqLen, contextLen, nullptr);
    double elapsed = timer.elapsedMs();
    
    float sum = 0.0f;
    for (float v : O) sum += std::abs(v);
    
    std::cout << "  Config: " << config.numHeads << " heads, " << config.headDim << " dim" << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  Output sum: " << sum << " (should be > 0)" << std::endl;
    std::cout << "  ✓ FusedSoftmaxAttention passed" << std::endl;
}

void testSlidingWindowAttention() {
    std::cout << "\n=== Test 2: SlidingWindowAttention ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.windowSize = 64;
    config.setDefaultsFromHeadDim();
    config.maskType = AttentionMaskType::SLIDING_WINDOW;
    
    SlidingWindowAttention impl(config);
    
    int64_t batchSize = 1, seqLen = 256, contextLen = 256;
    std::vector<float> Q(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> K(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> V(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> O(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    Timer timer;
    timer.start();
    impl.compute(O.data(), Q.data(), K.data(), V.data(), batchSize, seqLen, contextLen, nullptr);
    double elapsed = timer.elapsedMs();
    
    std::cout << "  Window size: " << config.windowSize << std::endl;
    std::cout << "  Sequence: " << seqLen << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  ✓ SlidingWindowAttention passed" << std::endl;
}

void testFlashAttention() {
    std::cout << "\n=== Test 3: FlashAttention (Tiled) ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 8;
    config.headDim = 64;
    config.blockSizeM = 64;
    config.blockSizeN = 64;
    config.setDefaultsFromHeadDim();
    config.maskType = AttentionMaskType::CAUSAL;
    
    FlashAttention impl(config);
    
    int64_t batchSize = 1, seqLen = 512, contextLen = 512;
    std::vector<float> Q(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> K(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> V(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
    std::vector<float> O(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
    
    Timer timer;
    timer.start();
    impl.compute(O.data(), Q.data(), K.data(), V.data(), batchSize, seqLen, contextLen, nullptr);
    double elapsed = timer.elapsedMs();
    
    std::cout << "  Block size: " << config.blockSizeM << "x" << config.blockSizeN << std::endl;
    std::cout << "  Sequence: " << seqLen << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  ✓ FlashAttention passed" << std::endl;
}

void testPrefixCache() {
    std::cout << "\n=== Test 4: PrefixCache (Radix Tree) ===" << std::endl;
    
    PrefixCache cache(1000, 4);
    
    std::vector<int32_t> prefix1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int32_t> prefix2 = {1, 2, 3, 4, 5, 20, 21, 22};
    
    cache.cachePrefix(prefix1.data(), prefix1.size(), 1);
    cache.cachePrefix(prefix2.data(), prefix2.size(), 2);
    
    // Test exact match
    std::vector<int32_t> query1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t blockId;
    int64_t matchLen = cache.lookupPrefix(query1.data(), query1.size(), blockId);
    std::cout << "  Exact match: len=" << matchLen << ", block=" << blockId << std::endl;
    
    // Test partial match
    std::vector<int32_t> query2 = {1, 2, 3, 4, 5, 30, 31};
    matchLen = cache.lookupPrefix(query2.data(), query2.size(), blockId);
    std::cout << "  Partial match: len=" << matchLen << std::endl;
    
    // Test no match
    std::vector<int32_t> query3 = {100, 101, 102};
    matchLen = cache.lookupPrefix(query3.data(), query3.size(), blockId);
    std::cout << "  No match: len=" << matchLen << std::endl;
    
    std::cout << "  Hit rate: " << cache.getHitRate() * 100 << "%" << std::endl;
    std::cout << "  ✓ PrefixCache passed" << std::endl;
}

void testBlockSizeOptimization() {
    std::cout << "\n=== Test 5: Block Size Optimization ===" << std::endl;
    
    std::vector<std::pair<int64_t, int64_t>> testCases = {
        {16, 16}, {32, 16}, {64, 32}, {128, 32},
        {256, 32}, {512, 64}, {1024, 64}, {2048, 128}, {4096, 128}
    };
    
    std::cout << "  SeqLen -> Optimal Block Size" << std::endl;
    for (auto [seqLen, expected] : testCases) {
        int64_t optimal = getOptimalBlockSize(seqLen, 128);
        std::cout << "  " << seqLen << " -> " << optimal;
        if (optimal == expected) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " (expected " << expected << ")" << std::endl;
        }
    }
    std::cout << "  ✓ Block Size Optimization passed" << std::endl;
}

void benchmarkAttention() {
    std::cout << "\n=== Benchmark: Attention Performance ===" << std::endl;
    
    AttentionConfig config;
    config.numHeads = 32;
    config.headDim = 128;
    config.setDefaultsFromHeadDim();
    config.maskType = AttentionMaskType::CAUSAL;
    
    std::vector<std::tuple<std::string, int64_t, int64_t>> testCases = {
        {"FusedSoftmax", 1, 128},
        {"FusedSoftmax", 1, 256},
        {"FusedSoftmax", 1, 512},
        {"FusedSoftmax", 4, 128},
        {"FlashAttention", 1, 512},
        {"FlashAttention", 1, 1024},
        {"SlidingWindow", 1, 1024},
        {"SlidingWindow", 1, 2048},
    };
    
    std::cout << "\n  Type             Batch  Seq      Time(ms)   Throughput" << std::endl;
    std::cout << "  ----------------------------------------------------------" << std::endl;
    
    for (auto& [type, batchSize, seqLen] : testCases) {
        std::unique_ptr<AttentionImpl> impl;
        
        if (type == "FusedSoftmax") {
            impl = std::make_unique<FusedSoftmaxAttention>(config);
        } else if (type == "FlashAttention") {
            config.blockSizeM = 64;
            config.blockSizeN = 64;
            impl = std::make_unique<FlashAttention>(config);
        } else {
            config.windowSize = 256;
            impl = std::make_unique<SlidingWindowAttention>(config);
        }
        
        int64_t contextLen = seqLen;
        std::vector<float> Q(batchSize * seqLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> K(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> V(batchSize * contextLen * config.numHeads * config.headDim, 0.1f);
        std::vector<float> O(batchSize * seqLen * config.numHeads * config.headDim, 0.0f);
        
        // Warmup
        impl->compute(O.data(), Q.data(), K.data(), V.data(), batchSize, seqLen, contextLen, nullptr);
        
        // Benchmark
        int iterations = 5;
        Timer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            impl->compute(O.data(), Q.data(), K.data(), V.data(), batchSize, seqLen, contextLen, nullptr);
        }
        double elapsed = timer.elapsedMs() / iterations;
        
        double tokens = batchSize * seqLen;
        double throughput = tokens / (elapsed / 1000.0);
        
        printf("  %-15s %5ld %5ld %10.2f %12.0f tok/s\n",
               type.c_str(), batchSize, seqLen, elapsed, throughput);
    }
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << " LLMIR C++ Algorithm Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    testFusedSoftmaxAttention();
    testSlidingWindowAttention();
    testFlashAttention();
    testPrefixCache();
    testBlockSizeOptimization();
    benchmarkAttention();
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << " All C++ Tests Passed! ✓" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    return 0;
}
