#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <memory>
#include <algorithm>
#include <string>

// Multi-Query Attention implementation for benchmarking
// MQA uses multiple query heads but only one key/value head
class MultiQueryAttention {
public:
    MultiQueryAttention(int64_t numQueryHeads, int64_t headDim) 
        : numQueryHeads_(numQueryHeads), headDim_(headDim) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute multi-query attention
    void compute(
        float* output,        // [batch, seqLen, numQueryHeads, headDim]
        const float* query,   // [batch, seqLen, numQueryHeads, headDim]
        const float* key,     // [batch, contextLen, 1, headDim] - shared across query heads
        const float* value,   // [batch, contextLen, 1, headDim] - shared across query heads
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen
    ) {
        // For each batch and head
        for (int64_t b = 0; b < batchSize; ++b) {
            // For each query head
            for (int64_t h = 0; h < numQueryHeads_; ++h) {
                // For each query position
                for (int64_t m = 0; m < seqLen; ++m) {
                    int64_t qIdx = ((b * seqLen + m) * numQueryHeads_ + h) * headDim_;
                    
                    // Compute attention scores and apply softmax
                    std::vector<float> scores(contextLen);
                    float maxScore = -std::numeric_limits<float>::infinity();
                    
                    // Compute scores - using shared KV heads
                    for (int64_t n = 0; n < contextLen; ++n) {
                        // Key is shared across all query heads
                        int64_t kIdx = (b * contextLen + n) * headDim_;
                        float dotProduct = 0.0f;
                        
                        for (int64_t d = 0; d < headDim_; ++d) {
                            dotProduct += query[qIdx + d] * key[kIdx + d];
                        }
                        
                        scores[n] = dotProduct * scale_;
                        
                        // Apply causal mask
                        if (m < n) {
                            scores[n] = -std::numeric_limits<float>::infinity();
                        }
                        
                        if (scores[n] > maxScore) {
                            maxScore = scores[n];
                        }
                    }
                    
                    // Apply softmax
                    float sum = 0.0f;
                    for (int64_t n = 0; n < contextLen; ++n) {
                        float expValue = 0.0f;
                        if (scores[n] != -std::numeric_limits<float>::infinity()) {
                            expValue = std::exp(scores[n] - maxScore);
                        }
                        scores[n] = expValue;
                        sum += expValue;
                    }
                    
                    // Normalize
                    if (sum > 0) {
                        for (int64_t n = 0; n < contextLen; ++n) {
                            scores[n] /= sum;
                        }
                    }
                    
                    // Compute weighted sum of values (shared across query heads)
                    int64_t outIdx = ((b * seqLen + m) * numQueryHeads_ + h) * headDim_;
                    for (int64_t d = 0; d < headDim_; ++d) {
                        output[outIdx + d] = 0.0f;
                        
                        for (int64_t n = 0; n < contextLen; ++n) {
                            // Value is shared across all query heads
                            int64_t vIdx = (b * contextLen + n) * headDim_ + d;
                            output[outIdx + d] += scores[n] * value[vIdx];
                        }
                    }
                }
            }
        }
    }

private:
    int64_t numQueryHeads_;
    int64_t headDim_;
    float scale_;
};

// Standard Multi-Head Attention for comparison
class MultiHeadAttention {
public:
    MultiHeadAttention(int64_t numHeads, int64_t headDim) 
        : numHeads_(numHeads), headDim_(headDim) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute standard multi-head attention
    void compute(
        float* output,        // [batch, seqLen, numHeads, headDim]
        const float* query,   // [batch, seqLen, numHeads, headDim]
        const float* key,     // [batch, contextLen, numHeads, headDim]
        const float* value,   // [batch, contextLen, numHeads, headDim]
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen
    ) {
        // For each batch and head
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t h = 0; h < numHeads_; ++h) {
                // For each query position
                for (int64_t m = 0; m < seqLen; ++m) {
                    int64_t qIdx = ((b * seqLen + m) * numHeads_ + h) * headDim_;
                    
                    // Compute attention scores and apply softmax
                    std::vector<float> scores(contextLen);
                    float maxScore = -std::numeric_limits<float>::infinity();
                    
                    // Compute scores
                    for (int64_t n = 0; n < contextLen; ++n) {
                        int64_t kIdx = ((b * contextLen + n) * numHeads_ + h) * headDim_;
                        float dotProduct = 0.0f;
                        
                        for (int64_t d = 0; d < headDim_; ++d) {
                            dotProduct += query[qIdx + d] * key[kIdx + d];
                        }
                        
                        scores[n] = dotProduct * scale_;
                        
                        // Apply causal mask
                        if (m < n) {
                            scores[n] = -std::numeric_limits<float>::infinity();
                        }
                        
                        if (scores[n] > maxScore) {
                            maxScore = scores[n];
                        }
                    }
                    
                    // Apply softmax
                    float sum = 0.0f;
                    for (int64_t n = 0; n < contextLen; ++n) {
                        float expValue = 0.0f;
                        if (scores[n] != -std::numeric_limits<float>::infinity()) {
                            expValue = std::exp(scores[n] - maxScore);
                        }
                        scores[n] = expValue;
                        sum += expValue;
                    }
                    
                    // Normalize
                    if (sum > 0) {
                        for (int64_t n = 0; n < contextLen; ++n) {
                            scores[n] /= sum;
                        }
                    }
                    
                    // Compute weighted sum of values
                    int64_t outIdx = ((b * seqLen + m) * numHeads_ + h) * headDim_;
                    for (int64_t d = 0; d < headDim_; ++d) {
                        output[outIdx + d] = 0.0f;
                        
                        for (int64_t n = 0; n < contextLen; ++n) {
                            int64_t vIdx = ((b * contextLen + n) * numHeads_ + h) * headDim_ + d;
                            output[outIdx + d] += scores[n] * value[vIdx];
                        }
                    }
                }
            }
        }
    }

private:
    int64_t numHeads_;
    int64_t headDim_;
    float scale_;
};

// Run benchmark
void runBenchmark(int64_t batchSize, int64_t seqLen, int64_t contextLen, 
                  int64_t numHeads, int64_t headDim, int64_t numTrials) {
    // Allocate tensors for standard MHA
    std::vector<float> query(batchSize * seqLen * numHeads * headDim);
    std::vector<float> key(batchSize * contextLen * numHeads * headDim);
    std::vector<float> value(batchSize * contextLen * numHeads * headDim);
    std::vector<float> output1(batchSize * seqLen * numHeads * headDim);
    
    // For MQA, we need single head for key/value
    std::vector<float> mqaKey(batchSize * contextLen * headDim);
    std::vector<float> mqaValue(batchSize * contextLen * headDim);
    std::vector<float> output2(batchSize * seqLen * numHeads * headDim);
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (auto& q : query) q = dist(rng);
    for (auto& k : key) k = dist(rng);
    for (auto& v : value) v = dist(rng);
    
    // Fill MQA key/value (average of the heads for this test)
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int64_t n = 0; n < contextLen; ++n) {
            for (int64_t d = 0; d < headDim; ++d) {
                float kSum = 0.0f;
                float vSum = 0.0f;
                
                // Average across heads
                for (int64_t h = 0; h < numHeads; ++h) {
                    int64_t idx = ((b * contextLen + n) * numHeads + h) * headDim + d;
                    kSum += key[idx];
                    vSum += value[idx];
                }
                
                int64_t mqaIdx = (b * contextLen + n) * headDim + d;
                mqaKey[mqaIdx] = kSum / numHeads;
                mqaValue[mqaIdx] = vSum / numHeads;
            }
        }
    }
    
    // Create implementations
    MultiQueryAttention mqaAttn(numHeads, headDim);
    MultiHeadAttention mhaAttn(numHeads, headDim);
    
    // Benchmark Multi-Head Attention
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        mhaAttn.compute(
            output1.data(),
            query.data(),
            key.data(),
            value.data(),
            batchSize,
            seqLen,
            contextLen
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mhaTime = end - start;
    
    // Benchmark Multi-Query Attention
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        mqaAttn.compute(
            output2.data(),
            query.data(),
            mqaKey.data(),
            mqaValue.data(),
            batchSize,
            seqLen,
            contextLen
        );
    }
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mqaTime = end - start;
    
    // Calculate memory usage
    size_t mhaMemory = (query.size() + key.size() + value.size() + output1.size()) * sizeof(float);
    size_t mqaMemory = (query.size() + mqaKey.size() + mqaValue.size() + output2.size()) * sizeof(float);
    
    // Calculate FLOPs
    // MHA: (QK^T + AV) * numHeads = 2 * batchSize * numHeads * seqLen * contextLen * headDim * 2
    // MQA: (QK^T + AV) * numHeads but with shared KV = 2 * batchSize * numHeads * seqLen * contextLen * headDim * 1
    double mhaFlops = 2.0 * batchSize * numHeads * seqLen * contextLen * headDim * 2;
    double mqaFlops = 2.0 * batchSize * numHeads * seqLen * contextLen * headDim;
    
    double mhaGFlopsPerSec = (mhaFlops * numTrials) / (mhaTime.count() * 1e9);
    double mqaGFlopsPerSec = (mqaFlops * numTrials) / (mqaTime.count() * 1e9);
    
    // Print results
    std::cout << "==== Multi-Query Attention Benchmark ====" << std::endl;
    std::cout << "Config: batch=" << batchSize 
              << ", seqLen=" << seqLen 
              << ", contextLen=" << contextLen 
              << ", heads=" << numHeads 
              << ", dim=" << headDim << std::endl;
    
    std::cout << "\nStandard Multi-Head Attention:" << std::endl;
    std::cout << "  Time: " << mhaTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << mhaGFlopsPerSec << " GFLOPs/sec" << std::endl;
    std::cout << "  Memory: " << mhaMemory / (1024.0 * 1024.0) << " MB" << std::endl;
    
    std::cout << "\nMulti-Query Attention:" << std::endl;
    std::cout << "  Time: " << mqaTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << mqaGFlopsPerSec << " GFLOPs/sec" << std::endl;
    std::cout << "  Memory: " << mqaMemory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Memory Reduction: " << (1.0 - (double)mqaMemory / mhaMemory) * 100.0 << "%" << std::endl;
    
    std::cout << "\nSpeedup: " << mhaTime.count() / mqaTime.count() << "x" << std::endl;
    
    // Check output quality
    double maxDiff = 0.0;
    double avgDiff = 0.0;
    
    for (size_t i = 0; i < output1.size(); ++i) {
        double diff = std::abs(output1[i] - output2[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
    }
    
    avgDiff /= output1.size();
    
    std::cout << "\nOutput Difference (MHA vs MQA):" << std::endl;
    std::cout << "  Max Difference: " << maxDiff << std::endl;
    std::cout << "  Average Difference: " << avgDiff << std::endl;
}

int main(int argc, char** argv) {
    // Default values
    int64_t batchSize = 2;
    int64_t seqLen = 512;
    int64_t contextLen = 512;
    int64_t numHeads = 12;  // Typical for medium models
    int64_t headDim = 64;
    int64_t numTrials = 5;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--batch") batchSize = std::stoi(argv[i+1]);
            else if (arg == "--seq") seqLen = std::stoi(argv[i+1]);
            else if (arg == "--context") contextLen = std::stoi(argv[i+1]);
            else if (arg == "--heads") numHeads = std::stoi(argv[i+1]);
            else if (arg == "--dim") headDim = std::stoi(argv[i+1]);
            else if (arg == "--trials") numTrials = std::stoi(argv[i+1]);
        }
    }
    
    // Run benchmark
    runBenchmark(batchSize, seqLen, contextLen, numHeads, headDim, numTrials);
    
    return 0;
} 