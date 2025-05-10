#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <memory>
#include <algorithm>
#include <string>

// Simple implementation of Flash Attention algorithm for benchmarking
// This is a standalone implementation that doesn't depend on external libraries
class FlashAttention {
public:
    FlashAttention(int64_t numHeads, int64_t headDim, int64_t blockSizeM = 64, int64_t blockSizeN = 64) 
        : numHeads_(numHeads), headDim_(headDim), blockSizeM_(blockSizeM), blockSizeN_(blockSizeN) {
        // Scale factor for attention
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute attention using the Flash Attention algorithm
    void compute(
        float* output,        // [batch, seqLen, numHeads, headDim]
        const float* query,   // [batch, seqLen, numHeads, headDim]
        const float* key,     // [batch, contextLen, numHeads, headDim]
        const float* value,   // [batch, contextLen, numHeads, headDim]
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen
    ) {
        // Temporary storage for softmax normalization
        std::vector<float> softmaxSums(batchSize * seqLen * numHeads_, 0.0f);
        std::vector<float> maxValues(batchSize * seqLen * numHeads_, -std::numeric_limits<float>::infinity());

        // Initialize output to zeros
        std::fill(output, output + batchSize * seqLen * numHeads_ * headDim_, 0.0f);

        // Process blocks of the matrices
        for (int64_t bn = 0; bn < contextLen; bn += blockSizeN_) {
            int64_t actualBlockSizeN = std::min(blockSizeN_, contextLen - bn);
            
            for (int64_t bm = 0; bm < seqLen; bm += blockSizeM_) {
                int64_t actualBlockSizeM = std::min(blockSizeM_, seqLen - bm);
                
                // Process each batch and head
                for (int64_t b = 0; b < batchSize; ++b) {
                    for (int64_t h = 0; h < numHeads_; ++h) {
                        // Block-based attention computation
                        computeAttentionBlock(
                            output, query, key, value,
                            softmaxSums.data(), maxValues.data(),
                            b, h, bm, bn,
                            actualBlockSizeM, actualBlockSizeN,
                            batchSize, seqLen, contextLen
                        );
                    }
                }
            }
        }
        
        // Normalize the output by the softmax sum
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t m = 0; m < seqLen; ++m) {
                for (int64_t h = 0; h < numHeads_; ++h) {
                    int64_t outIdx = ((b * seqLen + m) * numHeads_ + h) * headDim_;
                    int64_t sumIdx = (b * seqLen + m) * numHeads_ + h;
                    float sum = softmaxSums[sumIdx];
                    
                    if (sum > 0) {
                        for (int64_t d = 0; d < headDim_; ++d) {
                            output[outIdx + d] /= sum;
                        }
                    }
                }
            }
        }
    }

private:
    int64_t numHeads_;
    int64_t headDim_;
    int64_t blockSizeM_;
    int64_t blockSizeN_;
    float scale_;

    // Compute attention for a block of the matrices
    void computeAttentionBlock(
        float* output,        // [batch, seqLen, numHeads, headDim]
        const float* query,   // [batch, seqLen, numHeads, headDim]
        const float* key,     // [batch, contextLen, numHeads, headDim]
        const float* value,   // [batch, contextLen, numHeads, headDim]
        float* softmaxSums,
        float* maxValues,
        int64_t b, int64_t h, 
        int64_t bm, int64_t bn,
        int64_t blockSizeM, int64_t blockSizeN,
        int64_t batchSize, int64_t seqLen, int64_t contextLen
    ) {
        // Compute S = Q * K^T for this block
        std::vector<float> S(blockSizeM * blockSizeN, 0.0f);
        
        for (int64_t m = 0; m < blockSizeM; ++m) {
            for (int64_t n = 0; n < blockSizeN; ++n) {
                int64_t qIdx = ((b * seqLen + (bm + m)) * numHeads_ + h) * headDim_;
                int64_t kIdx = ((b * contextLen + (bn + n)) * numHeads_ + h) * headDim_;
                float dotProduct = 0.0f;
                
                // Compute dot product
                for (int64_t d = 0; d < headDim_; ++d) {
                    dotProduct += query[qIdx + d] * key[kIdx + d];
                }
                
                // Apply scale factor
                S[m * blockSizeN + n] = dotProduct * scale_;
                
                // Apply causal mask if needed (assuming causal attention)
                if ((bm + m) < (bn + n)) {
                    S[m * blockSizeN + n] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        
        // Apply softmax to S
        for (int64_t m = 0; m < blockSizeM; ++m) {
            int64_t sumIdx = (b * seqLen + (bm + m)) * numHeads_ + h;
            float& maxValue = maxValues[sumIdx];
            
            // Find max for numerical stability
            for (int64_t n = 0; n < blockSizeN; ++n) {
                float value = S[m * blockSizeN + n];
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            
            // Compute exp(S - max) and sum
            float sum = 0.0f;
            for (int64_t n = 0; n < blockSizeN; ++n) {
                float expValue = 0.0f;
                if (S[m * blockSizeN + n] != -std::numeric_limits<float>::infinity()) {
                    expValue = std::exp(S[m * blockSizeN + n] - maxValue);
                }
                S[m * blockSizeN + n] = expValue;
                sum += expValue;
            }
            
            // Update softmax sums
            softmaxSums[sumIdx] += sum;
        }
        
        // Compute O = S * V for this block
        for (int64_t m = 0; m < blockSizeM; ++m) {
            for (int64_t d = 0; d < headDim_; ++d) {
                int64_t outIdx = ((b * seqLen + (bm + m)) * numHeads_ + h) * headDim_ + d;
                
                for (int64_t n = 0; n < blockSizeN; ++n) {
                    int64_t vIdx = ((b * contextLen + (bn + n)) * numHeads_ + h) * headDim_ + d;
                    output[outIdx] += S[m * blockSizeN + n] * value[vIdx];
                }
            }
        }
    }
};

// Simple implementation of standard attention for comparison
class StandardAttention {
public:
    StandardAttention(int64_t numHeads, int64_t headDim) 
        : numHeads_(numHeads), headDim_(headDim) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute standard attention
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
    // Allocate tensors
    std::vector<float> query(batchSize * seqLen * numHeads * headDim);
    std::vector<float> key(batchSize * contextLen * numHeads * headDim);
    std::vector<float> value(batchSize * contextLen * numHeads * headDim);
    std::vector<float> output1(batchSize * seqLen * numHeads * headDim);
    std::vector<float> output2(batchSize * seqLen * numHeads * headDim);
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (auto& q : query) q = dist(rng);
    for (auto& k : key) k = dist(rng);
    for (auto& v : value) v = dist(rng);
    
    // Create implementations
    FlashAttention flashAttn(numHeads, headDim);
    StandardAttention stdAttn(numHeads, headDim);
    
    // Benchmark Flash Attention
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        flashAttn.compute(
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
    std::chrono::duration<double> flashTime = end - start;
    
    // Benchmark Standard Attention
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        stdAttn.compute(
            output2.data(),
            query.data(),
            key.data(),
            value.data(),
            batchSize,
            seqLen,
            contextLen
        );
    }
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> stdTime = end - start;
    
    // Calculate FLOPs
    // QK^T: 2 * batchSize * numHeads * seqLen * contextLen * headDim
    // Softmax: ~5 * batchSize * numHeads * seqLen * contextLen
    // AV: 2 * batchSize * numHeads * seqLen * contextLen * headDim
    double flops = 2.0 * batchSize * numHeads * seqLen * contextLen * headDim +
                  5.0 * batchSize * numHeads * seqLen * contextLen +
                  2.0 * batchSize * numHeads * seqLen * contextLen * headDim;
    
    double flashGFlopsPerSec = (flops * numTrials) / (flashTime.count() * 1e9);
    double stdGFlopsPerSec = (flops * numTrials) / (stdTime.count() * 1e9);
    
    // Print results
    std::cout << "==== Attention Benchmark ====" << std::endl;
    std::cout << "Config: batch=" << batchSize 
              << ", seqLen=" << seqLen 
              << ", contextLen=" << contextLen 
              << ", heads=" << numHeads 
              << ", dim=" << headDim << std::endl;
    
    std::cout << "\nFlash Attention:" << std::endl;
    std::cout << "  Time: " << flashTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << flashGFlopsPerSec << " GFLOPs/sec" << std::endl;
    
    std::cout << "\nStandard Attention:" << std::endl;
    std::cout << "  Time: " << stdTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << stdGFlopsPerSec << " GFLOPs/sec" << std::endl;
    
    std::cout << "\nSpeedup: " << stdTime.count() / flashTime.count() << "x" << std::endl;
    
    // Check correctness (approximate comparison)
    double maxDiff = 0.0;
    double sumDiff = 0.0;
    int diffCount = 0;
    
    for (size_t i = 0; i < output1.size(); ++i) {
        double diff = std::abs(output1[i] - output2[i]);
        maxDiff = std::max(maxDiff, diff);
        sumDiff += diff;
        if (diff > 1e-4) {
            diffCount++;
        }
    }
    
    std::cout << "\nCorrectness Check:" << std::endl;
    std::cout << "  Max difference: " << maxDiff << std::endl;
    std::cout << "  Average difference: " << sumDiff / output1.size() << std::endl;
    std::cout << "  Elements with diff > 1e-4: " << diffCount 
              << " (" << (100.0 * diffCount / output1.size()) << "%)" << std::endl;
}

int main(int argc, char** argv) {
    // Default values
    int64_t batchSize = 2;
    int64_t seqLen = 512;
    int64_t contextLen = 512;
    int64_t numHeads = 8;
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