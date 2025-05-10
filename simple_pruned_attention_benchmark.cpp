#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <memory>
#include <algorithm>
#include <string>

// Standard Attention implementation for comparison
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

// Threshold-based pruned attention implementation
class ThresholdPrunedAttention {
public:
    ThresholdPrunedAttention(int64_t numHeads, int64_t headDim, float threshold) 
        : numHeads_(numHeads), headDim_(headDim), threshold_(threshold) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute threshold-pruned attention (prune low attention weights)
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
                    
                    // Apply threshold pruning
                    for (int64_t n = 0; n < contextLen; ++n) {
                        if (scores[n] < threshold_) {
                            scores[n] = 0.0f;
                        }
                    }
                    
                    // Renormalize after pruning
                    sum = 0.0f;
                    for (int64_t n = 0; n < contextLen; ++n) {
                        sum += scores[n];
                    }
                    
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
                            if (scores[n] > 0) {  // Only compute for non-pruned
                                int64_t vIdx = ((b * contextLen + n) * numHeads_ + h) * headDim_ + d;
                                output[outIdx + d] += scores[n] * value[vIdx];
                            }
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
    float threshold_;
};

// Top-K pruned attention implementation
class TopKPrunedAttention {
public:
    TopKPrunedAttention(int64_t numHeads, int64_t headDim, int64_t topK) 
        : numHeads_(numHeads), headDim_(headDim), topK_(topK) {
        scale_ = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Compute Top-K pruned attention (keep only top K scores)
    void compute(
        float* output,        // [batch, seqLen, numHeads, headDim]
        const float* query,   // [batch, seqLen, numHeads, headDim]
        const float* key,     // [batch, contextLen, numHeads, headDim]
        const float* value,   // [batch, contextLen, numHeads, headDim]
        int64_t batchSize,
        int64_t seqLen,
        int64_t contextLen
    ) {
        int64_t actualTopK = std::min(topK_, contextLen);
        
        // For each batch and head
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t h = 0; h < numHeads_; ++h) {
                // For each query position
                for (int64_t m = 0; m < seqLen; ++m) {
                    int64_t qIdx = ((b * seqLen + m) * numHeads_ + h) * headDim_;
                    
                    // Compute attention scores
                    std::vector<std::pair<float, int64_t>> scoreIndices(contextLen);
                    float maxScore = -std::numeric_limits<float>::infinity();
                    
                    // Compute scores and store with indices
                    for (int64_t n = 0; n < contextLen; ++n) {
                        int64_t kIdx = ((b * contextLen + n) * numHeads_ + h) * headDim_;
                        float dotProduct = 0.0f;
                        
                        for (int64_t d = 0; d < headDim_; ++d) {
                            dotProduct += query[qIdx + d] * key[kIdx + d];
                        }
                        
                        float score = dotProduct * scale_;
                        
                        // Apply causal mask
                        if (m < n) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                        
                        scoreIndices[n] = {score, n};
                        if (score > maxScore) {
                            maxScore = score;
                        }
                    }
                    
                    // Sort by scores in descending order
                    std::sort(scoreIndices.begin(), scoreIndices.end(), 
                             [](const auto& a, const auto& b) {
                                 return a.first > b.first;
                             });
                    
                    // Apply softmax to top-K elements
                    std::vector<float> topKScores(contextLen, 0.0f);
                    float sum = 0.0f;
                    
                    for (int64_t k = 0; k < actualTopK; ++k) {
                        auto [score, idx] = scoreIndices[k];
                        if (score != -std::numeric_limits<float>::infinity()) {
                            float expValue = std::exp(score - maxScore);
                            topKScores[idx] = expValue;
                            sum += expValue;
                        }
                    }
                    
                    // Normalize
                    if (sum > 0) {
                        for (int64_t n = 0; n < contextLen; ++n) {
                            if (topKScores[n] > 0) {
                                topKScores[n] /= sum;
                            }
                        }
                    }
                    
                    // Compute weighted sum of values
                    int64_t outIdx = ((b * seqLen + m) * numHeads_ + h) * headDim_;
                    for (int64_t d = 0; d < headDim_; ++d) {
                        output[outIdx + d] = 0.0f;
                        
                        for (int64_t n = 0; n < contextLen; ++n) {
                            if (topKScores[n] > 0) {  // Only compute for non-pruned
                                int64_t vIdx = ((b * contextLen + n) * numHeads_ + h) * headDim_ + d;
                                output[outIdx + d] += topKScores[n] * value[vIdx];
                            }
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
    int64_t topK_;
};

// Run benchmark
void runBenchmark(int64_t batchSize, int64_t seqLen, int64_t contextLen, 
                  int64_t numHeads, int64_t headDim, float threshold, int64_t topK,
                  int64_t numTrials) {
    // Allocate tensors
    std::vector<float> query(batchSize * seqLen * numHeads * headDim);
    std::vector<float> key(batchSize * contextLen * numHeads * headDim);
    std::vector<float> value(batchSize * contextLen * numHeads * headDim);
    std::vector<float> output1(batchSize * seqLen * numHeads * headDim);  // Standard
    std::vector<float> output2(batchSize * seqLen * numHeads * headDim);  // Threshold pruned
    std::vector<float> output3(batchSize * seqLen * numHeads * headDim);  // Top-K pruned
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    for (auto& q : query) q = dist(rng);
    for (auto& k : key) k = dist(rng);
    for (auto& v : value) v = dist(rng);
    
    // Create implementations
    StandardAttention stdAttn(numHeads, headDim);
    ThresholdPrunedAttention threshAttn(numHeads, headDim, threshold);
    TopKPrunedAttention topKAttn(numHeads, headDim, topK);
    
    // Benchmark standard attention
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        stdAttn.compute(
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
    std::chrono::duration<double> stdTime = end - start;
    
    // Benchmark threshold-pruned attention
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        threshAttn.compute(
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
    std::chrono::duration<double> threshTime = end - start;
    
    // Benchmark top-K pruned attention
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; ++i) {
        topKAttn.compute(
            output3.data(),
            query.data(),
            key.data(),
            value.data(),
            batchSize,
            seqLen,
            contextLen
        );
    }
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> topKTime = end - start;
    
    // Calculate pruning stats for threshold-based pruning
    double threshPruneRatio = 0.0;
    int64_t threshZeroCount = 0;
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int64_t m = 0; m < seqLen; ++m) {
            for (int64_t h = 0; h < numHeads; ++h) {
                // Compare outputs to estimate pruning
                int64_t outIdx = ((b * seqLen + m) * numHeads + h) * headDim;
                for (int64_t d = 0; d < headDim; ++d) {
                    if (std::abs(output2[outIdx + d]) < 1e-6) {
                        threshZeroCount++;
                    }
                }
            }
        }
    }
    threshPruneRatio = static_cast<double>(threshZeroCount) / output2.size();
    
    // Calculate topK pruning ratio
    double topKPruneRatio = 1.0 - static_cast<double>(topK) / contextLen;
    
    // Calculate FLOPs
    double stdFlops = 2.0 * batchSize * numHeads * seqLen * contextLen * headDim * 2;
    double threshFlops = stdFlops * (1.0 - threshPruneRatio);
    double topKFlops = stdFlops * (1.0 - topKPruneRatio);
    
    double stdGFlopsPerSec = (stdFlops * numTrials) / (stdTime.count() * 1e9);
    double threshGFlopsPerSec = (threshFlops * numTrials) / (threshTime.count() * 1e9);
    double topKGFlopsPerSec = (topKFlops * numTrials) / (topKTime.count() * 1e9);
    
    // Print results
    std::cout << "==== Pruned Attention Benchmark ====" << std::endl;
    std::cout << "Config: batch=" << batchSize 
              << ", seqLen=" << seqLen 
              << ", contextLen=" << contextLen 
              << ", heads=" << numHeads 
              << ", dim=" << headDim << std::endl;
    std::cout << "Threshold: " << threshold << ", TopK: " << topK << std::endl;
    
    std::cout << "\nStandard Attention:" << std::endl;
    std::cout << "  Time: " << stdTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << stdGFlopsPerSec << " GFLOPs/sec" << std::endl;
    
    std::cout << "\nThreshold-Pruned Attention:" << std::endl;
    std::cout << "  Time: " << threshTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << threshGFlopsPerSec << " GFLOPs/sec" << std::endl;
    std::cout << "  Pruning Ratio: ~" << (threshPruneRatio * 100.0) << "%" << std::endl;
    std::cout << "  Speedup: " << stdTime.count() / threshTime.count() << "x" << std::endl;
    
    std::cout << "\nTop-K Pruned Attention:" << std::endl;
    std::cout << "  Time: " << topKTime.count() / numTrials * 1000 << " ms per run" << std::endl;
    std::cout << "  Performance: " << topKGFlopsPerSec << " GFLOPs/sec" << std::endl;
    std::cout << "  Pruning Ratio: " << (topKPruneRatio * 100.0) << "%" << std::endl;
    std::cout << "  Speedup: " << stdTime.count() / topKTime.count() << "x" << std::endl;
    
    // Check output quality
    std::cout << "\nOutput Differences:" << std::endl;
    
    double threshMaxDiff = 0.0;
    double threshAvgDiff = 0.0;
    double topKMaxDiff = 0.0;
    double topKAvgDiff = 0.0;
    
    for (size_t i = 0; i < output1.size(); ++i) {
        double threshDiff = std::abs(output1[i] - output2[i]);
        double topKDiff = std::abs(output1[i] - output3[i]);
        
        threshMaxDiff = std::max(threshMaxDiff, threshDiff);
        threshAvgDiff += threshDiff;
        
        topKMaxDiff = std::max(topKMaxDiff, topKDiff);
        topKAvgDiff += topKDiff;
    }
    
    threshAvgDiff /= output1.size();
    topKAvgDiff /= output1.size();
    
    std::cout << "  Threshold-Pruned vs Standard:" << std::endl;
    std::cout << "    Max Difference: " << threshMaxDiff << std::endl;
    std::cout << "    Average Difference: " << threshAvgDiff << std::endl;
    
    std::cout << "  Top-K Pruned vs Standard:" << std::endl;
    std::cout << "    Max Difference: " << topKMaxDiff << std::endl;
    std::cout << "    Average Difference: " << topKAvgDiff << std::endl;
}

int main(int argc, char** argv) {
    // Default values
    int64_t batchSize = 2;
    int64_t seqLen = 128;
    int64_t contextLen = 512;
    int64_t numHeads = 8;
    int64_t headDim = 64;
    float threshold = 0.01f;  // Keep attention weights > 1%
    int64_t topK = 128;       // Keep top 128 scores (25% of context for default)
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
            else if (arg == "--threshold") threshold = std::stof(argv[i+1]);
            else if (arg == "--topk") topK = std::stoi(argv[i+1]);
            else if (arg == "--trials") numTrials = std::stoi(argv[i+1]);
        }
    }
    
    // Run benchmark
    runBenchmark(batchSize, seqLen, contextLen, numHeads, headDim, threshold, topK, numTrials);
    
    return 0;
} 