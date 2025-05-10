# Attention Optimization Test Procedure

This document outlines the procedure for testing and evaluating different attention optimization techniques for LLM inference in the LLMIR project.

## 1. Test Objectives

The primary objectives of this test are:

1. Evaluate the performance benefits of different attention optimization techniques:
   - Flash Attention vs Standard Attention
   - Multi-Query Attention (MQA) vs Standard Multi-Head Attention (MHA)
   - Pruned Attention (Threshold-based and Top-K) vs Standard Attention

2. Measure the impact of these optimizations on:
   - Execution speed
   - Memory usage
   - Output accuracy
   - Scaling behavior with increasing sequence lengths

3. Determine the optimal parameters for pruning techniques:
   - Threshold values for threshold-based pruning
   - K values for Top-K pruning

## 2. Test Environment

- Hardware: Mac M3 ARM processor
- Compiler: g++ with -O3 optimization
- Project: LLMIR with standalone attention benchmarks

## 3. Benchmark Implementations

Three standalone benchmark implementations have been created:

1. **simple_attention_benchmark.cpp**: Compares Flash Attention vs Standard Attention
2. **simple_mqa_benchmark.cpp**: Compares Multi-Query Attention vs Standard Multi-Head Attention
3. **simple_pruned_attention_benchmark.cpp**: Compares Threshold-based and Top-K Pruning vs Standard Attention

## 4. Test Parameters

The benchmarks will be run with the following parameters:

- Batch size: 2
- Number of attention heads: 8-12
- Head dimension: 64
- Sequence lengths: 128, 256, 512, 1024, 2048
- Pruning thresholds: 0.001, 0.01, 0.05, 0.1
- Top-K values: 32, 64, 128, 256
- Number of trials per test: 10

## 5. Test Procedure

### 5.1 Pre-Test Setup

1. Compile the benchmark executables:
   ```bash
   g++ -std=c++17 -O3 simple_attention_benchmark.cpp -o simple_attention_benchmark
   g++ -std=c++17 -O3 simple_mqa_benchmark.cpp -o simple_mqa_benchmark
   g++ -std=c++17 -O3 simple_pruned_attention_benchmark.cpp -o simple_pruned_attention_benchmark
   ```

2. Create a directory for benchmark results:
   ```bash
   mkdir -p benchmark_results
   ```

### 5.2 Running the Benchmarks

All benchmarks can be run automatically using the provided script:

```bash
./run_benchmarks.sh
```

The script will:
1. Run Flash Attention benchmarks for all sequence lengths
2. Run Multi-Query Attention benchmarks for all sequence lengths
3. Run Pruned Attention benchmarks for all sequence lengths
4. Run Threshold Pruned Attention with varying threshold values
5. Run Top-K Pruned Attention with varying K values

All results will be saved to text files in the `benchmark_results` directory.

### 5.3 Alternative: Manual Benchmark Execution

If needed, benchmarks can be run manually with custom parameters:

```bash
# Flash Attention benchmark
./simple_attention_benchmark --batch 2 --seq 512 --context 512 --heads 8 --dim 64 --trials 10

# Multi-Query Attention benchmark
./simple_mqa_benchmark --batch 2 --seq 512 --context 512 --heads 12 --dim 64 --trials 10

# Pruned Attention benchmark
./simple_pruned_attention_benchmark --batch 2 --seq 512 --context 512 --heads 8 --dim 64 --threshold 0.01 --topk 128 --trials 10
```

Parameters:
- `--batch`: Batch size
- `--seq`: Sequence length (query length)
- `--context`: Context length (key/value length)
- `--heads`: Number of attention heads
- `--dim`: Head dimension
- `--threshold`: Pruning threshold (for pruned attention)
- `--topk`: K value for Top-K pruning (for pruned attention)
- `--trials`: Number of trials to average over

### 5.4 Analyzing Results

After running the benchmarks, parse the results and generate visualizations:

```bash
python parse_results.py
```

This script will:
1. Parse all benchmark result files in the `benchmark_results` directory
2. Generate comparative visualizations in the `plots` directory
3. Create a comprehensive test report in Markdown format

## 6. Expected Outputs

The test procedure will generate:

1. Raw benchmark results in text files (in `benchmark_results/`)
2. Visualization plots (in `plots/`):
   - Speedup vs sequence length comparison
   - Memory reduction for MQA
   - Accuracy comparison across methods
   - Effect of pruning threshold
   - Effect of Top-K value
   - Execution time comparison

3. A comprehensive test report (`attention_optimization_test_report.md`) with analysis and recommendations

## 7. Evaluation Criteria

The attention optimization techniques will be evaluated based on:

1. **Performance**: Speedup relative to standard attention
2. **Memory Efficiency**: Reduction in memory usage
3. **Accuracy**: Deviation from standard attention results
4. **Scalability**: How benefits scale with increasing sequence length

## 8. Troubleshooting

- If benchmarks run too slowly, reduce the sequence length or number of trials
- If parsing fails, check the raw result files for unexpected format changes
- For more detailed timing information, consider using a profiling tool

## 9. Notes

- All benchmarks use the same random seed (42) for reproducibility
- The benchmark implementations are simplified versions for demonstration and may not include all optimizations of production implementations
- The accuracy comparison assumes standard attention as ground truth, which is appropriate for these tests 