#!/bin/bash

# Create results directory
mkdir -p benchmark_results

# Define sequence lengths to test
SEQ_LENGTHS=(128 256 512 1024 2048)

# Run Flash Attention benchmark
echo "Running Flash Attention benchmarks..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Testing sequence length: $seq_len"
    ./simple_attention_benchmark --seq $seq_len --context $seq_len --trials 10 > benchmark_results/flash_attn_seq${seq_len}.txt
done

# Run Multi-Query Attention benchmark
echo "Running Multi-Query Attention benchmarks..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Testing sequence length: $seq_len"
    ./simple_mqa_benchmark --seq $seq_len --context $seq_len --trials 10 > benchmark_results/mqa_seq${seq_len}.txt
done

# Run Pruned Attention benchmark
echo "Running Pruned Attention benchmarks..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Testing sequence length: $seq_len"
    ./simple_pruned_attention_benchmark --seq $seq_len --context $seq_len --trials 10 > benchmark_results/pruned_attn_seq${seq_len}.txt
done

# Run additional benchmarks with varying thresholds
echo "Running Pruned Attention with varying thresholds..."
THRESHOLDS=(0.001 0.01 0.05 0.1)
for threshold in "${THRESHOLDS[@]}"; do
    echo "Testing threshold: $threshold"
    ./simple_pruned_attention_benchmark --seq 512 --context 512 --threshold $threshold --trials 10 > benchmark_results/pruned_attn_threshold${threshold}.txt
done

# Run additional benchmarks with varying topK values
echo "Running Pruned Attention with varying topK values..."
TOPK_VALUES=(32 64 128 256)
for topk in "${TOPK_VALUES[@]}"; do
    echo "Testing topK: $topk"
    ./simple_pruned_attention_benchmark --seq 512 --context 512 --topk $topk --trials 10 > benchmark_results/pruned_attn_topk${topk}.txt
done

echo "All benchmarks completed. Results are in the benchmark_results directory." 