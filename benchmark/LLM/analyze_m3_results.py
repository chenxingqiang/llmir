#!/usr/bin/env python3
# Simple analysis script for the Apple M3 KVCache benchmark results

import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_benchmark_results(file_path):
    results = []
    
    # Read the entire file into memory
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all benchmark result patterns
    pattern = r'KVCacheAppend/(.*?)/BlockSize=(\d+)/(\d+)/(\d+)/.*?/manual_time_mean\s+(\d+\.\d+)\s+ms.*?Tokens/s=(\d+\.\d+)k/s'
    config_matches = re.findall(pattern, content, re.DOTALL)
    
    # Parse configuration benchmark results
    for match in config_matches:
        config, block_size, batch_size, seq_len, time_ms, tokens_per_sec = match
        results.append({
            'test_type': "BlockSize",
            'benchmark': "KVCacheAppend",
            'config': config,
            'block_size': int(block_size),
            'batch_size': int(batch_size),
            'seq_len': int(seq_len),
            'total_tokens': int(batch_size) * int(seq_len),
            'time_ms': float(time_ms),
            'tokens_per_sec': float(tokens_per_sec) * 1000  # Convert from k/s to /s
        })
        print(f"Found BlockSize result: {config}, Block: {block_size}, Batch: {batch_size}, Seq: {seq_len}")
    
    # Find batch/seq benchmark results
    batch_pattern = r'KVCacheAppend/BatchSize=(\d+)/SeqLen=(\d+)/\d+/\d+/.*?/manual_time_mean\s+(\d+\.\d+)\s+ms.*?Tokens/s=(\d+\.\d+)k/s'
    batch_matches = re.findall(batch_pattern, content, re.DOTALL)
    
    # Parse batch/seq benchmark results
    for match in batch_matches:
        batch_size, seq_len, time_ms, tokens_per_sec = match
        results.append({
            'test_type': "BatchSeqLen",
            'benchmark': "KVCacheAppend",
            'config': 'Pool + Unified(128KB)',  # From the benchmark code
            'block_size': 64,  # Default block size for batch tests
            'batch_size': int(batch_size),
            'seq_len': int(seq_len),
            'total_tokens': int(batch_size) * int(seq_len),
            'time_ms': float(time_ms),
            'tokens_per_sec': float(tokens_per_sec) * 1000  # Convert from k/s to /s
        })
        print(f"Found BatchSeqLen result: Batch: {batch_size}, Seq: {seq_len}")
    
    return pd.DataFrame(results)

def plot_results(df):
    if df.empty:
        print("No data to plot")
        return
        
    print(f"Plotting data with {len(df)} results")
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create summary report
    with open('benchmark_summary.txt', 'w') as f:
        f.write("=== KVCache Benchmark Summary ===\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write(f"Total benchmarks: {len(df)}\n")
        f.write(f"Average tokens/sec: {df['tokens_per_sec'].mean():.2f}\n")
        f.write(f"Max tokens/sec: {df['tokens_per_sec'].max():.2f}\n\n")
        
        # Block size comparison
        if 'BlockSize' in df['test_type'].values:
            block_df = df[df['test_type'] == 'BlockSize']
            f.write("Block Size Performance:\n")
            for block_size in sorted(block_df['block_size'].unique()):
                avg_tokens = block_df[block_df['block_size'] == block_size]['tokens_per_sec'].mean()
                f.write(f"Block size {block_size}: {avg_tokens:.2f} tokens/sec\n")
            f.write("\n")
        
        # Configuration comparison
        f.write("Memory Configuration Performance:\n")
        for config in sorted(df['config'].unique()):
            avg_tokens = df[df['config'] == config]['tokens_per_sec'].mean()
            f.write(f"{config}: {avg_tokens:.2f} tokens/sec\n")
        f.write("\n")
        
        # Batch size and sequence length analysis
        if 'BatchSeqLen' in df['test_type'].values:
            batch_df = df[df['test_type'] == 'BatchSeqLen']
            f.write("Batch Size Performance:\n")
            for batch_size in sorted(batch_df['batch_size'].unique()):
                avg_tokens = batch_df[batch_df['batch_size'] == batch_size]['tokens_per_sec'].mean()
                f.write(f"Batch size {batch_size}: {avg_tokens:.2f} tokens/sec\n")
            f.write("\n")
    
    # Plot 1: Block size vs performance by configuration
    if 'BlockSize' in df['test_type'].values:
        plt.figure(figsize=(14, 8))
        block_df = df[df['test_type'] == 'BlockSize']
        sns.lineplot(data=block_df, x='block_size', y='tokens_per_sec', hue='config', marker='o')
        plt.title('Block Size vs. Performance by Memory Configuration')
        plt.xlabel('Block Size')
        plt.ylabel('Tokens/sec')
        plt.grid(True)
        plt.savefig('block_size_performance.png')
    
    # Plot 2: Batch size and sequence length effect
    if 'BatchSeqLen' in df['test_type'].values:
        batch_df = df[df['test_type'] == 'BatchSeqLen']
        
        plt.figure(figsize=(14, 8))
        # Create a plot with batch size and seq_len
        sns.lineplot(data=batch_df, x='seq_len', y='tokens_per_sec', hue='batch_size', marker='o')
        plt.title('Tokens/sec by Batch Size and Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('Tokens/sec')
        plt.grid(True)
        plt.savefig('batch_seq_performance.png')
    
    # Plot 3: Memory configuration performance
    plt.figure(figsize=(14, 8))
    config_performance = df.groupby('config')['tokens_per_sec'].mean().reset_index()
    config_performance = config_performance.sort_values('tokens_per_sec', ascending=False)
    
    sns.barplot(data=config_performance, x='config', y='tokens_per_sec')
    plt.title('Average Performance by Memory Configuration')
    plt.xlabel('Memory Configuration')
    plt.ylabel('Tokens/sec')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('config_performance.png')
    
    print("Plots saved to current directory.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_m3_results.py <results_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    # Parse benchmark results
    df = parse_benchmark_results(results_file)
    
    if df.empty:
        print("No valid benchmark results found in the file.")
        sys.exit(1)
    
    # Generate plots and summary
    plot_results(df)
    
    print(f"Analysis complete. Summary saved to benchmark_summary.txt")

if __name__ == "__main__":
    main()