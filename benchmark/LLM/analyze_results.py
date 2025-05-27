#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze and visualize results from the Llama-3 KVCache benchmark.
This script processes the benchmark output and generates plots for comparison.
"""

import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_benchmark_output(filename):
    """Parse the benchmark output file and extract results into a DataFrame."""
    with open(filename, 'r') as f:
        content = f.read()

    # Extract benchmark results
    pattern = r'(BM_KVCache\w+/[^/]+/BlockSize=\d+|BM_KVCache\w+/BatchSize=\d+/SeqLen=\d+)\s+(\d+) ms\s+\d+ ms\s+\d+ ms\s+(\d+) ms\s+(.+)'
    matches = re.findall(pattern, content)

    data = []
    for match in matches:
        name, iterations, time_ms, counters = match

        # Parse name components
        components = name.split('/')
        benchmark_type = components[0]

        if 'BlockSize' in components[1]:
            # Format: BM_KVCacheXXX/ConfigName/BlockSize=YYY
            config_name = components[1]
            block_size = int(components[2].split('=')[1])
            batch_size = 4  # Default
            seq_len = 1024  # Default
        else:
            # Format: BM_KVCacheXXX/BatchSize=YYY/SeqLen=ZZZ
            config_name = "Pool + Unified(128KB)"  # Default optimal config
            block_size = 64  # Default
            batch_size = int(components[1].split('=')[1])
            seq_len = int(components[2].split('=')[1])

        # Parse counters
        tokens_per_sec = 0
        mb_per_sec = 0
        gflops_per_sec = 0

        if 'Tokens/s=' in counters:
            tokens_per_sec = float(re.search(r'Tokens/s=([0-9.]+)', counters).group(1))

        if 'MB/s=' in counters:
            mb_per_sec = float(re.search(r'MB/s=([0-9.]+)', counters).group(1))

        if 'GFLOP/s=' in counters:
            gflops_per_sec = float(re.search(r'GFLOP/s=([0-9.]+)', counters).group(1))

        data.append({
            'benchmark': benchmark_type,
            'config': config_name,
            'block_size': block_size,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'time_ms': float(time_ms),
            'tokens_per_sec': tokens_per_sec,
            'mb_per_sec': mb_per_sec,
            'gflops_per_sec': gflops_per_sec
        })

    return pd.DataFrame(data)

def plot_block_size_comparison(df, metric='tokens_per_sec', benchmark_type='BM_KVCacheAppend'):
    """Plot comparison of different block sizes across GPU memory configurations."""
    subset = df[df['benchmark'] == benchmark_type]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=subset, x='block_size', y=metric, hue='config')
    plt.title(f'{benchmark_type}: {metric} vs Block Size')
    plt.xlabel('Block Size')

    if metric == 'tokens_per_sec':
        plt.ylabel('Tokens/s')
    elif metric == 'mb_per_sec':
        plt.ylabel('MB/s')
    elif metric == 'gflops_per_sec':
        plt.ylabel('GFLOP/s')
    else:
        plt.ylabel(metric)

    plt.legend(title='GPU Memory Config')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f'{benchmark_type}_{metric}_vs_block_size.png', dpi=300)
    plt.close()

def plot_config_comparison(df, benchmark_type='BM_KVCacheAppend'):
    """Plot comparison of different GPU memory configurations."""
    subset = df[df['benchmark'] == benchmark_type]
    optimal_block_size = 64  # Based on typical results

    subset = subset[subset['block_size'] == optimal_block_size]

    # Calculate speedup relative to no optimizations
    baseline = subset[subset['config'] == 'No optimizations']['tokens_per_sec'].values[0]
    subset['speedup'] = subset['tokens_per_sec'] / baseline

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=subset, x='config', y='speedup')
    plt.title(f'{benchmark_type}: Performance Speedup by GPU Memory Configuration')
    plt.xlabel('GPU Memory Configuration')
    plt.ylabel('Speedup (relative to no optimizations)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}x',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{benchmark_type}_config_speedup.png', dpi=300)
    plt.close()

def plot_batch_seq_heatmap(df, benchmark_type='BM_KVCacheAppend'):
    """Create heatmap showing performance across batch sizes and sequence lengths."""
    subset = df[(df['benchmark'] == benchmark_type) &
                (df['batch_size'] != 4 or df['seq_len'] != 1024)]

    pivot = subset.pivot_table(
        index='batch_size',
        columns='seq_len',
        values='tokens_per_sec',
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
    plt.title(f'{benchmark_type}: Tokens/s by Batch Size and Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(f'{benchmark_type}_batch_seq_heatmap.png', dpi=300)
    plt.close()

def generate_summary_report(df):
    """Generate a summary report of benchmark results."""
    report = []

    # Overall best configuration
    for benchmark_type in df['benchmark'].unique():
        subset = df[df['benchmark'] == benchmark_type]

        # Find optimal block size
        block_size_perf = subset.groupby('block_size')['tokens_per_sec'].mean().reset_index()
        best_block_size = block_size_perf.loc[block_size_perf['tokens_per_sec'].idxmax(), 'block_size']

        # Find optimal configuration
        config_perf = subset.groupby('config')['tokens_per_sec'].mean().reset_index()
        best_config = config_perf.loc[config_perf['tokens_per_sec'].idxmax(), 'config']

        # Calculate memory metrics for the best configuration
        best_subset = subset[(subset['block_size'] == best_block_size) &
                             (subset['config'] == best_config)]

        if len(best_subset) > 0:
            best_row = best_subset.iloc[0]

            # Calculate improvement over baseline (no optimizations)
            baseline = subset[(subset['block_size'] == best_block_size) &
                              (subset['config'] == 'No optimizations')]

            if len(baseline) > 0:
                baseline_perf = baseline.iloc[0]['tokens_per_sec']
                improvement = (best_row['tokens_per_sec'] / baseline_perf - 1) * 100
            else:
                improvement = 0

            report.append(f"Benchmark: {benchmark_type}")
            report.append(f"  Optimal Block Size: {best_block_size}")
            report.append(f"  Optimal GPU Config: {best_config}")
            report.append(f"  Performance: {best_row['tokens_per_sec']:.0f} tokens/s")
            report.append(f"  Improvement over baseline: {improvement:.1f}%")

            if 'gflops_per_sec' in best_row and best_row['gflops_per_sec'] > 0:
                report.append(f"  Compute Performance: {best_row['gflops_per_sec']:.1f} GFLOP/s")

            report.append("")

    # Write report to file
    with open('benchmark_summary.txt', 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Analyze KVCache benchmark results')
    parser.add_argument('filename', help='Benchmark output file')
    args = parser.parse_args()

    # Parse benchmark results
    df = parse_benchmark_output(args.filename)

    # Generate plots
    for benchmark_type in df['benchmark'].unique():
        plot_block_size_comparison(df, 'tokens_per_sec', benchmark_type)
        plot_block_size_comparison(df, 'mb_per_sec', benchmark_type)

        if benchmark_type == 'BM_KVCacheAttention':
            plot_block_size_comparison(df, 'gflops_per_sec', benchmark_type)

        plot_config_comparison(df, benchmark_type)

        # Only create heatmap if we have batch/seq variation data
        batch_seq_data = df[(df['benchmark'] == benchmark_type) &
                            ((df['batch_size'] != 4) | (df['seq_len'] != 1024))]
        if len(batch_seq_data) > 0:
            plot_batch_seq_heatmap(df, benchmark_type)

    # Generate summary report
    generate_summary_report(df)

if __name__ == '__main__':
    main()