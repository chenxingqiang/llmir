#!/usr/bin/env python3
"""
Attention Optimization Charts Generator
Creates professional charts for attention optimization benchmarks
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure matplotlib for better fonts and quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Set color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
sns.set_palette(colors)

def create_attention_speedup_chart():
    """Create attention optimization speedup comparison chart"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data from attention benchmarks
    techniques = ['Standard\nAttention', 'Flash\nAttention', 'Fused\nSoftmax',
                  'Optimized\nMasked', 'Sliding\nWindow']
    seq_lengths = [128, 256, 512, 1024, 2048]

    # Speedup data (relative to standard attention)
    speedups = {
        'Standard\nAttention': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Flash\nAttention': [1.28, 1.35, 1.42, 1.58, 1.69],
        'Fused\nSoftmax': [1.36, 1.39, 1.42, 1.45, 1.48],
        'Optimized\nMasked': [1.42, 1.55, 1.68, 1.78, 1.92],
        'Sliding\nWindow': [1.52, 1.68, 1.85, 2.02, 2.15]
    }

    x = np.arange(len(seq_lengths))
    width = 0.15

    for i, technique in enumerate(techniques):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, speedups[technique], width,
                     label=technique, color=colors[i], alpha=0.8)

        # Add value labels on bars for key points
        if technique != 'Standard\nAttention':
            for j, bar in enumerate(bars):
                if j == len(bars) - 1:  # Only label the last bar
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}×', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontweight='bold')
    ax.set_title('Attention Optimization Techniques Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    return fig

def create_memory_efficiency_chart():
    """Create memory efficiency comparison chart"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Memory usage data (GB)
    techniques = ['Standard', 'Flash Attention', 'Fused Softmax', 'Multi-Query', 'Sliding Window']
    seq_lengths = [512, 1024, 2048, 4096]

    memory_usage = {
        'Standard': [2.1, 4.8, 12.5, 28.2],
        'Flash Attention': [2.0, 4.6, 12.1, 27.8],
        'Fused Softmax': [1.5, 3.2, 8.1, 18.5],
        'Multi-Query': [0.8, 1.6, 3.8, 8.9],
        'Sliding Window': [1.2, 2.1, 4.2, 8.1]
    }

    x = np.arange(len(seq_lengths))

    for i, technique in enumerate(techniques):
        ax.plot(x, memory_usage[technique], marker='o', linewidth=2.5,
               markersize=8, label=technique, color=colors[i])

    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Memory Usage (GB)', fontweight='bold')
    ax.set_title('Memory Efficiency Comparison Across Attention Techniques', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()
    return fig

def create_accuracy_impact_chart():
    """Create accuracy impact analysis chart"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy retention data
    techniques = ['Flash\nAttention', 'Fused\nSoftmax', 'Optimized\nMasked',
                  'Sliding\nWindow', 'Threshold\nPruning']
    accuracy_retention = [99.8, 100.0, 100.0, 98.5, 95.2]
    speedup = [1.69, 1.48, 1.92, 2.15, 2.35]

    # Accuracy retention bar chart
    bars = ax1.bar(techniques, accuracy_retention, color=colors[:5], alpha=0.8)
    ax1.set_ylabel('Accuracy Retention (%)', fontweight='bold')
    ax1.set_title('Accuracy Retention by Optimization Technique', fontweight='bold')
    ax1.set_ylim(90, 101)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add value labels
    for bar, acc in zip(bars, accuracy_retention):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    # Accuracy vs Speedup scatter plot
    ax2.scatter(accuracy_retention, speedup, s=150, c=colors[:5], alpha=0.8)

    for i, technique in enumerate(['Flash', 'Fused', 'Masked', 'Sliding', 'Threshold']):
        ax2.annotate(technique, (accuracy_retention[i], speedup[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax2.set_xlabel('Accuracy Retention (%)', fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontweight='bold')
    ax2.set_title('Accuracy vs Performance Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(94, 101)
    ax2.set_ylim(1.3, 2.5)

    plt.tight_layout()
    return fig

def create_block_size_optimization_chart():
    """Create block size optimization analysis chart"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Block size performance data
    block_sizes = [16, 32, 64, 128, 256, 512]
    performance = [43479, 43318, 41670, 42181, 48407, 45234]
    memory_efficiency = [78, 82, 85, 88, 92, 87]

    # Create dual y-axis plot
    ax2 = ax.twinx()

    # Performance bars
    bars = ax.bar(block_sizes, performance, alpha=0.7, color=colors[0],
                  label='Performance (tokens/sec)')

    # Memory efficiency line
    line = ax2.plot(block_sizes, memory_efficiency, color=colors[1],
                   marker='o', linewidth=3, markersize=8,
                   label='Memory Efficiency (%)')

    ax.set_xlabel('Block Size', fontweight='bold')
    ax.set_ylabel('Performance (tokens/sec)', fontweight='bold', color=colors[0])
    ax2.set_ylabel('Memory Efficiency (%)', fontweight='bold', color=colors[1])
    ax.set_title('Block Size Optimization Analysis', fontweight='bold')

    # Highlight optimal block size
    optimal_idx = np.argmax(performance)
    bars[optimal_idx].set_color(colors[2])
    bars[optimal_idx].set_alpha(1.0)

    ax.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig

def main():
    """Generate all attention optimization charts"""

    print("Generating attention optimization charts...")

    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)

    # Generate charts
    charts = [
        ('attention_speedup_comparison', create_attention_speedup_chart()),
        ('attention_memory_efficiency', create_memory_efficiency_chart()),
        ('attention_accuracy_impact', create_accuracy_impact_chart()),
        ('block_size_optimization', create_block_size_optimization_chart())
    ]

    for name, fig in charts:
        fig.savefig(f'figures/{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
        print(f"✓ Generated {name}")
        plt.close(fig)

    print("\nAll attention optimization charts generated successfully!")

if __name__ == "__main__":
    main()