#!/usr/bin/env python3
"""
Block Size Optimization Analysis - Version 2
With LARGER FONTS for better readability (addresses reviewer feedback)
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up larger fonts globally
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

# Data from benchmark results
block_sizes = [16, 32, 64, 128, 256]
throughput = [43479, 43318, 41670, 42181, 48407]
memory_efficiency = [85, 88, 90, 91, 92]

# Colors
bar_color = '#4ECDC4'
line_color = '#E74C3C'
optimal_color = '#2ECC71'

# Left plot: Throughput by block size
bars = ax1.bar(block_sizes, throughput, color=bar_color, edgecolor='#2C3E50', linewidth=2, width=30)
# Highlight optimal block size
bars[4].set_color(optimal_color)

ax1.set_xlabel('Block Size (tokens)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=14, fontweight='bold')
ax1.set_title('Throughput by Block Size\n(LLaMA-2-13B, ShareGPT, A100-80GB)', fontsize=14, fontweight='bold')
ax1.set_xticks(block_sizes)
ax1.set_xticklabels([str(x) for x in block_sizes], fontsize=13)

# Add value labels on bars
for i, (bs, tp) in enumerate(zip(block_sizes, throughput)):
    ax1.annotate(f'{tp:,}', 
                xy=(bs, tp), 
                ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color='#2C3E50')

ax1.axhline(y=max(throughput), color='gray', linestyle='--', alpha=0.5)
ax1.set_ylim(38000, 52000)

# Right plot: Memory efficiency
ax2.plot(block_sizes, memory_efficiency, marker='o', markersize=10, 
         linewidth=3, color=line_color, markeredgecolor='#2C3E50', markeredgewidth=2)

ax2.set_xlabel('Block Size (tokens)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Memory Efficiency (%)', fontsize=14, fontweight='bold')
ax2.set_title('Memory Efficiency by Block Size\n(LLaMA-2-13B, ShareGPT, A100-80GB)', fontsize=14, fontweight='bold')
ax2.set_xticks(block_sizes)
ax2.set_xticklabels([str(x) for x in block_sizes], fontsize=13)
ax2.set_ylim(80, 95)

# Add value labels
for bs, me in zip(block_sizes, memory_efficiency):
    ax2.annotate(f'{me}%', 
                xy=(bs, me), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Add annotation for optimal point
ax2.annotate('Optimal:\n256 blocks', 
            xy=(256, 92), 
            xytext=(200, 86),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2),
            bbox=dict(boxstyle='round', facecolor='#FFEAA7', edgecolor='#2C3E50'))

plt.tight_layout()

plt.savefig('block_size_optimization_v2.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('block_size_optimization_v2.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("Created block_size_optimization_v2.pdf and block_size_optimization_v2.png")
