#!/usr/bin/env python3
"""
Attention Optimization Speedup Comparison - Version 2
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

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

# Sequence lengths
seq_lengths = [128, 256, 512, 1024, 2048, 4096]
x = np.arange(len(seq_lengths))
width = 0.15

# Speedup data for different techniques
flash_attention = [1.28, 1.35, 1.48, 1.58, 1.65, 1.69]
fused_softmax = [1.15, 1.22, 1.32, 1.40, 1.45, 1.48]
optimized_masked = [1.45, 1.55, 1.68, 1.78, 1.85, 1.92]
sliding_window = [1.10, 1.25, 1.55, 1.78, 1.95, 2.15]
multi_query = [1.35, 1.45, 1.58, 1.68, 1.78, 1.85]

# Colors with better contrast
colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#E74C3C', '#9B59B6']

# Create grouped bar chart
bars1 = ax.bar(x - 2*width, flash_attention, width, label='Flash Attention', 
               color=colors[0], edgecolor='#2C3E50', linewidth=1.5)
bars2 = ax.bar(x - width, fused_softmax, width, label='Fused Softmax', 
               color=colors[1], edgecolor='#2C3E50', linewidth=1.5)
bars3 = ax.bar(x, optimized_masked, width, label='Optimized Masked', 
               color=colors[2], edgecolor='#2C3E50', linewidth=1.5)
bars4 = ax.bar(x + width, sliding_window, width, label='Sliding Window', 
               color=colors[3], edgecolor='#2C3E50', linewidth=1.5)
bars5 = ax.bar(x + 2*width, multi_query, width, label='Multi-Query', 
               color=colors[4], edgecolor='#2C3E50', linewidth=1.5)

# Labels and formatting
ax.set_xlabel('Sequence Length (tokens)', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup vs Baseline', fontsize=14, fontweight='bold')
ax.set_title('Attention Optimization Speedup by Technique\n(A100-80GB, head_dim=128, batch_size=8)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seq_lengths], fontsize=13)

# Baseline reference line
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')

# Legend with larger font
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Y-axis limits
ax.set_ylim(0.9, 2.3)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Annotate peak speedup
ax.annotate('Peak: 2.15×', 
            xy=(5 + width, 2.15), 
            xytext=(4.5, 2.22),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5),
            color='#E74C3C')

plt.tight_layout()

plt.savefig('attention_speedup_v2.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('attention_speedup_v2.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("Created attention_speedup_v2.pdf and attention_speedup_v2.png")
