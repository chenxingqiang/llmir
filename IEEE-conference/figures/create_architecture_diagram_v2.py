#!/usr/bin/env python3
"""
Create LLMIR System Architecture Diagram - Version 2
Professional layered architecture with LARGER FONTS for better readability
Addresses reviewer feedback about font sizes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with high DPI for publication quality
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14  # Increased base font size

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Define colors with better contrast
colors = {
    'input': '#E8F6F3',      # Very light teal
    'system': '#F8F9FA',     # Very light gray
    'output': '#FDF2E9',     # Very light orange
    'process1': '#4ECDC4',   # Teal
    'process2': '#45B7D1',   # Blue
    'process3': '#96CEB4',   # Light green
    'border': '#2C3E50',     # Dark blue-gray
    'text': '#2C3E50',       # Dark blue-gray
    'highlight': '#FFEAA7'   # Yellow highlight
}

def create_rounded_rect(x, y, width, height, color, border_color, ax, 
                        text='', text_size=14, text_weight='normal'):
    """Create a rounded rectangle with properly positioned text"""
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(rect)
    
    if text:
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=text_size, 
                weight=text_weight, color=colors['text'],
                wrap=True)
    
    return rect

def create_arrow(start_x, start_y, end_x, end_y, ax):
    """Create an arrow between two points"""
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='-|>',
        color='#34495E',
        linewidth=2,
        alpha=1.0,
        mutation_scale=15
    )
    ax.add_patch(arrow)
    return arrow

# Set up the coordinate system
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')

# Title - LARGER
ax.text(7, 9.5, 'LLMIR Compilation Pipeline', 
        ha='center', va='center', fontsize=20, weight='bold', color=colors['text'])

# Stage boxes with larger text
stage_height = 1.2
stage_y = 7.5

# Stage 1: Model Import
create_rounded_rect(0.5, stage_y, 3, stage_height, colors['process1'], 
                   colors['border'], ax, 'Stage 1:\nModel Import', 14, 'bold')

# Stage 2: High-Level Optimization  
create_rounded_rect(4, stage_y, 3, stage_height, colors['process2'],
                   colors['border'], ax, 'Stage 2:\nLLM Dialect Opt.', 14, 'bold')

# Stage 3: Lowering
create_rounded_rect(7.5, stage_y, 3, stage_height, colors['process3'],
                   colors['border'], ax, 'Stage 3:\nCode Generation', 14, 'bold')

# Stage 4: Runtime
create_rounded_rect(11, stage_y, 2.5, stage_height, colors['process1'],
                   colors['border'], ax, 'Stage 4:\nRuntime', 14, 'bold')

# Arrows between stages
create_arrow(3.5, stage_y + stage_height/2, 4, stage_y + stage_height/2, ax)
create_arrow(7, stage_y + stage_height/2, 7.5, stage_y + stage_height/2, ax)
create_arrow(10.5, stage_y + stage_height/2, 11, stage_y + stage_height/2, ax)

# Detail boxes below each stage
detail_y = 5.5
detail_height = 1.5

# Input formats
create_rounded_rect(0.5, detail_y, 3, detail_height, colors['input'],
                   colors['border'], ax, 'PyTorch\nONNX\nvLLM Graphs', 13, 'normal')

# LLM Dialect operations
create_rounded_rect(4, detail_y, 3, detail_height, colors['input'],
                   colors['border'], ax, 'llm.attention\nllm.paged_attention\nllm.append_kv', 13, 'normal')

# Lowering targets
create_rounded_rect(7.5, detail_y, 3, detail_height, colors['input'],
                   colors['border'], ax, 'CUDA Kernels\nCPU SIMD\nMemory Layouts', 13, 'normal')

# Runtime components
create_rounded_rect(11, detail_y, 2.5, detail_height, colors['input'],
                   colors['border'], ax, 'Batch Mgmt\nMemory Pools\nvLLM API', 13, 'normal')

# Vertical arrows from stages to details
for x in [2, 5.5, 9, 12.25]:
    create_arrow(x, detail_y + detail_height + 0.1, x, stage_y - 0.1, ax)

# Key innovations at bottom
create_rounded_rect(1, 3, 5.5, 1.5, colors['highlight'], colors['border'], ax,
                   'Compile-Time: Block allocation,\nKernel selection, Sharing detection', 13, 'bold')

create_rounded_rect(7.5, 3, 5.5, 1.5, colors['highlight'], colors['border'], ax,
                   'Runtime: Dynamic growth,\nParameter binding, Communication', 13, 'bold')

# Performance summary
ax.text(7, 1.5, 'Performance: 22.4% over vLLM | 37.8% over SGLang | Perplexity within 0.1%',
        ha='center', va='center', fontsize=14, weight='bold', color=colors['text'],
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['border'], linewidth=2))

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# Save the figure
plt.savefig('llmir_architecture_v2.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('llmir_architecture_v2.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("Created llmir_architecture_v2.pdf and llmir_architecture_v2.png")
