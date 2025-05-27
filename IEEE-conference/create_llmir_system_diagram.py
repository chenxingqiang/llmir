#!/usr/bin/env python3
"""
Create LLMIR System Architecture Diagram in Robin System Style
Professional layered architecture with modular components and clear data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with high DPI for publication quality
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Define colors matching the Robin system style
colors = {
    'input': '#E8F4FD',      # Light blue
    'system': '#F0F0F0',     # Light gray
    'output': '#E8F5E8',     # Light green
    'process1': '#B3D9FF',   # Blue
    'process2': '#FFB3B3',   # Light red/pink
    'process3': '#FFE6B3',   # Light orange
    'border': '#333333',     # Dark gray
    'text': '#333333',       # Dark gray
    'arrow': '#666666'       # Medium gray
}

# Helper function to create rounded rectangles
def create_rounded_rect(x, y, width, height, color, border_color, ax, text='', text_size=10, text_weight='normal'):
    """Create a rounded rectangle with text"""
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(rect)
    
    if text:
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=text_size, 
                weight=text_weight, color=colors['text'],
                wrap=True)
    
    return rect

# Helper function to create arrows
def create_arrow(start_x, start_y, end_x, end_y, ax, style='->'):
    """Create an arrow between two points"""
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle=style,
        color=colors['arrow'],
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(arrow)
    return arrow

# Set up the coordinate system
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')

# Title
ax.text(7, 9.5, 'LLMIR System Architecture', 
        ha='center', va='center', fontsize=18, weight='bold', color=colors['text'])

# Main sections
# Input section (left)
input_rect = create_rounded_rect(0.5, 2, 2.5, 6, colors['input'], colors['border'], ax)
ax.text(1.75, 7.5, 'Input', ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])

# System section (center)
system_rect = create_rounded_rect(4, 1, 6, 8, colors['system'], colors['border'], ax)
ax.text(7, 8.5, 'LLMIR Compiler Infrastructure', ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])

# Output section (right)
output_rect = create_rounded_rect(11, 2, 2.5, 6, colors['output'], colors['border'], ax)
ax.text(12.25, 7.5, 'Output', ha='center', va='center', fontsize=14, weight='bold', color=colors['text'])

# Input components
create_rounded_rect(0.7, 6.5, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'PyTorch Models', 10, 'normal')
create_rounded_rect(0.7, 5.2, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'vLLM Graphs', 10, 'normal')
create_rounded_rect(0.7, 3.9, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'SGLang Programs', 10, 'normal')
create_rounded_rect(0.7, 2.6, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'ONNX Models', 10, 'normal')

# Input description
ax.text(1.75, 2.2, 'Frontend converters translate\nvarious model formats into\nLLMIR representation', 
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# System components - layered architecture
# Layer 1: Frontend
create_rounded_rect(4.3, 7.2, 5.4, 0.8, colors['process1'], colors['border'], ax, 
                   'Frontend Converters & Model Importers', 11, 'bold')

# Layer 2: MLIR Pipeline
create_rounded_rect(4.3, 6.2, 5.4, 0.8, colors['process2'], colors['border'], ax, 
                   'MLIR Optimization Pipeline', 11, 'bold')

# Layer 3: LLM Dialect
create_rounded_rect(4.3, 5.2, 5.4, 0.8, colors['process1'], colors['border'], ax, 
                   'LLM Dialect (Types & Operations)', 11, 'bold')

# Layer 4: Optimization Passes
create_rounded_rect(4.3, 3.8, 1.6, 1.2, colors['process3'], colors['border'], ax, 
                   'KV Cache\nOptimization', 9, 'normal')
create_rounded_rect(6.1, 3.8, 1.6, 1.2, colors['process3'], colors['border'], ax, 
                   'Multi-Precision\nComputation', 9, 'normal')
create_rounded_rect(7.9, 3.8, 1.6, 1.2, colors['process3'], colors['border'], ax, 
                   'Parallelization\nStrategies', 9, 'normal')

# Layer 5: Backend
create_rounded_rect(4.3, 2.6, 5.4, 0.8, colors['process2'], colors['border'], ax, 
                   'Backend Code Generation', 11, 'bold')

# Layer 6: Runtime Integration
create_rounded_rect(4.3, 1.6, 5.4, 0.8, colors['process1'], colors['border'], ax, 
                   'Runtime Integration Layer', 11, 'bold')

# Output components
create_rounded_rect(11.2, 6.5, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'Optimized CUDA\nKernels', 10, 'normal')
create_rounded_rect(11.2, 5.2, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'CPU Vectorized\nCode', 10, 'normal')
create_rounded_rect(11.2, 3.9, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'Runtime Libraries', 10, 'normal')
create_rounded_rect(11.2, 2.6, 2.1, 1, colors['process3'], colors['border'], ax, 
                   'Integration APIs', 10, 'normal')

# Output description
ax.text(12.25, 2.2, 'LLMIR generates optimized\ncode for multiple hardware\ntargets and frameworks', 
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Add arrows showing data flow
# Input to System
create_arrow(3.0, 5, 4.3, 5, ax)

# System to Output
create_arrow(9.7, 5, 11.0, 5, ax)

# Internal system flow arrows
create_arrow(7, 7.0, 7, 6.8, ax, '->')  # Frontend to Pipeline
create_arrow(7, 6.0, 7, 5.8, ax, '->')  # Pipeline to Dialect
create_arrow(7, 5.0, 7, 4.8, ax, '->')  # Dialect to Passes
create_arrow(7, 3.6, 7, 3.2, ax, '->')  # Passes to Backend
create_arrow(7, 2.4, 7, 2.2, ax, '->')  # Backend to Runtime

# Add key innovation callouts
# PagedAttention callout
callout_rect = create_rounded_rect(10.5, 0.2, 3, 1.2, '#FFFACD', colors['border'], ax)
ax.text(12, 0.8, 'Key Innovation:\nIR-level PagedAttention\nRepresentation', 
        ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])

# Performance callout
perf_rect = create_rounded_rect(0.5, 0.2, 3, 1.2, '#FFFACD', colors['border'], ax)
ax.text(2, 0.8, 'Performance:\n22.4% over vLLM\n37.8% over SGLang', 
        ha='center', va='center', fontsize=9, weight='bold', color=colors['text'])

# Add connecting lines for callouts
create_arrow(10.5, 0.8, 9.7, 2.0, ax, '->')
create_arrow(3.5, 0.8, 4.3, 2.0, ax, '->')

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add subtle grid for professional look
ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)

plt.tight_layout()

# Save the figure
plt.savefig('figures/llmir_system_architecture_robin_style.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/llmir_system_architecture_robin_style.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("LLMIR System Architecture diagram (Robin style) saved as:")
print("- figures/llmir_system_architecture_robin_style.pdf")
print("- figures/llmir_system_architecture_robin_style.png")

plt.show() 