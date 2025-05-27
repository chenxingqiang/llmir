#!/usr/bin/env python3
"""
Create LLMIR System Architecture Diagram - Fixed Version
Professional layered architecture with aligned connections and proper text positioning
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with high DPI for publication quality
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(16, 11))
fig.patch.set_facecolor('white')

# Define colors matching the memory optimization chart style
colors = {
    'input': '#E8F6F3',      # Very light teal
    'system': '#F8F9FA',     # Very light gray
    'output': '#FDF2E9',     # Very light orange
    'process1': '#4ECDC4',   # Teal (high performance color)
    'process2': '#45B7D1',   # Blue (medium performance color)
    'process3': '#96CEB4',   # Light green (accent color)
    'border': '#2C3E50',     # Dark blue-gray
    'text': '#2C3E50',       # Dark blue-gray
    'arrow': '#34495E'       # Medium blue-gray
}

# Helper function to create rounded rectangles with better text positioning
def create_rounded_rect(x, y, width, height, color, border_color, ax, text='', text_size=10, text_weight='normal', text_offset_y=0):
    """Create a rounded rectangle with properly positioned text"""
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(rect)
    
    if text:
        # Better text positioning without background box
        text_y = y + height/2 + text_offset_y
        ax.text(x + width/2, text_y, text, 
                ha='center', va='center', fontsize=text_size, 
                weight=text_weight, color=colors['text'],
                wrap=True)
    
    return rect

# Helper function to create properly aligned arrows
def create_arrow(start_x, start_y, end_x, end_y, ax, style='->', offset_start=0, offset_end=0):
    """Create an arrow between two points with proper alignment"""
    # Add small offsets to avoid overlapping with box edges
    if start_x == end_x:  # Vertical arrow
        if start_y > end_y:  # Downward
            start_y -= offset_start
            end_y += offset_end
        else:  # Upward
            start_y += offset_start
            end_y -= offset_end
    else:  # Horizontal arrow
        if start_x < end_x:  # Rightward
            start_x += offset_start
            end_x -= offset_end
        else:  # Leftward
            start_x -= offset_start
            end_x += offset_end
    
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='-|>',
        color='black',
        linewidth=1.0,
        alpha=1.0,
        mutation_scale=12
    )
    ax.add_patch(arrow)
    return arrow

# Set up the coordinate system with more space for bottom callouts
ax.set_xlim(0, 16)
ax.set_ylim(-2, 11)
ax.set_aspect('equal')

# Title with better positioning
ax.text(8, 10.3, 'LLMIR System Architecture', 
        ha='center', va='center', fontsize=20, weight='bold', color=colors['text'])

# Main sections with precise positioning
# Input section (left) - adjusted for better alignment
input_rect = create_rounded_rect(0.5, 2.5, 3, 6.5, colors['input'], colors['border'], ax)
ax.text(2, 8.7, 'Input', ha='center', va='center', fontsize=16, weight='bold', color=colors['text'])

# System section (center) - centered and aligned
system_rect = create_rounded_rect(4.5, 1.5, 7, 8, colors['system'], colors['border'], ax)
ax.text(8, 9.2, 'LLMIR Compiler Infrastructure', ha='center', va='center', fontsize=16, weight='bold', color=colors['text'])

# Output section (right) - aligned with input
output_rect = create_rounded_rect(12.5, 2.5, 3, 6.5, colors['output'], colors['border'], ax)
ax.text(14, 8.7, 'Output', ha='center', va='center', fontsize=16, weight='bold', color=colors['text'])

# Input components with consistent spacing
input_y_positions = [7.5, 6.2, 4.9, 3.6]
input_labels = ['PyTorch Models', 'vLLM Graphs', 'SGLang Programs', 'ONNX Models']

for i, (y_pos, label) in enumerate(zip(input_y_positions, input_labels)):
    create_rounded_rect(0.8, y_pos, 2.4, 0.9, colors['process3'], colors['border'], ax, 
                       label, 11, 'normal')

# Input description with better positioning
ax.text(2, 2.8, 'Frontend converters translate\nvarious model formats into\nLLMIR representation', 
        ha='center', va='center', fontsize=10, color=colors['text'], style='italic')

# System components - layered architecture with precise alignment
layer_width = 6.2
layer_x = 4.8
layer_spacing = 0.9

# Layer positions (top to bottom) - uniform increased spacing
layers = [
    (8.0, 'Frontend Converters & Model Importers', colors['process1'], 'bold'),
    (6.6, 'MLIR Optimization Pipeline', colors['process2'], 'bold'),
    (5.2, 'LLM Dialect (Types & Operations)', colors['process1'], 'bold'),
    (3.4, '', '', ''),  # Space for optimization passes
    (1.8, 'Backend Code Generation', colors['process2'], 'bold'),
    (0.4, 'Runtime Integration Layer', colors['process1'], 'bold')
]

# Create main layers
for y_pos, label, color, weight in layers:
    if label:  # Skip empty layer
        create_rounded_rect(layer_x, y_pos, layer_width, 0.7, color, colors['border'], ax, 
                           label, 12, weight)

# Optimization passes layer - three equal boxes
pass_width = 1.9
pass_spacing = 0.15
pass_y = 3.6
pass_start_x = layer_x + 0.2

passes = ['KV Cache\nOptimization', 'Multi-Precision\nComputation', 'Parallelization\nStrategies']
for i, pass_label in enumerate(passes):
    pass_x = pass_start_x + i * (pass_width + pass_spacing)
    create_rounded_rect(pass_x, pass_y, pass_width, 1.0, colors['process3'], colors['border'], ax, 
                       pass_label, 10, 'normal')

# Output components with consistent spacing and alignment
output_y_positions = [7.5, 6.2, 4.9, 3.6]
output_labels = ['Optimized CUDA\nKernels', 'CPU Vectorized\nCode', 'Runtime Libraries', 'Integration APIs']

for i, (y_pos, label) in enumerate(zip(output_y_positions, output_labels)):
    create_rounded_rect(12.8, y_pos, 2.4, 0.9, colors['process3'], colors['border'], ax, 
                       label, 11, 'normal')

# Output description with better positioning
ax.text(14, 2.8, 'LLMIR generates optimized\ncode for multiple hardware\ntargets and frameworks', 
        ha='center', va='center', fontsize=10, color=colors['text'], style='italic')

# Add properly aligned arrows showing data flow
# Input to System - horizontal alignment (precise edge connection)
input_center_y = 5.5
create_arrow(3.5, input_center_y, 4.5, input_center_y, ax)

# System to Output - horizontal alignment (precise edge connection)
create_arrow(11.5, input_center_y, 12.5, input_center_y, ax)

# Internal system flow arrows - perfectly vertical and centered with precise edge connections
system_center_x = 8

# Vertical flow arrows with precise edge-to-edge positioning
create_arrow(system_center_x, 8.0, system_center_x, 7.3, ax)  # Frontend to Pipeline
create_arrow(system_center_x, 6.6, system_center_x, 5.9, ax)  # Pipeline to Dialect  
create_arrow(system_center_x, 5.2, system_center_x, 4.6, ax)  # Dialect to Passes
create_arrow(system_center_x, 3.6, system_center_x, 2.5, ax)  # Passes to Backend
create_arrow(system_center_x, 1.8, system_center_x, 1.1, ax)  # Backend to Runtime

# Add key innovation callouts positioned at the bottom
# Performance callout - bottom left (wider box for longer text)
perf_rect = create_rounded_rect(0.5, -1.5, 7, 1.2, '#FFEAA7', colors['border'], ax)
ax.text(4.0, -0.9, 'Performance:\n22.4% over vLLM, 37.8% over SGLang', 
        ha='center', va='center', fontsize=10, weight='bold', color=colors['text'])

# PagedAttention callout - bottom right (wider box for longer text)
callout_rect1 = create_rounded_rect(8.5, -1.5, 7, 1.2, '#FFEAA7', colors['border'], ax)
ax.text(12.0, -0.9, 'Key Innovation:\nIR-level PagedAttention Representation', 
        ha='center', va='center', fontsize=10, weight='bold', color=colors['text'])

# Add connecting arrows from callouts to main system
create_arrow(4.0, -0.3, 6.0, 0.4, ax)  # Performance to Runtime layer
create_arrow(12.0, -0.3, 10.0, 0.4, ax)  # Innovation to Runtime layer

# Remove axes and grid
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Remove the grid for cleaner look
ax.grid(False)

plt.tight_layout()

# Save the figure with better file names
plt.savefig('llmir_architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('llmir_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("Updated LLMIR System Architecture diagram saved as:")
print("- llmir_architecture.pdf")
print("- llmir_architecture.png")

plt.show() 