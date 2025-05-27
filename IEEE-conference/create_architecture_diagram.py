#!/usr/bin/env python3
"""
LLMIR System Architecture Diagram Generator
Creates a professional system architecture diagram for the LLMIR paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, FancyArrowPatch
import numpy as np

# Configure matplotlib for better fonts and quality
plt.rcParams.update({
    'font.size': 10,
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

def create_architecture_diagram():
    """Create LLMIR system architecture diagram"""

    fig, ax = plt.subplots(figsize=(16, 16))

    # Define colors with better contrast
    colors = {
        'application': '#F8F9FA',    # Very light gray
        'frontend': '#E3F2FD',       # Light blue
        'compiler': '#BBDEFB',       # Medium blue
        'optimization': '#90CAF9',   # Darker blue
        'backend': '#42A5F5',        # Dark blue
        'execution': '#1976D2',      # Deep blue
        'border': '#263238',         # Dark border
        'text': '#263238',           # Dark text
        'arrow': '#37474F'           # Arrow color
    }

    # Application Layer - 顶部，大幅增加垂直间距
    app_box = FancyBboxPatch((2, 13.5), 12, 1.5,
                            boxstyle="round,pad=0.2",
                            facecolor=colors['application'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(app_box)
    ax.text(8, 14.25, 'Application Layer (vLLM / SGLang)',
            ha='center', va='center', fontweight='bold', fontsize=16, color=colors['text'])

    # Frontend Converters - 左侧，大幅增加垂直间距
    frontend_box = FancyBboxPatch((0.5, 10.5), 6, 2,
                                 boxstyle="round,pad=0.2",
                                 facecolor=colors['frontend'],
                                 edgecolor=colors['border'],
                                 linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(3.5, 11.8, 'Frontend Converters',
            ha='center', va='center', fontweight='bold', fontsize=14, color=colors['text'])
    ax.text(3.5, 11.3, '• PyTorch Models', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(3.5, 11.0, '• vLLM Integration', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(3.5, 10.7, '• SGLang Integration', ha='center', va='center', fontsize=12, color=colors['text'])

    # MLIR Optimization Pipeline - 右侧，大幅增加垂直间距
    opt_box = FancyBboxPatch((9.5, 10.5), 6, 2,
                            boxstyle="round,pad=0.2",
                            facecolor=colors['optimization'],
                            edgecolor=colors['border'],
                            linewidth=2)
    ax.add_patch(opt_box)
    ax.text(12.5, 11.8, 'MLIR Optimization Pipeline',
            ha='center', va='center', fontweight='bold', fontsize=14, color='white')
    ax.text(12.5, 11.3, '• LLM Dialect Operations', ha='center', va='center', fontsize=12, color='white')
    ax.text(12.5, 11.0, '• KV Cache Optimization', ha='center', va='center', fontsize=12, color='white')
    ax.text(12.5, 10.7, '• Multi-precision Passes', ha='center', va='center', fontsize=12, color='white')

    # Core LLMIR Compiler Box - 中间，大幅增加垂直间距
    compiler_box = FancyBboxPatch((1.5, 6.5), 13, 3,
                                 boxstyle="round,pad=0.25",
                                 facecolor=colors['compiler'],
                                 edgecolor=colors['border'],
                                 linewidth=3)
    ax.add_patch(compiler_box)
    ax.text(8, 8.7, 'LLMIR Compiler Core',
            ha='center', va='center', fontweight='bold', fontsize=18, color=colors['text'])

    # LLM Dialect Box - 左侧子模块
    dialect_box = FancyBboxPatch((2.5, 6.8), 5, 2.2,
                                boxstyle="round,pad=0.2",
                                facecolor='white',
                                edgecolor=colors['border'],
                                linewidth=2)
    ax.add_patch(dialect_box)
    ax.text(5, 8.3, 'LLM Dialect', ha='center', va='center', fontweight='bold', fontsize=14, color=colors['text'])
    ax.text(5, 7.9, '• PagedKVCacheType', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(5, 7.6, '• ShardedTensorType', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(5, 7.3, '• QuantizedTensorType', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(5, 7.0, '• llm.paged_attention', ha='center', va='center', fontsize=12, color=colors['text'])

    # Optimization Passes Box - 右侧子模块
    passes_box = FancyBboxPatch((8.5, 6.8), 5, 2.2,
                               boxstyle="round,pad=0.2",
                               facecolor='white',
                               edgecolor=colors['border'],
                               linewidth=2)
    ax.add_patch(passes_box)
    ax.text(11, 8.3, 'Optimization Passes', ha='center', va='center', fontweight='bold', fontsize=14, color=colors['text'])
    ax.text(11, 7.9, '• KV Cache Fusion', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(11, 7.6, '• Quantization-Aware', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(11, 7.3, '• Parallelization', ha='center', va='center', fontsize=12, color=colors['text'])
    ax.text(11, 7.0, '• Memory Optimization', ha='center', va='center', fontsize=12, color=colors['text'])

    # Backend Code Generators - 大幅增加垂直间距
    backend_box = FancyBboxPatch((1.5, 4), 13, 1.5,
                                boxstyle="round,pad=0.2",
                                facecolor=colors['backend'],
                                edgecolor=colors['border'],
                                linewidth=2)
    ax.add_patch(backend_box)
    ax.text(8, 5, 'Backend Code Generators',
            ha='center', va='center', fontweight='bold', fontsize=16, color='white')
    ax.text(4, 4.4, 'CUDA Backend', ha='center', va='center', fontsize=14, color='white')
    ax.text(8, 4.4, 'CPU Backend', ha='center', va='center', fontsize=14, color='white')
    ax.text(12, 4.4, 'ROCm Backend', ha='center', va='center', fontsize=14, color='white')

    # Execution Layer - 底部，大幅增加垂直间距
    exec_box = FancyBboxPatch((1.5, 1.5), 13, 1.2,
                             boxstyle="round,pad=0.2",
                             facecolor=colors['execution'],
                             edgecolor=colors['border'],
                             linewidth=2)
    ax.add_patch(exec_box)
    ax.text(8, 2.1, 'Execution Layer (CUDA / ROCm / CPU)',
            ha='center', va='center', fontweight='bold', fontsize=16, color='white')

    # Add curved arrows with better spacing and styling
    # Application to Frontend (curved left) - 调整箭头位置适应新间距
    arrow1 = FancyArrowPatch((5, 13.5), (3.5, 12.5),
                           connectionstyle="arc3,rad=-0.3",
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=3.5,
                           alpha=0.8)
    ax.add_patch(arrow1)

    # Application to Optimization (curved right) - 调整箭头位置
    arrow2 = FancyArrowPatch((11, 13.5), (12.5, 12.5),
                           connectionstyle="arc3,rad=0.3",
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=3.5,
                           alpha=0.8)
    ax.add_patch(arrow2)

    # Frontend to Compiler (curved) - 调整箭头位置
    arrow3 = FancyArrowPatch((5, 10.5), (6, 9.5),
                           connectionstyle="arc3,rad=0.2",
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=3.5,
                           alpha=0.8)
    ax.add_patch(arrow3)

    # Optimization to Compiler (curved) - 调整箭头位置
    arrow4 = FancyArrowPatch((11.5, 10.5), (10, 9.5),
                           connectionstyle="arc3,rad=-0.2",
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=3.5,
                           alpha=0.8)
    ax.add_patch(arrow4)

    # Compiler to Backend (straight but styled) - 调整箭头位置
    arrow5 = FancyArrowPatch((8, 6.5), (8, 5.5),
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=4,
                           alpha=0.8)
    ax.add_patch(arrow5)

    # Backend to Execution (straight but styled) - 调整箭头位置
    arrow6 = FancyArrowPatch((8, 4), (8, 2.7),
                           arrowstyle='->',
                           mutation_scale=30,
                           color=colors['arrow'],
                           linewidth=4,
                           alpha=0.8)
    ax.add_patch(arrow6)

    # Set axis properties with more space
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title with better spacing
    ax.text(8, 15.5, 'LLMIR System Architecture',
            ha='center', va='center', fontweight='bold', fontsize=22, color=colors['text'])

    plt.tight_layout()
    return fig

def main():
    """Generate architecture diagram"""

    print("Generating LLMIR architecture diagram...")

    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)

    # Generate architecture diagram
    fig = create_architecture_diagram()
    fig.savefig('figures/llmir_architecture.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/llmir_architecture.pdf', bbox_inches='tight')
    print("✓ Generated architecture diagram")

    print("\nArchitecture diagram generated successfully!")
    print("Files saved as 'figures/llmir_architecture.png' and 'figures/llmir_architecture.pdf'")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()