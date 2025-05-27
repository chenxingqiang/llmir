#!/usr/bin/env python3
"""
LLMIR Performance Comparison Chart Generator
Creates professional, publication-ready charts for the LLMIR paper
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set the style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for better fonts and quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_throughput_comparison():
    """Create the main throughput comparison chart"""
    
    # Data from the paper
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    # Performance data (tokens/sec)
    llmir_data = [78628, 83765, 84197, 84403, 82150, 79800, 61500]
    vllm_data = [64250, 68420, 69100, 69350, 68800, 65200, 61200]
    sglang_data = [57100, 60800, 61500, 61200, 59800, 56900, 53200]
    huggingface_data = [32150, 35200, 36800, 35800, 34200, 31500, 28900]
    
    # Create figure with specific size for IEEE format
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each framework
    colors = {
        'LLMIR': '#2E86AB',      # Professional blue
        'vLLM': '#A23B72',       # Deep pink
        'SGLang': '#F18F01',     # Orange
        'HuggingFace': '#C73E1D' # Red
    }
    
    # Plot lines with markers
    ax.plot(batch_sizes, llmir_data, 'o-', color=colors['LLMIR'], 
            label='LLMIR', linewidth=3, markersize=8, markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor=colors['LLMIR'])
    
    ax.plot(batch_sizes, vllm_data, 's-', color=colors['vLLM'], 
            label='vLLM', linewidth=3, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=colors['vLLM'])
    
    ax.plot(batch_sizes, sglang_data, '^-', color=colors['SGLang'], 
            label='SGLang', linewidth=3, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=colors['SGLang'])
    
    ax.plot(batch_sizes, huggingface_data, 'd-', color=colors['HuggingFace'], 
            label='HuggingFace', linewidth=3, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=colors['HuggingFace'])
    
    # Customize the plot
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax.set_title('LLM Inference Performance Comparison (LLaMA-2-13B)', 
                 fontweight='bold', pad=20)
    
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend with better positioning
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, ncol=1, bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add performance improvement annotation
    ax.annotate('LLMIR outperforms vLLM by 22.4%', 
                xy=(4, 75000), xytext=(8, 40000),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Set y-axis limits for better visualization
    ax.set_ylim(25000, 90000)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def create_memory_optimization_chart():
    """Create memory optimization impact chart"""
    
    configurations = ['No optimizations', 'Pool + Unified(128KB)', 
                     'Pool + Unified(256KB)', 'Pool only', 'Unified(128KB) only']
    performance = [45935, 72946, 39913, 41022, 48963]
    improvements = [0, 58.8, -13.1, -10.7, 6.6]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for bars
    colors = ['#FF6B6B' if imp < 0 else '#4ECDC4' if imp > 30 else '#45B7D1' 
              for imp in improvements]
    
    # Performance chart
    bars1 = ax1.bar(range(len(configurations)), performance, color=colors, alpha=0.8)
    ax1.set_xlabel('Memory Configuration', fontweight='bold')
    ax1.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax1.set_title('Memory Configuration Performance Impact', fontweight='bold')
    ax1.set_xticks(range(len(configurations)))
    ax1.set_xticklabels(configurations, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, perf in zip(bars1, performance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{perf:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement chart
    bars2 = ax2.bar(range(len(configurations)), improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Memory Configuration', fontweight='bold')
    ax2.set_ylabel('Performance Improvement (%)', fontweight='bold')
    ax2.set_title('Relative Performance Improvement', fontweight='bold')
    ax2.set_xticks(range(len(configurations)))
    ax2.set_xticklabels(configurations, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        y_pos = height + (2 if height >= 0 else -4)
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_scaling_efficiency_chart():
    """Create multi-GPU scaling efficiency chart"""
    
    gpu_counts = [1, 2, 4, 8]
    tensor_parallel = [1.0, 1.87, 3.65, 7.12]
    pipeline_parallel = [1.0, 1.92, 3.78, 7.41]
    hybrid = [1.0, 1.95, 3.82, 7.56]
    ideal = [1.0, 2.0, 4.0, 8.0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(gpu_counts, tensor_parallel, 'o-', label='Tensor Parallelism', 
            linewidth=3, markersize=8, color='#2E86AB')
    ax.plot(gpu_counts, pipeline_parallel, 's-', label='Pipeline Parallelism', 
            linewidth=3, markersize=8, color='#A23B72')
    ax.plot(gpu_counts, hybrid, '^-', label='Hybrid (TP+PP)', 
            linewidth=3, markersize=8, color='#F18F01')
    ax.plot(gpu_counts, ideal, '--', label='Ideal Scaling', 
            linewidth=2, alpha=0.7, color='gray')
    
    ax.set_xlabel('Number of GPUs', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Multi-GPU Scaling Efficiency', fontweight='bold', pad=20)
    
    # Add efficiency annotations
    efficiency_8gpu = hybrid[-1] / ideal[-1] * 100
    ax.annotate(f'94.5% efficiency\non 8 GPUs', 
                xy=(8, hybrid[-1]), xytext=(6.5, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_xticks(gpu_counts)
    ax.set_ylim(0, 9)
    
    plt.tight_layout()
    return fig

def create_ablation_study_chart():
    """Create ablation study chart"""
    
    configurations = ['Baseline', '+ KV Cache\nOptimization', '+ Multi-precision', 
                     '+ Parallelization', 'All Optimizations']
    performance = [32500, 45800, 52300, 58700, 58700]
    improvements = [0, 40.9, 14.2, 12.2, 80.6]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked effect visualization
    cumulative = [32500, 45800, 52300, 58700, 58700]
    
    # Colors for each optimization
    colors = ['#E8E8E8', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax.bar(range(len(configurations)), performance, color=colors, alpha=0.8)
    
    # Add improvement arrows and labels
    for i in range(1, len(configurations)):
        if i < len(configurations) - 1:  # Don't add arrow for last bar
            improvement = (performance[i] - performance[i-1]) / performance[i-1] * 100
            ax.annotate(f'+{improvement:.1f}%', 
                       xy=(i-0.5, (performance[i-1] + performance[i])/2),
                       ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for bar, perf in zip(bars, performance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{perf:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Optimization Configuration', fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax.set_title('Optimization Pass Contribution Analysis', fontweight='bold', pad=20)
    ax.set_xticks(range(len(configurations)))
    ax.set_xticklabels(configurations)
    
    # Add overall improvement annotation
    overall_improvement = (performance[-1] - performance[0]) / performance[0] * 100
    ax.annotate(f'Overall: +{overall_improvement:.1f}%', 
                xy=(len(configurations)-1, performance[-1]), 
                xytext=(len(configurations)-2, performance[-1] + 5000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 70000)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all charts"""
    
    print("Generating LLMIR performance charts...")
    
    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Generate main performance comparison chart
    fig1 = create_throughput_comparison()
    fig1.savefig('figures/llmir_performance_comparison.png', dpi=300, bbox_inches='tight')
    fig1.savefig('figures/llmir_performance_comparison.pdf', bbox_inches='tight')
    print("✓ Generated throughput comparison chart")
    
    # Generate memory optimization chart
    fig2 = create_memory_optimization_chart()
    fig2.savefig('figures/memory_optimization_impact.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figures/memory_optimization_impact.pdf', bbox_inches='tight')
    print("✓ Generated memory optimization chart")
    
    # Generate scaling efficiency chart
    fig3 = create_scaling_efficiency_chart()
    fig3.savefig('figures/scaling_efficiency.png', dpi=300, bbox_inches='tight')
    fig3.savefig('figures/scaling_efficiency.pdf', bbox_inches='tight')
    print("✓ Generated scaling efficiency chart")
    
    # Generate ablation study chart
    fig4 = create_ablation_study_chart()
    fig4.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
    fig4.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    print("✓ Generated ablation study chart")
    
    print("\nAll charts generated successfully!")
    print("Files saved in 'figures/' directory with both PNG and PDF formats")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main() 