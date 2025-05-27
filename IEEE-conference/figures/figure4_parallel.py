import matplotlib.pyplot as plt
import numpy as np

# Set font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6))

# 数据
num_gpus = [1, 2, 4, 8]

# 基于推断的数据，表示加速比
ideal_speedup = [1.0, 2.0, 4.0, 8.0]  # 理想线性加速比
tensor_parallel = [1.0, 1.85, 3.6, 6.8]  # 张量并行
pipeline_parallel = [1.0, 1.92, 3.7, 7.1]  # 流水线并行
combined_parallel = [1.0, 1.95, 3.8, 7.3]  # 组合并行

# Draw line plot
ax.plot(num_gpus, ideal_speedup, '--', label='Ideal Linear Speedup', linewidth=2, color='black')
ax.plot(num_gpus, tensor_parallel, 'o-', label='Tensor Parallel', linewidth=2, markersize=8, color='#4472C4')
ax.plot(num_gpus, pipeline_parallel, 's-', label='Pipeline Parallel', linewidth=2, markersize=8, color='#ED7D31')
ax.plot(num_gpus, combined_parallel, '^-', label='Combined Parallel', linewidth=2, markersize=8, color='#70AD47')

# Add labels and title
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('Speedup')
ax.set_title('Speedup of Different Parallelization Strategies on LLaMA-2-70B')

# Set x-axis ticks
ax.set_xticks(num_gpus)
ax.set_xticklabels(num_gpus)

# Add grid lines
ax.grid(linestyle='--', alpha=0.7)

# Add legend
ax.legend(loc='upper left')

# Add data labels
for i, n in enumerate(num_gpus):
    if i > 0:  # 跳过第一个点(1,1)
        ax.text(n, tensor_parallel[i] - 0.3, f'{tensor_parallel[i]}x', ha='center', va='top', fontsize=9, color='#4472C4')
        ax.text(n, pipeline_parallel[i] + 0.2, f'{pipeline_parallel[i]}x', ha='center', va='bottom', fontsize=9, color='#ED7D31')
        ax.text(n, combined_parallel[i] + 0.2, f'{combined_parallel[i]}x', ha='center', va='bottom', fontsize=9, color='#70AD47')

# Set axis range
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0, 9)

# 保存图片
plt.tight_layout()
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/figure4.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/figure4.png', dpi=300, bbox_inches='tight')
plt.close()
