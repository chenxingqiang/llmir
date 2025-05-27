import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(10, 8))

# 设置背景色为白色
ax.set_facecolor('white')

# 绘制框架
def draw_box(x, y, width, height, label, color, alpha=0.8):
    rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                             edgecolor='black', facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=12, fontweight='bold')

# 绘制箭头
def draw_arrow(x1, y1, x2, y2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=2, color='black'))

# 设置坐标范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Draw application layer
draw_box(3, 9, 4, 0.8, 'Application Layer (vLLM / SGLang)', '#8dd3c7')

# Draw LLMIR compiler framework
draw_box(1, 4, 8, 4, '', '#ffffb3', alpha=0.3)
ax.text(5, 7.7, 'LLMIR Compiler', ha='center', va='center', fontsize=14, fontweight='bold')

# Draw frontend converters
draw_box(1.5, 6.5, 2.5, 0.8, 'Frontend Converters', '#bebada')

# Draw MLIR optimization pipeline
draw_box(6, 6.5, 2.5, 0.8, 'MLIR Optimization Pipeline', '#fb8072')

# Draw backend generators
draw_box(6, 5, 2.5, 0.8, 'Backend Code Generators', '#80b1d3')

# Draw execution layer
draw_box(3, 2.5, 4, 0.8, 'Execution Layer (CUDA / ROCm / LLVM)', '#fdb462')

# 绘制连接线和箭头
draw_arrow(5, 9, 5, 8)  # 应用层到LLMIR
draw_arrow(2.75, 6.5, 5, 7)  # 前端到MLIR优化
draw_arrow(7.25, 6.5, 7.25, 5.8)  # MLIR优化到后端
draw_arrow(5, 5, 5, 3.3)  # 后端到执行层

# Add core components description
components = [
    "1. LLM Dialect: Specialized Ops & Types",
    "   - PagedKVCache, ShardedTensor",
    "   - Attention, KV Cache Operations",
    "",
    "2. Optimization Passes:",
    "   - KV Cache Optimization",
    "   - Multi-precision Computing",
    "   - Parallelization Strategies"
]

ax.text(1.2, 5.5, '\n'.join(components), ha='left', va='center', fontsize=10)

# 移除坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 保存图片
plt.tight_layout()
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/figure1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/figure1.png', dpi=300, bbox_inches='tight')
plt.close()
