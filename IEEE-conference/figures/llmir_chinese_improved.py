import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# 原始数据
x = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 65, 70]
y1 = [-3.26, -3.07, -2.28, -1.27, -0.33, 0.47, 1.64, 2.36, 2.67, 2.73, 2.72, 2.68]
y2 = [-3.256, -3.064, -2.283, -1.273, -0.332, 0.468, 1.638, 2.363, 2.675, 2.735, 2.716, 2.681]

# 样条插值
x_new = np.linspace(0, 70, 300)
y1_smooth = np.interp(x_new, x, y1)
y2_smooth = np.interp(x_new, x, y2)

# 设置学术风格的图表
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-paper')  # 使用seaborn的学术风格

# 设置字体和线条风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5
})

# 绘制平滑曲线
plt.plot(x_new, y1_smooth, label='LLMIR', color='#1f77b4', linewidth=2.5)
plt.plot(x_new, y2_smooth, label='vLLM', color='#ff7f0e', linewidth=2.5)

# 添加原始数据点
plt.scatter(x, y1, color='#1f77b4', s=80, zorder=5, edgecolors='black', linewidths=1)
plt.scatter(x, y2, color='#ff7f0e', s=80, zorder=5, edgecolors='black', linewidths=1)

# 设置图表标题和标签
plt.xlabel('Heel angle (°)', fontweight='bold')
plt.ylabel('Gz (m)', fontweight='bold')
plt.title('Performance Comparison', fontweight='bold', fontsize=16)

# 添加网格线，提高可读性
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(title='Framework', title_fontsize=14, frameon=True, fancybox=True, shadow=True)

# 设置y轴范围
plt.ylim(top=3)

# 自动调整布局
plt.tight_layout()

# 保存图表
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/heel_angle_improved.pdf')
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/heel_angle_improved.png', dpi=300)

print("英文图表已保存到 figures/heel_angle_improved.png 和 .pdf")

# 创建中文版本的图表
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-paper')  # 使用seaborn的学术风格

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 绘制平滑曲线
plt.plot(x_new, y1_smooth, label='装载手册值', color='#1f77b4', linewidth=2.5)
plt.plot(x_new, y2_smooth, label='本文方法', color='#ff7f0e', linewidth=2.5)

# 添加原始数据点
plt.scatter(x, y1, color='#1f77b4', s=80, zorder=5, edgecolors='black', linewidths=1)
plt.scatter(x, y2, color='#ff7f0e', s=80, zorder=5, edgecolors='black', linewidths=1)

# 设置图表标题和标签
plt.xlabel('倾角 (°)', fontweight='bold')
plt.ylabel('横向重心高度 (m)', fontweight='bold')
plt.title('性能对比', fontweight='bold', fontsize=16)

# 添加网格线，提高可读性
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(title='测试', title_fontsize=14, frameon=True, fancybox=True, shadow=True)

# 设置y轴范围
plt.ylim(top=3)

# 自动调整布局
plt.tight_layout()

# 保存图表
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/heel_angle_chinese_improved.pdf')
plt.savefig('/Users/xingqiangchen/llmir/llmir/IEEE-conference-template-062824/figures/heel_angle_chinese_improved.png', dpi=300)

print("中文图表已保存到 figures/heel_angle_chinese_improved.png 和 .pdf")
plt.show()
