import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# 原始数据
x = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 65, 70]
y1 = [-3.26, -3.07, -2.28, -1.27, -0.33, 0.47, 1.64, 2.36, 2.67, 2.73, 2.72, 2.68]
y2 = [-3.256, -3.064, -2.283, -1.273, -0.332, 0.468, 1.638, 2.363, 2.675, 2.735, 2.716, 2.681]

# 样条插值
tck1, u = interpolate.splprep([x, y1], s=0)
tck2, u = interpolate.splprep([x, y2], s=0)

# 生成更密集的点以获得平滑曲线
unew = np.linspace(0, 1, 100)
out1 = interpolate.splev(unew, tck1)
out2 = interpolate.splev(unew, tck2)

# 使用SciencePlots风格绘图
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制平滑曲线
    ax.plot(out1[0], out1[1], label='LLMIR', linewidth=2)
    ax.plot(out2[0], out2[1], label='vLLM', linewidth=2)
    
    # 添加原始数据点
    ax.scatter(x, y1, s=30, zorder=5)
    ax.scatter(x, y2, s=30, zorder=5)
    
    # 设置图表标题和标签
    ax.set(xlabel='Heel angle (°)')
    ax.set(ylabel='Gz (m)')
    
    # 添加图例
    ax.legend(title='Framework')
    
    # 自动调整布局
    ax.autoscale(tight=True)
    ax.set_ylim(top=3)
    fig.tight_layout()
    
    # 保存图表
    fig.savefig('figures/heel_angle_plot.pdf')
    fig.savefig('figures/heel_angle_plot.png', dpi=300)

# 创建中文版本的图表
try:
    # 尝试使用中文字体
    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制平滑曲线
        ax.plot(out1[0], out1[1], label='LLMIR方法', linewidth=2)
        ax.plot(out2[0], out2[1], label='基准方法', linewidth=2)
        
        # 添加原始数据点
        ax.scatter(x, y1, s=30, zorder=5)
        ax.scatter(x, y2, s=30, zorder=5)
        
        # 设置图表标题和标签
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        ax.set(xlabel='倾角 (°)')
        ax.set(ylabel='横向重心高度 (m)')
        ax.set_title('LLMIR性能对比')
        
        # 添加图例
        ax.legend(title='框架')
        
        # 自动调整布局
        ax.autoscale(tight=True)
        ax.set_ylim(top=3)
        fig.tight_layout()
        
        # 保存图表
        fig.savefig('figures/heel_angle_plot_chinese.pdf')
        fig.savefig('figures/heel_angle_plot_chinese.png', dpi=300)
except Exception as e:
    print(f"创建中文图表时出错: {e}")

plt.show()
