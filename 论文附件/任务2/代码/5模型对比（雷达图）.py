import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据预处理：整理模型评估结果
model_metrics = dic_result
#此处dic_result为上一部各个模型得出的结果的字典形式

fault_types = ["N", "OR", "IR", "B"]
fault_names = ["正常", "外圈故障", "内圈故障", "滚动体故障"]
model_colors = {"Random Forest (RF)": "#1f77b4", "XGBoost": "#ff7f0e",
                "Support Vector Machine (SVM)": "#2ca02c", "MLP": "#d62728"}
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 雷达图工具函数
def radar_factory(num_vars, frame='polygon'):
    """生成雷达图的角度与坐标轴配置，确保图形闭合"""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    theta = np.append(theta, theta[0])  # 闭合角度，使最后一个点回到起点
    return theta

def unit_poly_verts(theta):
    """生成雷达图多边形的顶点坐标"""
    x = np.cos(theta)
    y = np.sin(theta)
    return np.column_stack([x, y])

# 3. 修正后的雷达图绘制函数
def plot_model_radar_chart():
    categories = fault_names  # 4个故障类型作为雷达图维度
    N = len(categories)
    theta = radar_factory(N)  # 生成闭合的角度数组
    verts = unit_poly_verts(theta)  # 生成多边形顶点

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_title('各模型综合性能雷达图（维度：4类故障F1-score）',
                 fontsize=14, fontweight='bold', position=(0.5, 1.1))

    # 为每个模型绘制雷达曲线与填充
    for model, metrics in model_metrics.items():
        f1_scores = [metrics["f1-score"][ft] for ft in fault_types]
        f1_scores = f1_scores + [f1_scores[0]]  # 闭合数据，使曲线首尾相连
        ax.plot(theta, f1_scores, label=model, color=model_colors[model], linewidth=2)
        ax.fill(theta, f1_scores, color=model_colors[model], alpha=0.2)  # 填充区域增强对比

    # 设置角度标签
    ax.set_thetagrids(np.degrees(theta[:-1]), categories, fontsize=11)
    ax.set_theta_zero_location('N')  # 角度0°在正上方
    ax.set_theta_direction(-1)  # 角度顺时针增加
    ax.set_ylim(0.7, 1.05)  # 调整y轴范围，确保满分（1.0）不被截断
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3)  # 显示网格增强可读性

    plt.tight_layout()
    plt.savefig('图4_修正后_模型综合性能雷达图.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. 执行雷达图绘制
if __name__ == "__main__":
    plot_model_radar_chart()