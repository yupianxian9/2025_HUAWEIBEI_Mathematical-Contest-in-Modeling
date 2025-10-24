import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# 1. 数据预处理：整理模型评估结果

model_metrics = dic_result
# 此处dic_result为上一部各个模型得出的结果

# 故障类型与颜色映射（保持一致性）
fault_types = ["N", "OR", "IR", "B"]
fault_names = ["正常", "外圈故障", "内圈故障", "滚动体故障"]
model_colors = {"Random Forest (RF)": "#1f77b4", "XGBoost": "#ff7f0e",
                "Support Vector Machine (SVM)": "#2ca02c", "MLP": "#d62728"}
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# 2. 图1：单故障类型下多模型Precision对比（核心：看不同模型对同一故障的识别精度）
def plot_fault_precision_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算x轴位置（分组条形图）
    x = np.arange(len(fault_types))
    width = 0.2  # 条形宽度
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]  # 4个模型的偏移量

    # 绘制每个模型的Precision条形
    for idx, (model, metrics) in enumerate(model_metrics.items()):
        precisions = [metrics["precision"][ft] for ft in fault_types]
        bars = ax.bar(x + offsets[idx], precisions, width,
                      label=model, color=model_colors[model], alpha=0.8)

        # 在条形顶部添加数值标注
        for bar, p in zip(bars, precisions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{p:.2f}', ha='center', va='bottom', fontsize=10)

    # 图表美化
    ax.set_xlabel('故障类型', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision（精度）', fontsize=12, fontweight='bold')
    ax.set_title('不同故障类型下各模型的识别精度对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(fault_names, fontsize=11)
    ax.set_ylim(0.7, 1.05)  # 聚焦有效范围，突出差异
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加参考线（90%精度线）
    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, label='90%精度线')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('图1_故障类型-模型精度对比.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 图2：各模型整体Accuracy排名
def plot_model_accuracy_ranking():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 提取模型与对应准确率，按准确率降序排序
    models = list(model_metrics.keys())
    accuracies = [model_metrics[m]["accuracy"] for m in models]
    sorted_idx = np.argsort(accuracies)[::-1]  # 降序索引
    sorted_models = [models[i] for i in sorted_idx]
    sorted_accs = [accuracies[i] for i in sorted_idx]
    sorted_colors = [model_colors[m] for m in sorted_models]

    # 绘制水平条形图（排名更直观）
    bars = ax.barh(sorted_models, sorted_accs, color=sorted_colors, alpha=0.8)

    # 在条形右侧添加数值标注（保留4位小数）
    for bar, acc in zip(bars, sorted_accs):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2.,
                f'{acc:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')

    # 图表美化
    ax.set_xlabel('Accuracy（整体准确率）', fontsize=12, fontweight='bold')
    ax.set_title('各模型整体故障诊断准确率排名', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.8, 1.0)  # 聚焦有效范围
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加冠军标注（最高准确率模型）
    ax.text(sorted_accs[0] + 0.017, 0, '★ 最优模型',
            ha='left', va='center', fontsize=12, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('图2_模型准确率排名.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. 图3：单模型下多故障类型Recall对比
def plot_model_fault_recall_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算x轴位置
    x = np.arange(len(fault_types))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    # 绘制每个模型的Recall条形
    for idx, (model, metrics) in enumerate(model_metrics.items()):
        recalls = [metrics["recall"][ft] for ft in fault_types]
        bars = ax.bar(x + offsets[idx], recalls, width,
                      label=model, color=model_colors[model], alpha=0.8)

        # 在条形顶部添加数值标注
        for bar, r in zip(bars, recalls):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{r:.2f}', ha='center', va='bottom', fontsize=10)

    # 图表美化
    ax.set_xlabel('故障类型', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall（召回率）', fontsize=12, fontweight='bold')
    ax.set_title('各模型对不同故障类型的召回率对比（召回率越高，漏检越少）',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(fault_names, fontsize=11)
    ax.set_ylim(0.6, 1.05)  # 突出RF在IR上的低召回率问题
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加参考线（90%召回线，工业诊断最低要求）
    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, label='90%召回线')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('图3_模型-故障召回率对比.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. 执行所有可视化函数
if __name__ == "__main__":
    plot_fault_precision_comparison()  # 图1：故障-模型精度对比
    plot_model_accuracy_ranking()  # 图2：模型准确率排名
    plot_model_fault_recall_comparison()  # 图3：模型-故障召回率对比
