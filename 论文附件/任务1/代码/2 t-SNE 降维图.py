import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Dict

#  1. 加载特征数据 
file_path: str = 'source_domain_features_resampled_32k.csv'
try:
    df_features: pd.DataFrame = pd.read_csv(file_path)
    if df_features.empty:
        print("警告: CSV文件为空，无法进行可视化。")
        exit()
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请确保文件路径正确。")
    exit()

print("数据加载成功，总样本数:", len(df_features))

#  2. 准备数据：分离特征和标签 
X = df_features.drop('label', axis=1)
y = df_features['label']

samples_per_class: int = 500
df_sampled: pd.DataFrame = df_features.groupby('label').apply(
    lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
).reset_index(drop=True)

X_sampled = df_sampled.drop('label', axis=1)
y_sampled = df_sampled['label']

print(f"已从每种类别中抽取 {samples_per_class} 个样本进行t-SNE降维。")
print("抽样后各类别样本数量:")
print(y_sampled.value_counts())

#  3. 使用t-SNE进行降维 
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

# 对抽样后的数据进行降维
X_tsne = tsne.fit_transform(X_sampled)

#  4. 可视化绘制 
df_tsne = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['label'] = y_sampled

# 使用seaborn和matplotlib绘制散点图
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 10))

# 使用seaborn的scatterplot函数，并根据'label'列着色
sns.scatterplot(
    x='TSNE1',
    y='TSNE2',
    hue='label',
    palette=sns.color_palette("hsv", n_colors=len(df_tsne['label'].unique())),
    data=df_tsne,
    legend="full",
    alpha=0.7
)

plt.title('t-SNE Visualization of Bearing Fault Features', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 调整图例位置
plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

