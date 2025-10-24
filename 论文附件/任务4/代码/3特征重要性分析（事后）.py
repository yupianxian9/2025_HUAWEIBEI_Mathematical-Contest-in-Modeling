import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# 设置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#  0. 创建图表保存文件夹 
output_folder = "可视化图"
os.makedirs(output_folder, exist_ok=True)
print(f" 确保图表保存文件夹 '{output_folder}' 已创建 ")

#  1. 定义 MLP 模型（与训练时保持一致） 
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

#  2. 加载训练好的模型和数据 
try:
    save_dir = "trained_models_and_features"
    # 加载模型
    mlp_model = MLP(input_dim=10, output_dim=4) 
    mlp_model.load_state_dict(torch.load(os.path.join(save_dir, "mlp_model.pth")))
    mlp_model.eval()

    # 加载标准化器
    with open(os.path.join(save_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # 加载特征信息
    with open(os.path.join(save_dir, "feature_info.pkl"), "rb") as f:
        feature_info = pickle.load(f)
        selected_features_cols = feature_info['selected_features_cols']
        label_mapping = feature_info['label_mapping']

    # 加载源域数据用于评估
    df_source = pd.read_csv('combined_features_with_pca.csv')
    features_source = df_source[selected_features_cols]
    labels_source = df_source['fault_type']
    y_source = labels_source.map(label_mapping)

    # 标准化特征
    X_source_scaled = scaler.transform(features_source)
    X_source_tensor = torch.tensor(X_source_scaled, dtype=torch.float32)
    y_source_tensor = torch.tensor(y_source.values, dtype=torch.long)

    print("\n 模型和数据加载成功，开始特征重要性分析 ")
except FileNotFoundError as e:
    print(f"错误: 无法加载所需文件。请确保您已成功运行迁移学习脚本并生成了 '{save_dir}' 文件夹及其内容。")
    print(f"具体错误: {e}")
    exit()

#  3. 定义评估函数 
def evaluate_model(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        outputs, _ = model(data)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == labels).float().mean().item()
    return accuracy

#  4. 计算基准准确率 
baseline_accuracy = evaluate_model(mlp_model, X_source_tensor, y_source_tensor)
print(f"模型在源域数据集上的基准准确率: {baseline_accuracy:.4f}")

#  5. 计算特征重要性 
feature_importances = {}
for i, feature_name in enumerate(selected_features_cols):
    # 创建一个副本以避免修改原始数据
    X_shuffled = X_source_tensor.clone().detach()
    
    # 打乱当前特征列
    shuffled_indices = torch.randperm(X_shuffled.size(0))
    X_shuffled[:, i] = X_shuffled[shuffled_indices, i]
    
    # 评估打乱后的准确率
    shuffled_accuracy = evaluate_model(mlp_model, X_shuffled, y_source_tensor)
    
    # 计算重要性（准确率下降值）
    importance = baseline_accuracy - shuffled_accuracy
    feature_importances[feature_name] = importance
    print(f"特征 '{feature_name}' 打乱后，准确率下降: {importance:.4f}")

# 将结果按重要性降序排序
sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
sorted_features = [item[0] for item in sorted_importances]
sorted_scores = [item[1] for item in sorted_importances]

#  6. 可视化结果 
plt.figure(figsize=(12, 8))
# 使用不同的颜色来区分正负重要性
colors = ['red' if score < 0 else 'skyblue' for score in sorted_scores]
sns.barplot(x=sorted_scores, y=sorted_features, palette=colors)

plt.title('迁移诊断模型特征重要性排名', fontsize=16)
plt.xlabel('准确率下降值（重要性）', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "特征重要性柱状图.png"))
plt.show()

print("\n 特征重要性分析完成，图表已保存。")