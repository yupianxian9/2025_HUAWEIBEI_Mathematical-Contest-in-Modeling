import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 0. 创建图表保存文件夹 
output_folder = "可视化图"
os.makedirs(output_folder, exist_ok=True)
print(f" 确保图表保存文件夹 '{output_folder}' 已创建 ")

# 1. 定义 MLP 模型（作为特征提取器和分类器） 
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

# 2. 定义领域判别器 
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# 3. 数据准备 
# 加载源域数据（带标签）
print(" 准备源域数据 ")
try:
    df_source = pd.read_csv('combined_features_with_pca.csv')
except FileNotFoundError:
    print("错误：未找到 'combined_features_with_pca.csv' 文件。请确保文件路径正确。")
    exit()

selected_features_cols = df_source.drop(columns=['fault_type']).columns.tolist()
features_source = df_source[selected_features_cols]
labels_source = df_source['fault_type']

label_mapping_source = {
    'N': 0,
    'OR': 1,
    'IR': 2,
    'B': 3
}
y_source = labels_source.map(label_mapping_source)

# 加载目标域数据（无标签）
print(" 准备目标域数据（无标签） ")
try:
    df_target = pd.read_csv('target_domain_features_with_pca.csv')
except FileNotFoundError:
    print("错误：未找到 'target_domain_features_with_pca.csv' 文件。请确保文件路径正确。")
    exit()

if not all(col in df_target.columns for col in selected_features_cols):
    print("错误：目标域数据文件缺少关键特征列。请检查文件。")
    exit()

features_target = df_target[selected_features_cols]

# 特征标准化，使用源域的 StandardScaler
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(features_source)
X_target_scaled = scaler.transform(features_target)

# 转换为 PyTorch 张量
X_source_tensor = torch.tensor(X_source_scaled, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source.values, dtype=torch.long)
X_target_tensor = torch.tensor(X_target_scaled, dtype=torch.float32)

# 创建 DataLoader
source_dataset = TensorDataset(X_source_tensor, y_source_tensor)
target_dataset = TensorDataset(X_target_tensor, torch.zeros(len(X_target_tensor), dtype=torch.long))

source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)


# 4. 构建模型、损失函数和优化器 
input_dim = X_source_scaled.shape[1]
output_dim = len(label_mapping_source)

mlp_model = MLP(input_dim, output_dim)
domain_discriminator = DomainDiscriminator(64)

criterion_classifier = nn.CrossEntropyLoss()
criterion_domain = nn.BCEWithLogitsLoss()

optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=0.001)

# 4.5 训练前特征分布可视化 
def plot_tsne_features(features_source: np.ndarray, features_target: np.ndarray, title: str, filename: str):
    """使用 t-SNE 绘制特征分布图并保存"""
    all_features = np.vstack((features_source, features_target))
    domain_labels = ['源域'] * len(features_source) + ['目标域'] * len(features_target)
    
    # 随机采样以加速 t-SNE
    sample_size = min(2000, len(all_features))
    sample_indices = np.random.choice(len(all_features), size=sample_size, replace=False)
    sampled_features = all_features[sample_indices]
    sampled_domains = [domain_labels[i] for i in sample_indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(sampled_features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=sampled_domains, style=sampled_domains)
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    # 保存图像到指定文件夹
    plt.savefig(os.path.join(output_folder, filename))
    plt.show()

# 训练前
print("\n 训练前特征分布图 ")
with torch.no_grad():
    _, source_features_before = mlp_model(X_source_tensor)
    _, target_features_before = mlp_model(X_target_tensor)
plot_tsne_features(source_features_before.detach().numpy(), target_features_before.detach().numpy(), "训练前特征分布 (t-SNE)", "训练前特征分布图.png")

# 5. 对抗性训练 
epochs = 50
print("\n 正在进行无监督域适应训练 ")
# 记录损失以供后续绘图
classifier_losses = []
domain_losses = []

for epoch in range(epochs):
    mlp_model.train()
    domain_discriminator.train()
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    max_iter = max(len(source_loader), len(target_loader))
    
    total_classifier_loss = 0
    total_domain_loss = 0
    
    for i in range(max_iter):
        try:
            source_inputs, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_inputs, source_labels = next(source_iter)
            
        try:
            target_inputs, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_inputs, _ = next(target_iter)

        # 训练领域判别器
        optimizer_domain.zero_grad()
        _, source_features = mlp_model(source_inputs)
        _, target_features = mlp_model(target_inputs)
        combined_features = torch.cat((source_features, target_features), dim=0).detach()
        domain_labels = torch.cat((torch.zeros(source_features.size(0)),
                                   torch.ones(target_features.size(0))), dim=0).unsqueeze(1)
        domain_outputs = domain_discriminator(combined_features)
        domain_loss = criterion_domain(domain_outputs, domain_labels)
        domain_loss.backward()
        optimizer_domain.step()
        total_domain_loss += domain_loss.item()

        # 训练特征提取器和分类器
        optimizer_mlp.zero_grad()
        source_outputs, source_features = mlp_model(source_inputs)
        classifier_loss = criterion_classifier(source_outputs, source_labels)

        domain_outputs_adv = domain_discriminator(source_features)
        domain_loss_source = criterion_domain(domain_outputs_adv, torch.ones(source_features.size(0), 1))
        domain_outputs_adv = domain_discriminator(target_features)
        domain_loss_target = criterion_domain(domain_outputs_adv, torch.zeros(target_features.size(0), 1))
        adversarial_loss = domain_loss_source + domain_loss_target
        
        total_loss = classifier_loss + adversarial_loss
        total_loss.backward()
        optimizer_mlp.step()
        total_classifier_loss += classifier_loss.item()
    
    avg_classifier_loss = total_classifier_loss / max_iter
    avg_domain_loss = total_domain_loss / max_iter
    classifier_losses.append(avg_classifier_loss)
    domain_losses.append(avg_domain_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Classifier Loss: {avg_classifier_loss:.4f}, "
              f"Domain Loss: {avg_domain_loss:.4f}")

#  5.5 训练后可视化 
# 绘制损失曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), classifier_losses, label='分类器损失 (Source)')
plt.plot(range(1, epochs + 1), domain_losses, label='领域判别器损失 (Domain)')
plt.title('模型训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, "训练损失曲线图.png"))
plt.show()

# 训练后特征分布图
print("\n 训练后特征分布图 ")
with torch.no_grad():
    _, source_features_after = mlp_model(X_source_tensor)
    _, target_features_after = mlp_model(X_target_tensor)
plot_tsne_features(source_features_after.detach().numpy(), target_features_after.detach().numpy(), "训练后特征分布 (t-SNE)", "训练后特征分布图.png")

# 6. 使用训练好的模型进行预测并生成新的CSV文件 
print("\n 正在使用训练好的模型对目标域数据进行预测 ")
mlp_model.eval()

with torch.no_grad():
    # 对整个目标域数据集进行预测
    target_outputs, _ = mlp_model(X_target_tensor)
    _, target_preds_tensor = torch.max(target_outputs, 1)

# 将预测结果从 PyTorch 张量转换为 NumPy 数组
target_preds_np = target_preds_tensor.numpy()

# 逆向映射，将整数标签转回字符串标签
reverse_label_mapping = {v: k for k, v in label_mapping_source.items()}
predicted_labels = [reverse_label_mapping[pred] for pred in target_preds_np]

# 绘制预测结果柱状图
plt.figure(figsize=(8, 6))
sns.countplot(x=predicted_labels, order=list(label_mapping_source.keys()))
plt.title('目标域预测结果分布')
plt.xlabel('预测类别')
plt.ylabel('数量')
plt.grid(axis='y')
plt.savefig(os.path.join(output_folder, "预测结果柱状图.png"))
plt.show()

# 将预测结果添加到原始目标域数据框中
df_target['fault_type'] = predicted_labels

# 保存为新的CSV文件
output_filename = 'target_with_predictions.csv'
df_target.to_csv(output_filename, index=False)

print(f"\n 预测完成，新文件 '{output_filename}' 已成功生成。")
print(f"新文件包含 {df_target.shape[0]} 行和 {df_target.shape[1]} 列。")
import pickle
import os

# 创建保存目录
save_dir = "trained_models_and_features"
os.makedirs(save_dir, exist_ok=True)

# 1. 保存训练好的MLP模型
torch.save(mlp_model.state_dict(), os.path.join(save_dir, "mlp_model.pth"))
print(f"MLP模型保存至：{os.path.join(save_dir, 'mlp_model.pth')}")

# 2. 保存训练前后的特征（用于域间距离计算）
np.save(os.path.join(save_dir, "source_features_before.npy"), source_features_before.detach().numpy())
np.save(os.path.join(save_dir, "target_features_before.npy"), target_features_before.detach().numpy())
np.save(os.path.join(save_dir, "source_features_after.npy"), source_features_after.detach().numpy())
np.save(os.path.join(save_dir, "target_features_after.npy"), target_features_after.detach().numpy())
print(f"训练前后特征保存至：{save_dir}")

# 3. 保存标准化器
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"标准化器保存至：{save_dir}")

# 4. 保存特征列名
feature_info = {
    "selected_features_cols": selected_features_cols,
    "label_mapping": label_mapping_source
}
with open(os.path.join(save_dir, "feature_info.pkl"), "wb") as f:
    pickle.dump(feature_info, f)
print(f"特征信息保存至：{save_dir}")