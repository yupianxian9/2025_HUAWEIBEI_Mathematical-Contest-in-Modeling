import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 1. 数据准备 
# 假设您的特征数据已保存为CSV文件
try:
    df = pd.read_csv('combined_features_with_pca.csv')
except FileNotFoundError:
    print("错误：未找到 'combined_features_with_pca.csv' 文件。请确保文件路径正确。")
    exit()

# 字段选取：使用人工筛选后的核心特征
selected_features_cols = df.drop(columns=['fault_type']).columns.tolist()

features = df[selected_features_cols]
labels = df['fault_type']

# 标签编码
label_mapping = {
    'N': 0,
    'OR': 1,
    'IR': 2,
    'B': 3
}
y = labels.map(label_mapping)

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(features)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 构建 MLP 模型 
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 增加Dropout层防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# 模型初始化
input_dim = X_train.shape[1]
output_dim = len(label_mapping)
model = MLP(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 模型训练 
epochs = 50
print("\n正在训练深度学习模型 ")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# 4. 模型评估 
print("\n评估模型性能 ")
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    _, y_pred = torch.max(y_pred_probs, 1)

y_pred_np = y_pred.numpy()
y_test_np = y_test_tensor.numpy()

# 评估指标
accuracy = accuracy_score(y_test_np, y_pred_np)
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test_np, y_pred_np, target_names=label_mapping.keys()))

# 绘制混淆矩阵
cm = confusion_matrix(y_test_np, y_pred_np)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=label_mapping.keys(), 
    yticklabels=label_mapping.keys()
)
plt.title('MLP 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()