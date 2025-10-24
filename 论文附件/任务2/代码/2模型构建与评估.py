import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

df = pd.read_csv('combined_features_with_pca.csv')

# 1. 分离特征和标签
# 排除与分类无关的列：file_name, label, rpm
features = df.drop(columns=['fault_type'])
labels = df['fault_type']

# 2. 标签编码
# 将字符串标签（'N', 'OR', 'IR', 'B'）转换为数字
label_mapping = {
    'N': 0,
    'OR': 1,
    'IR': 2,
    'B': 3
}
y = labels.map(label_mapping)
print("\nEncoded labels:", y.unique())

# 3. 特征标准化
# 使用 StandardScaler 对特征进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(features)

# 4. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

def train_and_evaluate(model, model_name):
    """
    通用函数，用于训练和评估机器学习模型
    """
    print(f"\n--- Training {model_name} ---")
    
    # 模型训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    print(f"\n--- Evaluation Results for {model_name} ---")
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # 类别报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=label_mapping.keys(), 
        yticklabels=label_mapping.keys()
    )
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

# 随机森林分类器
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

train_and_evaluate(rf_model, "Random Forest")

# XGBoost 分类器
xgb_model = XGBClassifier(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10.0,  # L2 正则化
    reg_alpha=0.5,   # L1 正则化
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=56
)

train_and_evaluate(xgb_model, "XGBoost")

# 支持向量机
# SVC可能会因计算量大而需要较长时间
svm_model = SVC(
    kernel='rbf',
    C=0.5,        # 默认=1.0，调小降低过拟合
    gamma=0.1,    # 默认='scale'，可调小
    random_state=42
)

train_and_evaluate(svm_model, "Support Vector Machine")