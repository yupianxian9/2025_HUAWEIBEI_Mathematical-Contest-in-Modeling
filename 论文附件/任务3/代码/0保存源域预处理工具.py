import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os

# 核心配置
SOURCE_RAW_FEAT_FILE = "source_domain_features_resampled_32k.csv"  
TARGET_COL = "fault_type"  # 标签列
SAVE_DIR = "./source_utils"  # 源域工具保存目录
PCA_EXPLAINED_VARIANCE = 0.8  

# 定义PCA分组
PCA_GROUPS = {
    "group1_amplitude": ["rms", "std", "variance", "peak_to_peak"],  # 幅值相关高特征
    "group2_shape": ["kurtosis", "impulse_factor", "crest_factor", "shape_factor"],  # 形状相关高特征
    "group3_frequency": ["rms_freq", "freq_centroid"]  # 频率相关高特征
}

# 1. 加载并验证源域数据
if not os.path.exists(SOURCE_RAW_FEAT_FILE):
    raise FileNotFoundError(f"源域原始特征文件不存在：{SOURCE_RAW_FEAT_FILE}\n请确认第二问数据路径正确！")

df_source = pd.read_csv(SOURCE_RAW_FEAT_FILE)
# 分离特征与标签
df_source_feats = df_source.drop(columns=[TARGET_COL], errors="ignore")

# 验证PCA分组特征是否都在源域中
all_pca_feats = [f for group in PCA_GROUPS.values() for f in group]
missing_pca_feats = [f for f in all_pca_feats if f not in df_source_feats.columns]
if missing_pca_feats:
    raise ValueError(f"源域缺少PCA分组特征：{missing_pca_feats}\n请检查第二问特征提取结果！")

# 定义non_pca特征
NON_PCA_FEATURES = [
    "mean", "skewness", "std_freq", "kurtosis_psd",
    "env_peak_freq_1", "env_peak_freq_2", "env_peak_freq_3"
]
missing_non_pca_feats = [f for f in NON_PCA_FEATURES if f not in df_source_feats.columns]
if missing_non_pca_feats:
    raise ValueError(f"源域缺少non_pca特征：{missing_non_pca_feats}\n请重新生成第二问源域特征！")

# 2. 训练标准化器
scalers = {}
# 2.1 non_pca特征标准化器
scaler_non_pca = StandardScaler()
scaler_non_pca.fit(df_source_feats[NON_PCA_FEATURES])
scalers["non_pca"] = scaler_non_pca

# 2.2 各PCA组特征标准化器
for group_name, group_feats in PCA_GROUPS.items():
    scaler = StandardScaler()
    scaler.fit(df_source_feats[group_feats])
    scalers[group_name] = scaler

# 3. 训练PCA模型
pca_models = {}
for group_name, group_feats in PCA_GROUPS.items():
    # 用该组标准化器预处理源域特征
    X_scaled = scalers[group_name].transform(df_source_feats[group_feats])
    # 训练PCA
    pca = PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=42)
    pca.fit(X_scaled)
    pca_models[group_name] = pca
    print(f"{group_name}：{len(group_feats)}个原始特征 → PCA后{len(pca.components_)}个主成分（解释方差：{pca.explained_variance_ratio_.sum():.2%}）")

# 4. 保存预处理工具（供目标域复用）
os.makedirs(SAVE_DIR, exist_ok=True)

# 保存标准化器
with open(os.path.join(SAVE_DIR, "scalers.pkl"), "wb") as f:
    pickle.dump(scalers, f)

# 保存PCA模型
with open(os.path.join(SAVE_DIR, "pca_models.pkl"), "wb") as f:
    pickle.dump(pca_models, f)

# 保存特征信息
feature_info = {
    "non_pca_features": NON_PCA_FEATURES,
    "pca_groups": PCA_GROUPS,
    "source_feat_columns": df_source_feats.columns.tolist()
}
with open(os.path.join(SAVE_DIR, "feature_info.pkl"), "wb") as f:
    pickle.dump(feature_info, f)

print(f"\n源域预处理工具已全部保存至：{SAVE_DIR}")
print("包含文件：")
print("1. scalers.pkl → 标准化器（non_pca+3个PCA组）")
print("2. pca_models.pkl → PCA模型（3个特征组）")
print("3. feature_info.pkl → 特征分组与顺序信息")