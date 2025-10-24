# LIME局部可解释性分析
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import os
from lime import lime_tabular
import warnings

warnings.filterwarnings("ignore")
import seaborn as sns

# 配置
DOC_FAULT_CLASSES = ["正常(N)", "外圈故障(OR)", "内圈故障(IR)", "滚动体故障(B)"]  
DOC_RAW_FEATURES = ['mean', 'skewness', 'std_freq', 'kurtosis_psd', 'env_peak_freq_1',
                    'env_peak_freq_2', 'env_peak_freq_3', 'group1_amplitude_PC1',
                    'group2_shape_PC1', 'group3_frequency_PC1']  
DEVICE = torch.device("cpu")
FAULT_LABEL_MAP = {"N": 0, "OR": 1, "IR": 2, "B": 3}  
FAULT_LABEL_REV_MAP = {v: k for k, v in FAULT_LABEL_MAP.items()}

#  2. 加载文档目标域数据（带预测标签）
try:
    target_df = pd.read_csv("target_with_predictions.csv")
    print(f"目标域数据加载完成：形状{target_df.shape}，列名{target_df.columns.tolist()}")
    assert "fault_type" in target_df.columns, "缺失文档要求的'fault_type'列"
    assert all(feat in target_df.columns for feat in DOC_RAW_FEATURES), "特征列与文档不匹配"
except Exception as e:
    print(f"目标域数据加载失败：{str(e)}")
    exit()

# 3. 加载标准化器
try:
    with open("trained_models_and_features/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert list(scaler.feature_names_in_) == DOC_RAW_FEATURES, "标准化器特征与文档不匹配"
    print(f"标准化器加载完成：拟合特征与文档10个原始特征一致")
except Exception as e:
    print(f"标准化器加载失败：{str(e)}")
    exit()

# 4. 加载文档任务3的MLP迁移模型
class MLP(nn.Module):
    def __init__(self, input_dim=10, output_dim=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5)
        ).to(DEVICE)
        self.classifier = nn.Linear(64, output_dim).to(DEVICE) 

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        feats = self.feature_extractor(x)
        return self.classifier(feats)
try:
    mlp_model = MLP(input_dim=10, output_dim=4)
    mlp_model.load_state_dict(torch.load("trained_models_and_features/mlp_model.pth", map_location=DEVICE))
    mlp_model.eval()
    test_output = mlp_model(torch.randn(1, 10, device=DEVICE))
    assert test_output.shape == (1, 4), f"模型需输出(1,4)，当前{test_output.shape}"
    print(f"MLP迁移模型加载完成")
except Exception as e:
    print(f"❌ 模型加载失败：{str(e)}")
    exit()

# 5. 数据预处理
X_raw_df = target_df[DOC_RAW_FEATURES].copy()
X_scaled = scaler.transform(X_raw_df)  # 标准化特征矩阵
y_pred = target_df["fault_type"].map(FAULT_LABEL_MAP).values  
print(f"数据预处理完成：X_scaled形状{X_scaled.shape}，y_pred标签范围{np.min(y_pred)}-{np.max(y_pred)}")

# 6. 适配LIME的模型预测函数（输出概率）
def mlp_pred_proba(X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        logits = mlp_model(X_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba

# 7. 初始化LIME解释器
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=DOC_RAW_FEATURES,
    class_names=DOC_FAULT_CLASSES,
    mode="classification",
    random_state=42,
    discretize_continuous=False
)
print(f"LIME解释器初始化完成")

# 8. 选择并验证待解释样本
ir_sample_indices = np.where(y_pred == FAULT_LABEL_MAP["IR"])[0]
if len(ir_sample_indices) == 0:
    print(f"无IR样本，改用OR样本")
    ir_sample_indices = np.where(y_pred == FAULT_LABEL_MAP["OR"])[0]

ir_sample_idx = ir_sample_indices[0]
ir_sample = X_scaled[ir_sample_idx]
ir_sample_raw = X_raw_df.iloc[ir_sample_idx]
target_label = y_pred[ir_sample_idx]  # 待解释标签（如2=IR）
target_fault = DOC_FAULT_CLASSES[target_label]

print(f"\n待解释样本信息：")
print(f"    - 样本索引：{ir_sample_idx}")
print(f"    - 预测标签/故障类型：{target_label}/{target_fault}")
print(f"    - 文档关键特征值：")
print(f"      · std_freq（频率稳定性）：{ir_sample_raw['std_freq']:.4f}（文档：IR故障频率受转频调制）")
print(f"      · env_peak_freq_1（BPFI相关）：{ir_sample_raw['env_peak_freq_1']:.4f}")

# 9. 生成LIME解释
print(f"\n🔧 生成LIME解释（验证标签{target_label}有效性）...")
lime_explanation = lime_explainer.explain_instance(
    data_row=ir_sample,
    predict_fn=mlp_pred_proba,
    num_features=5,  
    num_samples=500,
    labels=[target_label]  
)

# 检查目标标签是否在局部解释中
if target_label not in lime_explanation.local_exp:
    print(f"标签{target_label}（{target_fault}）的局部解释缺失，改用LIME默认标签")
    # 获取LIME生成的有效标签（取第一个）
    valid_label = next(iter(lime_explanation.local_exp.keys()))
    target_label = valid_label
    target_fault = DOC_FAULT_CLASSES[target_label]
    print(f"    - 改用有效标签/故障类型：{target_label}/{target_fault}")
else:
    print(f"成功生成标签{target_label}（{target_fault}）的局部解释")

# 10. 可视化LIME解释
output_folder = "可视化图/事后可解释性_LIME"
os.makedirs(output_folder, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 为条形图添加值标签，并进行美化
def add_bar_labels_and_style(ax):
    """美化LIME条形图，添加带柔和颜色的标签，并调整外观"""
    positive_color = sns.xkcd_rgb["pale green"]
    negative_color = sns.xkcd_rgb["pale red"]

    # 调整条形宽度和间距
    ax.bar_width = 0.6

    # 获取所有条形并设置颜色
    bars = ax.patches
    for i, bar in enumerate(bars):
        bar_width = bar.get_width()
        x, y = bar.get_xy()

        # 设置条形颜色
        if bar_width > 0:
            bar.set_color(positive_color)
        else:
            bar.set_color(negative_color)

        # 在条形内部添加标签
        if bar_width >= 0:
            ax.text(bar_width - 0.005,  # 标签位置在条形内部，略微向左偏移
                    y + bar.get_height() / 2,
                    f'{bar_width:.2f}',  # 格式化为两位小数
                    ha='right',
                    va='center',
                    color='black',
                    fontsize=9,
                    weight='bold')
        else:
            ax.text(bar_width + 0.005,  # 标签位置在条形内部，略微向右偏移
                    y + bar.get_height() / 2,
                    f'{bar_width:.2f}',  # 格式化为两位小数
                    ha='left',
                    va='center',
                    color='black',
                    fontsize=9,
                    weight='bold')

# 绘制LIME解释图
try:
    fig = lime_explanation.as_pyplot_figure(label=target_label)
    ax = fig.gca()

    # 调整x轴范围，为内部标签留出空间
    max_abs_val = max(abs(bar.get_width()) for bar in ax.patches)
    ax.set_xlim(-max_abs_val - 0.05, max_abs_val + 0.05)

    # 添加背景网格线
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    add_bar_labels_and_style(ax)  # 调用新增函数进行美化和添加标签

    plt.title(
        f'LIME局部解释：样本{ir_sample_idx}诊断为{target_fault}的依据\n',
        fontsize=12, pad=20, weight='bold'
    )
    plt.xlabel('特征对诊断的贡献度', fontsize=10)

    # 隐藏边框
    sns.despine(ax=ax, top=True, right=True)

    fig_path = os.path.join(output_folder, f"LIME解释_样本{ir_sample_idx}_{target_fault}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LIME解释图保存至：{fig_path}")

except Exception as e:
    print(f"绘图过程异常：{str(e)}，改用文本输出关键特征")
    # 文本输出LIME结果（备选方案，确保不丢失文档机理信息）
    lime_top_feats = lime_explanation.as_list(label=target_label)
    print(f"\nLIME关键特征解释：")
    for i, (feat_name, weight) in enumerate(lime_top_feats, 1):
        print(f"    - Top{i}特征「{feat_name}」：权重{weight:.4f}")