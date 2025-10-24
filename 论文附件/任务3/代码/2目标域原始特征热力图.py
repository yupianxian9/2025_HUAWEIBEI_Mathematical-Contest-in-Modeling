import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def create_correlation_heatmap(file_path: str, save_path: str):
    """
    加载数据并生成所有数值型特征的相关性热力图。
    """
    try:
        # 1. 加载CSV文件
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件路径正确。")
        return

    # 2. 移除非数值型列
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # 3. 计算相关性矩阵
    correlation_matrix = numeric_df.corr()

    # 4. 绘制热力图
    plt.figure(figsize=(12, 10))  
    
    # 绘制热力图
    sns.heatmap(
        correlation_matrix,
        annot=True,         # 显示相关性数值
        fmt=".2f",          # 格式化数值，保留两位小数
        cmap="coolwarm",    # 选择颜色映射方案
        linewidths=.5,      # 设置网格线宽度
        cbar_kws={'label': '相关系数'} # 设置颜色条标签
    )
    
    # 设置图表标题
    plt.title("目标域原始特征相关性热力图", fontsize=16)
    
    # 调整布局，防止标签重叠
    plt.tight_layout()
    
    # 5. 保存图片并显示
    plt.savefig(save_path)
    print(f"相关性热力图已成功保存至：{save_path}")
    plt.show()

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_csv_file = "target_domain_raw_features_expanded.csv"
    output_image_file = "correlation_heatmap.png"
    # 执行函数
    create_correlation_heatmap(input_csv_file, output_image_file)