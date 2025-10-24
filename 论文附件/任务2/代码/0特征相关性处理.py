import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def perform_correlation_analysis(file_path: str):
    """
    对CSV文件中的特征进行相关性分析并可视化
    """
    try:
        # 加载特征数据
        features_df = pd.read_csv(file_path)
        
        # 排除非数值型特征和标签列 'fault_type'
        features_to_analyze = features_df.drop(columns=['fault_type'], errors='ignore')
        
        # 计算相关系数矩阵
        corr_matrix = features_to_analyze.corr()
        
        # 绘制热力图
        plt.figure(figsize=(25, 22))
        ax = sns.heatmap(corr_matrix, 
                         annot=True,  
                         cmap='coolwarm', 
                         fmt=".2f",
                         linewidths=.5, 
                         # 保持注释字体大小不变
                         annot_kws={"size": 8},
                         cbar_kws={'label': '相关系数'})
        
        # 设置标题字体大小
        plt.title('特征相关性热力图', fontsize=26, pad=30)
        
        # 调整x轴和y轴标签的倾斜角度和对齐方式
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')
        
        # 调整子图布局参数
        plt.subplots_adjust(top=0.9, bottom=0.2)
        
        plt.show()

        # 打印高度相关的特征对
        print("\n--- 高度相关的特征对 (绝对值 > 0.8) ---")
        corr_pairs = corr_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        
        high_corr_pairs = sorted_pairs[sorted_pairs.abs() > 0.8]
        # 过滤掉特征自身与自身的1.0相关性
        high_corr_pairs = high_corr_pairs[high_corr_pairs < 1.0] 
        print(high_corr_pairs.to_string())

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。请确保文件路径正确。")
    except Exception as e:
        print(f"发生错误：{e}")

# 执行相关性分析
if __name__ == '__main__':
    features_file = 'source_domain_features_resampled_32k.csv'
    perform_correlation_analysis(features_file)