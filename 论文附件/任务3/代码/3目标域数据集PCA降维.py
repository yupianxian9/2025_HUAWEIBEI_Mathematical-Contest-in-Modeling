import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def perform_grouped_pca(
    file_path: str,
    pca_groups: dict[str, list[str]],
    target_column: str,
    output_file: str,
    explained_variance: float = 0.8  
) -> pd.DataFrame:
    """
    对CSV文件中的指定特征组进行PCA降维，并将结果与原始非PCA特征合并。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()

    # 分离出要进行PCA的特征和保留的非PCA特征
    all_pca_features = [feature for group in pca_groups.values() for feature in group]
    all_columns = df.columns.tolist()
    non_pca_features = [col for col in all_columns if col not in all_pca_features and col != target_column]

    df_non_pca = df[non_pca_features].copy()
    df_target = df[[target_column]].copy()

    # 创建一个列表来存储每个PCA组的结果
    pca_results_dfs = []

    for group_name, features in pca_groups.items():
        if not all(col in df.columns for col in features):
            print(f"Warning: Not all features in group '{group_name}' were found in the DataFrame. Skipping PCA for this group.")
            continue
        
        # 标准化数据
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 执行PCA，使用80%的解释方差比例
        pca = PCA(n_components=explained_variance)
        principal_components = pca.fit_transform(X_scaled)
        
        # 为新的主成分创建列名
        pca_columns = [f'{group_name}_PC{i+1}' for i in range(principal_components.shape[1])]
        df_pca = pd.DataFrame(data=principal_components, columns=pca_columns)
        
        print(f"Group '{group_name}': {len(features)} original features were reduced to {df_pca.shape[1]} principal components.")
        
        pca_results_dfs.append(df_pca)

    # 水平合并所有结果
    if not pca_results_dfs:
        print("No PCA groups were processed. Returning original non-PCA features and target.")
        final_df = pd.concat([df_non_pca, df_target], axis=1)
    else:
        final_df = pd.concat([df_non_pca] + pca_results_dfs + [df_target], axis=1)
    
    # 保存为新的CSV文件
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved the new DataFrame to '{output_file}'.")
    
    return final_df

if __name__ == "__main__":
    # 定义您的文件路径和特征分组
    file_name = 'target_domain_raw_features_expanded.csv'
    output_file_name = 'target_domain_features_with_pca.csv'
    target_col = 'file_name'  # 用于排除无需降维的变量

    pca_groups_to_process = {
        'group1_amplitude': ['rms', 'std', 'variance', 'peak_to_peak'],
        'group2_shape': ['kurtosis', 'impulse_factor', 'crest_factor', 'shape_factor'],
        'group3_frequency': ['rms_freq', 'freq_centroid']
    }

    # 执行代码，明确指定explained_variance=0.8
    combined_df = perform_grouped_pca(
        file_path=file_name,
        pca_groups=pca_groups_to_process,
        target_column=target_col,
        output_file=output_file_name,
        explained_variance=0.8  # 明确指定80%的解释方差比例
    )

    if not combined_df.empty:
        print("\nFinal Merged DataFrame Head:")
        print(combined_df.head())
        print("\nFinal Merged DataFrame Columns:")
        print(combined_df.columns)