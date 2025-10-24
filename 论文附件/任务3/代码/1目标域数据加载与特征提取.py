import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft, hilbert

# 目标域文件路径
TARGET_MAT_DIR = r"D:\竞赛和项目\数学建模\25华为杯\中文赛题ACDEF题\E题\数据集\目标域数据集"
# 目标域信号参数
SAMPLE_RATE = 32000  # 采样频率（Hz）
COLLECT_TIME = 8     # 采集时间（秒）
FFT_WINDOW_SIZE = 2048  # 频域处理窗口
# 滑动窗口参数
WINDOW_SIZE = FFT_WINDOW_SIZE  # 分割窗口大小
OVERLAP_RATIO = 0.5  # 重叠率
# 输出路径
SAVE_RAW_FEAT_PATH = "target_domain_raw_features_expanded.csv"

# 工具函数：手动汉宁窗
def hanning_window(window_size: int) -> np.ndarray:
    n = np.arange(window_size)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_size - 1))

# 特征计算函数
def calc_time_domain_feats(signal: np.ndarray) -> dict:
    """计算10个时域特征"""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal ** 2))
    peak_to_peak_val = np.max(signal) - np.min(signal)
    kurtosis_val = kurtosis(signal, fisher=True)  
    skewness_val = skew(signal)
    abs_signal = np.abs(signal)
    mean_abs_val = np.mean(abs_signal)
    crest_factor_val = np.max(abs_signal) / rms_val if rms_val != 0 else 0.0
    shape_factor_val = rms_val / mean_abs_val if mean_abs_val != 0 else 0.0
    impulse_factor_val = np.max(abs_signal) / mean_abs_val if mean_abs_val != 0 else 0.0
    variance_val = np.var(signal)

    return {
        "mean": mean_val, "std": std_val, "rms": rms_val, "peak_to_peak": peak_to_peak_val,
        "kurtosis": kurtosis_val, "skewness": skewness_val, "crest_factor": crest_factor_val,
        "shape_factor": shape_factor_val, "impulse_factor": impulse_factor_val, "variance": variance_val
    }


def calc_freq_domain_feats(signal: np.ndarray) -> dict:
    """计算4个频域特征"""
    window = hanning_window(FFT_WINDOW_SIZE)
    signal_windowed = signal[:FFT_WINDOW_SIZE] * window  # 加窗处理

    # FFT计算
    n_fft = FFT_WINDOW_SIZE
    fft_vals = fft(signal_windowed)
    freq_axis = np.fft.fftfreq(n_fft, 1 / SAMPLE_RATE)[:n_fft // 2]
    psd = np.abs(fft_vals[:n_fft // 2]) ** 2 / n_fft  
    psd_normalized = psd / np.sum(psd)

    # 特征计算
    freq_centroid_val = np.sum(freq_axis * psd_normalized)  # 频率重心
    rms_freq_val = np.sqrt(np.sum(freq_axis ** 2 * psd_normalized))
    std_freq_val = np.sqrt(np.sum((freq_axis - freq_centroid_val) ** 2 * psd_normalized))
    psd_mean = np.mean(psd)
    psd_std = np.std(psd)
    kurtosis_psd_val = np.mean(((psd - psd_mean) / psd_std) ** 4) if psd_std != 0 else 0.0

    return {
        "freq_centroid": freq_centroid_val, "rms_freq": rms_freq_val,
        "std_freq": std_freq_val, "kurtosis_psd": kurtosis_psd_val
    }

def calc_envelope_spectrum_feats(signal: np.ndarray) -> dict:
    """计算3个包络谱特征"""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # 包络谱计算
    window = hanning_window(FFT_WINDOW_SIZE)
    envelope_windowed = envelope[:FFT_WINDOW_SIZE] * window
    n_fft = FFT_WINDOW_SIZE
    fft_env = fft(envelope_windowed)
    freq_axis = np.fft.fftfreq(n_fft, 1 / SAMPLE_RATE)[:n_fft // 2]
    env_psd = np.abs(fft_env[:n_fft // 2]) ** 2 / n_fft

    # 取前3个能量最大频率
    valid_idx = freq_axis > 1.0
    valid_freq = freq_axis[valid_idx]
    valid_env_psd = env_psd[valid_idx]
    if len(valid_env_psd) < 3:
        top3_freqs = np.pad(valid_freq, (0, 3 - len(valid_env_psd)), mode="constant")
    else:
        top3_idx = np.argsort(valid_env_psd)[::-1][:3]
        top3_freqs = valid_freq[top3_idx]

    return {
        "env_peak_freq_1": top3_freqs[0], "env_peak_freq_2": top3_freqs[1], "env_peak_freq_3": top3_freqs[2]
    }

# 批量加载与特征提取
def load_and_extract_target_feats() -> pd.DataFrame:
    """
    加载目标域的.mat文件，使用滑动窗口进行数据扩充，并提取特征。
    """
    mat_files = [f for f in os.listdir(TARGET_MAT_DIR)
                 if f.endswith(".mat") and f.split(".")[0] in "ABCDEFGHIJKLMNOP"]
    if len(mat_files) != 16:
        raise ValueError(f"目标域需16个文件（A-P），当前找到{len(mat_files)}个！路径：{TARGET_MAT_DIR}")

    target_feats = []
    expanded_file_names = []  

    # 计算滑动窗口参数
    step = int(WINDOW_SIZE * (1 - OVERLAP_RATIO)) 

    # 按A-P顺序处理每个文件
    for mat_file in sorted(mat_files, key=lambda x: x.split(".")[0]):
        file_prefix = mat_file.split(".")[0]  
        file_path = os.path.join(TARGET_MAT_DIR, mat_file)

        try:
            # 加载MAT文件
            mat_data = loadmat(file_path)
            if file_prefix not in mat_data:
                raise KeyError(f"{mat_file}缺少变量{file_prefix}！")

            # 提取信号
            signal = mat_data[file_prefix].flatten()
            if len(signal) != SAMPLE_RATE * COLLECT_TIME:
                print(f"{mat_file}信号长度异常（{len(signal)}点），预期256000点，仍继续处理")

            # 滑动窗口分割信号
            signal_len = len(signal)
            num_windows = (signal_len - WINDOW_SIZE) // step + 1  # 窗口总数
            print(f"处理{mat_file}：{num_windows}个滑动窗口（窗口{WINDOW_SIZE}点，重叠{OVERLAP_RATIO*100}%）")

            # 遍历每个窗口，提取特征
            for _ in range(num_windows):
                # 窗口截取
                start = _ * step
                end = start + WINDOW_SIZE
                if end > signal_len:
                    end = signal_len
                    start = end - WINDOW_SIZE
                win_signal = signal[start:end]

                # 计算17维特征
                time_feats = calc_time_domain_feats(win_signal)
                freq_feats = calc_freq_domain_feats(win_signal)
                env_feats = calc_envelope_spectrum_feats(win_signal)
                all_feats = {**time_feats, **freq_feats, **env_feats}

                # 保存特征与原始文件名
                target_feats.append(all_feats)
                expanded_file_names.append(mat_file)  # 直接保存完整文件名

            print(f"完成{mat_file}：生成{num_windows}个样本\n")

        except Exception as e:
            raise RuntimeError(f"处理{mat_file}失败：{str(e)}") from e

    # 转为DataFrame
    df_target = pd.DataFrame(target_feats)
    df_target.insert(0, "file_name", expanded_file_names)  

    # 校验特征数量
    if len(df_target.columns) - 1 != 17:
        raise ValueError(f"特征提取不完整！预期17个，实际{len(df_target.columns)-1}个\n请检查特征函数。")
    return df_target

# 主执行函数
if __name__ == "__main__":
    print("=" * 60)
    print("目标域数据加载与特征提取")
    print("=" * 60)

    # 执行提取与扩充
    df_target_raw = load_and_extract_target_feats()
    # 保存扩充后的数据
    df_target_raw.to_csv(SAVE_RAW_FEAT_PATH, index=False, encoding="utf-8-sig")
    # 输出结果统计
    total_samples = len(df_target_raw)
    original_files = len(set(df_target_raw["file_name"]))
    print(f"\n数据扩充完成！核心结果：")
    print(f"1. 数据规模：{original_files}个原始文件（A-P）→ {total_samples}个扩充样本 × 17个特征")
    print(f"2. 保存路径：{SAVE_RAW_FEAT_PATH}")
    print(f"\n3. 特征描述性统计：")
    stats_cols = ["mean", "std", "rms", "kurtosis", "freq_centroid"]
    print(df_target_raw[stats_cols].describe().round(4))  # 保留4位小数，与文档一致

    # 打印前5个扩充样本的对应关系
    print(f"\n4. 扩充样本示例：")
    sample_mapping = df_target_raw[["file_name"] + stats_cols].head(5)
    print(sample_mapping.to_string(index=False))