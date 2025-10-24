import os
import re
import typing
from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.stats import kurtosis, skew
from matplotlib import rcParams
# 替换您当前的字体设置
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei"]
#  配置参数 
# 轴承参数：滚动体数 n, 滚动体直径 d (英寸), 轴承节径 D (英寸), 接触角 alpha (度)
BEARING_PARAMS: Dict[str, Dict[str, float]] = {
    'SKF6205': {'n': 9, 'd': 0.3126, 'D': 1.537, 'alpha_deg': 0},
    'SKF6203': {'n': 9, 'd': 0.2656, 'D': 1.122, 'alpha_deg': 0}
}
# 目标采样频率
FS_TARGET: int = 32000
# 信号分段的窗口长度和重叠率（基于目标采样率设置）
WINDOW_SIZE: int = 4096
OVERLAP: float = 0.5


#  1. 数据加载与处理 
def find_data_key(mat_file_content: Dict[str, Any], sensor_type: str) -> str:
    """从.mat文件的键中找到包含特定传感器类型后缀的数据键"""
    suffix: str = f'_{sensor_type}_time'
    for key in mat_file_content.keys():
        if key.endswith(suffix):
            return key
    raise ValueError(f"在文件中未找到后缀为 '{suffix}' 的数据键")


def resample_signal(signal_data: np.ndarray, fs_original: int, fs_target: int) -> np.ndarray:
    """对信号进行重采样（从原始采样率到目标采样率）"""
    num_target = int(len(signal_data) * fs_target / fs_original)
    return signal.resample(signal_data, num_target)


def load_mat_data(file_path: str) -> Tuple[np.ndarray, int, int, Dict[str, float]]:
    """加载.mat文件，提取振动信号、RPM，重采样到FS_TARGET，并解析轴承参数"""
    try:
        # 解析原始采样率（从文件路径提取）
        fs_match = re.search(r'(\d+)kHz', file_path)
        if not fs_match:
            raise ValueError("无法从文件路径中解析出采样频率 (kHz)")
        fs_original = int(fs_match.group(1)) * 1000

        # 解析传感器类型和轴承类型
        path_parts = file_path.split(os.sep)
        sensor_type: str = ''
        bearing_type: str = ''
        for part in path_parts:
            if 'Normal_data' in part:
                sensor_type = 'DE'
                bearing_type = 'SKF6205'
            elif 'DE_data' in part:
                sensor_type = 'DE'
                bearing_type = 'SKF6205'
            elif 'FE_data' in part:
                sensor_type = 'FE'
                bearing_type = 'SKF6203'
        if not sensor_type:
            raise ValueError("无法从文件路径中解析出传感器类型")

        # 加载数据并提取信号
        data_content: Dict[str, Any] = scipy.io.loadmat(file_path)
        data_key: str = find_data_key(data_content, sensor_type)
        signal_data: np.ndarray = data_content[data_key].flatten()

        # 重采样到目标频率
        resampled_signal = resample_signal(signal_data, fs_original, FS_TARGET)

        # 解析RPM（优先从文件内容提取，其次从文件名提取）
        rpm_val: Any = data_content.get('RPM')
        if rpm_val is not None and rpm_val.size > 0:
            rpm = int(rpm_val.flatten()[0])
        else:
            try:
                rpm_str = file_path.split('(')[1].split('rpm')[0]
                rpm = int(rpm_str)
            except IndexError:
                rpm = 1797  # 默认常见RPM值

        return resampled_signal, rpm, fs_original, BEARING_PARAMS[bearing_type]
    except Exception as e:
        print(f"加载文件 {file_path} 出错: {e}")
        return np.array([]), 0, 0, {}


#  2. 故障机理计算 
def calculate_fault_frequencies(rpm: int, params: Dict[str, float]) -> Dict[str, float]:
    """根据轴承参数和转速计算理论故障频率（BPFI/BPFO/BSF/fr）"""
    fr: float = rpm / 60.0  # 转频
    alpha_rad: float = np.deg2rad(params['alpha_deg'])
    d_D_ratio: float = params['d'] / params['D'] * np.cos(alpha_rad)

    # 内圈故障频率（BPFI）、外圈故障频率（BPFO）、滚动体故障频率（BSF）
    bpfi: float = (params['n'] / 2.0) * fr * (1 + d_D_ratio)
    bpfo: float = (params['n'] / 2.0) * fr * (1 - d_D_ratio)
    bsf: float = (params['D'] / (2.0 * params['d'])) * fr * (1 - d_D_ratio ** 2)

    return {'fr': fr, 'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


#  3. 重构可视化：按特征类型归类绘图 
def plot_time_domain_summary(fault_data: Dict[str, Tuple[np.ndarray, int, int, Dict[str, float]]],
                             fs_target: int, max_points: int = 300) -> None:
    """绘制所有48kHz故障类型的时域波形汇总图（2x2子图布局）"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # 展平轴数组便于循环
    fig.suptitle('48kHz Bearing Fault Time Domain Summary (Resampled to 32kHz)', fontsize=16, fontweight='bold')

    fault_names = list(fault_data.keys())
    for idx, fault_name in enumerate(fault_names):
        sig, rpm, fs_original, _ = fault_data[fault_name]
        # 下采样信号（避免图形过于密集）
        downsample_factor = max(1, len(sig) // max_points)
        sampled_signal = sig[::downsample_factor]
        sampled_time = np.arange(len(sampled_signal)) * downsample_factor / fs_target

        # 绘制时域波形
        ax = axes[idx]
        ax.plot(sampled_time, sampled_signal, linewidth=0.8, color='blue', label='Time Domain')
        # 子图配置
        ax.set_title(f'{fault_name}\n(Original FS: {fs_original / 1000}kHz, RPM: {rpm})', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Acceleration (g)', fontsize=10)
        ax.set_xlim(0, 0.5)  # 统一x轴范围（便于对比）
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题预留空间
    plt.show()


def plot_frequency_spectrum_summary(fault_data: Dict[str, Tuple[np.ndarray, int, int, Dict[str, float]]],
                                    fs_target: int, max_points: int = 300) -> None:
    """绘制所有48kHz故障类型的频谱汇总图（2x2子图布局）"""
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('48kHz Bearing Fault Frequency Spectrum Summary (Resampled to 32kHz)', fontsize=16, fontweight='bold')

    # 故障频率颜色配置（统一视觉风格）
    freq_colors = {'BPFI': 'red', 'BPFO': 'green', 'BSF': 'magenta', 'fr': 'black'}
    fault_names = list(fault_data.keys())

    for idx, fault_name in enumerate(fault_names):
        sig, rpm, fs_original, bearing_params = fault_data[fault_name]
        fault_freqs = calculate_fault_frequencies(rpm, bearing_params)

        # 计算FFT
        fft_vals = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(len(sig), d=1 / fs_target)
        n_freq = len(fft_freq) // 2  # 取正频率部分
        # 下采样FFT结果
        downsample_factor = max(1, n_freq // max_points)
        freq_points = fft_freq[:n_freq:downsample_factor]
        amp_points = np.abs(fft_vals)[:n_freq:downsample_factor]

        # 绘制频谱
        ax = axes[idx]
        ax.plot(freq_points, amp_points, linewidth=0.8, color='purple', label='Spectrum')
        # 标记理论故障频率及其2、3次谐波
        for freq_name, freq_val in fault_freqs.items():
            if freq_name in freq_colors and freq_val <= 2000:  # 只标记x轴范围内的频率
                # 基频
                ax.axvline(x=freq_val, color=freq_colors[freq_name], linestyle='--', linewidth=2, alpha=0.8)
                # 2、3次谐波
                for harmonic in [2, 3]:
                    harm_freq = harmonic * freq_val
                    if harm_freq <= 2000:
                        ax.axvline(x=harm_freq, color=freq_colors[freq_name], linestyle=':', linewidth=1.5, alpha=0.6)
                # 添加图例（每个频率只显示一次）
                ax.plot([], [], color=freq_colors[freq_name], linestyle='--', linewidth=2,
                        label=f'{freq_name}: {freq_val:.2f}Hz')

        # 子图配置
        ax.set_title(f'{fault_name}\n(Original FS: {fs_original / 1000}kHz, RPM: {rpm})', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_xlim(0, 2000)  # 统一x轴范围（聚焦故障频率区间）
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_envelope_spectrum_summary(fault_data: Dict[str, Tuple[np.ndarray, int, int, Dict[str, float]]],
                                   fs_target: int, max_points: int = 300) -> None:
    """绘制所有48kHz故障类型的包络谱汇总图（2x2子图布局）"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('48kHz Bearing Fault Envelope Spectrum Summary (Resampled to 32kHz)', fontsize=16, fontweight='bold')

    # 故障频率颜色配置（与频谱图一致）
    freq_colors = {'BPFI': 'red', 'BPFO': 'green', 'BSF': 'magenta', 'fr': 'black'}
    fault_names = list(fault_data.keys())

    for idx, fault_name in enumerate(fault_names):
        sig, rpm, fs_original, bearing_params = fault_data[fault_name]
        fault_freqs = calculate_fault_frequencies(rpm, bearing_params)

        # 计算包络（希尔伯特变换）
        analytic_signal = signal.hilbert(sig)
        envelope = np.abs(analytic_signal)
        envelope = envelope - np.mean(envelope)  # 去除直流分量
        # 计算包络FFT
        fft_vals = np.fft.fft(envelope)
        fft_freq = np.fft.fftfreq(len(envelope), d=1 / fs_target)
        n_env = len(fft_freq) // 2  # 取正频率部分
        # 下采样包络FFT结果
        downsample_factor = max(1, n_env // max_points)
        freq_points = fft_freq[:n_env:downsample_factor]
        amp_points = np.abs(fft_vals)[:n_env:downsample_factor]

        # 绘制包络谱
        ax = axes[idx]
        ax.plot(freq_points, amp_points, linewidth=0.8, color='orange', label='Envelope')
        # 标记理论故障频率及其2、3次谐波
        for freq_name, freq_val in fault_freqs.items():
            if freq_name in freq_colors and freq_val <= 500:  # 包络谱聚焦低频率区间
                # 基频
                ax.axvline(x=freq_val, color=freq_colors[freq_name], linestyle='--', linewidth=2, alpha=0.8)
                # 2、3次谐波
                for harmonic in [2, 3]:
                    harm_freq = harmonic * freq_val
                    if harm_freq <= 500:
                        ax.axvline(x=harm_freq, color=freq_colors[freq_name], linestyle=':', linewidth=1.5, alpha=0.6)
                # 添加图例
                ax.plot([], [], color=freq_colors[freq_name], linestyle='--', linewidth=2,
                        label=f'{freq_name}: {freq_val:.2f}Hz')

        # 子图配置
        ax.set_title(f'{fault_name}\n(Original FS: {fs_original / 1000}kHz, RPM: {rpm})', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_xlim(0, 500)  # 统一x轴范围
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#  4. 特征提取
def get_time_domain_features(segment: np.ndarray) -> Dict[str, float]:
    """提取时域特征"""
    rms = np.sqrt(np.mean(segment ** 2))
    peak_to_peak = np.max(segment) - np.min(segment)
    mean_abs = np.mean(np.abs(segment))
    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'rms': rms,
        'peak_to_peak': peak_to_peak,
        'kurtosis': kurtosis(segment),
        'skewness': skew(segment),
        'crest_factor': np.max(np.abs(segment)) / rms if rms > 0 else 0,
        'shape_factor': rms / mean_abs if mean_abs > 0 else 0,
        'impulse_factor': np.max(np.abs(segment)) / mean_abs if mean_abs > 0 else 0,
        'variance': np.var(segment)
    }


def get_freq_domain_features(segment: np.ndarray, fs: int) -> Dict[str, float]:
    """提取频域特征"""
    n = len(segment)
    fft_vals = np.fft.fft(segment)
    psd = (np.abs(fft_vals) ** 2) / n  # 功率谱密度
    fft_freq = np.fft.fftfreq(n, d=1 / fs)
    psd = psd[:n // 2]
    fft_freq = fft_freq[:n // 2]

    # 频率重心、RMS频率、频率标准差
    sum_psd = np.sum(psd)
    freq_centroid = np.sum(fft_freq * psd) / sum_psd if sum_psd > 0 else 0
    rms_freq = np.sqrt(np.sum(psd * (fft_freq ** 2)) / sum_psd) if sum_psd > 0 else 0
    std_freq = np.sqrt(np.sum(psd * ((fft_freq - freq_centroid) ** 2)) / sum_psd) if sum_psd > 0 else 0

    return {
        'freq_centroid': freq_centroid,
        'rms_freq': rms_freq,
        'std_freq': std_freq,
        'kurtosis_psd': kurtosis(psd),
    }


def get_envelope_features(segment: np.ndarray, fs: int) -> Dict[str, float]:
    """提取包络谱峰值特征"""
    analytic_signal = signal.hilbert(segment)
    envelope = np.abs(analytic_signal)
    envelope = envelope - np.mean(envelope)
    n = len(envelope)
    fft_vals = np.fft.fft(envelope)
    fft_freq = np.fft.fftfreq(n, d=1 / fs)
    amps = np.abs(fft_vals[:n // 2])
    freqs = fft_freq[:n // 2]

    # 取前3个最大峰值的频率
    peak_indices = np.argsort(amps)[-3:]  # 降序排序后的索引
    peak_freqs = freqs[peak_indices]

    return {
        'env_peak_freq_1': peak_freqs[2],  # 最大峰值频率
        'env_peak_freq_2': peak_freqs[1],  # 第二大峰值频率
        'env_peak_freq_3': peak_freqs[0],  # 第三大峰值频率
    }


def extract_features_from_file(file_path: str, label: str) -> List[Dict[str, Any]]:
    """按窗口分段提取单个文件的所有特征（时域+频域+包络谱）"""
    signal_data, _, _, _ = load_mat_data(file_path)
    if signal_data.size == 0:
        return []

    features_list: List[Dict[str, Any]] = []
    step: int = int(WINDOW_SIZE * (1 - OVERLAP))  # 窗口步长（重叠率50%）
    for i in range(0, len(signal_data) - WINDOW_SIZE + 1, step):
        segment: np.ndarray = signal_data[i:i + WINDOW_SIZE]
        # 合并所有特征
        all_features: Dict[str, Any] = {}
        all_features.update(get_time_domain_features(segment))
        all_features.update(get_freq_domain_features(segment, FS_TARGET))
        all_features.update(get_envelope_features(segment, FS_TARGET))
        all_features['label'] = label  # 添加故障标签
        features_list.append(all_features)

    return features_list

if __name__ == '__main__':
    base_data_path: str = '源域数据集'
    print(" Processing 48kHz Bearing Fault Data Visualization ")

    # 1. 仅保留48kHz的代表性文件
    representative_48khz_files: Dict[str, str] = {
        'Normal': os.path.join(base_data_path, '48kHz_Normal_data', 'N_0.mat'),
        'IR Fault': os.path.join(base_data_path, '48kHz_DE_data', 'IR', '0007', 'IR007_1.mat'),
        'OR Fault': os.path.join(base_data_path, '48kHz_DE_data', 'OR', 'Centered', '0007',
                                                    'OR007@6_1.mat'),
        'Ball Fault': os.path.join(base_data_path, '48kHz_DE_data', 'B', '0007', 'B007_1.mat'),
    }

    # 2. 加载所有48kHz数据
    fault_data_48khz: Dict[str, Tuple[np.ndarray, int, int, Dict[str, float]]] = {}
    for fault_name, file_path in representative_48khz_files.items():
        if os.path.exists(file_path):
            sig, rpm, fs_original, bearing_params = load_mat_data(file_path)
            if sig.size > 0:
                fault_data_48khz[fault_name] = (sig, rpm, fs_original, bearing_params)
                print(f"Loaded: {fault_name} (48kHz)")
            else:
                print(f"Warning: {fault_name} file has no data, skipped")
        else:
            print(f"❌ Error: File not found: {file_path}")

    # 3. 按特征类型绘制汇总图（时域→频域→包络谱）
    if fault_data_48khz:
        print("\nPlotting Time Domain Summary...")
        plot_time_domain_summary(fault_data_48khz, FS_TARGET)

        print("Plotting Frequency Spectrum Summary...")
        plot_frequency_spectrum_summary(fault_data_48khz, FS_TARGET)

        print("Plotting Envelope Spectrum Summary...")
        plot_envelope_spectrum_summary(fault_data_48khz, FS_TARGET)
    else:
        print("\nNo 48kHz data loaded, visualization skipped")

    # 4. 批量特征提取（保持原逻辑不变，包含所有采样率数据）
    print("\n Batch Feature Extraction ")
    data_sources: Dict[str, str] = {
        'N': os.path.join(base_data_path, '48kHz_Normal_data'),
        'B': os.path.join(base_data_path, '12kHz_DE_data', 'B'),
        'IR': os.path.join(base_data_path, '12kHz_DE_data', 'IR'),
        'OR': os.path.join(base_data_path, '12kHz_DE_data', 'OR'),
        'B_48k': os.path.join(base_data_path, '48kHz_DE_data', 'B'),
        'IR_48k': os.path.join(base_data_path, '48kHz_DE_data', 'IR'),
        'OR_48k': os.path.join(base_data_path, '48kHz_DE_data', 'OR'),
        'B_FE': os.path.join(base_data_path, '12kHz_FE_data', 'B'),
        'IR_FE': os.path.join(base_data_path, '12kHz_FE_data', 'IR'),
        'OR_FE': os.path.join(base_data_path, '12kHz_FE_data', 'OR'),
    }

    all_data_features: List[Dict[str, Any]] = []
    for label, path in data_sources.items():
        cleaned_label = label.split('_')[0]  # 清洗标签（如B_48k→B）
        if not os.path.isdir(path):
            print(f"Warning: Directory not found {path}, skipped")
            continue

        print(f"Processing: {cleaned_label} (Path: {path})")
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    file_features = extract_features_from_file(file_path, cleaned_label)
                    all_data_features.extend(file_features)

    # 5. 保存特征到CSV
    if all_data_features:
        features_df = pd.DataFrame(all_data_features)
        output_path = 'source_domain_features_resampled_32k.csv'
        features_df.to_csv(output_path, index=False)
        print(f"\nFeature extraction completed! Generated {len(features_df)} samples")
        print(f"Features saved to: {output_path}")
        print("\nSample distribution by class:")
        print(features_df['label'].value_counts())
        print("\nFeature data preview:")
        print(features_df.head())
    else:
        print("\nNo features extracted, check data paths and files")