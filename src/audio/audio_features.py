# -*- coding: utf-8 -*-
"""
音频特征提取模块
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


class AudioFeatureExtractor:
    """音频特征提取器"""

    def __init__(self, sr: int = None):
        """
        初始化音频特征提取器

        Args:
            sr: 采样率，None表示保持原采样率
        """
        self.sr = sr

    def extract_features(self, audio_path: str) -> Dict:
        """
        提取单个音频文件的特征

        Args:
            audio_path: 音频文件路径

        Returns:
            Dict: 特征字典
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sr)

            features = {}

            # 1. 时长
            features['duration'] = librosa.get_duration(y=y, sr=sr)

            # 2. RMS能量
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)

            # 3. 过零率
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)

            # 4. 谱心
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)

            # 5. 谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

            # 6. 谱滚降
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)

            return features

        except Exception as e:
            print(f"提取特征时出错 ({audio_path}): {e}")
            return {}

    def extract_mfcc_features(self, audio_path: str, n_mfcc: int = 20) -> Dict:
        """
        提取MFCC特征（Level B）

        Args:
            audio_path: 音频文件路径
            n_mfcc: MFCC系数数量

        Returns:
            Dict: MFCC特征字典
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sr)

            # 提取MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            features = {}
            # 计算每个MFCC系数的均值和标准差
            for i in range(n_mfcc):
                features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc{i+1}_std'] = np.std(mfccs[i])

            return features

        except Exception as e:
            print(f"提取MFCC特征时出错 ({audio_path}): {e}")
            return {}

    def parse_filename(self, filename: str) -> Dict:
        """
        解析音频文件名，提取标签信息
        格式: weapon_distance_direction_id.mp3

        Args:
            filename: 文件名

        Returns:
            Dict: 标签信息
        """
        try:
            # 移除扩展名
            name = Path(filename).stem
            parts = name.split('_')

            if len(parts) >= 4:
                return {
                    'weapon': parts[0],
                    'distance': parts[1],
                    'direction': parts[2],
                    'id': parts[3]
                }
            else:
                return {
                    'weapon': 'unknown',
                    'distance': 'unknown',
                    'direction': 'unknown',
                    'id': '0'
                }
        except Exception as e:
            print(f"解析文件名时出错 ({filename}): {e}")
            return {}

    def process_audio_directory(self, audio_dir: str, output_csv: str,
                                 use_mfcc: bool = False) -> pd.DataFrame:
        """
        批量处理音频目录，提取所有音频的特征

        Args:
            audio_dir: 音频目录路径
            output_csv: 输出CSV文件路径
            use_mfcc: 是否使用MFCC特征

        Returns:
            pd.DataFrame: 特征数据框
        """
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob('**/*.mp3')) + list(audio_dir.glob('**/*.wav'))

        if not audio_files:
            print(f"在 {audio_dir} 中没有找到音频文件")
            return pd.DataFrame()

        print(f"找到 {len(audio_files)} 个音频文件")

        data = []
        for audio_file in tqdm(audio_files, desc="提取特征"):
            # 解析文件名
            labels = self.parse_filename(audio_file.name)

            # 提取基础特征
            features = self.extract_features(str(audio_file))

            # 如果需要，提取MFCC特征
            if use_mfcc:
                mfcc_features = self.extract_mfcc_features(str(audio_file))
                features.update(mfcc_features)

            # 合并标签和特征
            row = {**labels, **features, 'file_path': str(audio_file)}
            data.append(row)

        # 创建数据框
        df = pd.DataFrame(data)

        # 保存到CSV
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"特征已保存到: {output_csv}")

        return df


class AudioVisualizer:
    """音频可视化工具"""

    def __init__(self):
        """初始化可视化工具"""
        import matplotlib.pyplot as plt
        self.plt = plt

    def plot_waveform(self, audio_path: str, sr: int = None):
        """
        绘制波形图

        Args:
            audio_path: 音频文件路径
            sr: 采样率
        """
        import librosa.display

        y, sr = librosa.load(audio_path, sr=sr)

        self.plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=sr)
        self.plt.title(f'Waveform - {Path(audio_path).name}')
        self.plt.xlabel('Time (s)')
        self.plt.ylabel('Amplitude')
        self.plt.tight_layout()
        self.plt.show()

    def plot_spectrogram(self, audio_path: str, sr: int = None):
        """
        绘制频谱图

        Args:
            audio_path: 音频文件路径
            sr: 采样率
        """
        import librosa.display

        y, sr = librosa.load(audio_path, sr=sr)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        self.plt.figure(figsize=(14, 5))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        self.plt.colorbar(format='%+2.0f dB')
        self.plt.title(f'Spectrogram - {Path(audio_path).name}')
        self.plt.tight_layout()
        self.plt.show()

    def plot_features_comparison(self, df: pd.DataFrame, feature: str,
                                  group_by: str = 'weapon'):
        """
        绘制特征对比图

        Args:
            df: 特征数据框
            feature: 要对比的特征
            group_by: 分组依据（weapon, distance, direction）
        """
        import seaborn as sns

        self.plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=group_by, y=feature)
        self.plt.title(f'{feature} by {group_by}')
        self.plt.xticks(rotation=45)
        self.plt.tight_layout()
        self.plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """
        绘制特征相关性矩阵

        Args:
            df: 特征数据框
        """
        import seaborn as sns

        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()

        self.plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
        self.plt.title('Feature Correlation Matrix')
        self.plt.tight_layout()
        self.plt.show()
