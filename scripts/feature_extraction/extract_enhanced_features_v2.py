# -*- coding: utf-8 -*-
"""
改进版特征提取 - 添加Delta和Delta-Delta MFCC
特征维度: 60 → 140
预期提升: 2-4%
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


def extract_enhanced_features_v2_from_audio(y, sr, n_mfcc=20):
    """
    提取增强特征 v2

    新增:
        - Delta MFCC (40维)
        - Delta-Delta MFCC (40维)

    总特征: 140维
        - 基础: 13维（含谱平坦度 2 维）
        - MFCC静态: 40维
        - MFCC Delta: 40维
        - MFCC Delta2: 40维
        - 谱对比度: 7维
    """
    try:
        features = {}

        # === 1. 基础特征 (13维) ===
        features['duration'] = librosa.get_duration(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)

        # === 2. MFCC + Delta + Delta2 (120维) ===
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 计算一阶差分 (Delta)
        mfcc_delta = librosa.feature.delta(mfccs, order=1)

        # 计算二阶差分 (Delta-Delta)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        for i in range(n_mfcc):
            # 静态MFCC
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i])

            # Delta MFCC (变化速度)
            features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])

            # Delta-Delta MFCC (加速度)
            features[f'mfcc{i+1}_delta2_mean'] = np.mean(mfcc_delta2[i])
            features[f'mfcc{i+1}_delta2_std'] = np.std(mfcc_delta2[i])

        # === 3. 谱对比度 (7维) ===
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast{i+1}_mean'] = np.mean(spectral_contrast[i])

        return features

    except Exception as e:
        print(f"错误: feature_extraction - {e}")
        return None


def extract_enhanced_features_v2(audio_path, n_mfcc=20):
    """
    从音频文件提取增强特征 v2（对外入口，内部复用 extract_enhanced_features_v2_from_audio）
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        return extract_enhanced_features_v2_from_audio(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"错误: {audio_path} - {e}")
        return None


def parse_filename(filename):
    """解析音频文件名"""
    try:
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
        print(f"解析文件名错误: {filename} - {e}")
        return {}


def process_audio_directory(audio_dir, output_csv, n_mfcc=20):
    """批量处理音频目录"""
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('*.mp3'))

    if not audio_files:
        print(f"错误: 在 {audio_dir} 中未找到mp3文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"提取增强特征v2 (包含Delta MFCC)...")

    data = []
    failed = 0

    for audio_file in tqdm(audio_files, desc="提取特征"):
        # 解析文件名
        file_info = parse_filename(audio_file.name)

        # 提取特征
        features = extract_enhanced_features_v2(audio_file, n_mfcc=n_mfcc)

        if features is not None:
            # 合并文件信息和特征
            row = {**file_info, **features, 'file_path': str(audio_file)}
            data.append(row)
        else:
            failed += 1

    # 创建数据框
    df = pd.DataFrame(data)

    # 计算距离（米）
    if 'distance' in df.columns:
        df['distance_m'] = df['distance'].apply(
            lambda x: float(x.replace('m', '')) if x != 'None' else -1.0
        )
        df['distance_label'] = df['distance']

    # 保存到CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n完成!")
    print(f"成功: {len(data)} 个文件")
    print(f"失败: {failed} 个文件")
    exclude_cols = ['weapon', 'distance', 'direction', 'id',
                    'file_path', 'distance_m', 'distance_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"特征维度: {len(feature_cols)} 维")
    print(f"特征已保存到: {output_csv}")

    # 显示前10个特征名称
    print(f"\n特征列表 ({len(feature_cols)}维):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"  {i:2d}. {col}")
    print(f"  ... (还有{len(feature_cols)-10}个特征)")


if __name__ == "__main__":
    print("=" * 80)
    print("增强特征提取脚本 v2")
    print("新增: Delta MFCC + Delta-Delta MFCC")
    print("=" * 80)

    n_mfcc = 20

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 处理训练集
    print("\n处理训练集...")
    process_audio_directory(
        audio_dir=str(PROJECT_DIR / "data" / "gun_sound_train"),
        output_csv=str(FEATURES_DIR / "train_enhanced_features_v2.csv"),
        n_mfcc=n_mfcc
    )

    # 处理测试集
    print("\n处理测试集...")
    process_audio_directory(
        audio_dir=str(PROJECT_DIR / "data" / "gun_sound_test"),
        output_csv=str(FEATURES_DIR / "test_enhanced_features_v2.csv"),
        n_mfcc=n_mfcc
    )

    print("\n" + "=" * 80)
    print("全部完成!")
    print("=" * 80)

    print("\n下一步: 使用 train_enhanced_model_v2.py 训练改进模型")
