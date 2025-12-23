# -*- coding: utf-8 -*-
"""
训练集特征增强：专门针对 (weapon=ak, distance=600m, direction=front)

为什么这样做：
- 评估显示 ak 在 (600m,front) 场景下最容易高置信错判成 m24/m4
- 这些错误与 zcr + mfcc(delta2_std 等) 的统计特征强相关，典型符合“远距离衰减 + 带宽受限 + 噪声”

做法：
- 读取 data/features/train_enhanced_features_v2.csv
- 找到满足条件的样本行，读取其 file_path 对应音频
- 生成若干增强版本（低通 + 衰减 + 轻噪声 + 轻混响）
- 对增强音频重新提取 v2 特征，追加到新 CSV：
    data/features/train_enhanced_features_v2_aug.csv

注意：
- 不修改 test CSV（评估集保持不变）
- 不覆盖原始 train CSV（避免破坏原数据）；训练脚本可优先读取 *_aug.csv
"""

from __future__ import annotations

import zlib
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd

# 同目录导入（避免依赖 scripts/ 作为可导入包；也支持从项目根目录 import 本文件）
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from extract_enhanced_features_v2 import extract_enhanced_features_v2_from_audio


PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


TARGET_WEAPON = "ak"
TARGET_DISTANCE = "600m"
TARGET_DIRECTION = "front"

# 每条目标样本生成多少条增强样本（建议 2~4；过大会导致分布失衡）
NUM_AUG_PER_SAMPLE = 3


def _stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8", errors="ignore")
    return int(zlib.crc32(s) & 0xFFFFFFFF)


def _apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    return y * float(10.0 ** (gain_db / 20.0))


def _lowpass_fft(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """
    轻量低通：在频域做平滑衰减（避免依赖 scipy）
    """
    if cutoff_hz <= 0:
        return y

    n = int(len(y))
    if n <= 8:
        return y

    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    # smooth mask: 1 below cutoff, cosine roll-off until cutoff*1.2, then 0
    hi = cutoff_hz * 1.2
    mask = np.ones_like(freqs, dtype=float)
    mask[freqs >= hi] = 0.0
    mid = (freqs >= cutoff_hz) & (freqs < hi)
    if np.any(mid):
        x = (freqs[mid] - cutoff_hz) / max(hi - cutoff_hz, 1e-6)
        mask[mid] = 0.5 * (1.0 + np.cos(np.pi * x))

    y2 = np.fft.irfft(Y * mask, n=n)
    return y2.astype(y.dtype, copy=False)


def _add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    eps = 1e-10
    rms = float(np.sqrt(np.mean(y * y) + eps))
    if rms <= eps:
        return y

    noise = rng.standard_normal(size=y.shape).astype(y.dtype, copy=False)
    noise_rms = float(np.sqrt(np.mean(noise * noise) + eps))
    noise = noise / max(noise_rms, eps)

    # target noise rms so that 20*log10(signal/noise)=snr_db
    target_noise_rms = rms / float(10.0 ** (snr_db / 20.0))
    return y + noise * target_noise_rms


def _add_light_reverb(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    轻混响：用短脉冲响应做卷积（FFT），控制在很轻的强度，模拟远距离/环境反射。
    """
    n = len(y)
    ir_len = int(sr * rng.uniform(0.08, 0.18))
    ir_len = max(ir_len, 32)

    t = np.linspace(0, 1, ir_len, endpoint=False)
    decay = rng.uniform(3.0, 6.0)
    ir = np.exp(-decay * t).astype(np.float32)
    # sprinkle a few early reflections
    for _ in range(rng.integers(2, 6)):
        idx = int(rng.integers(0, ir_len))
        ir[idx] += float(rng.uniform(0.05, 0.20))

    ir = ir / float(np.sum(np.abs(ir)) + 1e-6)

    # FFT convolution
    size = int(2 ** np.ceil(np.log2(n + ir_len)))
    Y = np.fft.rfft(y, n=size)
    H = np.fft.rfft(ir, n=size)
    out = np.fft.irfft(Y * H, n=size)[:n]

    mix = float(rng.uniform(0.05, 0.15))
    return (1.0 - mix) * y + mix * out.astype(y.dtype, copy=False)


def augment_far_front(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    面向“600m/front”场景的增强：带宽受限 + 衰减 + 轻噪声 + 轻混响
    """
    y2 = np.asarray(y, dtype=np.float32)

    # attenuation: -6dB ~ -18dB
    y2 = _apply_gain_db(y2, gain_db=float(rng.uniform(-18.0, -6.0)))

    # band-limited: cutoff 2500~4500 Hz
    y2 = _lowpass_fft(y2, sr=sr, cutoff_hz=float(rng.uniform(2500.0, 4500.0)))

    # mild reverb
    y2 = _add_light_reverb(y2, sr=sr, rng=rng)

    # noise: SNR 12~25 dB
    y2 = _add_white_noise(y2, snr_db=float(rng.uniform(12.0, 25.0)), rng=rng)

    # avoid clipping
    peak = float(np.max(np.abs(y2)) + 1e-9)
    if peak > 0.98:
        y2 = y2 * (0.98 / peak)

    return y2.astype(np.float32, copy=False)


def main() -> int:
    train_csv = FEATURES_DIR / "train_enhanced_features_v2.csv"
    out_csv = FEATURES_DIR / "train_enhanced_features_v2_aug.csv"

    if not train_csv.exists():
        print(f"[FAIL] 缺少训练特征文件: {train_csv}")
        return 2

    df = pd.read_csv(train_csv)
    required = {"weapon", "distance", "direction", "id", "file_path"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[FAIL] 训练CSV缺少列: {missing}")
        return 2

    target = df[
        (df["weapon"].astype(str) == TARGET_WEAPON)
        & (df["distance"].astype(str) == TARGET_DISTANCE)
        & (df["direction"].astype(str) == TARGET_DIRECTION)
    ].copy()

    print("Train rows:", len(df))
    print("Target rows:", len(target))
    if target.empty:
        print("[WARN] 没有找到目标子集，未生成增强数据。")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("Wrote:", out_csv)
        return 0

    # Feature columns: keep exactly the same set as input CSV
    out_cols = list(df.columns)

    augmented_rows = []
    for _, row in target.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.exists():
            continue

        # load once
        y, sr = librosa.load(file_path, sr=None)
        base_id = str(row["id"])

        for k in range(1, NUM_AUG_PER_SAMPLE + 1):
            seed = _stable_seed(str(file_path), base_id, f"aug{k}")
            rng = np.random.default_rng(seed)

            y_aug = augment_far_front(y=y, sr=sr, rng=rng)
            feats = extract_enhanced_features_v2_from_audio(y=y_aug, sr=sr, n_mfcc=20)
            if feats is None:
                continue

            new_row = row.to_dict()
            new_row.update(feats)
            new_row["id"] = f"{base_id}_aug{k}"
            new_row["file_path"] = f"{file_path}#aug{k}"
            augmented_rows.append(new_row)

    print("Augmented rows generated:", len(augmented_rows))

    if augmented_rows:
        df2 = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    else:
        df2 = df

    # Ensure column order matches original
    for c in out_cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[out_cols]

    df2.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Wrote:", out_csv)
    print("Final rows:", len(df2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
