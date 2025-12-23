# -*- coding: utf-8 -*-
"""
Enhanced feature extraction v2.5

Based on v2 (140D) + adds temporal/transient features to better separate
single-shot vs multi-shot patterns at long distance (e.g. AK vs M24 at 600m/front).

Feature dim: 140 + 10 = 150
Extra (10):
  - rms_max, rms_p95
  - onset_env_mean, onset_env_std, onset_env_max
  - onset_count, onset_rate
  - tempo
  - ioi_mean, ioi_std (inter-onset interval)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


def _feature_order_v2_5(n_mfcc: int = 20) -> list[str]:
    base = [
        "duration",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_flatness_mean",
        "spectral_flatness_std",
    ]

    mfcc = []
    for i in range(1, n_mfcc + 1):
        mfcc.extend(
            [
                f"mfcc{i}_mean",
                f"mfcc{i}_std",
                f"mfcc{i}_delta_mean",
                f"mfcc{i}_delta_std",
                f"mfcc{i}_delta2_mean",
                f"mfcc{i}_delta2_std",
            ]
        )

    contrast = [f"spectral_contrast{i}_mean" for i in range(1, 8)]

    extra = [
        "rms_max",
        "rms_p95",
        "onset_env_mean",
        "onset_env_std",
        "onset_env_max",
        "onset_count",
        "onset_rate",
        "tempo",
        "ioi_mean",
        "ioi_std",
    ]

    return base + mfcc + contrast + extra


def extract_enhanced_features_v2_5_from_audio(y: np.ndarray, sr: int, n_mfcc: int = 20) -> dict | None:
    try:
        features: dict[str, float] = {}

        # Base (same as v2.1/v2.4 runtime)
        features["duration"] = float(librosa.get_duration(y=y, sr=sr))

        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
        features["spectral_flatness_std"] = float(np.std(spectral_flatness))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfccs, order=1)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        for i in range(n_mfcc):
            idx = i + 1
            features[f"mfcc{idx}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{idx}_std"] = float(np.std(mfccs[i]))
            features[f"mfcc{idx}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{idx}_delta_std"] = float(np.std(mfcc_delta[i]))
            features[f"mfcc{idx}_delta2_mean"] = float(np.mean(mfcc_delta2[i]))
            features[f"mfcc{idx}_delta2_std"] = float(np.std(mfcc_delta2[i]))

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            idx = i + 1
            features[f"spectral_contrast{idx}_mean"] = float(np.mean(spectral_contrast[i]))

        # Extra temporal/transient features
        features["rms_max"] = float(np.max(rms)) if len(rms) else 0.0
        features["rms_p95"] = float(np.quantile(rms, 0.95)) if len(rms) else 0.0

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["onset_env_mean"] = float(np.mean(onset_env)) if len(onset_env) else 0.0
        features["onset_env_std"] = float(np.std(onset_env)) if len(onset_env) else 0.0
        features["onset_env_max"] = float(np.max(onset_env)) if len(onset_env) else 0.0

        try:
            onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
            onset_times = np.asarray(onset_times, dtype=float).reshape(-1)
        except Exception:
            onset_times = np.asarray([], dtype=float)

        onset_count = int(onset_times.shape[0])
        dur = float(features["duration"]) if float(features["duration"]) > 1e-6 else 1e-6
        features["onset_count"] = float(onset_count)
        features["onset_rate"] = float(onset_count / dur)

        try:
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            features["tempo"] = float(tempo[0]) if len(tempo) else 0.0
        except Exception:
            features["tempo"] = 0.0

        if onset_count >= 2:
            ioi = np.diff(onset_times)
            features["ioi_mean"] = float(np.mean(ioi))
            features["ioi_std"] = float(np.std(ioi))
        else:
            features["ioi_mean"] = 0.0
            features["ioi_std"] = 0.0

        # Ensure stable column order and completeness
        order = _feature_order_v2_5(n_mfcc=n_mfcc)
        missing = [k for k in order if k not in features]
        if missing:
            raise ValueError(f"missing v2.5 features: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        return {k: features[k] for k in order}

    except Exception as e:
        print(f"[ERR] feature_extraction_v2_5: {e}")
        return None


def extract_enhanced_features_v2_5(audio_path: str | Path, n_mfcc: int = 20) -> dict | None:
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        return extract_enhanced_features_v2_5_from_audio(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"[ERR] {audio_path} - {e}")
        return None


def parse_filename(filename: str) -> dict:
    try:
        name = Path(filename).stem
        parts = name.split("_")
        if len(parts) >= 4:
            return {"weapon": parts[0], "distance": parts[1], "direction": parts[2], "id": parts[3]}
        return {"weapon": "unknown", "distance": "unknown", "direction": "unknown", "id": "0"}
    except Exception:
        return {"weapon": "unknown", "distance": "unknown", "direction": "unknown", "id": "0"}


def process_audio_directory(audio_dir: str | Path, output_csv: str | Path, n_mfcc: int = 20) -> None:
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)
    audio_files = list(audio_dir.glob("**/*.mp3")) + list(audio_dir.glob("**/*.wav"))

    data = []
    failed = 0
    for audio_file in tqdm(audio_files, desc="extract v2.5 features"):
        info = parse_filename(audio_file.name)
        feats = extract_enhanced_features_v2_5(audio_file, n_mfcc=n_mfcc)
        if feats is None:
            failed += 1
            continue
        row = {**info, **feats, "file_path": str(audio_file)}
        data.append(row)

    df = pd.DataFrame(data)
    if "distance" in df.columns:
        df["distance_m"] = df["distance"].apply(lambda x: float(str(x).replace("m", "")) if str(x) != "None" else -1.0)
        df["distance_label"] = df["distance"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    exclude_cols = ["weapon", "distance", "direction", "id", "file_path", "distance_m", "distance_label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"\nOK: {len(data)} files, FAIL: {failed} files")
    print(f"feature_dim: {len(feature_cols)}")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    process_audio_directory(
        audio_dir=PROJECT_DIR / "data" / "gun_sound_train",
        output_csv=FEATURES_DIR / "train_enhanced_features_v2_5.csv",
        n_mfcc=20,
    )
    process_audio_directory(
        audio_dir=PROJECT_DIR / "data" / "gun_sound_test",
        output_csv=FEATURES_DIR / "test_enhanced_features_v2_5.csv",
        n_mfcc=20,
    )

