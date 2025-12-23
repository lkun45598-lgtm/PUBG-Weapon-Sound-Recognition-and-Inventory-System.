# -*- coding: utf-8 -*-
"""
Build centroid artifacts for inference-time correction.

This script computes per-class centroids and p50/p95 radii in the same
"standardized feature space" used by v2.* models (StandardScaler output),
and writes them into `models/centroids_*.json`.

Why:
- The runtime recognizer can use these artifacts to correct a known hotspot:
  AK at (600m, front) being misclassified as M24/M4 with high confidence.

Note:
- Output lives under `models/` and is intentionally ignored by `.gitignore`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_DIR / "data" / "features"
MODELS_DIR = PROJECT_DIR / "models"


def _default_train_csv() -> Path:
    candidates = [
        FEATURES_DIR / "train_enhanced_features_v2_aug.csv",
        FEATURES_DIR / "train_enhanced_features_v2.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def _load_xy(train_csv: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(train_csv)
    exclude_cols = ["weapon", "distance", "direction", "id", "file_path", "distance_m", "distance_label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df["weapon"].values
    return X, y, feature_cols


def _build_centroid_artifacts(
    X_scaled: np.ndarray,
    y_encoded: np.ndarray,
    classes: list[str],
) -> dict:
    centroids: dict[str, list[float]] = {}
    radii_p50: dict[str, float] = {}
    radii_p95: dict[str, float] = {}
    counts: dict[str, int] = {}

    for class_idx, class_label in enumerate(classes):
        mask = (y_encoded == class_idx)
        n = int(np.sum(mask))
        if n <= 0:
            continue

        Xc = X_scaled[mask]
        centroid = np.mean(Xc, axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)

        centroids[str(class_label)] = centroid.astype(float).tolist()
        radii_p50[str(class_label)] = float(np.quantile(dists, 0.50))
        radii_p95[str(class_label)] = float(np.quantile(dists, 0.95))
        counts[str(class_label)] = n

    return {
        "space": "standardized",
        "feature_dim": int(X_scaled.shape[1]),
        "classes": list(classes),
        "centroids": centroids,
        "radii_p50": radii_p50,
        "radii_p95": radii_p95,
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["ensemble_v2_5", "ensemble_v2_4", "v2.1"],
        help="Which model family to build centroid artifacts for.",
    )
    parser.add_argument(
        "--train_csv",
        default=str(_default_train_csv()),
        help="Training feature CSV path (default: auto-pick augmented then base).",
    )
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    if not train_csv.is_absolute():
        train_csv = (PROJECT_DIR / train_csv).resolve()

    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")

    if args.model == "ensemble_v2_5":
        scaler_path = MODELS_DIR / "scaler_ensemble_v2_5.pkl"
        encoder_path = MODELS_DIR / "label_encoder_ensemble_v2_5.pkl"
        out_path = MODELS_DIR / "centroids_ensemble_v2_5.json"
    elif args.model == "ensemble_v2_4":
        scaler_path = MODELS_DIR / "scaler_ensemble_v2_4.pkl"
        encoder_path = MODELS_DIR / "label_encoder_ensemble_v2_4.pkl"
        out_path = MODELS_DIR / "centroids_ensemble_v2_4.json"
    else:
        scaler_path = MODELS_DIR / "scaler_enhanced_v2_1.pkl"
        encoder_path = MODELS_DIR / "label_encoder_enhanced_v2_1.pkl"
        out_path = MODELS_DIR / "centroids_enhanced_v2_1.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"label_encoder not found: {encoder_path}")

    scaler = joblib.load(str(scaler_path))
    label_encoder = joblib.load(str(encoder_path))

    X, y, feature_cols = _load_xy(train_csv)
    if int(getattr(scaler, "n_features_in_", X.shape[1])) != X.shape[1]:
        raise ValueError(
            f"feature dim mismatch: csv_dim={X.shape[1]} vs scaler.n_features_in_={getattr(scaler, 'n_features_in_', None)}"
        )

    y_encoded = label_encoder.transform(y)
    X_scaled = scaler.transform(X)

    artifacts = _build_centroid_artifacts(X_scaled=X_scaled, y_encoded=y_encoded, classes=list(label_encoder.classes_))
    artifacts["train_csv"] = str(train_csv)
    artifacts["feature_cols"] = list(feature_cols)

    MODELS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
