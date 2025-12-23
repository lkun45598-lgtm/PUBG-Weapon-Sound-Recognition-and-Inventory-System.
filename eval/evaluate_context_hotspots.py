# -*- coding: utf-8 -*-
"""
按 (distance, direction) 维度做误差热点分析（面向训练侧改进）

特点：
- 不需要读取音频文件，直接使用 data/features/test_enhanced_features_v2.csv 的特征
- 使用 src.audio.ModelLoader 加载当前最优模型（优先 v2.4 集成），并固定启用二阶段策略
- 输出整体准确率 + 各场景准确率 + 重点武器（ak/m4/m24）的场景误差
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# Ensure project root is on sys.path so `import src.*` works when running from eval/
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.audio import ModelLoader, AudioRecognizer


FEATURES_DIR = PROJECT_DIR / "data" / "features"


def _top_confusions(pairs: Counter[tuple[str, str]], top_n: int = 8) -> list[tuple[str, str, int]]:
    wrong = Counter({k: v for k, v in pairs.items() if k[0] != k[1]})
    return [(exp, pred, int(cnt)) for (exp, pred), cnt in wrong.most_common(top_n)]


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _maybe_filter_weapons(df: pd.DataFrame, weapons: set[str] | None) -> pd.DataFrame:
    if not weapons:
        return df
    return df[df["weapon"].isin(sorted(weapons))].copy()


def evaluate(
    *,
    focus_weapons: Iterable[str] = ("ak", "m4", "m24"),
    min_support: int = 15,
    only_distance: str | None = None,
    only_direction: str | None = None,
) -> int:
    if not _TORCH_AVAILABLE:
        print("[FAIL] 未安装 torch，无法做模型推理。")
        return 2

    model_loader = ModelLoader(PROJECT_DIR)
    ok, msg = model_loader.load_model()
    print("load_model:", ok, msg)
    if not ok:
        return 2

    recognizer = AudioRecognizer(model_loader)
    model_version = getattr(model_loader, "model_version", "") or ""
    expected_dim = getattr(model_loader.scaler, "n_features_in_", None)
    use_v2_5 = (
        model_version.startswith("v2.5")
        or model_version.startswith("ensemble_v2_5")
        or expected_dim == 150
    )

    if use_v2_5:
        test_csv_candidates = [
            FEATURES_DIR / "test_enhanced_features_v2_5.csv",
            FEATURES_DIR / "test_enhanced_features_v2.csv",
        ]
        order = recognizer._feature_order_v2_5()
    else:
        test_csv_candidates = [
            FEATURES_DIR / "test_enhanced_features_v2.csv",
            FEATURES_DIR / "test_enhanced_features_v2_5.csv",
        ]
        order = recognizer._feature_order_v2_1()

    test_csv = next((p for p in test_csv_candidates if p.exists()), None)
    if test_csv is None:
        print(f"[FAIL] 缺少测试特征文件: {test_csv_candidates[0]}")
        return 2

    df = pd.read_csv(test_csv)
    if only_distance is not None:
        df = df[df.get("distance", "") == only_distance]
    if only_direction is not None:
        df = df[df.get("direction", "") == only_direction]

    required_cols = {"weapon", "distance", "direction"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[FAIL] 测试特征缺少必要列: {missing}")
        return 2

    missing_feats = [c for c in order if c not in df.columns]
    if missing_feats:
        print(f"[FAIL] 测试特征缺少 v2 列（前 5 个）: {missing_feats[:5]}")
        return 2

    X = df[order].to_numpy(dtype=float, copy=True)
    y_true = df["weapon"].astype(str).to_numpy()
    distances = df["distance"].astype(str).to_numpy()
    directions = df["direction"].astype(str).to_numpy()

    expected_dim = getattr(model_loader.scaler, "n_features_in_", None)
    if isinstance(expected_dim, int) and X.shape[1] != expected_dim:
        print(f"[FAIL] 特征维度不匹配: got={X.shape[1]}, expected={expected_dim}")
        return 2

    X_scaled = model_loader.scaler.transform(X)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Forward + ensemble proba
    with torch.no_grad():
        if getattr(model_loader, "models", None):
            probs_sum = None
            for m in model_loader.models:
                logits = m(x_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
            probabilities = probs_sum / float(len(model_loader.models))
        else:
            logits = model_loader.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)

    # Two-stage decision (fixed enable for v2.*)
    model_version = getattr(model_loader, "model_version", "") or ""
    use_two_stage = model_version.startswith("v2") or model_version.startswith("ensemble_v2")

    classes = list(getattr(model_loader.label_encoder, "classes_", []))
    if not classes:
        print("[FAIL] label_encoder.classes_ 为空")
        return 2

    y_pred = []
    conf = []
    for i in range(probabilities.shape[0]):
        row = probabilities[i]
        if use_two_stage:
            pred_idx, c = recognizer._two_stage_predict_v2_1(row)
            y_pred.append(classes[int(pred_idx)])
            conf.append(float(c))
        else:
            c, pred_idx = torch.max(row, dim=0)
            y_pred.append(classes[int(pred_idx.item())])
            conf.append(float(c.item()))

    y_pred = np.asarray(y_pred, dtype=str)
    ok_mask = (y_pred == y_true)

    print("\n" + "=" * 80)
    print("Overall")
    print("=" * 80)
    print("samples:", len(y_true))
    print("accuracy:", _fmt_pct(float(ok_mask.mean())))
    print("avg_confidence:", float(np.mean(conf)))

    # Context accuracy
    by_ctx = defaultdict(Counter)  # (dist,dir) -> {'n','ok'}
    confusions = defaultdict(Counter)  # (dist,dir) -> Counter((exp,pred))
    for exp, pred, d, r, ok1 in zip(y_true, y_pred, distances, directions, ok_mask):
        by_ctx[(d, r)]["n"] += 1
        by_ctx[(d, r)]["ok"] += int(ok1)
        confusions[(d, r)][(exp, pred)] += 1

    ctx_rows = []
    for (d, r), cnt in by_ctx.items():
        n = int(cnt["n"])
        if n < min_support:
            continue
        acc = float(cnt["ok"]) / float(n)
        ctx_rows.append((acc, n, d, r))
    ctx_rows.sort(key=lambda t: (t[0], -t[1]))

    print("\n" + "=" * 80)
    print(f"Worst Contexts (n>={min_support})")
    print("=" * 80)
    for acc, n, d, r in ctx_rows[:12]:
        print(f"({d:5s},{r:6s}) acc={_fmt_pct(acc)} n={n}")
        for exp, pred, c in _top_confusions(confusions[(d, r)], top_n=6):
            print(f"  {exp:6s} -> {pred:6s}  {c}")

    # Focus weapons
    focus_set = set(focus_weapons)
    focus_df = _maybe_filter_weapons(df.assign(pred=y_pred, ok=ok_mask), focus_set)
    if not focus_df.empty:
        print("\n" + "=" * 80)
        print("Focus Weapons")
        print("=" * 80)
        for w in sorted(focus_set):
            w_df = focus_df[focus_df["weapon"] == w]
            if w_df.empty:
                continue
            w_acc = float(w_df["ok"].mean())
            print(f"\nweapon={w} acc={_fmt_pct(w_acc)} n={len(w_df)}")
            g = w_df.groupby(["distance", "direction"])["ok"].agg(["mean", "count"]).reset_index()
            g = g[g["count"] >= 3].sort_values(["mean", "count"], ascending=[True, False]).head(10)
            for _, row in g.iterrows():
                print(f"  ({row['distance']},{row['direction']}) acc={_fmt_pct(float(row['mean']))} n={int(row['count'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
