# -*- coding: utf-8 -*-
"""
武器声音分类 - LightGBM版本
最简单且高效的梯度提升模型
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import lightgbm as lgb
except ImportError:
    print("错误: 未安装 lightgbm，无法训练 LightGBM 模型。")
    print("安装方式: pip install lightgbm  或  conda install -c conda-forge lightgbm")
    raise SystemExit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[1]


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = f'confusion_matrix_{title.replace(" ", "_")}.png'
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / filename
    plt.savefig(out_path)
    print(f"混淆矩阵已保存: {out_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """绘制特征重要性"""
    importance = model.feature_importance()
    top_n = min(top_n, len(importance))  # 限制top_n不超过特征数
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title('Top Feature Importances')
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "feature_importance_lightgbm.png"
    plt.savefig(out_path)
    print(f"特征重要性已保存: {out_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("PUBG 武器声音分类 - LightGBM版本")
    print("=" * 60)

    # 数据路径
    base_dir = PROJECT_DIR
    train_csv = base_dir / "train_selected_features.csv"

    # 加载数据
    print(f"\n加载训练数据: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"训练数据形状: {train_df.shape}")

    # 数据探索
    print("\n武器类型分布:")
    weapon_counts = train_df['weapon'].value_counts()
    print(f"类别数: {len(weapon_counts)}")
    print(f"样本数: {len(train_df)}")

    # 准备数据
    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)

    target_col = 'weapon'
    exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                    'distance_m', 'distance_label']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    print(f"特征列: {feature_cols}")
    print(f"特征数: {len(feature_cols)}")

    X = train_df[feature_cols].values
    y = train_df[target_col].values

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"类别数: {num_classes}")

    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")

    # 创建LightGBM数据集
    print("\n" + "=" * 60)
    print("创建LightGBM数据集")
    print("=" * 60)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 设置参数
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }

    print("模型参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 训练模型
    print("\n" + "=" * 60)
    print("训练LightGBM模型")
    print("=" * 60)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    print(f"\n最佳迭代轮数: {model.best_iteration}")

    # 预测
    print("\n" + "=" * 60)
    print("评估模型")
    print("=" * 60)

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)

    # 计算指标
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"\n最终性能:")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, le.classes_, 'LightGBM')

    # 绘制特征重要性
    plot_feature_importance(model, feature_cols)

    # 分类报告
    print("\n分类报告:")
    report = classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0)
    print(report)

    # 保存模型
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    model.save_model(str(models_dir / "lightgbm_model.txt"))
    joblib.dump(le, models_dir / "label_encoder_lgb.pkl")

    print(f"\n模型已保存到: {models_dir}")
    print("  - lightgbm_model.txt")
    print("  - label_encoder_lgb.pkl")
    print("\n训练完成！")


if __name__ == "__main__":
    main()
