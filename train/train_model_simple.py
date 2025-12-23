# -*- coding: utf-8 -*-
"""
武器声音分类模型训练脚本（简化版 - 不需要librosa）
直接使用预提取的特征进行训练
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
    plt.savefig(results_dir / filename)
    print(f"混淆矩阵已保存: {filename}")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("PUBG 武器声音分类模型训练")
    print("=" * 60)

    # 数据路径
    train_csv = PROJECT_DIR / "train_selected_features.csv"
    test_csv = PROJECT_DIR / "test_selected_features.csv"

    # 检查文件是否存在
    if not train_csv.exists():
        print(f"错误: 找不到训练数据文件 {train_csv}")
        return

    if not test_csv.exists():
        print(f"错误: 找不到测试数据文件 {test_csv}")
        return

    # 加载数据
    print(f"\n加载训练数据: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"训练数据形状: {train_df.shape}")

    print(f"\n加载测试数据: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"测试数据形状: {test_df.shape}")

    # 数据探索
    print("\n" + "=" * 60)
    print("数据探索")
    print("=" * 60)

    print("\n武器类型分布:")
    weapon_counts = train_df['weapon'].value_counts()
    print(weapon_counts)

    # 准备数据
    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)

    target_col = 'weapon'
    exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                    'distance_m', 'distance_label']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    print(f"特征列: {feature_cols}")

    X = train_df[feature_cols].values
    y = train_df[target_col].values

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")
    print(f"类别数: {len(le.classes_)}")

    # 训练模型
    print("\n" + "=" * 60)
    print("训练分类模型")
    print("=" * 60)

    results = {}

    # 1. KNN
    print("\n1. KNN分类器")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    prec_knn = precision_score(y_test, y_pred_knn, average='macro', zero_division=0)
    rec_knn = recall_score(y_test, y_pred_knn, average='macro', zero_division=0)
    f1_knn = f1_score(y_test, y_pred_knn, average='macro', zero_division=0)

    print(f"准确率: {acc_knn:.4f}")
    print(f"精确率: {prec_knn:.4f}")
    print(f"召回率: {rec_knn:.4f}")
    print(f"F1分数: {f1_knn:.4f}")

    results['KNN'] = {
        'accuracy': acc_knn,
        'precision': prec_knn,
        'recall': rec_knn,
        'f1': f1_knn,
        'cm': confusion_matrix(y_test, y_pred_knn)
    }

    # 2. SVM
    print("\n2. SVM分类器")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
    rec_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)

    print(f"准确率: {acc_svm:.4f}")
    print(f"精确率: {prec_svm:.4f}")
    print(f"召回率: {rec_svm:.4f}")
    print(f"F1分数: {f1_svm:.4f}")

    results['SVM'] = {
        'accuracy': acc_svm,
        'precision': prec_svm,
        'recall': rec_svm,
        'f1': f1_svm,
        'cm': confusion_matrix(y_test, y_pred_svm)
    }

    # 3. 随机森林
    print("\n3. 随机森林分类器")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)

    print(f"准确率: {acc_rf:.4f}")
    print(f"精确率: {prec_rf:.4f}")
    print(f"召回率: {rec_rf:.4f}")
    print(f"F1分数: {f1_rf:.4f}")

    results['RandomForest'] = {
        'accuracy': acc_rf,
        'precision': prec_rf,
        'recall': rec_rf,
        'f1': f1_rf,
        'cm': confusion_matrix(y_test, y_pred_rf)
    }

    # 交叉验证
    print("\n" + "=" * 60)
    print("交叉验证")
    print("=" * 60)

    X_all = np.vstack([X_train_scaled, X_test_scaled])
    y_all = np.concatenate([y_train, y_test])

    print("\nKNN交叉验证:")
    cv_scores_knn = cross_val_score(knn, X_all, y_all, cv=5, scoring='accuracy')
    print(f"准确率: {cv_scores_knn}")
    print(f"平均: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std() * 2:.4f})")

    print("\nSVM交叉验证:")
    cv_scores_svm = cross_val_score(svm, X_all, y_all, cv=5, scoring='accuracy')
    print(f"准确率: {cv_scores_svm}")
    print(f"平均: {cv_scores_svm.mean():.4f} (+/- {cv_scores_svm.std() * 2:.4f})")

    print("\n随机森林交叉验证:")
    cv_scores_rf = cross_val_score(rf, X_all, y_all, cv=5, scoring='accuracy')
    print(f"准确率: {cv_scores_rf}")
    print(f"平均: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

    # 绘制混淆矩阵
    print("\n" + "=" * 60)
    print("绘制混淆矩阵")
    print("=" * 60)

    plot_confusion_matrix(results['KNN']['cm'], le.classes_, 'KNN')
    plot_confusion_matrix(results['SVM']['cm'], le.classes_, 'SVM')
    plot_confusion_matrix(results['RandomForest']['cm'], le.classes_, 'RandomForest')

    # 模型对比
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'RandomForest'],
        'Accuracy': [acc_knn, acc_svm, acc_rf],
        'Precision': [prec_knn, prec_svm, prec_rf],
        'Recall': [rec_knn, rec_svm, rec_rf],
        'F1-Score': [f1_knn, f1_svm, f1_rf]
    })

    print(comparison.to_string(index=False))

    # 绘制对比图
    comparison_melted = comparison.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_melted, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend(title='Model')
    plt.tight_layout()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "model_comparison.png"
    plt.savefig(out_path)
    print(f"\n模型对比图已保存: {out_path}")
    plt.close()

    # 保存模型
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)

    best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
    print(f"最佳模型: {best_model_name} (准确率: {comparison['Accuracy'].max():.4f})")

    models_dir = PROJECT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    # 保存所有模型
    joblib.dump(knn, models_dir / "knn_model.pkl")
    joblib.dump(svm, models_dir / "svm_model.pkl")
    joblib.dump(rf, models_dir / "randomforest_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(le, models_dir / "label_encoder.pkl")

    print(f"模型已保存到: {models_dir}")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
