# -*- coding: utf-8 -*-
"""
在真实测试集上评估调整版模型 v2.1
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


class EnhancedWeaponClassifierV2_1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(EnhancedWeaponClassifierV2_1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 80)
    print("在真实测试集上评估调整版模型 v2.1")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 路径（统一以项目根目录为准）
    test_csv = FEATURES_DIR / "test_enhanced_features_v2.csv"
    models_dir = PROJECT_DIR / "models"

    # 加载测试数据
    print(f"\n加载测试数据: {test_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"未找到测试特征文件: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"测试集大小: {test_df.shape}")

    # 加载预处理器
    scaler = joblib.load(models_dir / "scaler_enhanced_v2_1.pkl")
    le = joblib.load(models_dir / "label_encoder_enhanced_v2_1.pkl")

    # 准备特征
    exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                    'distance_m', 'distance_label']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[feature_cols].values
    y_test = test_df['weapon'].values
    y_test_encoded = le.transform(y_test)

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # 加载模型
    checkpoint = torch.load(models_dir / "best_model_enhanced_v2_1.pth", map_location=device)

    model = EnhancedWeaponClassifierV2_1(
        input_dim=X_test_scaled.shape[1],
        output_dim=len(le.classes_),
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n加载模型: Epoch {checkpoint['epoch']+1}, 验证集准确率: {checkpoint['val_acc']:.4f}")

    # 评估
    print("\n" + "=" * 80)
    print("在真实测试集上评估")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)

    y_pred = predicted.cpu().numpy()
    y_true = y_test_encoded

    # 计算指标
    test_acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 80)
    print("测试集性能指标")
    print("=" * 80)
    print(f"准确率 (Accuracy):   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"精确率 (Precision):  {precision:.4f} ({precision*100:.2f}%)")
    print(f"召回率 (Recall):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1分数 (F1-Score):   {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 80)

    # 对比分析
    baseline_acc = 0.9465
    v2_test_acc = 0.9374
    v2_1_val_acc = 0.9735
    improvement_from_baseline = (test_acc - baseline_acc) * 100
    improvement_from_v2 = (test_acc - v2_test_acc) * 100
    val_test_gap = (v2_1_val_acc - test_acc) * 100

    print(f"\n性能对比:")
    print(f"  基线版本测试集:     {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  v2版本测试集:       {v2_test_acc:.4f} ({v2_test_acc*100:.2f}%)")
    print(f"  v2.1版本验证集:     {v2_1_val_acc:.4f} ({v2_1_val_acc*100:.2f}%)")
    print(f"  v2.1版本测试集:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  相比基线提升:       {improvement_from_baseline:+.2f}%")
    print(f"  相比v2提升:         {improvement_from_v2:+.2f}%")
    print(f"  验证-测试差距:      {val_test_gap:.2f}%")

    if test_acc >= 0.97:
        print("\n优秀! 达到97%+目标! 过拟合问题已解决!")
    elif test_acc >= 0.96:
        print("\n良好! 接近目标，过拟合明显改善")
    elif test_acc >= baseline_acc:
        print("\n可接受，超过基线，但仍有改进空间")
    else:
        print("\n需要进一步调整")

    # 详细报告
    print("\n详细分类报告:")
    print("=" * 80)
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)
    print(report)

    # 分析问题类别
    print("\n问题类别分析 (准确率 < 90%):")
    print("=" * 80)

    cm = confusion_matrix(y_true, y_pred)
    test_dist = test_df['weapon'].value_counts()

    problem_count = 0
    for i, weapon in enumerate(le.classes_):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            if acc < 0.90:
                test_count = test_dist.get(weapon, 0)
                print(f"  {weapon:10s}: {acc:.2%} (测试样本: {test_count})")
                problem_count += 1

    if problem_count == 0:
        print("  所有类别准确率都 >= 90%!")

    print(f"\n总计: {problem_count} 个类别需要继续改进")

    # AK/M4混淆分析
    print("\n" + "=" * 80)
    print("AK/M4混淆分析")
    print("=" * 80)

    ak_idx = list(le.classes_).index('ak')
    m4_idx = list(le.classes_).index('m4')

    ak_to_m4 = cm[ak_idx, m4_idx]
    m4_to_ak = cm[m4_idx, ak_idx]
    total_confusion = ak_to_m4 + m4_to_ak

    print(f"AK误识别为M4: {ak_to_m4} 次")
    print(f"M4误识别为AK: {m4_to_ak} 次")
    print(f"总混淆次数: {total_confusion} 次")
    print(f"\n基线混淆: 24次")
    print(f"v2混淆: 未知")
    print(f"v2.1混淆: {total_confusion}次")

    if total_confusion < 12:
        print("优秀! AK/M4混淆已大幅减少!")
    elif total_confusion < 20:
        print("良好! AK/M4混淆有所改善")
    else:
        print("仍需改进AK/M4区分")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    print(f"\n调整效果:")
    print(f"  网络参数: 1,268,006 (v2) → 339,110 (v2.1), 减少 73%")
    print(f"  Dropout: 0.5 → 0.3")
    print(f"  类别权重比: 35:1 → 6:1")
    print(f"  测试准确率: {v2_test_acc:.4f} (v2) → {test_acc:.4f} (v2.1)")

    if test_acc > v2_test_acc:
        print(f"\n调整成功! 测试准确率提升 {improvement_from_v2:.2f}%")
    else:
        print(f"\n调整效果不明显，测试准确率变化 {improvement_from_v2:.2f}%")


if __name__ == "__main__":
    main()
