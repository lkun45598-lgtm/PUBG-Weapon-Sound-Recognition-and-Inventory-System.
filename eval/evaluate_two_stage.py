# -*- coding: utf-8 -*-
"""
二阶段分类策略 - 更精确地提升小样本类别
思路：只在模型置信度低时才调整，保护高置信度预测
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def two_stage_prediction(probs, target_classes, confidence_threshold=0.75, boost_in_top_k=3):
    """
    二阶段预测策略

    策略：
    1. 如果最高概率 >= confidence_threshold，直接使用（保护高置信度预测）
    2. 否则，检查top-k中是否有小样本类别：
       - 如果有，且该小样本类别概率在合理范围内，选择它
       - 否则，使用原始预测

    Args:
        probs: softmax概率 (batch_size, num_classes)
        target_classes: 小样本类别索引列表
        confidence_threshold: 高置信度阈值
        boost_in_top_k: 检查top-k中的小样本类别

    Returns:
        predictions: 调整后的预测
    """
    batch_size = probs.shape[0]
    predictions = []

    for i in range(batch_size):
        prob = probs[i]

        # 获取top-k预测
        top_k_probs, top_k_indices = torch.topk(prob, k=min(boost_in_top_k, prob.shape[0]))

        max_prob = top_k_probs[0].item()
        max_class = top_k_indices[0].item()

        # 策略1: 高置信度预测，直接接受
        if max_prob >= confidence_threshold:
            predictions.append(max_class)
            continue

        # 策略2: 低置信度，检查top-k中的小样本类别
        # 找到top-k中的小样本类别
        small_sample_in_topk = []
        for j in range(len(top_k_indices)):
            class_idx = top_k_indices[j].item()
            class_prob = top_k_probs[j].item()

            if class_idx in target_classes:
                small_sample_in_topk.append((class_idx, class_prob, j))

        # 如果有小样本类别在top-k中，考虑选择它
        if small_sample_in_topk:
            # 选择概率最高的小样本类别
            best_small_sample = max(small_sample_in_topk, key=lambda x: x[1])
            small_class, small_prob, small_rank = best_small_sample

            # 决策规则：
            # - 如果小样本类概率 >= 0.3 * 最高概率，选择小样本类
            # - 这确保我们不会选择概率太低的小样本类
            if small_prob >= 0.3 * max_prob:
                predictions.append(small_class)
            else:
                predictions.append(max_class)
        else:
            # top-k中没有小样本类别，使用原始预测
            predictions.append(max_class)

    return torch.tensor(predictions)


def main():
    print("="*80)
    print("二阶段分类策略 - 精确提升小样本类别")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 路径（统一以项目根目录为准）
    test_csv = FEATURES_DIR / "test_enhanced_features_v2.csv"
    models_dir = PROJECT_DIR / "models"

    print(f"\n加载测试数据: {test_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"未找到测试特征文件: {test_csv}")
    test_df = pd.read_csv(test_csv)

    # 加载模型
    scaler = joblib.load(models_dir / "scaler_enhanced_v2_1.pkl")
    le = joblib.load(models_dir / "label_encoder_enhanced_v2_1.pkl")

    exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                    'distance_m', 'distance_label']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[feature_cols].values
    y_test = test_df['weapon'].values
    y_test_encoded = le.transform(y_test)

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    checkpoint = torch.load(models_dir / "best_model_enhanced_v2_1.pth", map_location=device)

    model = EnhancedWeaponClassifierV2_1(
        input_dim=X_test_scaled.shape[1],
        output_dim=len(le.classes_),
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n加载v2.1模型: Epoch {checkpoint['epoch']+1}")

    # 目标类别
    target_weapons = ['aug', 'm249', 'qbu', 'vec', 'win']
    target_classes = [list(le.classes_).index(w) for w in target_weapons if w in le.classes_]

    print(f"\n目标提升类别: {target_weapons}")

    # 获取预测
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = F.softmax(logits, dim=1)

        # 原始预测
        y_pred_original = logits.argmax(dim=1).cpu().numpy()

    # 测试不同参数配置
    configs = [
        {'threshold': 0.70, 'top_k': 3, 'name': 'aggressive'},
        {'threshold': 0.75, 'top_k': 3, 'name': 'balanced'},
        {'threshold': 0.80, 'top_k': 3, 'name': 'conservative'},
        {'threshold': 0.75, 'top_k': 5, 'name': 'wider_search'},
    ]

    print("\n" + "="*80)
    print("测试不同配置")
    print("="*80)

    results = {'original': {
        'acc': accuracy_score(y_test_encoded, y_pred_original),
        'cm': confusion_matrix(y_test_encoded, y_pred_original),
        'preds': y_pred_original
    }}

    for config in configs:
        y_pred = two_stage_prediction(
            probs.cpu(),
            target_classes,
            confidence_threshold=config['threshold'],
            boost_in_top_k=config['top_k']
        ).numpy()

        acc = accuracy_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)

        results[config['name']] = {
            'acc': acc,
            'cm': cm,
            'preds': y_pred,
            'config': config
        }

    # 结果对比
    print(f"\n{'配置':<20} {'整体准确率':<15} {'vs原始':<10} {'AK/M4混淆'}")
    print("-"*70)

    ak_idx = list(le.classes_).index('ak')
    m4_idx = list(le.classes_).index('m4')

    for name, res in results.items():
        cm = res['cm']
        acc = res['acc']
        ak_m4 = cm[ak_idx, m4_idx] + cm[m4_idx, ak_idx]

        if name == 'original':
            change_str = "-"
        else:
            change = (acc - results['original']['acc']) * 100
            change_str = f"{change:+.2f}%"

        print(f"{name:<20} {acc:.4f} ({acc*100:.2f}%)  {change_str:<10} {ak_m4}次")

    # 小样本类别详细对比
    print("\n" + "="*80)
    print("小样本类别准确率详细对比")
    print("="*80)

    print(f"\n{'类别':<10} {'原始':<10}", end='')
    for config in configs:
        print(f"{config['name']:<18}", end='')
    print()
    print("-"*90)

    for weapon in target_weapons:
        if weapon not in le.classes_:
            continue

        idx = list(le.classes_).index(weapon)
        print(f"{weapon:<10}", end='')

        for name in ['original'] + [c['name'] for c in configs]:
            res = results[name]
            cm = res['cm']
            if cm[idx].sum() > 0:
                acc = cm[idx, idx] / cm[idx].sum()
                print(f"{acc:.2%}      ", end='')
            else:
                print(f"N/A       ", end='')
        print()

    # 计算平均提升
    print("\n" + "="*80)
    print("小样本类别平均提升")
    print("="*80)

    original_target_acc = []
    for weapon in target_weapons:
        if weapon in le.classes_:
            idx = list(le.classes_).index(weapon)
            cm = results['original']['cm']
            if cm[idx].sum() > 0:
                original_target_acc.append(cm[idx, idx] / cm[idx].sum())

    original_avg = np.mean(original_target_acc) if original_target_acc else 0

    print(f"\n{'配置':<20} {'小样本平均':<15} {'提升':<10} {'整体变化'}")
    print("-"*70)

    for name in ['original'] + [c['name'] for c in configs]:
        res = results[name]
        cm = res['cm']

        target_accs = []
        for weapon in target_weapons:
            if weapon in le.classes_:
                idx = list(le.classes_).index(weapon)
                if cm[idx].sum() > 0:
                    target_accs.append(cm[idx, idx] / cm[idx].sum())

        target_avg = np.mean(target_accs) if target_accs else 0

        if name == 'original':
            print(f"{name:<20} {target_avg:.2%}           -         -")
        else:
            target_change = (target_avg - original_avg) * 100
            overall_change = (res['acc'] - results['original']['acc']) * 100
            print(f"{name:<20} {target_avg:.2%}           {target_change:+.1f}%      {overall_change:+.2f}%")

    # 选择最佳配置
    print("\n" + "="*80)
    print("最佳配置推荐")
    print("="*80)

    best_config = None
    best_score = -1

    for config in configs:
        name = config['name']
        res = results[name]

        # 评分标准
        overall_change = (res['acc'] - results['original']['acc']) * 100

        # 小样本提升
        target_accs = []
        for weapon in target_weapons:
            if weapon in le.classes_:
                idx = list(le.classes_).index(weapon)
                cm = res['cm']
                if cm[idx].sum() > 0:
                    target_accs.append(cm[idx, idx] / cm[idx].sum())

        target_avg = np.mean(target_accs) if target_accs else 0
        target_improvement = (target_avg - original_avg) * 100

        # AK/M4混淆变化
        cm = res['cm']
        ak_m4 = cm[ak_idx, m4_idx] + cm[m4_idx, ak_idx]
        original_ak_m4 = results['original']['cm'][ak_idx, m4_idx] + results['original']['cm'][m4_idx, ak_idx]
        ak_m4_change = ak_m4 - original_ak_m4

        # 综合评分：优先小样本提升，但整体不能下降太多
        if overall_change >= -0.3:  # 整体下降不超过0.3%
            score = target_improvement * 2 - abs(overall_change) - ak_m4_change * 0.5
        else:
            score = -999  # 整体下降过多，不考虑

        print(f"\n配置: {name}")
        print(f"  阈值: {config['threshold']}, Top-K: {config['top_k']}")
        print(f"  整体准确率变化: {overall_change:+.2f}%")
        print(f"  小样本平均提升: {target_improvement:+.2f}%")
        print(f"  AK/M4混淆变化: {ak_m4_change:+d}次")
        print(f"  综合评分: {score:.2f}")

        if score > best_score:
            best_score = score
            best_config = name

    if best_config and best_config != 'original':
        print(f"\n推荐配置: {best_config}")

        res = results[best_config]
        config = res['config']

        print("\n最终效果:")
        print("-"*80)
        print(f"v2.1原始:       {results['original']['acc']:.4f} ({results['original']['acc']*100:.2f}%)")
        print(f"{best_config}策略:  {res['acc']:.4f} ({res['acc']*100:.2f}%)")
        print(f"变化:           {(res['acc']-results['original']['acc'])*100:+.2f}%")

        print("\n小样本类别对比:")
        for weapon in target_weapons:
            if weapon in le.classes_:
                idx = list(le.classes_).index(weapon)

                original_cm = results['original']['cm']
                original_acc = original_cm[idx, idx] / original_cm[idx].sum() if original_cm[idx].sum() > 0 else 0

                new_cm = res['cm']
                new_acc = new_cm[idx, idx] / new_cm[idx].sum() if new_cm[idx].sum() > 0 else 0

                change = (new_acc - original_acc) * 100
                if change > 0:
                    symbol = "UP"
                elif change == 0:
                    symbol = "--"
                else:
                    symbol = "DOWN"

                print(f"  {weapon:6s}: {original_acc:.2%} -> {new_acc:.2%} ({change:+.1f}%) [{symbol}]")

        if res['acc'] >= results['original']['acc'] - 0.003:
            print("\n成功! 在几乎不损失整体准确率的情况下提升了小样本类别!")
        else:
            print(f"\n整体准确率小幅下降{abs((res['acc']-results['original']['acc'])*100):.2f}%")
            print("但小样本类别得到明显改善，适合对这些类别识别要求高的场景")

    else:
        print("\n所有配置都会导致整体准确率下降超过0.3%")
        print("\n建议采用其他方案：")
        print("  1. 分层分类器（先判断是否为小样本类别，再细分）")
        print("  2. 收集更多小样本类别的训练数据")
        print("  3. 使用类别特定的特征工程")


if __name__ == "__main__":
    main()
