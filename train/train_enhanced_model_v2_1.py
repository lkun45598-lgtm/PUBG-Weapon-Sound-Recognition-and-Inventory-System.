# -*- coding: utf-8 -*-
"""
改进版训练脚本 v2.1 - 调整参数版本
调整:
1. 网络更浅更窄 (避免过拟合)
2. Dropout降低到0.3
3. 类别权重使用平方根缩放 (避免过于激进)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_DIR / "data" / "features"

# 训练侧的“场景纠偏”（不影响推理逻辑）：对已知误差热点子分布做轻量重采样
# 依据：评估显示 (600m,front) 的 AK→M24/M4 错误集中且高置信。
# 如需关闭：将这些倍数改为 1.0 或清空字典即可。
CONTEXT_MULTIPLIERS: dict[tuple[str, str, str], float] = {
    ("ak", "600m", "front"): 8.0,
    ("ak", "600m", "left"): 3.0,
    ("ak", "600m", "right"): 3.0,
    ("ak", "600m", "back"): 3.0,
    ("m4", "600m", "left"): 1.5,
    ("m4", "600m", "right"): 1.5,
}


def _build_context_sample_weights(
    weapons: np.ndarray,
    distances: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    weights = np.ones(len(weapons), dtype=np.float32)
    for i in range(len(weapons)):
        key = (str(weapons[i]), str(distances[i]), str(directions[i]))
        weights[i] *= float(CONTEXT_MULTIPLIERS.get(key, 1.0))
    return weights


def _build_centroid_artifacts(
    X_scaled: np.ndarray,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    classes = list(getattr(label_encoder, "classes_", []))
    if not classes:
        raise ValueError("LabelEncoder is not fitted; classes_ is empty.")

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
        "classes": classes,
        "centroids": centroids,
        "radii_p50": radii_p50,
        "radii_p95": radii_p95,
        "counts": counts,
    }


def _save_centroid_artifacts(models_dir: Path, filename: str, artifacts: dict) -> None:
    out_path = models_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)


class EnhancedWeaponClassifierV2_1(nn.Module):
    """
    调整后的分类器 v2.1
    - 更浅的网络 (256→512→256→128)
    - 更低的Dropout (0.3)
    - 参数量: ~50万 (比v2的127万少很多)
    """

    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(EnhancedWeaponClassifierV2_1, self).__init__()

        self.network = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            # 第二层
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            # 第三层
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            # 第四层
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            # 输出层
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc


def main():
    print("=" * 80)
    print("PUBG 枪械音频识别 - 调整版v2.1 (优化参数)")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 路径（统一以项目根目录为准，避免脚本移动后相对路径失效）
    train_csv_candidates = [
        FEATURES_DIR / "train_enhanced_features_v2_aug.csv",
        FEATURES_DIR / "train_enhanced_features_v2.csv",
    ]
    train_csv = next((p for p in train_csv_candidates if p.exists()), train_csv_candidates[-1])
    models_dir = PROJECT_DIR / "models"

    print(f"\n加载训练数据: {train_csv}")
    if not train_csv.exists():
        raise FileNotFoundError(f"未找到训练特征文件: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"数据形状: {train_df.shape}")

    # 准备特征
    exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                    'distance_m', 'distance_label']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X = train_df[feature_cols].values
    y = train_df['weapon'].values

    print(f"特征维度: {len(feature_cols)}")

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"\n类别: {num_classes}")

    # === 代价敏感矩阵：针对已定位热点混淆对加罚 ===
    # 数据分割（同时拆分元数据，便于做场景重采样）
    distances = train_df.get('distance', pd.Series(["unknown"] * len(train_df))).values
    directions = train_df.get('direction', pd.Series(["unknown"] * len(train_df))).values

    X_train, X_val, y_train, y_val, dist_train, dist_val, dir_train, dir_val, weapon_train, weapon_val = train_test_split(
        X,
        y_encoded,
        distances,
        directions,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # === 调整后的类别权重计算 ===
    print("\n" + "=" * 80)
    print("计算类别权重 (平方根缩放)")
    print("=" * 80)

    # 先计算标准权重
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # 使用平方根缩放来降低极端值
    # 原理: sqrt(weight) 会让大权重变小，小权重变化不大
    class_weights_array_sqrt = np.sqrt(class_weights_array)

    # 重新归一化，使平均权重为1
    class_weights_array_sqrt = class_weights_array_sqrt / class_weights_array_sqrt.mean()

    class_weights = torch.FloatTensor(class_weights_array_sqrt).to(device)

    print(f"\n权重统计:")
    print(f"  原始权重比: {class_weights_array.max()/class_weights_array.min():.2f}:1")
    print(f"  平方根后权重比: {class_weights_array_sqrt.max()/class_weights_array_sqrt.min():.2f}:1")
    print(f"  最小权重: {class_weights_array_sqrt.min():.4f}")
    print(f"  最大权重: {class_weights_array_sqrt.max():.4f}")

    # 显示部分类别的权重对比
    class_counts = train_df['weapon'].value_counts()
    print(f"\n部分类别权重对比:")
    for idx in [class_counts.idxmin(), class_counts.idxmax()]:
        weapon_idx = list(le.classes_).index(idx)
        weight_original = class_weights_array[weapon_idx]
        weight_sqrt = class_weights_array_sqrt[weapon_idx]
        count = class_counts[idx]
        print(f"  {idx:10s}: 样本数={count:3d}, 原始权重={weight_original:.2f}, 调整后={weight_sqrt:.2f}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    centroid_artifacts = _build_centroid_artifacts(X_train_scaled, y_train, le)

    print(f"\n训练集: {X_train_scaled.shape}")
    print(f"验证集: {X_val_scaled.shape}")

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    # 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    sample_weights = _build_context_sample_weights(weapon_train, dist_train, dir_train)
    boosted = int(np.sum(sample_weights != 1.0))
    if boosted > 0:
        print(
            f"\n场景重采样: 启用 (boosted_samples={boosted}/{len(sample_weights)}, "
            f"min_w={sample_weights.min():.2f}, max_w={sample_weights.max():.2f})"
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nBatch Size: {batch_size}")

    # 创建调整后的模型
    print("\n" + "=" * 80)
    print("创建调整模型 v2.1 (更浅,Dropout=0.3)")
    print("=" * 80)

    input_dim = X_train_scaled.shape[1]
    model = EnhancedWeaponClassifierV2_1(
        input_dim=input_dim,
        output_dim=num_classes,
        dropout=0.3  # 降低到0.3
    ).to(device)

    print(f"\n模型架构:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"  (v2: 1,268,006, v2.1: {total_params:,})")

    # 使用调整后的加权损失（代价敏感）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n使用平方根缩放的类别权重")
    print(f"  权重比从 35:1 降低到 {class_weights_array_sqrt.max()/class_weights_array_sqrt.min():.1f}:1")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # 训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    num_epochs = 150
    best_val_acc = 0.0
    best_epoch = 0
    patience = 20
    no_improve = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        # 打印进度
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0

            # 保存最佳模型
            models_dir.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, models_dir / "best_model_enhanced_v2_1.pth")

            joblib.dump(scaler, models_dir / "scaler_enhanced_v2_1.pkl")
            joblib.dump(le, models_dir / "label_encoder_enhanced_v2_1.pkl")
            _save_centroid_artifacts(
                models_dir=models_dir,
                filename="centroids_enhanced_v2_1.json",
                artifacts=centroid_artifacts,
            )

            status = "*** Best ***"
        else:
            no_improve += 1

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {status}")

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 加载最佳模型
    print("\n" + "=" * 80)
    print("评估最佳模型")
    print("=" * 80)

    checkpoint = torch.load(models_dir / "best_model_enhanced_v2_1.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型 (Epoch {best_epoch+1}, Acc: {best_val_acc:.4f})")

    # 最终评估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "=" * 80)
    print("验证集性能指标")
    print("=" * 80)
    print(f"准确率 (Accuracy):   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率 (Precision):  {precision:.4f} ({precision*100:.2f}%)")
    print(f"召回率 (Recall):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1分数 (F1-Score):   {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 80)

    # 对比
    baseline_acc = 0.9465
    v2_val_acc = 0.9659

    print(f"\n性能对比:")
    print(f"  基线版本测试集:     {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  v2版本验证集:       {v2_val_acc:.4f} ({v2_val_acc*100:.2f}%)")
    print(f"  v2.1版本验证集:     {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 详细报告
    print("\n详细分类报告:")
    print("=" * 80)
    report = classification_report(all_labels, all_preds, target_names=le.classes_, zero_division=0)
    print(report)

    print(f"\n模型已保存到: {models_dir.absolute()}")
    print("\n下一步: 使用 evaluate_on_test_v2_1.py 在真实测试集上验证")


if __name__ == "__main__":
    main()
