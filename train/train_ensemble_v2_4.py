# -*- coding: utf-8 -*-
"""
训练集成模型 v2.4 - 基于v2.1架构
训练3个不同随机种子的模型，通过投票提升稳定性
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
    """v2.1架构 - 用于集成学习"""
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


def set_seed(seed):
    """设置随机种子以保证可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, train_loader, criterion, optimizer, device):
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


def train_single_model(
    model_id,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    input_dim,
    num_classes,
    device,
    models_dir,
    train_sample_weights=None,
):
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"训练模型 #{model_id}")
    print(f"{'='*80}")

    # 设置随机种子
    seed = 42 + model_id * 100
    set_seed(seed)
    print(f"随机种子: {seed}")

    # 创建模型
    model = EnhancedWeaponClassifierV2_1(
        input_dim=input_dim,
        output_dim=num_classes,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # 数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if train_sample_weights is not None:
        sample_weights = np.asarray(train_sample_weights, dtype=np.float32)
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
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 训练
    num_epochs = 150
    best_val_acc = 0.0
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0

            # 保存最佳模型
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'seed': seed
            }

            # 与 src/audio/model_loader.py 的默认加载文件名保持一致
            torch.save(ckpt, models_dir / f"best_model_ensemble_v2_4_{model_id}.pth")

            # 兼容旧命名（可选）
            torch.save(ckpt, models_dir / f"ensemble_model_{model_id}.pth")

            status = "*** Best ***"
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or status:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {status}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n模型 #{model_id} 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch+1})")

    return best_val_acc


def main():
    print("="*80)
    print("PUBG 枪械音频识别 - v2.4 (集成学习)")
    print("="*80)

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

    print(f"\n特征维度: {len(feature_cols)}")

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"类别数: {num_classes}")

    # 数据分割（使用相同的随机种子以保证所有模型使用相同的验证集）
    # 同步拆分元数据（distance/direction）以便做场景重采样
    distances = train_df.get('distance', pd.Series(["unknown"] * len(train_df))).values
    directions = train_df.get('direction', pd.Series(["unknown"] * len(train_df))).values

    set_seed(42)
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

    # 计算类别权重（平方根缩放）
    print("\n" + "="*80)
    print("计算类别权重 (平方根缩放)")
    print("="*80)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights_array_sqrt = np.sqrt(class_weights_array)
    class_weights_array_sqrt = class_weights_array_sqrt / class_weights_array_sqrt.mean()

    class_weights = torch.FloatTensor(class_weights_array_sqrt).to(device)

    print(f"\n权重比: {class_weights_array_sqrt.max()/class_weights_array_sqrt.min():.2f}:1")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"\n训练集: {X_train_scaled.shape}")
    print(f"验证集: {X_val_scaled.shape}")

    train_sample_weights = _build_context_sample_weights(weapon_train, dist_train, dir_train)

    # 创建模型目录
    models_dir.mkdir(exist_ok=True)

    # 保存预处理器
    joblib.dump(scaler, models_dir / "scaler_ensemble_v2_4.pkl")
    joblib.dump(le, models_dir / "label_encoder_ensemble_v2_4.pkl")
    _save_centroid_artifacts(
        models_dir=models_dir,
        filename="centroids_ensemble_v2_4.json",
        artifacts=_build_centroid_artifacts(X_train_scaled, y_train, le),
    )

    # 训练集成模型
    print("\n" + "="*80)
    print("训练集成模型 (3个模型)")
    print("="*80)

    num_models = 3
    val_accuracies = []

    for i in range(num_models):
        val_acc = train_single_model(
            model_id=i+1,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            class_weights=class_weights,
            input_dim=X_train_scaled.shape[1],
            num_classes=num_classes,
            device=device,
            models_dir=models_dir,
            train_sample_weights=train_sample_weights,
        )
        val_accuracies.append(val_acc)

    # 总结
    print("\n" + "="*80)
    print("集成模型训练完成")
    print("="*80)

    print(f"\n各模型验证准确率:")
    for i, acc in enumerate(val_accuracies):
        print(f"  模型 #{i+1}: {acc:.4f} ({acc*100:.2f}%)")

    avg_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)

    print(f"\n平均准确率: {avg_val_acc:.4f} ({avg_val_acc*100:.2f}%)")
    print(f"标准差: {std_val_acc:.4f}")

    print(f"\n模型已保存到: {models_dir.absolute()}")
    print("  - ensemble_model_1.pth")
    print("  - ensemble_model_2.pth")
    print("  - ensemble_model_3.pth")
    print("  - scaler_ensemble_v2_4.pkl")
    print("  - label_encoder_ensemble_v2_4.pkl")

    print("\n" + "="*80)
    print("下一步: 使用 evaluate_ensemble_v2_4.py 在测试集上评估集成效果")
    print("="*80)


if __name__ == "__main__":
    main()
