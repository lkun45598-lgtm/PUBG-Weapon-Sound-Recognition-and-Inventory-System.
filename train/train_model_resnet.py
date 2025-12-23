# -*- coding: utf-8 -*-
"""
武器声音分类 - ResNet风格DNN
带残差连接的深度神经网络
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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("错误: 未安装 torch，无法训练 ResNet 模型。")
    print("建议: 使用 conda 安装 PyTorch，或按 PyTorch 官网指引安装对应版本。")
    raise SystemExit(1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ResidualBlock(nn.Module):
    """残差块 - ResNet的核心组件"""

    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()

        # 主路径: 两层全连接
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        # 跳跃连接是恒等映射（直接传递）

    def forward(self, x):
        # 保存输入（跳跃连接）
        identity = x

        # 主路径
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # 残差连接: 将输入加到输出上
        out += identity  # 这是ResNet的关键！
        out = self.relu(out)

        return out


class ResNetClassifier(nn.Module):
    """ResNet风格的武器分类器"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度（残差块的维度）
            output_dim: 输出类别数
            num_blocks: 残差块的数量
            dropout: Dropout比例
        """
        super(ResNetClassifier, self).__init__()

        # 输入层: 将输入映射到hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入层
        x = self.input_layer(x)

        # 通过所有残差块
        for block in self.residual_blocks:
            x = block(x)

        # 输出层
        x = self.output_layer(x)

        return x


def train_model(model, train_loader, criterion, optimizer, device):
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
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "training_history_resnet.png"
    plt.savefig(out_path)
    print(f"训练历史已保存: {out_path}")
    plt.close()


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


def main():
    """主函数"""
    print("=" * 60)
    print("PUBG 武器声音分类 - ResNet风格DNN")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据路径（训练脚本位于 train/ 目录，数据在项目根目录）
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

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"验证集大小: {X_val_scaled.shape}")

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    # 创建数据加载器
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"批次大小: {batch_size}")
    print(f"训练批次数: {len(train_loader)}")

    # 创建ResNet模型
    print("\n" + "=" * 60)
    print("构建ResNet风格神经网络")
    print("=" * 60)

    input_dim = X_train_scaled.shape[1]
    hidden_dim = 128  # 残差块的维度
    num_blocks = 4    # 4个残差块

    model = ResNetClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_blocks=num_blocks,
        dropout=0.3
    ).to(device)

    print(f"模型架构:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  残差块数量: {num_blocks}")
    print(f"  输出类别数: {num_classes}")
    print(f"\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 训练模型
    print("\n" + "=" * 60)
    print("训练ResNet模型")
    print("=" * 60)

    num_epochs = 100
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            models_dir = base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            best_model_path = models_dir / "best_model_resnet.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f'>>> 保存最佳模型 (Val Acc: {val_acc:.4f}) -> {best_model_path}')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    # 加载最佳模型
    print("\n" + "=" * 60)
    print("评估最佳模型")
    print("=" * 60)

    models_dir = base_dir / "models"
    checkpoint = torch.load(models_dir / "best_model_resnet.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型 (Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['val_acc']:.4f})")

    # 最终评估
    _, val_acc, y_pred, y_true = evaluate_model(model, val_loader, criterion, device)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n最终性能:")
    print(f"准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 绘制训练历史
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, le.classes_, 'ResNet')

    # 分类报告
    print("\n分类报告:")
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)
    print(report)

    # 保存模型和预处理器
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler_resnet.pkl")
    joblib.dump(le, models_dir / "label_encoder_resnet.pkl")

    print(f"\n模型和预处理器已保存到: {models_dir}")
    print(f"最佳模型: {models_dir / 'best_model_resnet.pth'}")
    print("\n训练完成！")


if __name__ == "__main__":
    main()
