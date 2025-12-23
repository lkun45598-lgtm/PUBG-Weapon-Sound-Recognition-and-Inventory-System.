# -*- coding: utf-8 -*-
"""
Train ensemble model v2.5 (based on v2.1 architecture).

v2.5 uses 150D features (v2.1 140D + temporal/transient extras).
This is intended to reduce long-distance hotspot confusions such as:
  AK@(600m,front) -> M24/M4 high-confidence mistakes.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


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
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed: int) -> None:
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

        running_loss += float(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += int(labels.size(0))
        correct += int((predicted == labels).sum().item())

    return running_loss / max(len(train_loader), 1), correct / max(total, 1)


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
            running_loss += float(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += int(labels.size(0))
            correct += int((predicted == labels).sum().item())

    return running_loss / max(len(val_loader), 1), correct / max(total, 1)


def train_single_model(
    model_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device,
    models_dir: Path,
    train_sample_weights: np.ndarray | None = None,
) -> float:
    print(f"\n{'='*80}")
    print(f"Training model #{model_id}")
    print(f"{'='*80}")

    seed = 42 + model_id * 100
    set_seed(seed)

    model = EnhancedWeaponClassifierV2_1(input_dim=input_dim, output_dim=num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if train_sample_weights is not None:
        sample_weights = np.asarray(train_sample_weights, dtype=np.float32)
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

    num_epochs = 150
    best_val_acc = 0.0
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "seed": seed,
            }
            torch.save(ckpt, models_dir / f"ensemble_v2_5_model_{model_id}.pth")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1:3d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch+1})")
    return best_val_acc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_csv = FEATURES_DIR / "train_enhanced_features_v2_5.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing v2.5 feature CSV: {train_csv} (run extract_enhanced_features_v2_5.py first)")

    models_dir = PROJECT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(train_csv)
    exclude_cols = ["weapon", "distance", "direction", "id", "file_path", "distance_m", "distance_label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df["weapon"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    distances = df.get("distance", pd.Series(["unknown"] * len(df))).values
    directions = df.get("direction", pd.Series(["unknown"] * len(df))).values

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

    class_weights_array = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_array_sqrt = np.sqrt(class_weights_array)
    class_weights_array_sqrt = class_weights_array_sqrt / class_weights_array_sqrt.mean()
    class_weights = torch.FloatTensor(class_weights_array_sqrt).to(device)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_sample_weights = _build_context_sample_weights(weapon_train, dist_train, dir_train)

    joblib.dump(scaler, models_dir / "scaler_ensemble_v2_5.pkl")
    joblib.dump(le, models_dir / "label_encoder_ensemble_v2_5.pkl")
    _save_centroid_artifacts(
        models_dir=models_dir,
        filename="centroids_ensemble_v2_5.json",
        artifacts=_build_centroid_artifacts(X_train_scaled, y_train, le),
    )

    val_accuracies = []
    for i in range(3):
        val_acc = train_single_model(
            model_id=i + 1,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            class_weights=class_weights,
            input_dim=int(X_train_scaled.shape[1]),
            num_classes=num_classes,
            device=device,
            models_dir=models_dir,
            train_sample_weights=train_sample_weights,
        )
        val_accuracies.append(val_acc)

    print("\nDone.")
    print("val_accuracies:", val_accuracies)
    print("avg:", float(np.mean(val_accuracies)))


if __name__ == "__main__":
    main()

