# -*- coding: utf-8 -*-
"""
模型加载模块
负责加载训练好的音频识别模型
"""

from pathlib import Path
import joblib
from typing import Tuple, Optional
import warnings
import json
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class WeaponClassifierNN(nn.Module):
        """武器分类神经网络模型"""

        def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.3):
            super(WeaponClassifierNN, self).__init__()
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class EnhancedWeaponClassifierV2_1(nn.Module):
        """
        v2.1 模型架构（输入 140 维特征）

        说明：
        - 该结构与 train_enhanced_model_v2_1.py 中保持一致
        """

        def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
            super(EnhancedWeaponClassifierV2_1, self).__init__()
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
else:
    class WeaponClassifierNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装PyTorch，无法创建神经网络模型，请先安装 torch。")

    class EnhancedWeaponClassifierV2_1:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装PyTorch，无法创建 v2.1 音频模型，请先安装 torch。")


class ModelLoader:
    """模型加载器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.model = None
        self.models = None  # 集成模型（list[nn.Module]），与 self.model 互斥
        self.scaler = None
        self.label_encoder = None
        self.model_info = ""
        self.model_version: str = ""
        self.input_dim: Optional[int] = None
        self.centroids: Optional[dict[str, np.ndarray]] = None
        self.centroid_radii_p50: Optional[dict[str, float]] = None
        self.centroid_radii_p95: Optional[dict[str, float]] = None

    def _try_load_centroid_artifacts(self, model_version: str) -> None:
        self.centroids = None
        self.centroid_radii_p50 = None
        self.centroid_radii_p95 = None

        models_dir = self.base_dir / "models"
        safe_version = str(model_version).replace(".", "_")

        candidates: list[Path] = []
        if model_version == "ensemble_v2_5":
            candidates.append(models_dir / "centroids_ensemble_v2_5.json")
        if model_version == "ensemble_v2_4":
            candidates.append(models_dir / "centroids_ensemble_v2_4.json")
        if model_version == "v2.1":
            candidates.append(models_dir / "centroids_enhanced_v2_1.json")
        candidates.append(models_dir / f"centroids_{safe_version}.json")

        for path in candidates:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue
            centroids_raw = data.get("centroids")
            if not isinstance(centroids_raw, dict) or not centroids_raw:
                continue

            dim_expected = self.input_dim
            centroids: dict[str, np.ndarray] = {}
            for label, vector in centroids_raw.items():
                arr = np.asarray(vector, dtype=float).reshape(-1)
                if isinstance(dim_expected, int) and arr.shape[0] != dim_expected:
                    centroids = {}
                    break
                centroids[str(label)] = arr

            if not centroids:
                continue

            self.centroids = centroids
            self.centroid_radii_p50 = {str(k): float(v) for k, v in (data.get("radii_p50") or {}).items()}
            self.centroid_radii_p95 = {str(k): float(v) for k, v in (data.get("radii_p95") or {}).items()}
            return

    def load_model(self) -> Tuple[bool, str]:
        """
        加载训练好的模型

        Returns:
            (成功标志, 消息)
        """
        try:
            if not _TORCH_AVAILABLE:
                return False, "未安装PyTorch，音频识别模型无法加载（请安装 torch）。"

            # 尝试多个可能的路径（优先加载更高版本/更高准确率的模型）
            # 规则：只要文件存在，也可能因为“架构/维度/版本不一致”导致加载失败；
            # 因此这里按候选顺序逐个尝试，失败则继续尝试下一个，避免“有文件但不能用”时直接退出。
            candidates = [
                # v2.4 集成（3个模型投票/平均概率）
                (
                    "ensemble_v2_5",
                    (
                        self.base_dir / "models" / "ensemble_v2_5_model_1.pth",
                        self.base_dir / "models" / "ensemble_v2_5_model_2.pth",
                        self.base_dir / "models" / "ensemble_v2_5_model_3.pth",
                    ),
                    self.base_dir / "models" / "scaler_ensemble_v2_5.pkl",
                    self.base_dir / "models" / "label_encoder_ensemble_v2_5.pkl",
                ),
                (
                    "ensemble_v2_4",
                    (
                        self.base_dir / "models" / "ensemble_model_1.pth",
                        self.base_dir / "models" / "ensemble_model_2.pth",
                        self.base_dir / "models" / "ensemble_model_3.pth",
                    ),
                    self.base_dir / "models" / "scaler_ensemble_v2_4.pkl",
                    self.base_dir / "models" / "label_encoder_ensemble_v2_4.pkl",
                ),
                # v2.3 / v2.2 / v2.1 / v2（单模型）
                (
                    "v2.3",
                    self.base_dir / "models" / "best_model_enhanced_v2_3.pth",
                    self.base_dir / "models" / "scaler_enhanced_v2_3.pkl",
                    self.base_dir / "models" / "label_encoder_enhanced_v2_3.pkl",
                ),
                (
                    "v2.2",
                    self.base_dir / "models" / "best_model_enhanced_v2_2.pth",
                    self.base_dir / "models" / "scaler_enhanced_v2_2.pkl",
                    self.base_dir / "models" / "label_encoder_enhanced_v2_2.pkl",
                ),
                (
                    "v2.1",
                    self.base_dir / "models" / "best_model_enhanced_v2_1.pth",
                    self.base_dir / "models" / "scaler_enhanced_v2_1.pkl",
                    self.base_dir / "models" / "label_encoder_enhanced_v2_1.pkl",
                ),
                (
                    "v2",
                    self.base_dir / "models" / "best_model_enhanced_v2.pth",
                    self.base_dir / "models" / "scaler_enhanced_v2.pkl",
                    self.base_dir / "models" / "label_encoder_enhanced_v2.pkl",
                ),
                # 兼容老版本：DNN（10维特征）
                (
                    "dnn_v1",
                    self.base_dir / "best_model_nn.pth",
                    self.base_dir / "models" / "scaler_nn.pkl",
                    self.base_dir / "models" / "label_encoder_nn.pkl",
                ),
                (
                    "dnn_v1",
                    self.base_dir / "models" / "best_model_nn.pth",
                    self.base_dir / "models" / "scaler_nn.pkl",
                    self.base_dir / "models" / "label_encoder_nn.pkl",
                ),
            ]

            errors = []

            for model_version, model_path, scaler_path, encoder_path in candidates:
                # 跳过不存在的候选
                if model_version.startswith("ensemble_v2"):
                    mp1, mp2, mp3 = model_path
                    if not (mp1.exists() and mp2.exists() and mp3.exists() and scaler_path.exists() and encoder_path.exists()):
                        continue
                else:
                    if not (Path(model_path).exists() and scaler_path.exists() and encoder_path.exists()):
                        continue

                try:
                    # 每次尝试前先清空状态，避免半加载状态泄漏
                    self.model = None
                    self.models = None
                    self.scaler = None
                    self.label_encoder = None
                    self.model_info = ""
                    self.model_version = ""
                    self.input_dim = None

                    # 加载标准化器和标签编码器
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.scaler = joblib.load(str(scaler_path))
                        self.label_encoder = joblib.load(str(encoder_path))

                        for w in caught:
                            if "InconsistentVersionWarning" in str(type(w.message)) or "InconsistentVersionWarning" in str(w.message):
                                print(
                                    "提示: 检测到 scikit-learn 版本与模型训练时不一致，"
                                    "可能会影响 scaler/label_encoder 的加载或结果稳定性。"
                                    "建议在同一环境中重新导出模型，或固定 scikit-learn 版本。"
                                )
                                break

                    output_dim = len(self.label_encoder.classes_)

                    # v2.x ensemble
                    if model_version.startswith("ensemble_v2"):
                        mp1, mp2, mp3 = model_path
                        input_dim = int(getattr(self.scaler, "n_features_in_", 140))
                        self.models = []

                        for mp in (mp1, mp2, mp3):
                            checkpoint = torch.load(str(mp), map_location="cpu")
                            model = EnhancedWeaponClassifierV2_1(input_dim=input_dim, output_dim=output_dim, dropout=0.3)
                            model.load_state_dict(checkpoint["model_state_dict"])
                            model.eval()
                            self.models.append(model)

                        self.model_version = model_version
                        self.input_dim = input_dim
                        self._try_load_centroid_artifacts(self.model_version)
                        self.model_info = (
                            f"模型: {model_version} 集成(3模型, 输入维度: {input_dim}, 含二阶段策略)\n"
                            f"模型文件: {Path(mp1).name}/{Path(mp2).name}/{Path(mp3).name}"
                        )
                        return True, f"成功加载模型({model_version}集成): {self.base_dir / 'models'}"

                    # v2.x single
                    if model_version.startswith("v2"):
                        checkpoint = torch.load(str(model_path), map_location="cpu")
                        input_dim = int(getattr(self.scaler, "n_features_in_", 140))
                        self.model = EnhancedWeaponClassifierV2_1(input_dim=input_dim, output_dim=output_dim, dropout=0.3)
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        self.model.eval()
                        self.model_version = model_version
                        self.input_dim = input_dim
                        self._try_load_centroid_artifacts(self.model_version)
                        self.model_info = f"模型: {model_version} (输入维度: {input_dim}, 含二阶段策略)\n模型文件: {Path(model_path).name}"
                        return True, f"成功加载模型({model_version}): {model_path}"

                    # dnn_v1
                    checkpoint = torch.load(str(model_path), map_location="cpu")
                    input_dim = 10
                    hidden_dims = [128, 256, 128, 64]
                    self.model = WeaponClassifierNN(input_dim, hidden_dims, output_dim, dropout=0.3)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.model.eval()
                    self.model_version = "dnn_v1"
                    self.input_dim = input_dim
                    self.model_info = f"模型: DNN (准确率: 78.79%)\n模型文件: {Path(model_path).name}"
                    return True, f"成功加载模型(DNN): {model_path}"

                except Exception as e:
                    errors.append(f"{model_version}: {e}")
                    continue

            # 走到这里说明：没有任何可用候选（可能是文件缺失或存在但加载失败）
            if errors:
                details = "\n".join(f"  - {line}" for line in errors[:5])
                more = "\n  - ..." if len(errors) > 5 else ""
                return False, (
                    "未能加载任何可用音频识别模型（文件可能存在但加载失败）。\n"
                    "失败原因（前几条）:\n"
                    f"{details}{more}\n\n"
                    "请检查 models/ 目录下的模型文件是否与当前代码/依赖匹配。"
                )

            return False, (
                "未找到模型文件。\n"
                "请将训练产物放入 models/ 目录，或先运行训练脚本生成模型。"
            )

        except Exception as e:
            self.model = None
            return (
                False,
                "加载模型失败: "
                f"{e}\n\n"
                "可尝试重新训练并生成模型文件:\n"
                "  - v2.1（推荐）: python train_enhanced_model_v2_1.py\n"
                "  - DNN v1:      python train/train_model_deep.py"
            )

    def get_model_info(self) -> str:
        """获取模型信息"""
        if self.model or self.models:
            return self.model_info
        return "模型未加载"

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None or bool(self.models)
