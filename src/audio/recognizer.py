# -*- coding: utf-8 -*-
"""
音频识别模块
负责音频特征提取和武器识别
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


class AudioRecognizer:
    """音频识别器"""

    def __init__(self, model_loader):
        """
        初始化音频识别器

        Args:
            model_loader: ModelLoader实例
        """
        self.model_loader = model_loader

    def extract_features(self, audio_path: str) -> Optional[Dict[str, float]]:
        """
        从音频文件提取特征

        Args:
            audio_path: 音频文件路径

        Returns:
            特征字典或None(失败时)
        """
        model_version = getattr(self.model_loader, "model_version", "") or ""
        input_dim = getattr(self.model_loader, "input_dim", None)

        # 优先按模型版本选择特征方案；若未设置，则根据输入维度兜底判断
        use_v2 = (
            model_version.startswith("v2")
            or model_version.startswith("ensemble_v2")
            or (isinstance(input_dim, int) and input_dim >= 100)
        )

        if use_v2:
            if (
                model_version.startswith("v2.5")
                or model_version.startswith("ensemble_v2_5")
                or (isinstance(input_dim, int) and input_dim == 150)
            ):
                return self._extract_features_v2_5(audio_path)
            return self._extract_features_v2_1(audio_path)
        return self._extract_features_v1(audio_path)

    def _extract_features_v1(self, audio_path: str) -> Dict[str, float]:
        """基础特征（10维，与 DNN v1 模型一致）"""
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None)

            duration = librosa.get_duration(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            return {
                "duration": float(duration),
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "zcr_mean": float(np.mean(zcr)),
                "zcr_std": float(np.std(zcr)),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_centroid_std": float(np.std(spectral_centroid)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            }

        except ImportError:
            raise ImportError("未安装 librosa 库，无法提取音频特征。请安装: pip install librosa soundfile")
        except Exception as e:
            raise Exception(f"特征提取失败(v1): {e}")

    @staticmethod
    def _feature_order_v2_1(n_mfcc: int = 20) -> list[str]:
        base = [
            "duration",
            "rms_mean",
            "rms_std",
            "zcr_mean",
            "zcr_std",
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "spectral_flatness_mean",
            "spectral_flatness_std",
        ]

        mfcc = []
        for i in range(1, n_mfcc + 1):
            mfcc.extend(
                [
                    f"mfcc{i}_mean",
                    f"mfcc{i}_std",
                    f"mfcc{i}_delta_mean",
                    f"mfcc{i}_delta_std",
                    f"mfcc{i}_delta2_mean",
                    f"mfcc{i}_delta2_std",
                ]
            )

        contrast = [f"spectral_contrast{i}_mean" for i in range(1, 8)]
        return base + mfcc + contrast

    @staticmethod
    def _feature_order_v2_5(n_mfcc: int = 20) -> list[str]:
        extra = [
            "rms_max",
            "rms_p95",
            "onset_env_mean",
            "onset_env_std",
            "onset_env_max",
            "onset_count",
            "onset_rate",
            "tempo",
            "ioi_mean",
            "ioi_std",
        ]
        return AudioRecognizer._feature_order_v2_1(n_mfcc=n_mfcc) + extra

    def _extract_features_v2_1(self, audio_path: str, n_mfcc: int = 20) -> Dict[str, float]:
        """
        v2.1 特征（140维）

        与 `extract_enhanced_features_v2.py` / `train_enhanced_model_v2_1.py` 使用的列一致：
        - 基础 13 维
        - MFCC + Delta + Delta2：20 * 6 = 120 维
        - 谱对比度 7 维
        合计 140 维
        """
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None)
            features: Dict[str, float] = {}

            # 基础特征（13维）
            features["duration"] = float(librosa.get_duration(y=y, sr=sr))

            rms = librosa.feature.rms(y=y)[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
            features["spectral_centroid_std"] = float(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
            features["spectral_flatness_std"] = float(np.std(spectral_flatness))

            # MFCC + Delta + Delta2（120维）
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_delta = librosa.feature.delta(mfccs, order=1)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            for i in range(n_mfcc):
                idx = i + 1
                features[f"mfcc{idx}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc{idx}_std"] = float(np.std(mfccs[i]))
                features[f"mfcc{idx}_delta_mean"] = float(np.mean(mfcc_delta[i]))
                features[f"mfcc{idx}_delta_std"] = float(np.std(mfcc_delta[i]))
                features[f"mfcc{idx}_delta2_mean"] = float(np.mean(mfcc_delta2[i]))
                features[f"mfcc{idx}_delta2_std"] = float(np.std(mfcc_delta2[i]))

            # 谱对比度（7维）
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(spectral_contrast.shape[0]):
                idx = i + 1
                features[f"spectral_contrast{idx}_mean"] = float(np.mean(spectral_contrast[i]))

            # 最终校验：确保特征齐全
            order = self._feature_order_v2_1(n_mfcc=n_mfcc)
            missing = [k for k in order if k not in features]
            if missing:
                raise ValueError(f"v2.1 特征缺失: {missing[:5]}{'...' if len(missing) > 5 else ''}")

            return features

        except ImportError:
            raise ImportError("未安装 librosa 库，无法提取 v2.1 特征。请安装: pip install librosa soundfile")
        except Exception as e:
            raise Exception(f"特征提取失败(v2.1): {e}")

    def _extract_features_v2_5(self, audio_path: str, n_mfcc: int = 20) -> Dict[str, float]:
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None)
            features: Dict[str, float] = {}

            features["duration"] = float(librosa.get_duration(y=y, sr=sr))

            rms = librosa.feature.rms(y=y)[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
            features["spectral_centroid_std"] = float(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
            features["spectral_flatness_std"] = float(np.std(spectral_flatness))

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_delta = librosa.feature.delta(mfccs, order=1)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            for i in range(n_mfcc):
                idx = i + 1
                features[f"mfcc{idx}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc{idx}_std"] = float(np.std(mfccs[i]))
                features[f"mfcc{idx}_delta_mean"] = float(np.mean(mfcc_delta[i]))
                features[f"mfcc{idx}_delta_std"] = float(np.std(mfcc_delta[i]))
                features[f"mfcc{idx}_delta2_mean"] = float(np.mean(mfcc_delta2[i]))
                features[f"mfcc{idx}_delta2_std"] = float(np.std(mfcc_delta2[i]))

            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(spectral_contrast.shape[0]):
                idx = i + 1
                features[f"spectral_contrast{idx}_mean"] = float(np.mean(spectral_contrast[i]))

            # Temporal/transient extras (10D)
            features["rms_max"] = float(np.max(rms)) if rms.size else 0.0
            features["rms_p95"] = float(np.quantile(rms, 0.95)) if rms.size else 0.0

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features["onset_env_mean"] = float(np.mean(onset_env)) if onset_env.size else 0.0
            features["onset_env_std"] = float(np.std(onset_env)) if onset_env.size else 0.0
            features["onset_env_max"] = float(np.max(onset_env)) if onset_env.size else 0.0

            try:
                onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
                onset_times = np.asarray(onset_times, dtype=float).reshape(-1)
            except Exception:
                onset_times = np.asarray([], dtype=float)

            onset_count = int(onset_times.shape[0])
            duration = float(features["duration"]) if float(features["duration"]) > 1e-6 else 1e-6
            features["onset_count"] = float(onset_count)
            features["onset_rate"] = float(onset_count / duration)

            try:
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                features["tempo"] = float(tempo[0]) if len(tempo) else 0.0
            except Exception:
                features["tempo"] = 0.0

            if onset_count >= 2:
                ioi = np.diff(onset_times)
                features["ioi_mean"] = float(np.mean(ioi))
                features["ioi_std"] = float(np.std(ioi))
            else:
                features["ioi_mean"] = 0.0
                features["ioi_std"] = 0.0

            order = self._feature_order_v2_5(n_mfcc=n_mfcc)
            missing = [k for k in order if k not in features]
            if missing:
                raise ValueError(f"v2.5 特征缺失: {missing[:5]}{'...' if len(missing) > 5 else ''}")

            return features

        except ImportError:
            raise ImportError("未安装 librosa 库，无法提取 v2.5 特征。请安装: pip install librosa soundfile")
        except Exception as e:
            raise Exception(f"特征提取失败(v2.5): {e}")

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        预测武器类型

        Args:
            features: 音频特征字典

        Returns:
            (武器名称, 置信度)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("未安装PyTorch，无法进行模型推理（请安装 torch）。")

        if not self.model_loader.is_loaded():
            raise RuntimeError("模型未加载")

        model_version = getattr(self.model_loader, "model_version", "") or ""
        input_dim = getattr(self.model_loader, "input_dim", None)

        use_v2 = (
            model_version.startswith("v2")
            or model_version.startswith("ensemble_v2")
            or (isinstance(input_dim, int) and input_dim >= 100)
        )

        if use_v2:
            model_version = getattr(self.model_loader, "model_version", "") or ""
            expected = getattr(self.model_loader.scaler, "n_features_in_", None)
            if (
                model_version.startswith("v2.5")
                or model_version.startswith("ensemble_v2_5")
                or expected == 150
            ):
                order = self._feature_order_v2_5()
            else:
                order = self._feature_order_v2_1()
        else:
            order = [
                "duration",
                "rms_mean",
                "rms_std",
                "zcr_mean",
                "zcr_std",
                "spectral_centroid_mean",
                "spectral_centroid_std",
                "spectral_bandwidth_mean",
                "spectral_bandwidth_std",
                "spectral_rolloff_mean",
            ]

        try:
            feature_vector = np.array([features[k] for k in order], dtype=float).reshape(1, -1)
        except KeyError as e:
            raise KeyError(f"特征缺失，无法预测: {e}")

        # 维度校验（避免“模型升级了但特征仍按旧维度”的静默错误）
        expected = getattr(self.model_loader.scaler, "n_features_in_", None)
        if isinstance(expected, int) and feature_vector.shape[1] != expected:
            raise ValueError(f"特征维度不匹配: got={feature_vector.shape[1]}, expected={expected}")

        # 标准化
        feature_vector = self.model_loader.scaler.transform(feature_vector)
        feature_vector_scaled = feature_vector.reshape(-1)

        # 转换为Tensor
        feature_tensor = torch.FloatTensor(feature_vector)

        # 预测
        with torch.no_grad():
            model_version = getattr(self.model_loader, "model_version", "") or ""

            # 集成模型：平均概率
            if getattr(self.model_loader, "models", None):
                probs_sum = None
                for m in self.model_loader.models:
                    logits = m(feature_tensor)
                    probs = torch.softmax(logits, dim=1)
                    probs_sum = probs if probs_sum is None else (probs_sum + probs)
                probabilities = probs_sum / float(len(self.model_loader.models))
            else:
                output = self.model_loader.model(feature_tensor)
                probabilities = torch.softmax(output, dim=1)

            probs_row = probabilities[0]

            # v2.x：固定启用二阶段策略（不提供开关）
            if model_version.startswith("v2") or model_version.startswith("ensemble_v2"):
                pred_idx, conf = self._two_stage_predict_v2_1(probs_row)
                predicted = torch.tensor([pred_idx], dtype=torch.long)
                confidence = torch.tensor([conf], dtype=torch.float)
            else:
                confidence, predicted = torch.max(probabilities, 1)

        # 解码标签
        weapon = self.model_loader.label_encoder.inverse_transform([predicted.item()])[0]
        conf_value = float(confidence.item())

        if use_v2:
            weapon, conf_value = self._apply_centroid_correction(
                weapon=weapon,
                confidence=conf_value,
                probs_row=probs_row,
                feature_vector_scaled=feature_vector_scaled,
            )

        return weapon, conf_value

    def _two_stage_predict_v2_1(
        self,
        probs_row: "torch.Tensor",
        confidence_threshold: float = 0.75,
        boost_in_top_k: int = 3,
    ) -> tuple[int, float]:
        """
        v2.1 二阶段策略（与 weapon_classifier_final.py 思路一致）

        - 高置信度：直接接受 argmax
        - 低置信度：在 top-k 内优先提升小样本类别（aug/m249/qbu/vec/win）
        """
        top_k = min(boost_in_top_k, int(probs_row.shape[0]))
        top_k_probs, top_k_indices = torch.topk(probs_row, k=top_k)

        max_prob = float(top_k_probs[0].item())
        max_class = int(top_k_indices[0].item())

        if max_prob >= confidence_threshold:
            return max_class, max_prob

        # 小样本类别提升
        target_weapons = ["aug", "m249", "qbu", "vec", "win"]
        classes = list(getattr(self.model_loader.label_encoder, "classes_", []))
        target_classes = {classes.index(w) for w in target_weapons if w in classes}
        if not target_classes:
            return max_class, max_prob

        for j in range(top_k):
            class_idx = int(top_k_indices[j].item())
            class_prob = float(top_k_probs[j].item())
            if class_idx in target_classes and class_prob >= 0.3 * max_prob:
                return class_idx, class_prob

        return max_class, max_prob

    def _apply_centroid_correction(
        self,
        weapon: str,
        confidence: float,
        probs_row: "torch.Tensor",
        feature_vector_scaled: np.ndarray,
    ) -> tuple[str, float]:
        centroids = getattr(self.model_loader, "centroids", None)
        if not isinstance(centroids, dict) or not centroids:
            return weapon, confidence

        if weapon not in {"m24", "m4"}:
            return weapon, confidence

        if "ak" not in centroids or weapon not in centroids:
            return weapon, confidence

        if float(confidence) < 0.75:
            return weapon, confidence

        try:
            x = np.asarray(feature_vector_scaled, dtype=float).reshape(-1)
            d_pred = float(np.linalg.norm(x - np.asarray(centroids[weapon], dtype=float).reshape(-1)))
            d_ak = float(np.linalg.norm(x - np.asarray(centroids["ak"], dtype=float).reshape(-1)))
        except Exception:
            return weapon, confidence

        if not (np.isfinite(d_pred) and np.isfinite(d_ak)):
            return weapon, confidence

        ratio = d_ak / max(d_pred, 1e-9)

        radii_p95 = getattr(self.model_loader, "centroid_radii_p95", None) or {}
        pred_p95 = float(radii_p95.get(weapon, float("inf")))
        ak_p95 = float(radii_p95.get("ak", float("inf")))

        pred_atypical = np.isfinite(pred_p95) and d_pred > pred_p95 * 1.05
        ak_typical = np.isfinite(ak_p95) and d_ak < ak_p95 * 1.05

        if ratio > 0.92:
            return weapon, confidence

        if not (pred_atypical or ak_typical):
            return weapon, confidence

        classes = list(getattr(self.model_loader.label_encoder, "classes_", []))
        p_ak = 0.0
        if "ak" in classes:
            try:
                p_ak = float(probs_row[classes.index("ak")].item())
            except Exception:
                p_ak = 0.0

        try:
            score_ak = float(np.exp(-d_ak))
            score_pred = float(np.exp(-d_pred))
            centroid_conf = score_ak / max(score_ak + score_pred, 1e-12)
        except Exception:
            centroid_conf = 0.5

        return "ak", float(max(p_ak, centroid_conf))

    def predict_from_file(self, audio_path: str) -> Tuple[str, float]:
        """
        从音频文件直接预测

        Args:
            audio_path: 音频文件路径

        Returns:
            (武器名称, 置信度)
        """
        features = self.extract_features(audio_path)
        if features is None:
            raise ValueError("特征提取失败")

        return self.predict(features)
