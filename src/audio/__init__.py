# -*- coding: utf-8 -*-
"""
音频处理模块

说明：
- 该包包含音频特征提取（librosa）、传统ML、PyTorch模型加载/推理等。
- 为了让“缺少可选依赖”时也能正常导入项目其它功能，这里对部分依赖做了延迟/可选导入。
"""

# librosa 相关（可选导入：缺少 librosa 时，仍允许导入 ModelLoader 等）
try:
    from .audio_features import AudioFeatureExtractor, AudioVisualizer
except ImportError:
    AudioFeatureExtractor = None
    AudioVisualizer = None

from .ml_models import WeaponSoundClassifier
from .model_loader import ModelLoader, WeaponClassifierNN
from .recognizer import AudioRecognizer

__all__ = [
    'AudioFeatureExtractor',
    'AudioVisualizer',
    'WeaponSoundClassifier',
    'ModelLoader',
    'WeaponClassifierNN',
    'AudioRecognizer',
]
