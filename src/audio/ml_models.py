# -*- coding: utf-8 -*-
"""
机器学习模型训练模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
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


class WeaponSoundClassifier:
    """武器声音分类器"""

    def __init__(self):
        """初始化分类器"""
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'weapon',
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        准备训练数据

        Args:
            df: 特征数据框
            target_col: 目标列名（weapon, distance_label, direction）
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        # 选择特征列（排除标签列和文件路径）
        exclude_cols = ['weapon', 'distance', 'direction', 'id', 'file_path',
                        'distance_m', 'distance_label']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        X = df[self.feature_columns].values
        y = df[target_col].values

        # 标签编码
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            y = self.label_encoders[target_col].fit_transform(y)
        else:
            y = self.label_encoders[target_col].transform(y)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 标准化
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_knn(self, X_train, y_train, n_neighbors: int = 5) -> KNeighborsClassifier:
        """
        训练KNN分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_neighbors: 邻居数量

        Returns:
            KNeighborsClassifier: 训练好的模型
        """
        print(f"训练KNN分类器 (n_neighbors={n_neighbors})...")
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        self.models['KNN'] = model
        return model

    def train_svm(self, X_train, y_train, kernel: str = 'rbf',
                  C: float = 1.0) -> SVC:
        """
        训练SVM分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            kernel: 核函数类型
            C: 正则化参数

        Returns:
            SVC: 训练好的模型
        """
        print(f"训练SVM分类器 (kernel={kernel}, C={C})...")
        model = SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        return model

    def train_random_forest(self, X_train, y_train, n_estimators: int = 100,
                            max_depth: Optional[int] = None) -> RandomForestClassifier:
        """
        训练随机森林分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_estimators: 树的数量
            max_depth: 最大深度

        Returns:
            RandomForestClassifier: 训练好的模型
        """
        print(f"训练随机森林分类器 (n_estimators={n_estimators})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        return model

    def evaluate_model(self, model, X_test, y_test, model_name: str,
                       target_col: str = 'weapon') -> Dict:
        """
        评估模型性能

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            target_col: 目标列名

        Returns:
            Dict: 评估指标
        """
        print(f"\n评估 {model_name} 模型...")

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 分类报告
        class_names = self.label_encoders[target_col].classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0
        )

        results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names
        }

        # 打印结果
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision Macro): {precision:.4f}")
        print(f"召回率 (Recall Macro): {recall:.4f}")
        print(f"F1分数 (F1 Macro): {f1:.4f}")
        print("\n分类报告:")
        print(report)

        return results

    def cross_validate(self, model, X, y, cv: int = 5) -> Dict:
        """
        交叉验证

        Args:
            model: 模型
            X: 特征
            y: 标签
            cv: 折数

        Returns:
            Dict: 交叉验证结果
        """
        print(f"进行 {cv} 折交叉验证...")

        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

        print(f"交叉验证准确率: {scores}")
        print(f"平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return results

    def save_model(self, model_name: str, save_dir: str):
        """
        保存模型

        Args:
            model_name: 模型名称
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in self.models:
            print(f"模型 {model_name} 不存在")
            return

        model_path = save_dir / f"{model_name.lower()}_model.pkl"
        scaler_path = save_dir / "scaler.pkl"
        encoder_path = save_dir / "label_encoders.pkl"

        # 保存模型
        joblib.dump(self.models[model_name], model_path)
        print(f"模型已保存: {model_path}")

        # 保存标准化器
        joblib.dump(self.scaler, scaler_path)

        # 保存标签编码器
        joblib.dump(self.label_encoders, encoder_path)

    def load_model(self, model_name: str, load_dir: str):
        """
        加载模型

        Args:
            model_name: 模型名称
            load_dir: 加载目录
        """
        load_dir = Path(load_dir)

        model_path = load_dir / f"{model_name.lower()}_model.pkl"
        scaler_path = load_dir / "scaler.pkl"
        encoder_path = load_dir / "label_encoders.pkl"

        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return

        # 加载模型
        self.models[model_name] = joblib.load(model_path)
        print(f"模型已加载: {model_path}")

        # 加载标准化器
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # 加载标签编码器
        if encoder_path.exists():
            self.label_encoders = joblib.load(encoder_path)

    def predict(self, audio_path: str, model_name: str = 'RandomForest',
                target_col: str = 'weapon') -> Tuple[str, float]:
        """
        预测单个音频文件

        Args:
            audio_path: 音频文件路径
            model_name: 使用的模型名称
            target_col: 目标列名

        Returns:
            Tuple[str, float]: (预测标签, 置信度)
        """
        from .audio_features import AudioFeatureExtractor

        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未训练或加载")

        # 提取特征
        extractor = AudioFeatureExtractor()
        features = extractor.extract_features(audio_path)

        # 准备特征向量
        feature_vector = np.array([features[col] for col in self.feature_columns])
        feature_vector = feature_vector.reshape(1, -1)

        # 标准化
        feature_vector = self.scaler.transform(feature_vector)

        # 预测
        model = self.models[model_name]
        prediction = model.predict(feature_vector)[0]

        # 解码标签
        label = self.label_encoders[target_col].inverse_transform([prediction])[0]

        # 获取置信度（如果模型支持）
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_vector)[0]
            confidence = np.max(probabilities)

        return label, confidence
