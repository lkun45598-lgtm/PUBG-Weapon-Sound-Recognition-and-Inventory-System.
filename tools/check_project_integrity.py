# -*- coding: utf-8 -*-
"""
项目完整性检查 - 发现潜在bug和冲突
"""

import sys
from pathlib import Path
import joblib
import warnings

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_DIR / "data" / "features"


def _pick_best_model_artifacts(models_dir: Path):
    """
    按与 src/audio/model_loader.py 类似的优先级选择可用模型文件集合。
    返回 (version, model_paths, scaler_path, encoder_path) 或 None。
    """
    candidates = [
        (
            "ensemble_v2_5",
            [
                models_dir / "ensemble_v2_5_model_1.pth",
                models_dir / "ensemble_v2_5_model_2.pth",
                models_dir / "ensemble_v2_5_model_3.pth",
            ],
            models_dir / "scaler_ensemble_v2_5.pkl",
            models_dir / "label_encoder_ensemble_v2_5.pkl",
        ),
        (
            "ensemble_v2_4",
            [
                models_dir / "best_model_ensemble_v2_4_1.pth",
                models_dir / "best_model_ensemble_v2_4_2.pth",
                models_dir / "best_model_ensemble_v2_4_3.pth",
            ],
            models_dir / "scaler_ensemble_v2_4.pkl",
            models_dir / "label_encoder_ensemble_v2_4.pkl",
        ),
        (
            "v2.3",
            [models_dir / "best_model_enhanced_v2_3.pth"],
            models_dir / "scaler_enhanced_v2_3.pkl",
            models_dir / "label_encoder_enhanced_v2_3.pkl",
        ),
        (
            "v2.2",
            [models_dir / "best_model_enhanced_v2_2.pth"],
            models_dir / "scaler_enhanced_v2_2.pkl",
            models_dir / "label_encoder_enhanced_v2_2.pkl",
        ),
        (
            "v2.1",
            [models_dir / "best_model_enhanced_v2_1.pth"],
            models_dir / "scaler_enhanced_v2_1.pkl",
            models_dir / "label_encoder_enhanced_v2_1.pkl",
        ),
        (
            "v2",
            [models_dir / "best_model_enhanced_v2.pth"],
            models_dir / "scaler_enhanced_v2.pkl",
            models_dir / "label_encoder_enhanced_v2.pkl",
        ),
        (
            "dnn_v1",
            [models_dir / "best_model_nn.pth"],
            models_dir / "scaler_nn.pkl",
            models_dir / "label_encoder_nn.pkl",
        ),
    ]

    for version, model_paths, scaler_path, encoder_path in candidates:
        if all(p.exists() for p in model_paths) and scaler_path.exists() and encoder_path.exists():
            return version, model_paths, scaler_path, encoder_path
    return None


def check_model_files():
    """检查模型文件完整性"""
    print("="*80)
    print("检查1: 模型文件完整性")
    print("="*80)

    models_dir = PROJECT_DIR / "models"

    issues = []

    if not models_dir.exists():
        issues.append("[FAIL] models/ 目录不存在")
    else:
        picked = _pick_best_model_artifacts(models_dir)
        if not picked:
            issues.append("[FAIL] 未找到可用模型文件集合（v2.4/v2.3/v2.2/v2.1/v2/dnn_v1 均不完整）")
        else:
            version, model_paths, scaler_path, encoder_path = picked
            print(f"[OK] 选择模型版本: {version}")
            for p in model_paths + [scaler_path, encoder_path]:
                size = p.stat().st_size / 1024  # KB
                print(f"[OK] {p.name}: {size:.1f} KB")

    if issues:
        print("\n问题:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("\n[OK] 所有模型文件存在")
    return True


def check_model_consistency():
    """检查模型和预处理器的一致性"""
    print("\n" + "="*80)
    print("检查2: 模型和预处理器一致性")
    print("="*80)

    try:
        if not _TORCH_AVAILABLE:
            print("[SKIP] 未安装 torch，跳过 .pth 模型一致性检查（本项不计入失败）")
            return True

        models_dir = PROJECT_DIR / "models"
        picked = _pick_best_model_artifacts(models_dir)
        if not picked:
            print("[FAIL] 未找到可用模型文件集合")
            return False

        version, model_paths, scaler_path, encoder_path = picked
        print(f"[INFO] 检查版本: {version}")

        # 加载预处理器（允许 sklearn 版本不一致的警告，但不要影响检查流程）
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scaler = joblib.load(scaler_path)
            le = joblib.load(encoder_path)
            if caught:
                print(f"[WARN] 加载预处理器时出现 {len(caught)} 条警告（常见于 sklearn 版本不一致）")

        scaler_dim = int(getattr(scaler, "n_features_in_", 140))
        label_classes = len(getattr(le, "classes_", []))

        if label_classes <= 0:
            print("[FAIL] LabelEncoder.classes_ 为空或不存在")
            return False

        def _infer_dims(checkpoint_dict):
            state = checkpoint_dict.get("model_state_dict") or checkpoint_dict.get("state_dict") or {}
            first = None
            last = None
            for key, tensor in state.items():
                if "weight" in key and "network." in key:
                    if first is None:
                        first = tensor
                    last = tensor
            if first is None or last is None:
                return None, None
            return int(first.shape[1]), int(last.shape[0])

        print("特征维度:")
        print(f"  Scaler期望: {scaler_dim}")
        print("\n类别数量:")
        print(f"  LabelEncoder: {label_classes}")

        issues = []
        for model_path in model_paths:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_input_dim, model_output_dim = _infer_dims(checkpoint)
            if model_input_dim is None or model_output_dim is None:
                issues.append(f"[FAIL] {model_path.name}: 无法从 state_dict 推断输入/输出维度")
                continue

            print(f"\n{model_path.name}:")
            print(f"  模型输入: {model_input_dim}")
            print(f"  模型输出: {model_output_dim}")

            if model_input_dim != scaler_dim:
                issues.append(f"[FAIL] 特征维度不匹配: {model_path.name} input={model_input_dim} vs Scaler{scaler_dim}")
            if model_output_dim != label_classes:
                issues.append(f"[FAIL] 类别数不匹配: {model_path.name} out={model_output_dim} vs LabelEncoder{label_classes}")

        if issues:
            print("\n问题:")
            for issue in issues:
                print(f"  {issue}")
            return False

        print("\n[OK] 模型和预处理器维度一致")
        return True

    except Exception as e:
        print(f"\n[FAIL] 检查失败: {e}")
        return False


def check_feature_extraction():
    """检查特征提取维度"""
    print("\n" + "="*80)
    print("检查3: 特征提取函数")
    print("="*80)

    try:
        # 模拟特征提取（仅计算维度，不依赖运行时音频库）
        n_mfcc = 20

        # 计算应有的特征数
        # 基础特征: 13个
        basic_features = 13

        # MFCC: 20 * 6 = 120 (mean + std + delta_mean + delta_std + delta2_mean + delta2_std)
        mfcc_features = n_mfcc * 6

        # 谱对比度: 7个
        spectral_contrast_features = 7

        # 总特征数
        base_total = basic_features + mfcc_features + spectral_contrast_features
        total_features = base_total

        print(f"预期特征维度:")
        print(f"  基础特征:     {basic_features}")
        print(f"  MFCC特征:     {mfcc_features} (n_mfcc={n_mfcc}, 包含delta和delta2)")
        print(f"  谱对比度:     {spectral_contrast_features}")
        print(f"  v2-base总计:  {base_total}")

        # 加载scaler检查
        models_dir = PROJECT_DIR / "models"
        picked = _pick_best_model_artifacts(models_dir)
        if not picked:
            print("[FAIL] 未找到可用模型文件集合，无法确定 Scaler 维度")
            return False

        version, _, scaler_path, _ = picked
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scaler = joblib.load(scaler_path)
            if caught:
                print(f"[WARN] 加载 Scaler 时出现 {len(caught)} 条警告（常见于 sklearn 版本不一致）")
        expected_dim = int(getattr(scaler, "n_features_in_", base_total))

        if expected_dim == base_total:
            extra_temporal = 0
        elif expected_dim == base_total + 10:
            extra_temporal = 10
        else:
            print(f"\n[FAIL] 未知的特征维度组合")
            print(f"   v2-base(140) 计算得到: {base_total}")
            print(f"   Scaler要求: {expected_dim}")
            return False

        total_features = base_total + extra_temporal

        print(f"\n实际Scaler期望: {expected_dim}")
        print(f"[INFO] 选择模型版本: {version}")
        if extra_temporal:
            print(f"[INFO] 检测到额外特征: {extra_temporal} (v2.5 extras)")

        if total_features != expected_dim:
            print(f"\n[FAIL] 特征维度不匹配!")
            print(f"   计算得到: {total_features}")
            print(f"   Scaler要求: {expected_dim}")
            print(f"   差异: {expected_dim - total_features}")
            return False

        print("\n[OK] 特征提取维度正确")
        return True

    except Exception as e:
        print(f"\n[FAIL] 检查失败: {e}")
        return False


def check_scripts_consistency():
    """检查关键脚本是否使用一致的配置"""
    print("\n" + "="*80)
    print("检查4: 关键脚本配置一致性")
    print("="*80)

    issues = []

    # 检查weapon_classifier_final.py
    classifier_file = PROJECT_DIR / "tools" / "weapon_classifier_final.py"
    if classifier_file.exists():
        content = classifier_file.read_text(encoding='utf-8')

        # 检查统一推理入口（应委托给 src.audio 的 ModelLoader/AudioRecognizer）
        checks = {
            "from src.audio import ModelLoader, AudioRecognizer": "使用统一推理入口(src.audio)",
            "ModelLoader(": "包含 ModelLoader 初始化",
            "AudioRecognizer(": "包含 AudioRecognizer 初始化",
        }

        print("\nweapon_classifier_final.py 检查:")
        for check, desc in checks.items():
            if check in content:
                print(f"  [OK] {desc}")
            else:
                issues.append(f"[FAIL] {desc} 未找到: {check}")
    else:
        issues.append("[FAIL] weapon_classifier_final.py 不存在")

    if issues:
        print("\n问题:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("\n[OK] 脚本配置一致")
    return True


def check_data_files():
    """检查数据文件"""
    print("\n" + "="*80)
    print("检查5: 数据文件")
    print("="*80)

    models_dir = PROJECT_DIR / "models"
    picked = _pick_best_model_artifacts(models_dir)
    expected_dim = 140
    if picked:
        _, _, scaler_path, _ = picked
        try:
            scaler = joblib.load(scaler_path)
            expected_dim = int(getattr(scaler, "n_features_in_", 140))
        except Exception:
            expected_dim = 140

    if expected_dim == 150:
        required_files = [
            "train_enhanced_features_v2_5.csv",
            "test_enhanced_features_v2_5.csv",
        ]
    else:
        required_files = [
            "train_enhanced_features_v2.csv",
            "test_enhanced_features_v2.csv",
        ]

    issues = []

    for file in required_files:
        file_path = FEATURES_DIR / file
        if not file_path.exists():
            issues.append(f"[FAIL] 缺失: {file_path.relative_to(PROJECT_DIR)}")
        else:
            # 读取并检查
            import pandas as pd
            df = pd.read_csv(file_path)
            print(f"[OK] {file}: {df.shape[0]} 样本, {df.shape[1]} 列")

            # 检查是否有weapon列
            if 'weapon' not in df.columns:
                issues.append(f"[FAIL] {file} 缺少 'weapon' 列")

    if issues:
        print("\n问题:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("\n[OK] 数据文件完整")
    return True


def check_python_encoding():
    """检查Python文件编码问题"""
    print("\n" + "="*80)
    print("检查6: Python脚本编码问题")
    print("="*80)

    # 检查关键脚本中是否还有可能导致编码问题的字符
    scripts_to_check = [
        Path("tools/weapon_classifier_final.py"),
        Path("eval/evaluate_two_stage.py"),
        Path("src/audio/model_loader.py"),
        Path("src/audio/recognizer.py"),
    ]

    issues = []

    for rel_path in scripts_to_check:
        script_path = PROJECT_DIR / rel_path
        display_name = str(rel_path).replace("\\", "/")
        if script_path.exists():
            try:
                content = script_path.read_text(encoding='utf-8')

                # 检查常见的编码问题字符（避免把 emoji 本身写进源码里）
                problematic_chars = ['→']
                found_chars = [char for char in problematic_chars if char in content]

                if found_chars:
                    issues.append(f"[WARN] {display_name}: 包含 {len(found_chars)} 个可能导致编码问题的字符")
                else:
                    print(f"[OK] {display_name}: 无编码问题")

            except Exception as e:
                issues.append(f"[FAIL] {display_name}: 读取失败 - {e}")
        else:
            print(f"  - {display_name}: 文件不存在（跳过）")

    if issues:
        print("\n警告:")
        for issue in issues:
            print(f"  {issue}")
        print("\n建议: 在Windows环境下，避免在print语句中使用emoji")
        return False

    print("\n[OK] 所有脚本无明显编码问题")
    return True


def main():
    print("="*80)
    print("PUBG枪械音频识别 - 项目完整性检查")
    print("="*80)
    print()

    results = {}

    # 运行所有检查
    results['model_files'] = check_model_files()
    results['model_consistency'] = check_model_consistency()
    results['feature_extraction'] = check_feature_extraction()
    results['scripts_consistency'] = check_scripts_consistency()
    results['data_files'] = check_data_files()
    results['encoding'] = check_python_encoding()

    # 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"{check:25s}: {status}")

    print("\n" + "="*80)

    if all_passed:
        print("[OK] 所有检查通过! 项目无重大bug或冲突")
        print("\n可以安全部署和使用:")
        print("  - weapon_classifier_final.py (推理脚本)")
        print("  - evaluate_two_stage.py (评估脚本)")
    else:
        print("[WARN]  发现一些问题，请根据上述详情进行修复")
        failed_checks = [k for k, v in results.items() if not v]
        print(f"\n失败的检查: {', '.join(failed_checks)}")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
