# PUBG 武器管理与声音识别系统

课程设计项目 - 综合实践武器管理系统与声音识别功能

## 项目简介

本项目整合了"武器管理系统"与"武器声音识别"两大模块，基于PUBG游戏场景，实现：
- 玩家账号管理系统（注册/登录）
- 武器数据管理（增删改查、排序、统计）
- 音频特征提取与可视化
- 深度学习音频分类（DNN、ResNet、LightGBM等）
- 完整的图形用户界面（GUI）

## 快速开始

### 方式1: 运行Web版本(推荐✨)

```bash
# 使用conda环境
conda activate pytorch

# 运行Web服务器
python app.py
```

然后在浏览器访问:
- **本地访问**: http://localhost:5000
- **局域网访问**: http://你的IP地址:5000 (如 http://192.168.1.118:5000)

Web版本功能:
- ✅ 在线注册/登录系统
- ✅ 武器库浏览(搜索、排序)
- ✅ 背包管理
- ✅ 音频上传识别
- ✅ 美观的响应式界面
- ✅ 支持多用户同时访问

### 方式2: 运行桌面GUI版本

```bash
# 使用conda环境
conda activate pytorch

# 运行完整版GUI程序
python main_gui.py
```

程序包含：
- ✅ 武器管理系统（浏览、添加、删除、排序）
- ✅ 音频识别功能（文件识别、手动输入、批量测试）
- ✅ 用户账号系统（注册/登录）
- ✅ 数据持久化

## 项目结构

```
D:\Python课设设计\
├── main_gui.py               # ⭐ 主程序入口（GUI版）
├── main_gui_complete.py      # 兼容旧入口（可选）
├── README.md                  # 项目说明
├── requirements.txt           # 依赖库
├── Arms.xlsx                  # 武器数据（6种武器）
├── train_selected_features.csv  # 训练特征
├── test_selected_features.csv   # 测试特征
│
├── src/                       # 源代码模块
│   ├── models/               # 数据模型
│   │   ├── weapon.py         # 武器类
│   │   └── player.py         # 玩家类
│   ├── data/                 # 数据管理
│   │   └── data_manager.py   # 数据持久化
│   ├── auth/                 # 认证模块
│   │   └── auth_manager.py   # 登录注册
│   ├── services/             # 业务逻辑
│   │   └── weapon_service.py # 武器服务
│   └── audio/                # 音频处理模块
│       ├── audio_features.py # 特征提取
│       ├── ml_models.py      # ML模型
│       ├── model_loader.py   # 模型加载器（重构）
│       └── recognizer.py     # 音频识别器（重构）
│
├── train/                     # 训练脚本
│   ├── train_model_simple.py   # 传统ML（KNN/SVM/RF）
│   ├── train_model_deep.py     # ⭐ DNN深度学习（78.79%）
│   ├── train_model_resnet.py   # ResNet架构（72.73%）
│   └── train_model_lightgbm.py # LightGBM（69.70%）
│
├── models/                    # 训练好的模型文件
│   ├── best_model_nn.pth      # DNN模型
│   ├── best_model_resnet.pth  # ResNet模型
│   ├── scaler_nn.pkl          # 特征标准化器
│   ├── label_encoder_nn.pkl   # 标签编码器
│   └── ...                    # 其他模型文件
│
├── data/                      # 数据文件
│   ├── gun_sound_train/       # 训练音频（38种武器）
│   ├── gun_sound_test/        # 测试音频
│   ├── players.json           # 玩家数据
│   └── weapons.json           # 武器数据
│
├── demo/                      # 演示音频文件
│   ├── ak_demo.mp3           # AKM演示
│   ├── m4_demo.mp3           # M416演示
│   ├── awm_demo.mp3          # AWM演示
│   ├── sks_demo.mp3          # SKS演示
│   └── ump_demo.mp3          # UMP9演示
│
├── results/                   # 训练结果
│   ├── confusion_matrix_*.png # 混淆矩阵
│   ├── training_history_*.png # 训练曲线
│   └── model_comparison.png   # 模型对比
│
├── docs/                      # 文档
│   ├── 完整版GUI使用说明.md
│   ├── 模型对比总结报告.md
│   └── 课程设计相关PDF
│
└── archive/                   # 归档文件
    ├── main.py                # 旧版终端程序
    ├── main_gui.py            # 旧版基础GUI
    └── train_model.py         # 旧版训练脚本
```

## 环境配置

### 1. Python版本
- 需要 Python 3.10 或更高版本

### 2. 安装依赖

使用conda环境（推荐）：
```bash
conda activate pytorch
pip install -r requirements.txt
```

或使用pip直接安装：
```bash
pip install -r requirements.txt
```

### 3. 主要依赖库
- `pandas`, `numpy` - 数据处理
- `openpyxl` - Excel读取
- `torch` - PyTorch深度学习框架
- `scikit-learn` - 机器学习
- `lightgbm` - 梯度提升框架
- `librosa`, `soundfile` - 音频处理
- `matplotlib`, `seaborn` - 可视化
- `tkinter` - GUI界面（Python内置）

## 使用说明

### 主程序（推荐）

运行完整版GUI程序：
```bash
python main_gui.py
```

功能包括：
1. **用户注册/登录** - 学号作为账号，密码SHA256加密存储
2. **武器浏览** - 查看所有武器，支持搜索和排序
3. **背包管理** - 添加/删除武器，查看背包，按DPS排序
4. **弹药管理** - 添加弹药，查看弹药库存
5. **音频识别** - 三种识别方式：
   - 📁 选择音频文件识别
   - 📊 手动输入特征识别
   - 🔍 批量测试模型

### 模型训练

#### 方式1：使用预提取特征（Level A）

直接使用CSV特征文件训练：
```bash
# 传统机器学习（KNN/SVM/RF）
python train/train_model_simple.py

# 深度神经网络（推荐，准确率78.79%）
python train/train_model_deep.py

# ResNet架构
python train/train_model_resnet.py

# LightGBM
python train/train_model_lightgbm.py
```

#### 方式2：从原始音频提取特征（Level B）

如果有原始音频文件：
```bash
python extract_features.py
```

该脚本会：
- 从音频文件提取特征（duration, RMS, ZCR, spectral特征等）
- 保存特征到CSV文件
- 可选：进行可视化分析

## 模型性能对比

| 模型 | 准确率 | 说明 |
|------|--------|------|
| **DNN** | **78.79%** | ⭐ 最佳模型，4层全连接网络 |
| ResNet | 72.73% | 残差网络，带跳跃连接 |
| RandomForest | 71.59% | 传统集成学习 |
| LightGBM | 69.70% | 梯度提升树 |
| KNN | 64.02% | K近邻算法 |
| SVM | 63.64% | 支持向量机 |

训练结果保存在 [results/](results/) 目录

## 功能特性

### 玩家管理
- ✅ 学号注册，密码SHA256加密
- ✅ 登录界面密码遮蔽
- ✅ 玩家数据JSON持久化
- ✅ 修改密码功能

### 武器管理
- ✅ 从 Arms.xlsx 读取武器数据
- ✅ 武器增删改查
- ✅ 多维度排序（伤害、射速、弹匣、射程、DPS）
- ✅ 按类型筛选
- ✅ 武器搜索
- ✅ 弹药统计与换弹
- ✅ 武器统计信息

### 音频识别
- ✅ 音频特征提取（duration, RMS, ZCR, spectral特征）
- ✅ 深度神经网络（DNN）分类器 - **78.79%准确率**
- ✅ ResNet架构 - 72.73%准确率
- ✅ 传统ML（KNN、SVM、随机森林）
- ✅ 混淆矩阵可视化
- ✅ 训练曲线可视化
- ✅ 模型保存与加载
- ✅ 实时音频识别（GUI集成）

### GUI界面
- ✅ 完整的图形用户界面
- ✅ 登录/注册窗口
- ✅ 武器库浏览（支持搜索、排序）
- ✅ 背包管理
- ✅ 音频识别窗口
- ✅ 三种识别模式（文件/手动/批量）

## 设计模式

### 面向对象设计
- `Weapon` 类 - 武器数据封装
- `Player` 类 - 玩家数据和武器管理
- `DataManager` 类 - 数据持久化
- `AuthManager` 类 - 认证管理
- `WeaponService` 类 - 武器业务逻辑
- `AudioFeatureExtractor` 类 - 特征提取
- `ModelLoader` 类 - 模型加载器（重构）
- `AudioRecognizer` 类 - 音频识别器（重构）

### 异常处理
- 文件读写异常捕获
- 数据验证（学号、密码格式）
- 用户输入异常处理
- 模型加载异常处理

### 数据持久化
- JSON格式存储玩家和武器数据
- CSV格式导入导出
- Excel读取武器初始数据
- PyTorch模型文件（.pth）
- joblib序列化（scaler, encoder）

## 注意事项

1. 首次运行需要 `Arms.xlsx` 文件来加载武器数据
2. 音频识别需要已训练的模型文件（位于 `models/` 目录）
3. 如果模型不存在，先运行 `python train/train_model_deep.py` 训练
4. 所有运行数据文件会自动保存到 `data/` 目录（以 `data/players.json`、`data/weapons.json` 为准）
5. 根目录下如果存在 `players.json` / `weapons.json`，属于旧版本遗留文件：
   - 当前程序不会直接使用它们
   - 若 `data/` 下文件缺失，程序会尝试自动迁移（复制）到 `data/`
   - 若两处文件都存在且内容不同，程序只提示，不自动覆盖（避免误覆盖数据）
5. 演示音频文件位于 `demo/` 目录

## 开发团队

课程设计项目 - 2025

## 许可证

本项目仅用于学习和教育目的
