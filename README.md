# LottoProphet

一个使用深度学习模型进行彩票号码预测的应用程序。本项目支持两种主要的彩票类型：**双色球 (SSQ)** 和 **大乐透 (DLT)**，并使用先进的机器学习技术（如条件随机场 CRF）进行序列建模。

[English README](./README_EN.md) | 中文说明

## 20241210更新内容  [查看文档](./docs/train_ssq_update.md)
## 20241210更新内容  [查看文档](./docs/train_ssq_update.md)
## 20250114更新内容  [查看文档](./update/README.md) 主要修复内容：每次多个的结果都一样
## 20250322更新内容  [查看文档](./update/20250322Update.md) 主要更新内容：UI优化、功能增强和性能优化
## 20250323更新内容  [查看文档](./update/20250323Update.md) 代码重构，模型增加，修复已知问题
## 20250325更新内容  [查看文档](./update/20250325Update.md) 期望值模型
## 20250413更新内容  [查看文档](./update/blue_ball_fix_update.md) 修复已知问题
## 多个小伙伴遇到安装 torchcrf 失败，解决方案：从 GitHub 克隆并手动安装
1. **克隆仓库**：
   ```bash
   git clone https://github.com/kmkurn/pytorch-crf.git
   cd pytorch-crf
   python setup.py install
   ```
---

## 目录结构

```plaintext
LottoProphet/
├── main.py                         # 主程序入口点，支持命令行参数启动不同功能
├── lottery_predictor_app_new.py    # 主程序新版本，实现了完整的GUI界面和功能
├── ui_components.py                # UI组件模块，包含tab页面创建和布局相关功能
├── thread_utils.py                 # 模型训练和数据更新的线程工具，支持后台运行
├── model_utils.py                  # 模型加载和预测工具，管理模型资源
├── prediction_utils.py             # 号码生成和预测工具，处理预测结果
├── data_processing.py              # 数据处理模块，统计分析和特征工程
├── ml_models.py                    # 机器学习模型实现，包含多种预测模型和集成学习
├── fetch_and_train.py              # 数据获取和模型训练脚本，自动化训练流程
├── train_models.py                 # 模型训练脚本，用于命令行单独训练模型
├── theme_manager.py                # UI主题管理器，支持深色和浅色主题切换
├── model.py                        # 模型定义文件，包含神经网络结构
├── scripts/
│   ├── data_analysis.py            # 数据分析工具，提供数据可视化和统计分析功能
│   ├── advanced_statistics.py      # 高级统计分析，包含复杂统计指标计算和图表生成
│   ├── dlt/
│   │   ├── train_dlt_model.py      # 大乐透模型训练脚本，使用LSTM-CRF序列建模
│   │   ├── fetch_dlt_data.py       # 大乐透数据获取脚本，爬取最新历史数据
│   │   ├── dlt_history.csv         # 大乐透历史数据，包含开奖号码和日期信息
│   │   └── training_loss.png       # 大乐透训练损失可视化，展示模型训练过程
│   └── ssq/
│       ├── train_ssq_model.py      # 双色球模型训练脚本，使用LSTM-CRF序列建模
│       ├── fetch_ssq_data.py       # 双色球数据获取脚本，爬取最新历史数据
│       └── ssq_history.csv         # 双色球历史数据，包含开奖号码和日期信息
├── model/
│   ├── README.md                   # 模型文档，解释模型架构和使用方法
│   ├── MODEL_UPDATE_SUMMARY.md     # 模型更新摘要，记录模型改进历史
│   ├── generate_placeholder_models.py # 生成占位模型脚本，用于测试和初始化
│   ├── test_model_loading.py       # 测试模型加载脚本，验证模型文件完整性
│   ├── dlt/                        # 大乐透模型目录，用于存储训练好的模型
│   └── ssq/                        # 双色球模型目录，用于存储训练好的模型
├── update/
│   ├── README.md                   # 更新文档，综合介绍各版本更新内容
│   ├── 20250217Update.md           # 2025年2月17日更新说明
│   ├── 20250219Update.md           # 2025年2月19日更新说明
│   ├── 20250322Update.md           # 2025年3月22日更新，新增UI和功能优化
│   ├── 20250323Update.md           # 2025年3月23日更新，新增UI和功能优化
│   └── 20250325Update.md           # 2025年3月25日更新，新增期望值模型
└── requirements.txt                # 项目依赖列表，包含所需Python库
```

只需要运行main.py（ssq_history.csv，ssq_model.pth，scaler_X.pkl等都会通过调用自动生成 ）

## 软件截图
> **注意**: 最新版本(2025年3月22日更新)已优化UI界面，使用QTabWidget分离预测和分析功能。如下
![image](https://github.com/user-attachments/assets/187a8e17-fb40-4cc1-9892-36fa1401d386)
![image](https://github.com/user-attachments/assets/faec6855-2599-46c6-b78f-292748130c12)

# 彩票预测应用程序整体流程

彩票预测应用程序的主要步骤，包括数据获取、模型训练和用户界面交互。应用程序支持两种彩票类型：**双色球（ssq）**和**大乐透（dlt）**。

## 1. 组件概述

### 数据获取脚本

- **双色球数据获取 (`fetch_ssq_data.py`)**
    - 爬取双色球历史数据。
    - 保存数据为 `ssq_history.csv`。

- **大乐透数据获取 (`fetch_dlt_data.py`)**
    - 爬取大乐透历史数据。
    - 保存数据为 `dlt_history.csv`。

### 模型训练脚本

- **双色球模型训练 (`train_ssq_model.py`)**
    - 检查并下载 `ssq_history.csv`（若不存在）。
    - 预处理数据并训练 LSTM-CRF 模型。
    - 保存训练好的模型和特征缩放器。

- **大乐透模型训练 (`train_dlt_model.py`)**
    - 检查并下载 `dlt_history.csv`（若不存在）。
    - 预处理数据并训练 LSTM-CRF 模型。
    - 保存训练好的模型和特征缩放器。

### 主应用程序

- **主界面 (`lottery_predictor_app_new.py/main.py`)**
    - 提供用户界面，选择彩票类型（双色球或大乐透）。
    - 输入特征值并选择生成的预测数量。
    - 按钮操作：
        - **训练模型**：启动训练线程，调用相应的训练脚本。
        - **生成预测**：加载模型，处理输入特征，生成并显示预测号码。
        - **数据分析**：使用高级统计方法分析历史数据。
    - 日志显示框：实时展示训练和预测过程中的日志信息。
    - 支持深色/浅色主题切换。

## 2. 整体流程

### 训练模型流程

1. **用户操作**：
    - 在主界面选择彩票类型（双色球或大乐透）。
    - 点击"训练模型"按钮。

2. **启动训练线程**：
    - `TrainModelThread` 启动，调用对应的训练脚本 (`train_ssq_model.py` 或 `train_dlt_model.py`)。

3. **数据检查与获取**：
    - 训练脚本检查历史数据文件 (`ssq_history.csv` 或 `dlt_history.csv`) 是否存在。
    - 若不存在，调用对应的数据获取脚本下载数据。

4. **数据预处理与模型训练**：
    - 加载并预处理数据（特征缩放、划分训练集和验证集）。
    - 初始化并训练 LSTM-CRF 模型或其他机器学习模型。
    - 实施早停机制，保存最佳模型权重和缩放器。
    - 支持GPU加速训练（如果可用）。

5. **日志更新与完成**：
    - 训练过程中的日志通过信号传递到主界面的日志框。
    - 训练完成后，恢复界面操作并提示用户。

### 生成预测流程

1. **用户操作**：
    - 在主界面选择彩票类型（双色球或大乐透）。
    - 输入特征值（如和值、奇数个数等）或使用默认值。
    - 选择生成的预测数量和预测模型（LSTM-CRF或其他ML模型）。
    - 点击"生成预测"按钮。

2. **特征处理与模型加载**：
    - 获取并缩放用户输入的特征值。
    - 添加随机噪声以增强多样性（基于正态分布的随机性）。
    - 加载对应的彩票模型和特征缩放器。

3. **号码预测**：
    - 将处理后的特征输入模型，生成红球和蓝球的预测类别。
    - 通过温度采样和Top-K参数增强预测多样性。
    - 通过 CRF 解码获取具体的号码预测。

4. **结果处理与展示**：
    - 确保预测号码在有效范围内且红球号码唯一。
    - 在主界面显示预测结果。
    - 日志框实时记录预测过程中的详细信息。
    - 支持保存和导出预测结果。

### 数据分析流程

1. **用户操作**：
    - 切换到"数据分析"选项卡。
    - 选择分析类型和参数。

2. **数据加载与处理**：
    - 加载历史数据并进行质量检查。
    - 使用缓存机制（memoization）优化频繁数据分析操作的响应速度。

3. **统计分析与可视化**：
    - 频率分析：展示每个号码的历史出现频率。
    - 热冷号分析：识别热门和冷门号码。
    - 间隔统计：分析号码出现间隔规律。
    - 模式分析：发现重复模式和序列。
    - 趋势分析：分析长期趋势和变化。
    - 高级统计：进行复杂的统计验证和假设检验。

4. **结果呈现**：
    - 生成高质量的统计图表。
    - 提供数据解释和建议。
    - 支持图表保存和导出。

## 3. 错误处理和日志记录

- **数据获取脚本**：
    - 捕获网络请求异常和解析错误，记录详细日志。
    - 支持自动重试和恢复数据获取。

- **训练脚本**：
    - 检查数据和脚本存在性，捕获训练过程中的异常。
    - 保存训练检查点，支持从断点处恢复训练。

- **主应用程序**：
    - 训练和预测过程中捕获异常，通过日志框显示错误信息。
    - 提供详细的错误诊断信息以帮助用户解决问题。

## 4. 环境要求

- Python 3.9 或更高版本
- PyTorch (可选择GPU版本) - 现支持GPU加速训练和预测
- torchcrf
- PyQt5
- pandas
- numpy
- scikit-learn
- matplotlib (数据可视化)
- seaborn (增强统计图表)
- xgboost (高级机器学习模型)
- lightgbm (可选)
- catboost (可选)
- joblib (模型序列化)

## 5. 安装步骤

1. **克隆仓库**：
   ```bash
   git clone git@github.com:zhaoyangpp/LottoProphet.git
   cd LottoProphet
   ```

2. **安装依赖**:
   ```bash
   # 推荐使用虚拟环境
   pip install -r requirements.txt 
   ```

3. **运行主程序**: 
   ```bash
   python main.py app
   # 或直接从命令行运行特定功能
   python main.py fetch ssq  # 获取双色球数据
   python main.py train dlt  # 训练大乐透模型
   python main.py predict ssq --model lightgbm  # 使用LightGBM模型预测双色球
   ```
