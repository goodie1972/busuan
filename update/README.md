# LottoProphet


一个使用深度学习模型进行彩票号码预测的应用程序。本项目支持两种主要的彩票类型：**双色球 (SSQ)** 和 **大乐透 (DLT)**，并使用先进的机器学习技术（如条件随机场 CRF）进行序列建模。

## 20241210更新内容  [查看文档](./train_ssq_update.md)
## 20250114更新内容  [查看文档](./update.md) 主要修复内容：每次多个的结果都一样
## 20250322更新内容  [查看文档](./20250322README.md) 主要更新内容：UI优化、功能增强和性能优化





## 多个小伙伴遇到安装 torchcrf 失败，解决方案：从 GitHub 克隆并手动安装
1. **克隆仓库**：
   ```bash
   git clone https://github.com/kmkurn/pytorch-crf.git
   cd pytorch-crf
   python setup.py install

-------------------------

---

## 目录结构

```plaintext
LottoProphet-main/
├── model.py                        # 模型定义，包含彩票预测模型的训练和实现
├── lottery_predictor_app.py        # 主程序，负责用户交互和启动训练、预测
├── scripts/
│   ├── __init__.py                 # 主程序，用于初始化 scripts 目录，标记为 Python 包
│   ├── dlt/
│   │   ├── __init__.py             # 主程序，用于初始化 dlt 目录，标记为 Python 包
│   │   ├── train_dlt_model.py      # 模型训练脚本，训练大乐透彩票预测模型
│   │   ├── fetch_dlt_data.py       # 数据获取脚本，爬取大乐透历史开奖数据
│   │   ├── dlt_history.csv         # 保存大乐透历史开奖数据（默认不存在，需要运行lottery_predictor_app.py进行爬取）
│   │   ├── dlt_model.pth          # 保存训练好的大乐透模型权重（默认不存在，需要运行lottery_predictor_app.py进行模型训练）
│   │   └── scaler_X.pkl           # 保存训练过程中用于特征缩放的 scaler（默认不存在，需要运行lottery_predictor_app.py进行生成特征缩放）
│   └── ssq/
│       ├── __init__.py             # 主程序，用于初始化 ssq 目录，标记为 Python 包
│       ├── train_ssq_model.py      # 模型训练脚本，训练双色球彩票预测模型
│       ├── fetch_ssq_data.py       # 数据获取脚本，爬取双色球历史开奖数据
│       ├── ssq_history.csv         # 保存双色球历史开奖数据（默认不存在，需要运行lottery_predictor_app.py进行爬取）
│       ├── ssq_model.pth          # 保存训练好的双色球模型权重（默认不存在，需要运行lottery_predictor_app.py进行模型训练）
│       └── scaler_X.pkl           # 保存训练过程中用于特征缩放的 scaler（默认不存在，需要运行lottery_predictor_app.py进行生成特征缩放）
└── requirements.txt                # 项目依赖管理文件，列出所有必需的 Python 库

只需要运行lottery_predictor_app.py（ssq_history.csv，ssq_model.pth，scaler_X.pkl等都会通过调用自动生成 ）

```

## 软件截图
![原始界面](https://github.com/user-attachments/assets/e7e8bc1e-3f4f-472b-91ed-303fce6d3b01)

> **注意**: 最新版本(2025年3月22日更新)已优化UI界面，使用QTabWidget分离预测和分析功能。如下
![image](https://github.com/user-attachments/assets/4e986db9-83fd-4650-92c2-b91d4dbbcb41) ![image](https://github.com/user-attachments/assets/b2ce1770-b6fc-4df2-b2c5-f14b008ef42f)




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

- **主界面 (`lottery_predictor_app.py`)**
    - 提供用户界面，选择彩票类型（双色球或大乐透）。
    - 输入特征值并选择生成的预测数量。
    - 按钮操作：
        - **训练模型**：启动训练线程，调用相应的训练脚本。
        - **生成预测**：加载模型，处理输入特征，生成并显示预测号码。
    - 日志显示框：实时展示训练和预测过程中的日志信息。

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
    - 初始化并训练 LSTM-CRF 模型。
    - 实施早停机制，保存最佳模型权重和缩放器。

5. **日志更新与完成**：
    - 训练过程中的日志通过信号传递到主界面的日志框。
    - 训练完成后，恢复界面操作并提示用户。

### 生成预测流程

1. **用户操作**：
    - 在主界面选择彩票类型（双色球或大乐透）。
    - 输入特征值（如和值、奇数个数等）。
    - 选择生成的预测数量。
    - 点击"生成预测"按钮。

2. **特征处理与模型加载**：
    - 获取并缩放用户输入的特征值。
    - 添加随机噪声以增强鲁棒性。
    - 加载对应的彩票模型和特征缩放器。

3. **号码预测**：
    - 将处理后的特征输入模型，生成红球和蓝球的预测类别。
    - 通过 CRF 解码获取具体的号码预测。

4. **结果处理与展示**：
    - 确保预测号码在有效范围内且红球号码唯一。
    - 在主界面显示预测结果。
    - 日志框实时记录预测过程中的详细信息。

## 3. 错误处理与日志记录

- **数据获取脚本**：
    - 捕获网络请求异常和解析错误，记录详细日志。

- **训练脚本**：
    - 检查数据和脚本存在性，捕获训练过程中的异常。

- **主应用程序**：
    - 训练和预测过程中捕获异常，通过日志框显示错误信息。


## 4. 测试与验证

1. **独立测试数据获取脚本**：
    - 运行 `fetch_ssq_data.py` 和 `fetch_dlt_data.py`，验证数据下载和保存。

2. **独立测试训练脚本**：
    - 运行 `train_ssq_model.py` 和 `train_dlt_model.py`，验证模型训练和保存。

3. **集成测试主应用程序**：
    - 启动 `lottery_predictor_app.py`，进行训练和预测操作，观察日志和结果展示。

4. **异常情况测试**：
    - 模拟网络中断、数据文件缺失或训练脚本缺失，验证错误处理机制。

## 5. 总结

彩票预测应用程序通过集成数据获取、模型训练和预测功能，为用户提供便捷的双色球和大乐透号码预测服务。通过模块化设计、详细的错误处理与日志记录。


## 6. 最新更新摘要 (2025年3月22日)

### UI优化
- 优化窗口大小为960x680像素，更符合现代桌面应用标准
- 重新设计主界面布局，使用QTabWidget分离预测和分析功能
- 统一按钮样式和优化字体大小，提升整体界面美观度

### 功能增强
- 数据分析模块新增缓存机制(memoization)，提高频繁数据分析操作的响应速度
- 改进图表生成功能和可视化效果
- 优化了模型推理过程，支持GPU加速
- 增强输入随机性：将随机输入的生成方式从均匀分布改为正态分布，解决了每次多个预测结果相同的问题
- 引入温度采样参数和提高Top-K值，增强预测结果的多样性

### 性能和稳定性
- 清理冗余代码和注释，重构关键功能模块
- 增强了错误处理机制，特别是在数据加载和网络请求过程中
- 优化日志系统，提高日志信息的可读性

详细更新内容请查看[更新文档](./20250322README.md)。



## 功能

- 支持双色球 (SSQ) 和大乐透 (DLT) 的彩票号码预测。
- 使用 LSTM 和 CRF 模型进行训练，实现序列化建模。
- 提供基于 PyQt5 的图形用户界面 (GUI)，便于操作。
- 支持数据自动抓取和实时训练日志显示。
---

## 环境要求

- Python 3.9 或更高版本
- PyTorch(可选择GPU版本) - 现支持GPU加速训练和预测
- torchcrf
- PyQt5
- pandas
- numpy
- scikit-learn
- matplotlib (数据可视化)
- seaborn (增强统计图表)
- functools (用于缓存支持)
- joblib (模型序列化)

---

## 安装步骤

1. **克隆仓库**：
   ```bash
   git clone git@github.com:zhaoyangpp/LottoProphet.git
   cd LottoProphet

2. **安装依赖**:
    ```bash
    推荐使用虚拟环境
    pip install -r requirements.txt 
    ```
3. **运行主程序**: 
   ```bash
   python3 lottery_predictor_app.py
    ```
