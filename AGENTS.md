# LottoProphet — Agent Guide

## 项目概览

彩票预测桌面应用（PyQt5 GUI），支持双色球(SSQ)和大乐透(DLT)。使用 PyTorch LSTM-CRF 及多种 sklearn ML 模型。

## 入口点

- `main.py` — 统一入口，子命令模式
- `lottery_predictor_app_new.py` — GUI 主窗口（~1200行），能被 `main.py app` 或直接 `python lottery_predictor_app_new.py` 启动

## 命令行速查

```bash
python main.py app                          # GUI 启动
python main.py fetch ssq                    # 获取双色球数据
python main.py fetch dlt                    # 获取大乐透数据
python main.py train ssq --model lightgbm   # 训练 ML 模型
python main.py predict dlt --model ensemble # 命令行预测
python train_models.py --type all --gpu --epochs 200     # 独立训练 LSTM-CRF
python fetch_and_train.py --type all --gpu --skip-fetch  # 仅训练（跳过拉数据）
```

`--model` 可选：`random_forest`, `xgboost`, `gbdt`, `lightgbm`, `catboost`, `ensemble`, `expected_value`

## 架构要点

- 所有非 LSTM-CRF 的模型（含 expected_value）走 `LotteryMLModels` 类，训练时 `subprocess` 调对应脚本
- 线程模型：`TrainModelThread` 和 `UpdateDataThread` 通过 `pyqtSignal` + `LogEmitter` 回传日志
- 数据文件：`scripts/{dlt,ssq}/{dlt,ssq}_history.csv`
- 模型存于 `model/{dlt,ssq}/`（PyTorch .pth + scaler.pkl）或 `model/{dlt,ssq}/{model_type}/`（ML .pkl + scaler）
- `model_utils.py:12-29` `name_path` 字典是彩票类型配置的唯一权威来源
- GUI 在 `ui_components.py` 中以函数形式组织（`create_main_tab`, `create_analysis_tab` 等）

## 号码规则（必须遵守）

| 彩票 | 红球范围 | 红球个数 | 蓝球范围 | 蓝球个数 |
|------|---------|---------|---------|---------|
| SSQ  | 1-33    | 6       | 1-16    | 1       |
| DLT  | 1-35    | 5       | 1-12    | 2       |

## 依赖坑

- PyPI 上的 `torchcrf` 实际上是 `TorchCRF`（大写），导入语句必须用 `from TorchCRF import CRF`，与 `model.py` 一致
- `pip install torchcrf` 可能失败，备选方案：`git clone https://github.com/kmkurn/pytorch-crf.git && cd pytorch-crf && python setup.py install`
- `lightgbm` / `catboost` 可选缺失，代码内有 `try/except ImportError` 逻辑
- 日志使用 `loguru` + stdlib `logging` 双轨道

## 代码风格

- 所有 docstring 和注释为中文
- 变量名使用英文
- 主程序约 25 个 `.py` 文件，无子包，所有 import 为扁平相对路径

## 测试

本项目无测试文件。修改后手动运行验证。
