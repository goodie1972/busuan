# 模型反馈优化算法白皮书

> 版本: v2026-10-14 ｜ 适用范围: SSQ(双色球) / DLT(大乐透) 机器学习预测链路
> 本模块不改变"开奖独立随机"的事实——它做的是**自适应模型管理**：用每次预测的真实前向表现，持续优化"该听谁的"。

## 1. 闭环总览

```
生成预测(含模型原始概率) ──→ prediction_history.json 存档
        │                          │
        ▼                          ▼ 开奖后核验(自动/手动)
模型校准(等距回归)  ◄──── 核验记录: 预测概率 vs 实际命中
        │                          │
        ▼                          ▼ EWMA超额命中
期望值模型使用校准概率      动态权重 model_weights.json
                                   │
                                   ▼
             ensemble 加权投票 / 推荐模型 / 近期窗口重训(抗漂移)
```

## 2. 增强特征模式（`ml_models.py`）

在 70 维 0/1 位图特征（feature_window × 红/蓝球位图）之外，新增 **21 个时序衍生特征**（每个号码位 2 列 + 全局统计）：

| 特征族 | 说明 | 防泄漏 |
|---|---|---|
| 遗漏值 omission | 该号距上次出现的间隔期数（红/蓝各一组） | 逐行 shift(1)：只用 t-1 期前信息 |
| 滚动统计 | 窗口内和值/跨度/奇偶比/大小比的滚动均值与标准差 | 同上 |
| 重号率 repeat | 上期号码在本期重现的比例 | 同上 |

- 训练侧：`feature_mode='enhanced'` 时特征维度 `10*(7+21)`=280；暖机丢弃最早 50 期，样本不足自动回退 `original` 并记录日志。
- 持久化：`model_info.json` 记录 `feature_mode` 与 `aug_feature_names`；`load_models` 恢复，旧模型无标记=`original`，行为与升级前完全一致。
- 预测侧：`_aug_row_features` 按**期号对齐**取历史行，无法对齐则硬中止（拒绝用错位历史产生伪概率）。

## 3. 动态权重（`model_weights.py`）

对每个可加权模型（random_forest / xgboost / gbdt / lightgbm / catboost）：

1. **得分**：每注已核验记录 `score = (红命中 - 红基线) + (蓝命中 - 蓝基线)`；
   随机基线 = `k·m/总数`（SSQ 红 6·6/33，DLT 红 5·5/35 等）。
2. **EWMA**：`α=0.05`（半衰期≈14注），每模型最多取最近 200 注；样本 <10 按中性 0 参与（线性衰减 `n/MIN_SAMPLES` 软过渡）。
3. **权重**：`w_i = softmax(EWMA_i / τ)`，`τ=0.15`；归一化写入 `model_weights.json`。
4. **应用**：ensemble 概率平均与投票按同名子模型权重加权（缺名/非法权重回退 1.0 等权）；GUI 表现榜展示 EWMA 超额并推荐权重最高者。

触发时机：核验完成后自动 `update_all_weights()`；模型优化页手动重算；重训练完成后自动刷新。

**诚实声明**：该机制不提升中奖概率，只是把话语权向近期前向表现好的模型倾斜。

## 4. 模型校准（`model_calibration.py`）

- 样本：已核验且含概率存档的记录，`(预测概率, 实际是否命中)` 对。
- 方法：`sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` 按彩种×球种分别拟合，存 `model_calibration/{lt}_{ball}_isotonic.pkl`。
- 用途：期望值模型计算 EV 时用 `calibrate_probability` 替换原始概率，缓解过度自信。
- 门槛：无概率存档的旧记录不可补；建议积累 ≥30 注再训练。

## 5. 反馈重训练（`retrain_models_with_feedback.py`）

- **真实训练路径**：直接实例化 `LotteryMLModels` 并 `train()`（旧版 subprocess 调 LSTM 脚本且传 `--model` 参数，从未真正训练过 ML 模型）。
- **近期窗口重训(抗漂移)**：`--recent N` 仅用最近 N 期重训；下限自动钳制（original 120 期 / enhanced 170 期 = 暖机+最少样本）。
- **特征模式**：`--feature-mode enhanced` 与训练/预测端 feature_mode 一致。
- **LSTM-CRF**：默认跳过（约 2 小时）；`--with-lstm` 显式开启（正确参数，超时 2h）。
- 训练成功后自动 `update_weights`；特征重要性(top-K 增益贡献)与漂移检测(PSI)写入 `feature_analysis/`，报告写 `retrain_reports/`。

## 6. 参数速查

| 参数 | 值 | 位置 |
|---|---|---|
| EWMA_ALPHA | 0.05 | model_weights.py |
| SOFTMAX_TAU | 0.15 | model_weights.py |
| MIN_SAMPLES | 10 | model_weights.py |
| MAX_RECORDS | 200 | model_weights.py |
| 增强暖机期 | 50 | ml_models._augment_features_frame |
| 近窗下限 | 120 / 170(增强) | retrain_models_with_feedback.py |
| 校准最低样本 | 30(建议) | model_calibration.py |

## 7. 数据文件

| 文件 | 内容 |
|---|---|
| `prediction_history.json` | 预测存档(含 red/blue 概率) + 核验结果 |
| `model_weights.json` | 各彩种模型权重/表现统计/推荐模型 |
| `model_calibration/*.pkl` | 等距回归校准模型 |
| `model/{lt}/{mt}/model_info.json` | 模型特征模式与维度标记 |
| `feature_analysis/*.json` | 特征重要性/漂移检测 |
