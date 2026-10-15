# busuan 模型优化系统 v2 设计方案

> **版本**: v2.0  
> **日期**: 2026-10-15  
> **状态**: 设计审核中  
> **作者**: AI 辅助开发  
> **审核人**: 用户  

---

## 一、背景与动机

### 1.1 当前系统现状

busuan 彩票预测应用已有以下优化模块（v1）:

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 模型校准 | `model_calibration.py` | Isotonic Regression 校准概率 | 可运行但缺数据 |
| 动态权重 | `model_weights.py` | EWMA 超额命中率加权 | 可运行但缺数据 |
| 特征分析 | `feature_analysis.py` | 重要性 + 漂移检测 | 可运行 |
| 反馈重训 | `retrain_models_with_feedback.py` | 近期窗口重训 | 可运行 |
| 对抗安全 | `json_utils.py` / `input_validator.py` / `model_integrity.py` | 原子写入/输入校验/模型防损坏 | 已完成测试 |

### 1.2 核心痛点

| # | 痛点 | 影响 |
|---|------|------|
| P1 | **数据荒漠**：仅1条预测记录，0条核验，校准/权重无法运行 | 模块形同虚设 |
| P2 | **纯手动触发**：用户须手动点每个按钮，无自动监控 | 体验差、易遗忘 |
| P3 | **无智能决策**：漂移了怎么办？哪个模型该重训？没有建议 | 用户无所适从 |
| P4 | **字段不一致**：`prediction_history.json` 用 `target_period`，部分代码查 `has_drawn` | 校准数据收集失败 |
| P5 | **无闭环验证**：优化后效果如何？没有前后对比 | 无法量化收益 |

### 1.3 用户诉求

1. **半自动化**：低风险操作自动执行，高风险操作一键确认
2. **大模型辅助**：用 LLM 分析系统健康度，生成可解释的建议
3. **可追溯审核**：完整设计方案 + 执行日志，方便事后追溯

---

## 二、设计目标

| 目标 | 衡量标准 |
|------|----------|
| 半自动化 | 低风险操作（权重更新/校准）自动执行率 100% |
| 智能决策 | LLM 生成诊断报告 + 优先级建议，用户可读懂 |
| 可追溯 | 每次优化操作有完整日志（时间/原因/参数/结果） |
| 零回归 | 不破坏现有 475 条预测记录和现有 UI 功能 |
| 可量化 | 优化前后有命中率对比报告 |

---

## 三、总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                        UI 层 (ui_components.py)              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ 健康度面板  │ │ LLM建议面板 │ │ 待确认操作  │ │ 优化日志   │ │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬─────┘ │
└────────┼─────────────┼──────────────┼───────────────┼───────┘
         │             │              │               │
┌────────▼─────────────▼──────────────▼───────────────▼───────┐
│                    调度层 (lottery_predictor_app_new.py)       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │AutoOptThread │  │ on_diagnose  │  │ on_execute_plan    │  │
│  │(每日自动)     │  │ (手动诊断)    │  │ (一键执行)         │  │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘  │
└─────────┼─────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                     逻辑层 (新增3模块)                         │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ optimization_     │  │ llm_optimizer.py │  │ optimization│ │
│  │ monitor.py        │  │                  │  │ _executor.py│ │
│  │                  │  │ - analyze_report │  │            │  │
│  │ - check_data     │  │ - generate_plan  │  │- auto_exec │  │
│  │ - check_perf     │  │ - explain        │  │- request   │  │
│  │ - check_drift    │  │   recommendation │  │  _confirm  │  │
│  │ - check_calib    │  │                  │  │- log_action │  │
│  │ - gen_report     │  │ (调用 OpenAI/    │  │            │  │
│  │                  │  │  Claude API)     │  │            │  │
│  └──────────────────┘  └──────────────────┘  └────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │             optimization_log.py (优化日志)                │ │
│  │  - log_optimization_action(action, params, result)     │ │
│  │  - get_optimization_history(limit)                      │ │
│  │  - generate_diff_report(before, after)                  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                     数据层 (现有模块复用)                      │
│  prediction_records.py  model_weights.py  model_calibration.py│
│  feature_analysis.py    retrain_models_with_feedback.py        │
│  prediction_history.json  model_weights.json  scripts/*.csv   │
└──────────────────────────────────────────────────────────────┘
```

---

## 四、模块详细设计

### 4.1 智能监控引擎 `optimization_monitor.py`

**职责**: 定期扫描系统健康度，生成结构化报告

#### 4.1.1 接口定义

```python
def generate_health_report(lottery_type: str = 'all') -> dict:
    """
    生成系统健康报告
    
    Returns:
        {
            "timestamp": "2026-10-15T10:00:00",
            "lottery_type": "dlt",
            "health_score": 6.5,           # 0-10 分
            "checks": {
                "data_volume": {
                    "status": "warning",    # ok / warning / critical
                    "value": 15,            # 核验记录数
                    "threshold": 30,        # 建议最低
                    "message": "核验记录不足30条，校准精度受限"
                },
                "model_performance": {
                    "status": "ok",
                    "details": {
                        "xgboost": {"hit_rate": 0.32, "trend": "stable"},
                        "lightgbm": {"hit_rate": 0.28, "trend": "declining"}
                    }
                },
                "feature_drift": {
                    "status": "critical",
                    "drifted_features": ["red_miss_5", "blue_hot_10"],
                    "max_kl": 0.45,
                    "message": "2个特征显著漂移，建议重训"
                },
                "calibration_quality": {
                    "status": "ok",
                    "red_brier": 0.21,
                    "blue_brier": 0.18,
                    "message": "校准误差在可接受范围"
                },
                "data_freshness": {
                    "status": "ok",
                    "last_draw": "2026305",
                    "days_since": 2
                }
            },
            "needs_action": True,
            "priority": "medium"
        }
    """
```

#### 4.1.2 检查项

| 检查项 | 数据来源 | 阈值 | 状态判定 |
|--------|----------|------|----------|
| 数据量 | `prediction_records.load_records()` | ≥30 ok, ≥10 warn, <10 crit | 核验记录数 |
| 模型表现 | `model_weights.get_performance_report()` | 命中率≥基准 ok, 连续3期下降 warn | 各模型 EWMA |
| 特征漂移 | `feature_analysis.detect_feature_drift()` | KL<0.1 ok, <0.3 warn, ≥0.3 crit | 漂移特征数+最大KL |
| 校准质量 | `model_calibration.get_calibration_info()` | Brier<0.25 ok, <0.35 warn | 红蓝球 Brier score |
| 数据新鲜度 | CSV 最新期号 vs 今天 | ≤3天 ok, ≤7 warn, >7 crit | 距今天数 |

#### 4.1.3 健康度评分公式

```
health_score = (
    data_volume_score * 0.25 +
    model_performance_score * 0.30 +
    feature_drift_score * 0.20 +
    calibration_quality_score * 0.15 +
    data_freshness_score * 0.10
)

# 各项 0-10: ok=10, warning=5, critical=0
```

---

### 4.2 LLM 决策助手 `llm_optimizer.py`

**职责**: 调用大模型分析健康报告，生成可解释的优化建议

#### 4.2.1 接口定义

```python
def analyze_health_report(report: dict, lottery_type: str) -> dict:
    """
    调用 LLM 分析健康报告，生成优化建议
    
    Returns:
        {
            "diagnosis": "系统健康度6.5/10。数据量偏少...",
            "health_score": 6.5,
            "issues": [
                {"name": "数据不足", "severity": "medium", "detail": "..."},
                {"name": "特征漂移", "severity": "high", "detail": "..."}
            ],
            "recommendations": [
                {
                    "action": "update_weights",       # 自动执行
                    "priority": "high",
                    "auto_executable": True,
                    "reason": "lightgbm 连续3期表现下降，重算权重可降低其影响",
                    "params": {"lottery_type": "dlt"}
                },
                {
                    "action": "retrain_model",        # 需用户确认
                    "priority": "medium",
                    "auto_executable": False,
                    "reason": "2个特征漂移，建议用近期50期数据重训",
                    "params": {
                        "lottery_type": "dlt",
                        "model_types": ["xgboost", "lightgbm"],
                        "recent_periods": 50
                    }
                }
            ],
            "explanation": "当前 DLT 预测系统健康度中等。主要问题是..."
        }
    """
```

#### 4.2.2 LLM 调用策略

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Provider | OpenAI 兼容 API | 支持 OpenAI / DeepSeek / 本地 Ollama |
| Model | gpt-4o-mini / deepseek-chat | 可配置 |
| Temperature | 0.3 | 低温度保证输出稳定 |
| Max tokens | 2000 | 足够输出结构化 JSON |
| Timeout | 30s | 避免长时间等待 |
| Fallback | 规则引擎 | API 不可用时用规则匹配 |

#### 4.2.3 Prompt 模板

```
你是彩票预测系统的优化专家。请分析以下系统健康报告，给出诊断和建议。

## 系统健康报告
{report_json}

## 输出要求
请输出 JSON 格式（不要 markdown 代码块），包含以下字段：
- diagnosis: 系统健康度评估（一句话）
- health_score: 0-10 分
- issues: 问题列表 [{name, severity, detail}]
- recommendations: 建议列表 [{
    action: "update_weights"|"calibrate"|"retrain_model"|"update_data"|"noop",
    priority: "high"|"medium"|"low",
    auto_executable: true|false,
    reason: "为什么",
    params: {具体参数}
  }]
- explanation: 给用户的通俗解释（2-3句话）

## 判定规则
- auto_executable=true: update_weights, calibrate（低风险，可自动执行）
- auto_executable=false: retrain_model, update_data（高风险，需用户确认）
- 如果健康度≥8且无 critical 问题，建议 noop
```

#### 4.2.4 降级方案（LLM 不可用时）

```python
def _rule_based_analysis(report: dict) -> dict:
    """规则引擎降级：无 LLM 时用规则匹配生成建议"""
    recommendations = []
    checks = report.get('checks', {})
    
    if checks.get('data_volume', {}).get('status') == 'critical':
        recommendations.append({
            'action': 'noop', 'priority': 'high',
            'auto_executable': False,
            'reason': '数据不足，建议先积累预测+核验记录',
            'params': {}
        })
    
    if checks.get('feature_drift', {}).get('status') == 'critical':
        recommendations.append({
            'action': 'retrain_model', 'priority': 'high',
            'auto_executable': False,
            'reason': '特征漂移显著，建议近期窗口重训',
            'params': {'recent_periods': 50}
        })
    
    if checks.get('model_performance', {}).get('status') == 'warning':
        recommendations.append({
            'action': 'update_weights', 'priority': 'medium',
            'auto_executable': True,
            'reason': '部分模型表现下降，重算权重',
            'params': {}
        })
    
    # ... 更多规则
    return {'recommendations': recommendations, ...}
```

---

### 4.3 自动化执行器 `optimization_executor.py`

**职责**: 执行优化计划，低风险自动、高风险请求确认

#### 4.3.1 接口定义

```python
def execute_plan(plan: dict, auto_confirm: bool = False) -> dict:
    """
    执行优化计划
    
    Args:
        plan: LLM 生成的建议列表
        auto_confirm: True=自动确认高风险操作（用于定时任务）
    
    Returns:
        {
            "executed": ["update_weights(dlt)"],
            "skipped": ["retrain_model(dlt) - 需用户确认"],
            "results": {
                "update_weights(dlt)": {"success": True, "detail": "..."}
            },
            "log_id": "opt_20261015_100000"
        }
    """
```

#### 4.3.2 操作分级

| 操作 | 风险 | 自动执行 | 说明 |
|------|------|----------|------|
| `update_weights` | 低 | ✅ | 重算动态权重（毫秒级） |
| `calibrate` | 低 | ✅ | 训练校准模型（秒级） |
| `retrain_model` | 高 | ❌ | 重训 ML 模型（分钟级） |
| `update_data` | 高 | ❌ | 更新开奖数据（网络请求） |
| `noop` | 无 | ✅ | 无需操作 |

#### 4.3.3 执行流程

```
for rec in plan['recommendations']:
    action = rec['action']
    
    if action == 'noop':
        log("无需操作")
        continue
    
    if rec['auto_executable']:
        # 低风险：直接执行
        result = _execute_action(action, rec['params'])
        log_optimization_action(action, rec, result)
    else:
        # 高风险：请求确认
        if auto_confirm:
            result = _execute_action(action, rec['params'])
        else:
            # 加入待确认队列，等用户在 UI 上确认
            pending_actions.append(rec)
```

---

### 4.4 优化日志 `optimization_log.py`

**职责**: 记录每次优化操作，支持追溯和效果对比

#### 4.4.1 日志格式

```json
// optimization_log.json
{
  "logs": [
    {
      "id": "opt_20261015_100000",
      "timestamp": "2026-10-15T10:00:00",
      "trigger": "auto",           // auto / manual / llm
      "lottery_type": "dlt",
      "action": "update_weights",
      "reason": "lightgbm 连续3期表现下降",
      "params": {"lottery_type": "dlt"},
      "result": {
        "success": true,
        "before": {"lightgbm_weight": 0.25},
        "after": {"lightgbm_weight": 0.18},
        "detail": "权重已从0.25降至0.18"
      },
      "health_before": 6.5,
      "health_after": 7.0
    }
  ]
}
```

#### 4.4.2 接口

```python
def log_optimization_action(action: str, recommendation: dict, 
                            result: dict, health_before: float = None) -> str:
    """记录一条优化操作日志，返回 log_id"""

def get_optimization_history(limit: int = 20) -> list:
    """获取最近的优化操作历史"""

def generate_effect_report(lottery_type: str, days: int = 30) -> dict:
    """生成优化效果报告（对比优化前后命中率变化）"""
```

---

### 4.5 UI 改造 `ui_components.py`

**职责**: 在现有"模型优化"标签页中集成新功能

#### 4.5.1 新增 UI 控件

在 `create_optimization_tab()` 中新增一个顶部区域：

```
┌─────────────────────────────────────────────────────────────────┐
│  🏥 系统健康度                                                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │ 数据量    │ 模型表现   │ 特征漂移  │ 校准质量  │ 数据新鲜度    │  │
│  │ 15/30   │ 稳定      │ 2个漂移   │ 良好     │ 2天前        │  │
│  │ ⚠️       │ ✅        │ 🔴        │ ✅       │ ✅           │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
│  健康度: 6.5/10    [智能诊断]  [一键优化]                        │
│                                                                  │
│  📋 优化建议                                              [刷新]  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 💡 系统健康度6.5/10。数据量偏少，建议继续积累。            │ │
│  │                                                             │ │
│  │ 📌 建议操作:                                               │ │
│  │  [✅] 更新权重 (自动) - lightgbm表现下降，重算权重          │ │
│  │  [☐ ] 重训模型 (需确认) - 2个特征漂移，建议近期窗口重训    │ │
│  │  [☐ ] 更新数据 (需确认) - 距上次更新已7天                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  [执行选中操作]                                                   │
│                                                                  │
│  📊 优化日志                               [查看历史] [效果报告]  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 10:00 [auto] 更新权重 DLT - lightgbm 0.25→0.18  ✅         │ │
│  │ 09:30 [llm]  智能诊断完成 - 健康度6.5                       │ │
│  │ ...                                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

（下方保留原有的：模型校准区 / 特征分析区 / 反馈重训区 / 表现榜区）
```

#### 4.5.2 交互流程

| 步骤 | 触发 | 动作 |
|------|------|------|
| 1 | 用户切换到"模型优化"标签 | 自动刷新健康度面板 |
| 2 | 用户点击"智能诊断" | 调用 monitor → llm → 显示建议 |
| 3 | 系统自动执行 `auto_executable=true` 的建议 | 日志记录 |
| 4 | 用户勾选高风险操作 + 点"执行选中" | 调用 executor 执行 |
| 5 | 执行完成 | 刷新健康度面板 + 日志 |

#### 4.5.3 后台自动监控

```python
class AutoOptimizationThread(QThread):
    """后台自动优化线程（每24小时触发一次）"""
    diagnosis_signal = pyqtSignal(dict)  # 发送诊断结果到UI
    
    def __init__(self, parent):
        super().__init__(parent)
        self._running = True
    
    def run(self):
        while self._running:
            time.sleep(86400)  # 24小时
            for lt in ['ssq', 'dlt']:
                report = optimization_monitor.generate_health_report(lt)
                if report.get('needs_action'):
                    plan = llm_optimizer.analyze_health_report(report, lt)
                    # 自动执行低风险操作
                    result = optimization_executor.execute_plan(plan)
                    # 通知UI
                    self.diagnosis_signal.emit({
                        'lottery_type': lt,
                        'report': report,
                        'plan': plan,
                        'result': result
                    })
    
    def stop(self):
        self._running = False
```

---

### 4.6 字段一致性修复

#### 4.6.1 问题

`prediction_history.json` 中的记录字段：

```json
{
  "target_period": null,      // 核验目标期号（null=未核验）
  "red_hits": null,           // 红球命中数
  "blue_hits": null,          // 蓝球命中数
  "prize_name": null,         // 奖级
  "prize_amount": null        // 奖金
}
```

但 `model_calibration.py` 中 `collect_calibration_data()` 检查的字段是 `has_drawn`（不存在）。

#### 4.6.2 修复方案

统一使用 `target_period is not None` 作为"已核验"判定标准：

```python
# model_calibration.py - collect_calibration_data() 中
# 旧: if record.get('has_drawn'):
# 新: if record.get('target_period') is not None:
```

同时排查 `model_weights.py`、`prediction_records.py` 中的同类问题。

---

## 五、数据流

### 5.1 预测 → 核验 → 优化闭环

```
用户预测 → 保存记录(prediction_history.json)
                ↓
开奖更新 → CSV更新
                ↓
核验 → 更新记录(target_period, red_hits, blue_hits, prize_*)
                ↓
自动触发权重更新 → model_weights.json更新
                ↓
定期监控 → health_report
                ↓
LLM分析 → recommendations
                ↓
执行 → 自动(权重/校准) + 手动确认(重训)
                ↓
日志记录 → optimization_log.json
                ↓
效果对比 → before/after 命中率
```

### 5.2 冷启动策略

由于当前仅1条记录，需要冷启动：

| 阶段 | 记录数 | 启用功能 |
|------|--------|----------|
| 冷启动 | 0-9 | 仅监控+提示"积累数据中" |
| 预热 | 10-29 | 权重计算(中性) + 特征分析 |
| 正常 | 30+ | 校准 + 权重 + LLM诊断 + 半自动执行 |

---

## 六、配置项

```python
# config/optimization_config.py

OPTIMIZATION_CONFIG = {
    # 监控阈值
    "min_verified_records": 30,       # 校准最低记录数
    "drift_kl_threshold": 0.3,        # 漂移KL阈值
    "performance_decline_periods": 3,  # 连续下降期数
    "data_stale_days": 7,             # 数据过期天数
    
    # 自动化
    "auto_check_interval_hours": 24,  # 自动检查间隔
    "auto_executable_actions": ["update_weights", "calibrate"],
    "require_confirm_actions": ["retrain_model", "update_data"],
    
    # LLM
    "llm_provider": "openai",         # openai / deepseek / ollama
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.3,
    "llm_max_tokens": 2000,
    "llm_timeout_seconds": 30,
    "llm_fallback_to_rules": True,    # API不可用时降级
    
    # 日志
    "max_log_entries": 500,           # 最大日志条数
    "log_file": "optimization_log.json"
}
```

---

## 七、测试计划

### 7.1 单元测试

| 测试文件 | 覆盖模块 | 测试项 |
|----------|----------|--------|
| `test_optimization_monitor.py` | 监控引擎 | 5项检查各状态判定 + 健康度评分 |
| `test_llm_optimizer.py` | LLM决策 | API调用(mock) + 降级规则 + JSON解析 |
| `test_optimization_executor.py` | 执行器 | 自动执行/手动确认/日志记录 |
| `test_optimization_log.py` | 优化日志 | 写入/读取/效果报告 |

### 7.2 集成测试

| 测试项 | 流程 |
|--------|------|
| 冷启动 | 0条记录 → 诊断提示"积累数据" |
| 预热 | 15条记录 → 权重中性 + 特征分析 |
| 正常 | 50条记录 → 完整闭环 |
| 漂移场景 | 模拟漂移 → LLM建议重训 → 用户确认 → 执行 |
| LLM降级 | API不可用 → 规则引擎接管 |

### 7.3 对抗测试

| 场景 | 预期 |
|------|------|
| 损坏的 prediction_history.json | safe_json_load 恢复 + 继续运行 |
| LLM 返回非法JSON | 降级到规则引擎 |
| 校准数据全为同一值 | IsotonicRegression 处理边界 |
| 漂移检测无特征 | 返回"无显著漂移" |
| 并发写入 optimization_log.json | 原子写入 + 线程安全 |

---

## 八、实施计划

### 阶段1: 基础设施（2-3h）

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1.1 | `optimization_monitor.py` | 5项检查 + 健康报告 |
| 1.2 | `optimization_log.py` | 日志写入/读取/效果报告 |
| 1.3 | `config/optimization_config.py` | 配置项 |
| 1.4 | `test_optimization_monitor.py` | 单元测试 |
| 1.5 | `test_optimization_log.py` | 单元测试 |

### 阶段2: LLM决策（1-2h）

| 步骤 | 文件 | 内容 |
|------|------|------|
| 2.1 | `llm_optimizer.py` | API调用 + Prompt + 降级 |
| 2.2 | `test_llm_optimizer.py` | mock测试 + 降级测试 |

### 阶段3: 执行器（1h）

| 步骤 | 文件 | 内容 |
|------|------|------|
| 3.1 | `optimization_executor.py` | 执行 + 分级 + 日志 |
| 3.2 | `test_optimization_executor.py` | 单元测试 |

### 阶段4: UI集成（2h）

| 步骤 | 文件 | 内容 |
|------|------|------|
| 4.1 | `ui_components.py` | 健康度面板 + 建议面板 + 日志面板 |
| 4.2 | `lottery_predictor_app_new.py` | 信号槽 + AutoOptimizationThread |
| 4.3 | 字段一致性修复 | model_calibration.py 等 |

### 阶段5: 验证（1h）

| 步骤 | 内容 |
|------|------|
| 5.1 | 全套单元测试 |
| 5.2 | 冒烟测试（不破坏现有功能） |
| 5.3 | 对抗测试 |
| 5.4 | 冷启动场景模拟 |

### 阶段6: 文档与提交

| 步骤 | 内容 |
|------|------|
| 6.1 | 更新 `docs/algorithm-model-feedback.md` |
| 6.2 | 更新 `AGENTS.md` |
| 6.3 | Git 提交 |

---

## 九、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| LLM API 不可用 | 规则引擎降级，系统仍可运行 |
| API Key 泄露 | 通过环境变量读取，不硬编码 |
| 优化操作破坏模型 | 高风险操作需用户确认 + 回滚机制 |
| 日志文件膨胀 | 最大500条，超出自动清理旧条目 |
| 并发写入冲突 | 原子写入(json_utils) + 线程锁 |
| 冷启动无数据 | 分阶段启用，数据不足时仅提示 |

---

## 十、追溯与审核

### 10.1 审核清单

- [ ] 设计方案已审核（本文档）
- [ ] 每个模块有单元测试且通过
- [ ] 冒烟测试无回归
- [ ] 对抗测试覆盖异常场景
- [ ] 优化日志可追溯每次操作
- [ ] AGENTS.md 已更新
- [ ] Git 提交包含完整变更

### 10.2 追溯路径

```
优化操作 → optimization_log.json (log_id)
    → 健康报告 (health_before/after)
    → LLM建议 (reason)
    → 执行结果 (success/detail)
    → 效果报告 (命中率对比)
```

---

## 附录 A: 现有模块复用映射

| 现有模块 | v2 中的角色 | 改动 |
|----------|-------------|------|
| `model_calibration.py` | 执行器的 calibrate 操作 | 修复字段引用 |
| `model_weights.py` | 执行器的 update_weights 操作 | 无改动 |
| `feature_analysis.py` | 监控引擎的 drift 检查 | 无改动 |
| `retrain_models_with_feedback.py` | 执行器的 retrain 操作 | 无改动 |
| `prediction_records.py` | 监控引擎的数据源 | 无改动 |
| `json_utils.py` | 所有 JSON 读写 | 无改动 |
| `input_validator.py` | 参数校验 | 无改动 |
| `model_integrity.py` | 模型加载校验 | 无改动 |

---

*文档结束*
