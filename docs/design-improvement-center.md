# 预测改进中心 UI 设计方案（V3）

> 对应 todo 6。目标：用简单的三栏布局替代旧的"模型优化页"，专注提高预测概率。

## 设计目标

1. **简单**：用户打开页面一眼看清「系统状态 → 该做什么 → 做了后的效果」
2. **半自动**：低风险操作一键自动执行，高风险操作显示确认按钮
3. **可追溯**：所有建议、执行、效果都记录在案
4. **联动高级统计**：把数据分析/高级统计页的洞察（热冷号、遗漏、趋势）融入建议

## 布局结构（三栏 + 顶部工具栏）

```
┌─────────────────────────────────────────────────────────────┐
│ 顶部工具栏: [彩票类型 Combo] [刷新] [立即诊断]               │
│           [✓ 自动执行低风险] [立即备份]                       │
├──────────────┬──────────────────────────┬───────────────────┤
│ ① 状态栏      │ ② 改进建议区               │ ③ 效果跟踪区       │
│ (系统健康)    │ (可执行建议列表)           │ (历史优化效果)     │
│              │                          │                   │
│ · 核验样本数  │ 每行: [建议类型] [说明]    │ 表格:              │
│ · 校准样本数  │      [说明]               │ 时间/操作/状态/效果 │
│ · 数据新鲜度  │ [执行] [详情] 按钮        │                   │
│ · 当前推荐模型│                          │                   │
│              │ 低风险→绿 [一键执行]        │                   │
│              │ 高风险→橙 [手动确认]        │                   │
├──────────────┴──────────────────────────┴───────────────────┤
│ 底部: 联动统计洞察建议（来自高级统计页）                      │
│       [热号] [冷号] [遗漏] [趋势] 卡片式建议                  │
└─────────────────────────────────────────────────────────────┘
```

## 控件清单与信号连接

### 顶部工具栏
| 控件 | 类型 | 信号 | 操作 |
|------|------|------|------|
| lottery_combo | QComboBox | currentIndexChanged | 切换时刷新三栏 |
| refresh_btn | QPushButton | clicked → refresh_all | 刷新状态+建议 |
| diagnose_btn | QPushButton | clicked → run_rules | 运行规则引擎生成建议 |
| auto_check | QCheckBox | toggled | 控制是否自动执行低风险 |
| backup_btn | QPushButton | clicked → db_backup.backup_database | 手动备份 |

### ① 状态栏（QGroupBox "系统状态"）
| 控件 | 类型 | 显示内容 |
|------|------|---------|
| status_verified_label | QLabel | 已核验记录数 / 待核验数 |
| status_calib_label | QLabel | 校准样本数（带概率已核验）/ 30 |
| status_data_label | QLabel | 最新期 vs CSV 最新期（数据新鲜度） |
| status_reco_label | QLabel | 当前推荐模型 + 权重更新时间 |

### ② 改进建议区（QGroupBox "改进建议（半自动）"）
| 控件 | 类型 | 显示内容 |
|------|------|---------|
| suggestion_table | QTableWidget | 列：类型/说明/风险/操作 |
| | | 行内 [执行] 按钮（弹确认） |
| auto_exec_btn | QPushButton | 一键执行所有低风险建议 |
| | | 高风险行按钮单独显示 |

### ③ 效果跟踪区（QGroupBox "优化效果跟踪"）
| 控件 | 类型 | 显示内容 |
|------|------|---------|
| effect_table | QTableWidget | 列：时间/操作/状态/效果摘要 |
| | | 数据来自 optimization_log 表（DB） |

### 底部统计洞察区
| 控件 | 类型 | 显示内容 |
|------|------|---------|
| insight_text | QTextEdit | 来自 statistics_insights 的卡片文案 |
| | | 热冷号/遗漏/趋势 三项指标 |
| insight_apply_btn | QPushButton | 将洞察指标应用到选号（省略号） |

## 半自动逻辑（与 improvement_rules 配合）

- `run_rules()` 生成 `auto_exec_list` + `manual_confirm_list`
- 若 `auto_check` 勾选且 `auto_exec_list` 非空 → 自动执行并把结果写入 effect_table
- 手动确认区始终展示，用户点「执行」走 `optimization_executor.execute_single_action`

## 现有代码兼容

- 在 `ui_components.py` 新增 `create_improvement_center()` 函数
- 在 `lottery_predictor_app_new.py` 的优化 tab 替换旧 health_panel + 各操作区，
  但保留旧名称的属性引用（health_score_label 等仍指向新控件）避免大改
- 复用 `optimization_executor` 的 `execute_single_action` 与 `optimization_log` 的
  DB 记录

## 实现优先级（本次）

1. 三栏框架 + 信号连接
2. 规则引擎输出 → 建议区表格
3. 状态栏实时刷新
4. 效果跟踪区从 DB 读日志
5. 统计洞察卡片（先读 statistics_insights，后建模块）
