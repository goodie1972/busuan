# -*- coding: utf-8 -*-
"""
UI/UX 模拟测试：覆盖全部 9 个标签页的切换、控件遍历、安全交互与主题切换。
验证 UI 布局完整、控件可见可用、交互不崩溃、状态反馈正常。

安全设计（遵守"测试禁触真实数据文件"）：
- 预测记录/投注计划写入重定向到沙盒
- 真实数据文件(CSV/真实json)只读
- 结束时 MD5 校验真实数据文件未被触碰

注意: 不 enable faulthandler (offscreen 下会放大后台线程原生 fault)。
运行: QT_QPA_PLATFORM=offscreen python test_ui_ux_sim.py
"""
import sys, os, json, time, hashlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from PyQt5.QtWidgets import (QApplication, QMessageBox, QWidget, QTabWidget,
                             QPushButton, QComboBox, QSpinBox, QLineEdit,
                             QCheckBox, QTableWidget, QLabel, QTextEdit,
                             QPlainTextEdit)
app = QApplication(sys.argv)
_QMB = QMessageBox
_QMB.question = staticmethod(lambda *a, **k: _QMB.Yes)
_QMB.information = staticmethod(lambda *a, **k: _QMB.Ok)
_QMB.warning = staticmethod(lambda *a, **k: _QMB.Ok)
_QMB.critical = staticmethod(lambda *a, **k: _QMB.Ok)

PASS, FAIL = [], []
def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

# ============ 0. 真实数据文件 MD5 基线 ============
REAL_FILES = ['scripts/ssq/ssq_history.csv', 'scripts/dlt/dlt_history.csv',
              'prediction_history.json', 'investment_history.json']
def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
BASELINE = {p: md5(p) for p in REAL_FILES}

# ============ 1. 沙盒重定向 ============
SANDBOX = tempfile.mkdtemp(prefix='busuan_uiux_')
import prediction_records as pr
pr.HISTORY_FILE = os.path.join(SANDBOX, 'prediction_history.json')

from lottery_predictor_app_new import LotteryPredictorApp
win = LotteryPredictorApp()
win.show()
# 投注文件也重定向(若窗口有该属性)
if hasattr(win, 'investment_file'):
    win.investment_file = os.path.join(SANDBOX, 'investment_history.json')

def pump(ms=150):
    for _ in range(max(1, ms // 10)):
        app.processEvents(); time.sleep(0.01)

# ============ 阶段1: 9个标签页切换 ============
print("=" * 60)
print("阶段1: 全部9个标签页逐一切换(触发各自的自动刷新逻辑)")
print("=" * 60)
tab_names = ["预测", "期望值模型", "数据分析", "高级统计", "历史回测",
             "紫微斗数", "梅花易数", "预测记录", "投注计划", "模型优化"]
check("标签页数量=10(含模型优化)", win.tab_widget.count() == len(tab_names),
      f"actual={win.tab_widget.count()}")
for i, name in enumerate(tab_names):
    win.tab_widget.setCurrentIndex(i)
    pump(300)
    actual = win.tab_widget.tabText(i)
    check(f"切换到[{name}]: 标签名一致且无异常", actual == name, f"actual={actual}")
# 停在第一个标签
win.tab_widget.setCurrentIndex(0)
pump(200)

# ============ 阶段2: 控件可见性/完整性遍历 ============
print()
print("=" * 60)
print("阶段2: 各标签页关键控件完整性(非空/可见/可用)")
print("=" * 60)
def walk(label, names, tab_idx=None):
    # 控件可见性依赖于所在标签页: 先切到对应页再检查
    if tab_idx is not None:
        win.tab_widget.setCurrentIndex(tab_idx)
        pump(120)
    for n in names:
        w = getattr(win, n, None)
        ok = w is not None
        if ok and isinstance(w, (QPushButton, QComboBox, QSpinBox, QLineEdit, QCheckBox)):
            ok = w.isVisibleTo(win)
        check(f"[{label}] 控件 {n}", ok)

walk("预测", ['lottery_combo', 'model_combo', 'prediction_spin', 'predict_button',
             'train_button', 'auto_predict_button', 'result_label', 'log_box',
             'compound_red_spin', 'compound_blue_spin',
             'dt_red_dan', 'dt_red_tuo', 'dt_blue_dan', 'dt_blue_tuo'], tab_idx=0)
walk("期望值", ['ev_result_label', 'ev_log_box'], tab_idx=1)
walk("数据分析", ['analysis_combo', 'chart_label', 'stats_text', 'advanced_stats_button'], tab_idx=2)
walk("高级统计", ['advanced_stats_lottery_combo', 'run_advanced_stats_button', 'stats_result_label'], tab_idx=3)
walk("历史回测", ['bt_start_button', 'bt_lottery_combo', 'bt_model_combo',
                 'bt_periods_spin', 'bt_result_text', 'bt_status_label', 'bt_verify_button'], tab_idx=4)
walk("紫微斗数", ['zw_predict_btn', 'zw_reset_btn', 'zw_name', 'zw_gender', 'zw_year'], tab_idx=5)
walk("梅花易数", ['mh_predict_btn', 'mh_reset_btn', 'mh_num_check', 'mh_num1', 'mh_target'], tab_idx=6)
walk("预测记录", ['rec_refresh_btn', 'rec_verify_btn', 'rec_verify_all_btn',
                 'rec_export_btn', 'rec_delete_btn', 'records_table',
                 'rec_filter_lottery', 'rec_filter_model', 'rec_filter_status',
                 'rec_stats_label'], tab_idx=7)
walk("投注计划", ['invest_load_btn', 'invest_confirm_btn', 'invest_verify_btn',
                 'invest_export_btn', 'invest_clear_btn', 'prediction_select_table',
                 'investment_table', 'total_invested_label', 'roi_label', 'win_rate_label',
                 'invest_sel_all_pred_btn', 'invest_clear_all_pred_btn',
                 'invest_sel_all_rec_btn', 'invest_clear_all_rec_btn'], tab_idx=8)
walk("模型优化", ['opt_lottery_combo', 'opt_feat_table', 'opt_top_k_spin',
                 'opt_train_calib_btn', 'opt_run_feat_btn', 'opt_log_box'], tab_idx=9)
# 表格列头
for tattr, tname in [('records_table', '预测记录'), ('investment_table', '投注计划'),
                     ('prediction_select_table', '投注选择'), ('opt_feat_table', '特征列表')]:
    t = getattr(win, tattr)
    check(f"[{tname}] 表格列头>=3列", t.columnCount() >= 3, f"cols={t.columnCount()}")

# ============ 阶段3: 安全交互(真实点击/切换, 只写沙盒) ============
print()
print("=" * 60)
print("阶段3: 安全交互 (主题/彩种切换/模式卡片/筛选/加载/核验)")
print("=" * 60)
# 3a. 主题全切换(不含自定义, 自定义会弹对话框)
from theme_manager import ThemeManager
theme_list = list(ThemeManager().themes.keys())
_orig_theme = win.theme_manager.current_theme  # 保存原主题, 结束时恢复
win.tab_widget.setCurrentIndex(0); pump(150)
for th in theme_list:
    if th == "自定义":
        continue
    win.theme_combo.setCurrentText(th)
    pump(150)
    check(f"主题切换: {th}", win.theme_manager.current_theme == th,
          f"actual={win.theme_manager.current_theme}")
check("主题数量=7", len(theme_list) == 7, f"actual={len(theme_list)}")
win.theme_combo.setCurrentText(_orig_theme); pump(120)

# 3b. 彩种切换触发模型列表联动
win.model_combo.setCurrentIndex(0)
win.lottery_combo.setCurrentIndex(1)  # dlt
pump(300)
n_dlt = win.model_combo.count()
win.lottery_combo.setCurrentIndex(0)  # ssq
pump(300)
n_ssq = win.model_combo.count()
check("彩种切换: 模型列表刷新(数量一致或合理)", n_dlt >= 5 and n_ssq >= 5,
      f"dlt={n_dlt} ssq={n_ssq}")

# 3c. 模式卡片互斥切换(单式→复式→胆拖→单式)
win.model_combo.setCurrentIndex(0)  # 保证是支持的模型
pump(150)
# 通过 mode_combo 数据载体验证切换
win.mode_combo.setCurrentText("复式"); pump(100)
check("模式切换: mode_combo数据载体=复式", win.mode_combo.currentText() == "复式")
win.mode_combo.setCurrentText("胆拖"); pump(100)
check("模式切换: mode_combo数据载体=胆拖", win.mode_combo.currentText() == "胆拖")
win.mode_combo.setCurrentText("单式"); pump(100)
check("模式切回单式", win.mode_combo.currentText() == "单式")

# 3d. 预测记录页: 筛选器切换触发刷新(表格行数=沙盒0条)
win.tab_widget.setCurrentIndex(7); pump(200)
for f in ['rec_filter_lottery', 'rec_filter_model', 'rec_filter_status']:
    combo = getattr(win, f)
    for i in range(min(2, combo.count())):
        combo.setCurrentIndex(i)
        pump(120)
    check(f"筛选器 {f}: 切换不崩溃", win.records_table.rowCount() >= 0)
win.rec_refresh_btn.click(); pump(200)
stats = win.rec_stats_label.text()
check("预测记录: 刷新后统计标签非空", len(stats.strip()) > 0, f"stats={stats[:40]!r}")

# 3e. 历史回测页: 点击核验(走 verify_prediction_records, 只写沙盒)
win.tab_widget.setCurrentIndex(4); pump(200)
# 先往沙盒放2条待核验记录
from prediction_records import save_records
rec_base = {'predict_time': '2026-10-01T10:00:00', 'lottery_type': 'ssq',
            'model_type': 'lightgbm', 'latest_period': 26108,
            'red_numbers': [1, 2, 3, 4, 5, 6], 'blue_numbers': [1],
            'red_probabilities': None, 'blue_probabilities': None,
            'target_period': None, 'red_hits': None, 'blue_hits': None,
            'prize_name': None, 'prize_amount': None}
save_records([dict(rec_base), dict(rec_base)])
win.bt_verify_button.click(); pump(500)
recs = pr.load_records()
check("历史回测: 点击核验后记录被核验",
      all(r['target_period'] is not None for r in recs),
      f"n={len(recs)}")
check("历史回测: 状态标签有反馈", len(win.bt_status_label.text().strip()) > 0,
      f"text={win.bt_status_label.text()[:30]!r}")

# 3f. 投注计划页: 加载最近预测到选择表
win.tab_widget.setCurrentIndex(8); pump(200)
win.invest_load_btn.click(); pump(300)
n_sel = win.prediction_select_table.rowCount()
check("投注计划: 加载最近预测到选择表", n_sel >= 1, f"rows={n_sel}")
# 统计标签初始态可见
check("投注计划: ROI/胜率标签存在", win.roi_label is not None and win.win_rate_label is not None)

# 3f-2. 预测选择表: 全选/取消全选(复选框)
win.invest_sel_all_pred_btn.click(); pump(120)
checked = sum(1 for r in range(win.prediction_select_table.rowCount())
              if win.prediction_select_table.cellWidget(r, 0).isChecked())
check("投注计划: 全选预测→全部勾选", checked == n_sel and n_sel >= 1,
      f"checked={checked}/{n_sel}")
win.invest_clear_all_pred_btn.click(); pump(120)
checked = sum(1 for r in range(win.prediction_select_table.rowCount())
              if win.prediction_select_table.cellWidget(r, 0).isChecked())
check("投注计划: 取消全选→全部未勾选", checked == 0, f"checked={checked}")
# 勾一行→确认投注(写沙盒)→投注记录表有行, 验证第二步全选
first_cb = win.prediction_select_table.cellWidget(0, 0)
if first_cb:
    first_cb.setChecked(True)
    rows_before = win.investment_table.rowCount()
    win.invest_confirm_btn.click(); pump(200)
    check("投注计划: 确认投注后记录表+1行",
          win.investment_table.rowCount() == rows_before + 1,
          f"{rows_before}->{win.investment_table.rowCount()}")
    # 第二步: 投注记录表全选/取消全选
    win.invest_sel_all_rec_btn.click(); pump(120)
    sel_rows = set(i.row() for i in win.investment_table.selectedItems())
    check("投注计划: 全选投注记录→选中全部行",
          len(sel_rows) == win.investment_table.rowCount(),
          f"sel={len(sel_rows)}/{win.investment_table.rowCount()}")
    win.invest_clear_all_rec_btn.click(); pump(120)
    check("投注计划: 取消选中→0选中行",
          len(win.investment_table.selectedItems()) == 0)

# 3g. 梅花易数: 直觉数字勾选联动启用3个数字框
win.tab_widget.setCurrentIndex(6); pump(200)
win.mh_num_check.setChecked(True); pump(80)
check("梅花: 勾选直觉数字→数字框启用", win.mh_num1.isEnabled())
win.mh_num_check.setChecked(False); pump(80)
check("梅花: 取消勾选→数字框禁用", not win.mh_num1.isEnabled())

# 3h. 紫微斗数: reset 按钮重置结果卡片(其语义是重置结果, 非清空表单)
win.tab_widget.setCurrentIndex(5); pump(200)
# 先把某张卡片设成非默认值, 再点重置, 应回到"待排盘"
r0, b0, d0 = win.zw_cards[0]
r0.setText("红球: 99,98,97"); b0.setText("蓝球: 99"); d0.setText("推演: 测试")
win.zw_reset_btn.click(); pump(150)
reset_ok = (r0.text() == "红球: 待排盘" and b0.text() == "蓝球: 待排盘"
            and d0.text() == "推演: 待排盘")
check("紫微: reset重置结果卡片回待排盘", reset_ok,
      f"text={r0.text()!r}/{b0.text()!r}/{d0.text()!r}")

# 3i. 模型优化页: 彩种切换联动刷新校准状态
win.tab_widget.setCurrentIndex(9); pump(200)
win.opt_lottery_combo.setCurrentIndex(1); pump(300)
check("模型优化: 切换dlt后校准状态已刷新(日志非空)",
      len(win.opt_log_box.toPlainText().strip()) > 0)
win.opt_lottery_combo.setCurrentIndex(0); pump(200)

# ============ 阶段4: 真实数据文件完整性终检 ============
print()
print("=" * 60)
print("阶段4: 真实数据文件完整性终检")
print("=" * 60)
for p, h in BASELINE.items():
    check(f"未被触碰: {p}", md5(p) == h)

import shutil
shutil.rmtree(SANDBOX, ignore_errors=True)

print()
print("=" * 60)
print(f"UI/UX 模拟结果: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 60)
sys.exit(0 if not FAIL else 1)
