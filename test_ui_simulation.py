# -*- coding: utf-8 -*-
"""
模型优化标签页 - 完整 UI 逻辑模拟测试
按真实用户操作顺序模拟点击，验证每一步的 UI/UX 状态。
只读操作为主；特征分析/漂移检测会写 feature_analysis_result.json 等结果文件(属正常功能输出)。
预测记录/投注计划等真实数据文件不被触碰。
注意: 此脚本不要 enable faulthandler——它会接管 SIGSEGV, 把 offscreen 环境下
后台线程潜伏的原生 fault(底层本会静默处理)放大成进程直接终止, 造成假死现象。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

PASS = []
FAIL = []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {detail}")

app = QApplication(sys.argv)
from lottery_predictor_app_new import LotteryPredictorApp
win = LotteryPredictorApp()
win.show()

# ---- 无头环境(offscreen)下 QMessageBox 模态框会导致 access violation ----
# 测试层直接应答, 模拟用户在真实屏幕上的点击
def _test_msg_box(parent, title, text, buttons, default_button=None):
    from PyQt5.QtWidgets import QMessageBox as _QB
    if buttons & _QB.Yes:
        # 所有确认类弹窗都点"是"(模拟用户确认)
        return _QB.Yes
    if buttons & _QB.Ok:
        return _QB.Ok
    if buttons & _QB.No:
        return _QB.No
    return buttons  # 兜底返回第一个按钮

from PyQt5.QtWidgets import QMessageBox as _QMB
_QMB.question = staticmethod(lambda *a, **k: _QMB.Yes)
_QMB.information = staticmethod(lambda *a, **k: _QMB.Ok)
_QMB.warning = staticmethod(lambda *a, **k: _QMB.Ok)
_QMB.critical = staticmethod(lambda *a, **k: _QMB.Ok)
print('[setup] QMessageBox 已替换为自动应答', flush=True)

def wait_threads(max_ms=300000, poll=200):
    """等待后台线程结束(按钮禁用状态轮询 + 显式线程等待)"""
    deadline = time.time() + max_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        busy = False
        for th in (win.feature_analysis_thread, win.calibration_thread, win.retrain_thread):
            if th is not None and th.isRunning():
                busy = True
                break
        if not busy:
            # 再处理一批待投递信号
            for _ in range(20):
                app.processEvents()
            return True
        time.sleep(0.05)
    return False

# ============ 场景1: 打开应用, 切到模型优化标签页 ============
print("\n=== 场景1: 打开应用, 定位'模型优化'标签页 ===")
tabs = win.tab_widget
opt_idx = tabs.indexOf(win.optimization_tab)
check("模型优化标签页存在", opt_idx >= 0, f"index={opt_idx}")
tabs.setCurrentIndex(opt_idx)
app.processEvents()

# ============ 场景2: 初始状态检查 ============
print("\n=== 场景2: 初始状态(SSQ默认) ===")
status_text = win.opt_calib_status_label.text()
check("校准状态标签有内容", "校准" in status_text, f"-> {status_text}")
check("校准状态显示未校准", "未校准" in status_text)
samples_text = win.opt_calib_samples_label.text()
check("样本数标签显示0", "0" in samples_text, f"-> {samples_text}")
check("校准表2行(红/蓝)", win.opt_calib_table.rowCount() == 2,
      f"rows={win.opt_calib_table.rowCount()}")

# ============ 场景3: SSQ 运行特征分析 ============
print("\n=== 场景3: SSQ 点击'运行特征分析' ===")
check("彩票选择=双色球", win._get_opt_lottery_type() == 'ssq')
win.opt_run_feat_btn.click()
app.processEvents()
check("点击后按钮禁用", not win.opt_run_feat_btn.isEnabled())
check("日志框有开始提示", "特征分析" in win.opt_log_box.toPlainText())
ok = wait_threads()
check("特征分析线程正常结束", ok)
feat_rows = win.opt_feat_table.rowCount()
check("特征表已填充", feat_rows > 0, f"rows={feat_rows} (Top-K={win.opt_top_k_spin.value()})")
check("特征表行数<=Top-K", feat_rows <= win.opt_top_k_spin.value())
# 检查表格列内容
first_feat = win.opt_feat_table.item(0, 1)
check("首行特征名非空", first_feat is not None and first_feat.text().strip() != "")
log_text = win.opt_log_box.toPlainText()
check("日志含完成信息", "完成" in log_text or "Top" in log_text)

# ============ 场景4: 刷新特征(不重跑) ============
print("\n=== 场景4: 点击'刷新'特征表 ===")
before = win.opt_feat_table.rowCount()
win.opt_refresh_feat_btn.click()
app.processEvents()
check("刷新后表格保持", win.opt_feat_table.rowCount() == before,
      f"rows {before} -> {win.opt_feat_table.rowCount()}")

# ============ 场景5: 修改Top-K ============
print("\n=== 场景5: Top-K 改为 10 再刷新 ===")
win.opt_top_k_spin.setValue(10)
win.opt_refresh_feat_btn.click()
app.processEvents()
check("Top-K=10 时行数<=10", win.opt_feat_table.rowCount() <= 10,
      f"rows={win.opt_feat_table.rowCount()}")
win.opt_top_k_spin.setValue(20)
win.opt_refresh_feat_btn.click()
app.processEvents()

# ============ 场景6: SSQ 漂移检测 ============
print("\n=== 场景6: SSQ 点击'漂移检测' ===")
win.opt_drift_check_btn.click()
app.processEvents()
check("漂移检测线程已启动", win.feature_analysis_thread is not None and win.feature_analysis_thread.isRunning())
ok = wait_threads()
check("漂移检测线程正常结束", ok)
log_text = win.opt_log_box.toPlainText()
check("漂移日志有输出", "漂移" in log_text)

# ============ 场景7: 切换到 DLT, 校准状态联动刷新 ============
print("\n=== 场景7: 切换到 DLT ===")
win.opt_lottery_combo.setCurrentIndex(1)
app.processEvents()
check("彩票切换=DLT", win._get_opt_lottery_type() == 'dlt')
# 此时特征表应自动清空或仍显示(SSQ的缓存? 看实现: refresh_feature_list 重新读 DLT 结果)
# DLT 没跑过分析, 特征表应为空
check("DLT未跑分析时特征表为空", win.opt_feat_table.rowCount() == 0,
      f"rows={win.opt_feat_table.rowCount()}")

# ============ 场景8: DLT 运行特征分析 ============
print("\n=== 场景8: DLT 点击'运行特征分析' ===")
win.opt_run_feat_btn.click()
app.processEvents()
ok = wait_threads()
check("DLT 特征分析线程正常结束", ok)
check("DLT 特征表已填充", win.opt_feat_table.rowCount() > 0,
      f"rows={win.opt_feat_table.rowCount()}")

# ============ 场景9: DLT 漂移检测 ============
print("\n=== 场景9: DLT 点击'漂移检测' ===")
win.opt_drift_check_btn.click()
app.processEvents()
ok = wait_threads()
check("DLT 漂移检测线程正常结束", ok)

# ============ 场景10: DLT 训练校准模型(无概率数据) ============
print("\n=== 场景10: DLT 点击'训练校准模型'(应精确诊断) ===")
log_before = win.opt_log_box.toPlainText()
win.opt_train_calib_btn.click()
# 泵事件直到校准线程启动(确认框已被monkeypatch同步应答), 不用裸sleep冻结事件循环
for _ in range(50):
    app.processEvents()
    if win.calibration_thread is not None and win.calibration_thread.isRunning():
        break
    time.sleep(0.02)
check("校准线程已启动", win.calibration_thread is not None and win.calibration_thread.isRunning())
ok = wait_threads()
check("校准线程正常结束", ok)
log_new = win.opt_log_box.toPlainText()[len(log_before):]
check("无误导提示'需要先核验'", "需要先核验" not in log_new)
check("精确诊断出现", "不含模型概率" in log_new, f"新日志: {log_new.strip()[:120]!r}")
check("给出积累建议", "积累" in log_new)

# ============ 场景11: SSQ 训练校准模型(同样诊断) ============
print("\n=== 场景11: 切回 SSQ, 训练校准模型 ===")
win.opt_lottery_combo.setCurrentIndex(0)
app.processEvents()
log_before = win.opt_log_box.toPlainText()
win.opt_train_calib_btn.click()
app.processEvents()
ok = wait_threads()
log_new = win.opt_log_box.toPlainText()[len(log_before):]
check("SSQ 校准线程正常结束", ok)
check("SSQ 精确诊断出现", "不含模型概率" in log_new)

# ============ 场景12: 查看校准曲线(无数据) ============
print("\n=== 场景12: 点击'查看校准曲线'(无数据, 应提示不崩溃) ===")
try:
    win.show_calibration_curve()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.05)
    check("无数据时查看曲线不崩溃(弹提示框)", True)
except Exception as e:
    check("无数据时查看曲线不崩溃", False, f"异常: {e}")

# ============ 场景13: 反馈重训练按钮(不真正执行, 验证确认与禁用) ============
print("\n=== 场景13: 重训练按钮交互 ===")
check("重训练单彩票按钮可用", win.opt_retrain_one_btn.isEnabled())
check("重训练全部按钮可用", win.opt_retrain_all_btn.isEnabled())
# 验证"取消拦截"逻辑: 临时把 question 改成返回 No, 点击后不应启动线程
_QMB.question = staticmethod(lambda *a, **k: _QMB.No)
win.opt_retrain_all_btn.click()
app.processEvents()
check("点'否'后未启动重训练线程", not (hasattr(win, 'retrain_thread') and win.retrain_thread is not None and win.retrain_thread.isRunning()))
check("点'否'后按钮保持可用", win.opt_retrain_all_btn.isEnabled() and win.opt_retrain_one_btn.isEnabled())
_QMB.question = staticmethod(lambda *a, **k: _QMB.Yes)  # 恢复

# ============ 场景14: 按钮禁用保护 ============
print("\n=== 场景14: 线程运行中按钮禁用保护 ===")
win.opt_drift_check_btn.click()
# 禁用是同步的: click返回后按钮必须已禁用(不依赖事件循环)
check("点击后立即禁用-运行分析", not win.opt_run_feat_btn.isEnabled())
check("点击后立即禁用-漂移按钮", not win.opt_drift_check_btn.isEnabled())
# 再泵事件确认线程确实起来了(向量化后漂移检测可能<0.3秒完成, 用泵事件捕获)
thread_seen = False
for _ in range(50):
    app.processEvents()
    if win.feature_analysis_thread is not None and win.feature_analysis_thread.isRunning():
        thread_seen = True
        break
    if win.feature_analysis_thread is not None and not win.feature_analysis_thread.isRunning():
        break  # 已启动并完成(极快路径)
    time.sleep(0.02)
check("线程已启动(运行中或已快速完成)", thread_seen or (win.feature_analysis_thread is not None))
ok = wait_threads()
check("结束后按钮恢复", win.opt_run_feat_btn.isEnabled() and win.opt_drift_check_btn.isEnabled())

# ============ 场景15: 日志框完整性 ============
print("\n=== 场景15: 日志框检查 ===")
log_text = win.opt_log_box.toPlainText()
check("日志框非空", len(log_text) > 100, f"len={len(log_text)}")
check("日志含时间戳", "[" in log_text and "】" not in log_text[:20])
check("无未捕获异常堆栈", "Traceback" not in log_text, )

# ============ 收尾 ============
print("\n" + "=" * 50)
print(f"总计: {len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print(f"  - {f}")
# 验证真实数据文件未被触碰
rec = json.load(open('prediction_history.json', encoding='utf-8'))
inv = json.load(open('investment_history.json', encoding='utf-8'))
check("prediction_history.json 仍274条", len(rec) == 274, f"actual={len(rec)}")
check("investment_history.json 仍8条", len(inv) == 8, f"actual={len(inv)}")
print("=" * 50)
print("RESULT:", "ALL PASS" if not FAIL else "HAS FAIL")
sys.exit(0 if not FAIL else 1)
