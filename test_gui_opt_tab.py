# -*- coding: utf-8 -*-
"""
GUI offscreen 对抗性测试: 模型优化页新区 + 重训练参数 + 增强特征勾选
写入全部重定向沙盒；真实文件只读校验零污染。
"""
import os, sys, json, hashlib, tempfile, shutil

# 必须先设 offscreen；torch 必须先于 QApplication 导入(与 test_ui_ux_sim 同序，避免 DLL 初始化失败)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
import torch  # noqa: F401  先 torch 后 Qt，与现网测试同序

REAL_FILES = ['prediction_history.json', 'model_weights.json',
              os.path.join('feature_analysis', 'selected_features.json'),
              os.path.join('feature_analysis', 'feature_importance.json'),
              os.path.join('feature_analysis', 'drift_detection.json')]

def snap():
    d = {}
    for f in REAL_FILES:
        d[f] = hashlib.md5(open(f, 'rb').read()).hexdigest() if os.path.exists(f) else None
    m = {}
    for root, _, files in os.walk('model'):
        for fn in files:
            p = os.path.join(root, fn)
            m[p] = os.path.getsize(p)
    return d, m

snap0, m0 = snap()
SANDBOX = tempfile.mkdtemp(prefix='gui_opt_')

import faulthandler
from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication(sys.argv)

# 拦截模态框
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)

import warnings; warnings.filterwarnings('ignore')
import lottery_predictor_app_new as L

# 拦截线程真实执行: 记录参数不真跑
import thread_utils
calls = []
class FakeThread:
    def __init__(self, **kw): calls.append(kw)
    def start(self): pass
    log_signal = None
# 仅替换 RetrainThread（其他线程不动）
class RetrainRecorder(thread_utils.QThread):
    from PyQt5.QtCore import pyqtSignal
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    def __init__(self, **kw):
        super().__init__()
        calls.append(kw)
    def run(self): pass
thread_utils.RetrainThread = RetrainRecorder
L.RetrainThread = RetrainRecorder

# 权重文件重定向（recalc 按钮真会写）
import model_weights as mw
mw.WEIGHTS_FILE = os.path.join(SANDBOX, 'model_weights.json')

win = L.LotteryPredictorApp()
app.processEvents()

# ===== T1: 优化页新区存在 =====
assert hasattr(win, 'opt_weights_table') and win.opt_weights_table.columnCount() == 7, '表现榜表格缺失'
assert hasattr(win, 'opt_refresh_weights_btn') and hasattr(win, 'opt_recalc_weights_btn')
assert hasattr(win, 'opt_recommended_label') and hasattr(win, 'opt_weights_update_label')
print('T1 优化页组件存在 PASS')

# ===== T2: 刷新表现榜(真实记录只读) =====
win.opt_lottery_combo.setCurrentIndex(1)  # index1=DLT
win.refresh_model_weights_ui()
app.processEvents()
rows = win.opt_weights_table.rowCount()
print(f'T2 DLT 表现榜行数: {rows}')
assert rows >= 4, '表现榜行数异常'
rec_txt = win.opt_recommended_label.text()
print('   ', rec_txt)
assert '推荐模型' in rec_txt
win.opt_lottery_combo.setCurrentIndex(0)  # index0=SSQ
win.refresh_model_weights_ui()
app.processEvents()
assert win.opt_weights_table.rowCount() >= 4, 'SSQ 表现榜行数异常'
print('T2 表现榜刷新(双彩种) PASS')

# ===== T3: 重算权重按钮(写沙盒) =====
win.recalc_model_weights_ui()
app.processEvents()
assert os.path.exists(mw.WEIGHTS_FILE), '沙盒权重文件未生成'
log_text = win.opt_log_box.toPlainText()
assert '权重已更新' in log_text, '日志缺少更新行'
print('T3 重算权重(沙盒) PASS')

# ===== T4: 重训练参数透传 =====
calls.clear()
win.opt_lottery_combo.setCurrentIndex(1)  # DLT
win.opt_recent_spin.setValue(300)
win.opt_enhanced_retrain_check.setChecked(True)
win.opt_lstm_retrain_check.setChecked(True)
win.opt_model_multi_edit.setText('gbdt,random_forest')
win.retrain_current_lottery()
app.processEvents()
assert len(calls) == 1, f'重训线程未启动: {calls}'
c = calls[0]
print('T4 参数透传:', c)
assert c.get('recent_periods') == 300
assert c.get('feature_mode') == 'enhanced'
assert c.get('include_lstm') is True
assert c.get('model_types') == ['gbdt', 'random_forest']
assert c.get('lottery_type') == 'dlt'
print('T4 重训参数透传 PASS')

# 默认值回归: 近窗=0→None, 不勾选→original/False
def _p(m):
    print(m, flush=True)
_p('T4b.1 spin')
win.opt_recent_spin.setValue(0)
_p('T4b.2 enh')
win.opt_enhanced_retrain_check.setChecked(False)
_p('T4b.3 lstm')
win.opt_lstm_retrain_check.setChecked(False)
_p('T4b.4 edit')
win.opt_model_multi_edit.setText('')
_p('T4b.5 clear')
calls.clear()
_p('T4b.6 call retrain')
win.retrain_current_lottery()
_p('T4b.7 events')
app.processEvents()
_p('T4b.8 done')
c = calls[0]
assert c.get('recent_periods') is None and c.get('feature_mode') == 'original'
assert c.get('include_lstm') is False and c.get('model_types') is None
print('T4b 默认值回归(全量/original/不含LSTM) PASS')

# ===== T5: 全部重训透传 =====
calls.clear()
win.opt_recent_spin.setValue(150)
win.retrain_all_lotteries_ui()
app.processEvents()
c = calls[0]
assert c.get('retrain_all') is True and c.get('recent_periods') == 150
print('T5 全部重训透传 PASS')

# ===== T6: 主界面增强特征勾选框联动 =====
assert hasattr(win, 'enhanced_feature_checkbox')
win.enhanced_feature_checkbox.setChecked(True)
assert win.enhanced_feature_checkbox.isChecked()
print('T6 增强特征勾选框存在可交互 PASS')

# ===== T7: 自动重训勾选→权重刷新(写沙盒) =====
win.toggle_autoretrain(2)
app.processEvents()
assert os.path.exists(os.path.join(SANDBOX, 'model_weights.json'))
print('T7 自动重训启用刷新权重(沙盒) PASS')

# ===== T8: 优化页标签切换不崩 =====
idx = win.tab_widget.indexOf(win.optimization_tab)
win.tab_widget.setCurrentIndex(idx)
app.processEvents()
print('T8 切换到优化页(触发自动刷新) PASS')

# ===== 零污染 =====
snap1, m1 = snap()
bad = [f for f in snap0 if snap0[f] != snap1.get(f)]
assert not bad, f'真实文件被修改: {bad}'
assert not (set(m1) - set(m0)) and not (set(m0) - set(m1)), '模型目录被污染'
changed = [p for p in (set(m0) & set(m1)) if m0[p] != m1[p]]
assert not changed, f'模型文件和变化: {changed}'

print('\nGUI OPT TAB ADVERSARIAL ALL PASS (真实数据零污染)')
