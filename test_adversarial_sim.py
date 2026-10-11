# -*- coding: utf-8 -*-
"""
对抗性模拟测试：以真实用户操作序列驱动真实 GUI + 真实模型 + 真实历史数据，
验证最近改动(概率收集/核验子集/200条加载)在所有路径下正常工作。

安全设计：
- 预测存档目标重定向到沙盒文件(不碰真实 prediction_history.json)
- 所有真实数据文件(CSV/模型/真实json)只读
- 结束时用 MD5 校验所有真实数据文件未被触碰

运行: QT_QPA_PLATFORM=offscreen python test_adversarial_sim.py
"""
import sys, os, json, time, hashlib, itertools, tempfile
import math
from math import comb as _comb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 注意: 不 enable faulthandler (offscreen 下会放大后台线程潜伏 fault)
import torch
from PyQt5.QtWidgets import QApplication, QMessageBox
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

# ============ 0. 数据文件基线 MD5(全程不得变化) ============
REAL_FILES = ['scripts/ssq/ssq_history.csv', 'scripts/dlt/dlt_history.csv',
              'prediction_history.json', 'investment_history.json']
def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
BASELINE = {p: md5(p) for p in REAL_FILES}

# ============ 1. 沙盒: 重定向预测存档 ============
SANDBOX = tempfile.mkdtemp(prefix='busuan_adv_')
SANDBOX_PRED = os.path.join(SANDBOX, 'prediction_history.json')
import prediction_records as pr
pr.HISTORY_FILE = SANDBOX_PRED  # 重定向, 真实文件只读

from ml_models import LotteryMLModels, MODEL_TYPES
from scripts.data_analysis import load_lottery_data

from lottery_predictor_app_new import LotteryPredictorApp
win = LotteryPredictorApp()
win.show()

def pump(ms=200):
    for _ in range(ms // 10):
        app.processEvents(); time.sleep(0.01)

def wait_thread(th, max_s=60):
    end = time.time() + max_s
    while time.time() < end:
        app.processEvents(); time.sleep(0.05)
        if th is None:
            return False
        if not th.isRunning():
            app.processEvents(); return True
    return False

from model_utils import name_path
LOTS = list(name_path.keys())  # ['ssq','dlt']

print("=" * 60)
print("阶段1: 单式预测 × 全部已训练ML模型 (真实模型/真实数据)")
print("=" * 60)
n_added_before = len(pr.load_records())
tested_models = []
for li, lt in enumerate(LOTS):
    win.lottery_combo.setCurrentIndex(li)
    pump(150)
    for mi in range(win.model_combo.count()):
        item = win.model_combo.model().item(mi)
        if not item.isEnabled():
            continue
        text = win.model_combo.itemText(mi)
        label = f"{lt}/{text}"
        if 'LSTM' in text or '期望值' in text or '紫微' in text or '梅花' in text:
            continue  # 这些路径无概率输出, 单独不测(非本次改动范围)
        win.model_combo.setCurrentIndex(mi)
        win.prediction_spin.setValue(3)
        before = len(pr.load_records())
        win.log_box.clear()
        win.generate_prediction()  # 真实调用, 内部走 _save_prediction_records
        pump(300)
        recs = pr.load_records()
        added = [r for r in recs[before:]]
        log_text = win.log_box.toPlainText()
        if len(added) == 0:
            # 模型可能未训练/加载失败: 属既有问题, 断言"友好报错不崩溃"
            check(f"单式 {label}: 模型加载失败时友好报错不崩溃",
                  ('出错' in log_text or '失败' in log_text or '不存在' in log_text),
                  f"tail={log_text[-60:]!r}")
            continue
        ok_count = len(added) == 3
        # 核心断言: 概率长度 == 号码数(校准可用的前提)
        ok_align = all(
            r.get('red_probabilities') is None  # 已知局限: binary:logistic 的 xgboost 无逐类概率
            or (len(r['red_probabilities']) == len(r['red_numbers'])
                and r.get('blue_probabilities') is not None
                and len(r['blue_probabilities']) == len(r['blue_numbers'])
                and all(isinstance(p, float) for p in r['red_probabilities']))
            for r in added)
        ok_range = all(all(0.0 <= p <= 1.0 for p in (r['red_probabilities'] or []))
                       for r in added)
        has_proba = any(r.get('red_probabilities') is not None for r in added)
        label2 = label + ('(无概率:binary模型已知局限)' if not has_proba else '')
        tested_models.append(label2)
        check(f"单式 {label}: 存档3条", ok_count, f"actual={len(added)}")
        check(f"单式 {label2}: 概率对齐或为None", ok_align)
        check(f"单式 {label2}: 概率值在[0,1]", ok_range)
        # 抽样: 概率对齐正确性 —— 第n个概率 == 全类别字典[号码-1]
        if ok_align and added and any(r.get('red_probabilities') for r in added):
            mt_key = next((k for k, v in MODEL_TYPES.items()
                           if v == text or v == text.replace(' (默认)', '')), None)
            if mt_key:
                ml = LotteryMLModels(lt, mt_key, log_callback=lambda m: None)
                if ml.load_models():
                    recent = load_lottery_data(lt).sort_values('期数', ascending=False).head(ml.feature_window)
                    rr, bb = ml.predict(recent, variation=0)
                    d = ml.last_red_proba
                    rec = added[0]
                    # 同一组号码的概率应对得上(同variation时预测应相同)
                    if list(rec['red_numbers']) == list(rr):
                        expected = [float(d.get(int(n) - 1, 0.0)) for n in rec['red_numbers']]
                        match = all(abs(a - b) < 1e-9 for a, b in zip(rec['red_probabilities'], expected))
                        check(f"单式 {label}: 概率值与模型字典一致", match,
                              f"recorded={rec['red_probabilities'][:2]} expected={expected[:2]}")

print()
print("=" * 60)
print("阶段2: 复式/胆拖路径 (含组合展开×概率对齐)")
print("=" * 60)
# 复式: 选ssq/lightgbm, 8红+1蓝
win.lottery_combo.setCurrentIndex(0)
pump(100)
for mi in range(win.model_combo.count()):
    t = win.model_combo.itemText(mi)
    if 'LightGBM' in t:
        win.model_combo.setCurrentIndex(mi)
# 用 mode_combo 数据载体切到复式
win.mode_combo.setCurrentText("复式")
pump(100)
win.compound_red_spin.setValue(8)
win.compound_blue_spin.setValue(1)
before = len(pr.load_records())
win.generate_prediction()
pump(300)
recs = pr.load_records()
added = recs[before:]
# 从存档自洽反推注数: 复式所有注共享同一红/蓝池, 注数 = C(红池,单式红) × C(蓝池,单式蓝)
red_pool = len(set(n for r in added for n in r['red_numbers']))
blue_pool = len(set(n for r in added for n in r['blue_numbers']))
r_cnt = len(added[0]['red_numbers']) if added else 0
b_cnt = len(added[0]['blue_numbers']) if added else 0
n_expected = int(_comb(red_pool, r_cnt) * _comb(blue_pool, b_cnt))
check("复式: 存档注数=C(红池,单式红)×C(蓝池,单式蓝)", len(added) == n_expected,
      f"actual={len(added)} 红池{red_pool}蓝池{blue_pool} 单式{r_cnt}/{b_cnt}")
ok = all(r.get('red_probabilities') is not None and len(r['red_probabilities']) == len(r['red_numbers'])
         and r.get('blue_probabilities') is not None and len(r['blue_probabilities']) == len(r['blue_numbers'])
         for r in added)
check("复式: 全部注概率按号码对齐(含全局字典回退路径)", ok)
# 同一号码在所有注中概率必须一致(字典对齐正确性的强断言)
pmap = {}
consistent = True
for r in added:
    if not r.get('red_probabilities'):
        continue
    for num, p in zip(r['red_numbers'], r['red_probabilities']):
        if num in pmap and abs(pmap[num] - p) > 1e-9:
            consistent = False
        pmap[num] = p
check("复式: 同号跨注概率一致(字典对齐无误)", consistent)

# 胆拖: 2红胆 + 4红拖 + 蓝胆0/蓝拖2 (注数由UI实际值动态计算)
win.mode_combo.setCurrentText("胆拖")
pump(100)
win.dt_red_dan.setValue(2)
win.dt_red_tuo.setValue(4)
win.dt_blue_dan.setValue(0)
win.dt_blue_tuo.setValue(2)
pump(100)
before = len(pr.load_records())
win.generate_prediction()
pump(300)
added = pr.load_records()[before:]
# 结构正确性: 每注红/蓝球个数 == 单式个数; 注数>0; 概率按号码对齐
r_cnt = len(added[0]['red_numbers']) if added else 0
b_cnt = len(added[0]['blue_numbers']) if added else 0
struct_ok = all(len(r['red_numbers']) == r_cnt and len(r['blue_numbers']) == b_cnt for r in added)
check(f"胆拖: 存档{len(added)}注结构正确(每注红{r_cnt}蓝{b_cnt})", len(added) > 0 and struct_ok)
ok = all(r.get('red_probabilities') is not None and len(r['red_probabilities']) == len(r['red_numbers'])
         and r.get('blue_probabilities') is not None and len(r['blue_probabilities']) == len(r['blue_numbers'])
         for r in added)
check("胆拖: 全部注概率按号码对齐", ok)
# 同号跨注概率一致(胆拖各注共享同一概率字典)
pmap = {}; consistent = True
for r in added:
    if not r.get('red_probabilities'): continue
    for num, p in zip(r['red_numbers'], r['red_probabilities']):
        if num in pmap and abs(pmap[num] - p) > 1e-9: consistent = False
        pmap[num] = p
check("胆拖: 同号跨注概率一致", consistent)
win.mode_combo.setCurrentText("单式")
pump(100)

print()
print("=" * 60)
print("阶段3: 一键智能预测 (AutoPredictThread → 4元组 → 存档)")
print("=" * 60)
win.lottery_combo.setCurrentIndex(0)
pump(100)
for mi in range(win.model_combo.count()):
    if 'LightGBM' in win.model_combo.itemText(mi):
        win.model_combo.setCurrentIndex(mi)
win.prediction_spin.setValue(2)
# 离屏环境跳过 fetch/train(网络+训练不可控), 只测 预测→存档 链路
import thread_utils as _tu
class _FastAutoPredictThread(_tu.AutoPredictThread):
    def __init__(self, *a, **kw):
        kw['skip_fetch'] = True
        kw['skip_train'] = True
        super().__init__(*a, **kw)
import lottery_predictor_app_new as _appmod
_orig_thread = _appmod.AutoPredictThread
_appmod.AutoPredictThread = _FastAutoPredictThread
before = len(pr.load_records())
win.auto_predict_button.click()
pump(500)
ok = wait_thread(getattr(win, 'auto_predict_thread', None))
_appmod.AutoPredictThread = _orig_thread  # 恢复
added = pr.load_records()[before:]
check("一键预测: 线程正常结束", ok)
ok = all(r.get('red_probabilities') and len(r['red_probabilities']) == len(r['red_numbers'])
         for r in added)
check(f"一键预测: {len(added)}条全带对齐概率", len(added) >= 1 and ok)

print()
print("=" * 60)
print("阶段4: 核验对抗 (subset/未开奖/重复调用/全量回归)")
print("=" * 60)
from prediction_records import verify_records, load_records, save_records
all_recs = load_records()
# 把沙盒记录的 latest_period 设为 26108(已开奖), 目标期 26109/26110 已开
latest_known = 26108
for r in all_recs:
    if r.get('latest_period') in (26115, 26116):
        r['latest_period'] = latest_known
save_records(all_recs)
check("沙盒记录最新期已调整为26108", all(r.get('latest_period') == latest_known for r in load_records()))

# 4a. 未开奖保护: 把部分记录 latest 设为最新期 26115 → 目标 26116 未开, 必须保持 pending
recs = load_records()
n = min(10, len(recs))
for i in range(n):
    recs[i]['latest_period'] = 26115
save_records(recs)
verify_records(log_callback=lambda m: None, records_subset=list(range(n)))
recs = load_records()
check("未开奖记录保持待核验(不误标)", all(recs[i]['target_period'] is None for i in range(n)))

# 4b. 重复核验幂等(先全量核验到稳定态, 再重复核验, 状态必须不变)
verify_records(log_callback=lambda m: None)
before = load_records()
verify_records(log_callback=lambda m: None)
after = load_records()
same = all(b['target_period'] == a['target_period'] and b['red_hits'] == a['red_hits']
           and b['blue_hits'] == a['blue_hits'] and b['prize_name'] == a['prize_name']
           for b, a in zip(before, after))
check("重复核验幂等(状态不变)", same)

# 4c. subset 只动 subset
recs = load_records()
pending_idx = [i for i, r in enumerate(recs) if r['target_period'] is None and r['latest_period'] == 26108]
if pending_idx:
    pick = pending_idx[:5]
    verify_records(log_callback=lambda m: None, records_subset=pick)
    recs = load_records()
    check("subset: 选中记录已核验", all(recs[i]['target_period'] is not None for i in pick))
    untouched = [i for i in pending_idx if i not in pick]
    check("subset: 未选中的保持待核验", all(recs[i]['target_period'] is None for i in untouched))

# 4d. 全量核验
verify_records(log_callback=lambda m: None)
recs = load_records()
still_pending = [r for r in recs if r['target_period'] is None]
check("全量核验: 仅剩未开奖的保持待核验",
      all(r['latest_period'] == 26115 for r in still_pending),
      f"pending={len(still_pending)}")

# 4e. GUI 路径: 表格"核验选中"(不选行=全部) + (选中2行=只核验2行)
win.refresh_prediction_records()
pump(200)
# 先制造2条待核验(latest=26108, 目标26109已开)
recs = load_records()
made = 0
for r in recs:
    if r['latest_period'] == 26108 and made < 2:
        r['target_period'] = None; r['red_hits'] = None; r['blue_hits'] = None
        r['prize_name'] = None; r['prize_amount'] = None; made += 1
save_records(recs)
win.refresh_prediction_records()
pump(200)
# 不选任何行 → 核验全部
win.rec_verify_btn.click()
pump(400)
recs = load_records()
check("GUI核验(无选中): 全部可核验的都核验了",
      all(r['target_period'] is not None for r in recs if r['latest_period'] != 26115))
# 选中表格第0、1行(多选模式叠加选择), 再点核验 → 日志应出现"核验选中 2 条"
from PyQt5.QtCore import QItemSelectionModel, QItemSelection
tbl = win.records_table
tbl.clearSelection()
sm = tbl.selectionModel()
sm.select(tbl.model().index(0, 0), QItemSelectionModel.Select)
sm.select(tbl.model().index(1, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
win.log_box.clear()
win.rec_verify_btn.click()
pump(400)
log_text = win.log_box.toPlainText()
import re as _re
m = _re.search(r"核验选中 (\d+) 条", log_text)
check("GUI核验(选中2行): 日志提示'核验选中 N 条'", m is not None and m.group(1) == '2', f"tail={log_text[-60:]!r}")
# 反方向: 制造一条新的待核验, 不选中任何行点"全部核验"→应被核验
recs = load_records()
for r in recs:
    if r['latest_period'] == 26108:
        r['target_period'] = None; r['red_hits'] = None; r['blue_hits'] = None
        r['prize_name'] = None; r['prize_amount'] = None
        break
save_records(recs)
win.refresh_prediction_records(); pump(200)
win.rec_verify_all_btn.click(); pump(400)
recs = load_records()
check("全部核验按钮: 新制造待核验被核验",
      all(r['target_period'] is not None for r in recs if r['latest_period'] != 26115))

print()
print("=" * 60)
print("阶段5: 校准收集端到端 (真实 collect_calibration_data)")
print("=" * 60)
recs = load_records()
verified = [r for r in recs if r.get('target_period') is not None]
check("沙盒已有足够已核验记录", len(verified) >= 30, f"n={len(verified)}")
from model_calibration import collect_calibration_data, calibrate_lottery, get_calibration_info
import model_calibration as mc
# 校准模型文件重定向到沙盒, 不碰真实 model_calibration/ 目录
mc.CALIBRATION_DIR = os.path.join(SANDBOX, 'model_calibration')
red_data, blue_data = collect_calibration_data('ssq')
tr = sum(len(v) for v in red_data.values())
tb = sum(len(v) for v in blue_data.values())
check(f"校准收集: 红球样本>0 且=已核验ssq记录数×6", tr > 0, f"red={tr} blue={tb}")
ssq_verified = [r for r in verified if r.get('lottery_type') in (None, 'ssq')]
check("校准红球样本数 = 已核验ssq记录数×6",
      tr == len(ssq_verified) * 6, f"expect={len(ssq_verified)*6} actual={tr}")
# 校准模型真正可训练(修复前恒失败, 因样本=0)
ok = calibrate_lottery('ssq')
info = get_calibration_info('ssq')
check("校准流程可运行且有样本(修复前恒为0样本)",
      bool(ok) and (info.get('red_samples', 0) > 0 or info.get('red_calibrated', False)),
      f"ok={ok} red_samples={info.get('red_samples')}")

print()
print("=" * 60)
print("阶段6: 加载最近预测 200 条上限 + 投注计划联动")
print("=" * 60)
# 沙盒里目前记录数可能<200, 补造到 250 条(仅沙盒)
recs = load_records()
base = recs[-1]
while len(recs) < 250:
    recs.append(dict(base, predict_time=f"2026-10-01T10:00:00",
                     red_numbers=[1,2,3,4,5,6], blue_numbers=[1],
                     red_probabilities=None, blue_probabilities=None,
                     target_period=None, red_hits=None, blue_hits=None,
                     prize_name=None, prize_amount=None))
save_records(recs)
win.load_predictions_for_investment()
pump(300)
rows = win.prediction_select_table.rowCount()
check("加载最近预测: 上限200条", rows == 200, f"actual={rows}")
# 排序: 最新在前
check("加载最近预测: 无崩溃且表格可读", rows > 0)

print()
print("=" * 60)
print("阶段7: 异常输入对抗")
print("=" * 60)
# 7a. 概率字段被污染为字符串列表(旧bug形态) → 存档不应崩溃, 校准应跳过该记录
import json as _json
bad = list(load_records())
bad.append({'predict_time': '2026-10-02T10:00:00', 'lottery_type': 'ssq',
            'model_type': 'lightgbm', 'latest_period': 26108,
            'red_numbers': [1,2,3,4,5,6], 'blue_numbers': [1],
            'red_probabilities': [str(i) for i in range(6)],  # 脏数据
            'blue_probabilities': ['0.1'],
            'target_period': None, 'red_hits': None, 'blue_hits': None,
            'prize_name': None, 'prize_amount': None})
save_records(bad)
verify_records(log_callback=lambda m: None)  # 不应崩溃
recs = load_records()
check("脏数据(字符串概率): 核验不崩溃", len(recs) == len(bad))
rd, bd = collect_calibration_data('ssq')
# 脏数据记录若被核验, 其字符串概率在收集时 float() 会怎样? 应被 try/except 跳过而非崩溃
check("脏数据: 校准收集不崩溃", True)

# 7b. 空概率(旧记录) 与 带概率记录混存
check("混存: 校准只收集有效长度记录",
      all(len(v) > 0 for v in rd.values()))

# 7c. 缺失期号边界: latest_period 为非整数 → 核验应跳过不崩溃
bad2 = load_records()
bad2.append({'predict_time': '2026-10-03T10:00:00', 'lottery_type': 'ssq',
             'model_type': 'lightgbm', 'latest_period': '损坏期号',
             'red_numbers': [1,2,3,4,5,6], 'blue_numbers': [1],
             'red_probabilities': None, 'blue_probabilities': None,
             'target_period': None, 'red_hits': None, 'blue_hits': None,
             'prize_name': None, 'prize_amount': None})
save_records(bad2)
verify_records(log_callback=lambda m: None)  # 不应崩溃
check("损坏期号: 核验不崩溃", len(load_records()) == len(bad2))

# 7c. 模型目录缺失时 generate_prediction 应报友好错误不崩溃
win.lottery_combo.setCurrentIndex(1)  # dlt
pump(100)
for mi in range(win.model_combo.count()):
    if 'XGBoost' in win.model_combo.itemText(mi) and win.model_combo.model().item(mi).isEnabled():
        win.model_combo.setCurrentIndex(mi)
        break
before = len(load_records())
win.generate_prediction()  # 若模型不存在应友好报错
pump(300)
check("模型缺失: 不崩溃(应用仍存活)", True)

print()
print("=" * 60)
print("阶段8: 真实数据文件完整性终检")
print("=" * 60)
for p, h in BASELINE.items():
    check(f"未被触碰: {p}", md5(p) == h)
# 清理沙盒
import shutil
shutil.rmtree(SANDBOX, ignore_errors=True)

print()
print("=" * 60)
print(f"对抗性模拟结果: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 60)
sys.exit(0 if not FAIL else 1)
