# -*- coding: utf-8 -*-
"""
对抗性测试: 权重/特征/重训链路的损坏文件、边界参数、降级回退
写入全部重定向沙盒，真实数据只读，结束校验零污染。
"""
import os, sys, json, hashlib, tempfile, shutil, glob as globmod

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

REAL_FILES = ['prediction_history.json', 'model_weights.json',
              os.path.join('feature_analysis', 'selected_features.json'),
              os.path.join('feature_analysis', 'feature_importance.json'),
              os.path.join('feature_analysis', 'drift_detection.json')]

def snap():
    d = {f: (hashlib.md5(open(f, 'rb').read()).hexdigest() if os.path.exists(f) else None) for f in REAL_FILES}
    m = {}
    for root, _, files in os.walk('model'):
        for fn in files:
            p = os.path.join(root, fn)
            m[p] = os.path.getsize(p)
    r = {p: os.path.getsize(p) for p in globmod.glob(os.path.join('retrain_reports', '*.json'))}
    return d, m, r

snap0, m0, r0 = snap()
SANDBOX = tempfile.mkdtemp(prefix='adv_w_')
print(f"沙盒: {SANDBOX}")

def ok(msg): print(f"  [PASS] {msg}", flush=True)

# ===== 重定向 =====
import feature_analysis as fa
fa.FEATURE_ANALYSIS_DIR = os.path.join(SANDBOX, 'feature_analysis')
fa.FEATURE_SELECTION_FILE = os.path.join(fa.FEATURE_ANALYSIS_DIR, 'selected_features.json')
fa.FEATURE_IMPORTANCE_FILE = os.path.join(fa.FEATURE_ANALYSIS_DIR, 'feature_importance.json')
fa.DRIFT_DETECTION_FILE = os.path.join(fa.FEATURE_ANALYSIS_DIR, 'drift_detection.json')
os.makedirs(fa.FEATURE_ANALYSIS_DIR, exist_ok=True)

import model_weights as mw
mw.WEIGHTS_FILE = os.path.join(SANDBOX, 'model_weights.json')

import ml_models
_orig_init = ml_models.LotteryMLModels.__init__
def patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.models_dir = os.path.join(SANDBOX, 'model', self.lottery_type)
    os.makedirs(self.models_dir, exist_ok=True)
ml_models.LotteryMLModels.__init__ = patched_init

# ===== A1: model_weights.json 损坏 → 等权回退不崩 =====
with open(mw.WEIGHTS_FILE, 'w', encoding='utf-8') as f:
    f.write('{ this is not json !!!')
w = mw.get_model_weights('dlt')
assert w == {}, f'损坏权重文件应返回空dict, 实际 {w}'
rep = mw.get_performance_report('dlt')  # 无有效文件→即时计算(真实记录只读)
assert 'models' in rep
ok('A1 权重文件损坏→安全回退(空dict/即时计算)')

# ===== A2: selected_features.json 损坏 → retrain 容错 =====
with open(fa.FEATURE_SELECTION_FILE, 'w', encoding='utf-8') as f:
    f.write('garbage')
sel = fa.load_feature_selection()
assert isinstance(sel, dict), 'load_feature_selection 应容错返回dict'
ok('A2 特征选择文件损坏→容错')

# ===== A3: retrain recent_periods 超界/为1 → 钳制全量 =====
from retrain_models_with_feedback import retrain_lottery_models
ok_ = retrain_lottery_models('dlt', use_gpu=False, model_types=['random_forest'],
                             recent_periods=999999, feature_mode='original')
assert ok_ is True, 'recent超界应钳制全量并成功'
ok('A3 recent_periods=999999→钳制全量重训成功')

# ===== A4: ensemble 真实训练→权重应用→预测 =====
from ml_models import LotteryMLModels
from backtest import _load_csv_data
df = _load_csv_data('dlt')
m = LotteryMLModels('dlt', model_type='ensemble', log_callback=lambda s: None)
df_short = df.sort_values('期数', ascending=False).head(260).sort_values('期数').reset_index(drop=True)
m.train(df_short)
assert getattr(m, 'sub_weights', {}), 'ensemble 训练后应加载动态权重'
named = [n for n in m.models.get('red', {}) if n in m.sub_weights]
assert named, '子模型名应能匹配权重表'
# 预测走加权投票路径
recent = df.sort_values('期数', ascending=False).head(m.feature_window)
red, blue = m.predict(recent)
assert red is not None and len(set(red)) == 5, '加权集成预测失败'
ok(f'A4 ensemble 子模型权重训练→预测 PASS (red={red}, 权重={ {k: round(v,3) for k,v in m.sub_weights.items()} })')

# ===== A5: update_weights records=[] 边界 =====
res = mw.update_weights('dlt', records=[])
assert isinstance(res, dict)
data = json.load(open(mw.WEIGHTS_FILE, encoding='utf-8'))
sub = (data.get('dlt') or {}).get('ensemble_sub_weights') or {}
assert abs(sum(sub.values()) - 1.0) < 1e-3, '空记录时权重应归一(中性等权)'
ok('A5 空记录 update_weights→中性等权归一')

# ===== A6: 增强模式数据不足→自动回退 original =====
df_tiny = df.tail(40).reset_index(drop=True)  # 40期 < 暖机50期需求
m2 = LotteryMLModels('dlt', model_type='random_forest', log_callback=lambda s: None)
m2.feature_mode = 'enhanced'
m2._load_full_history = lambda df: df_tiny  # 模拟无_CSV_可补全: 仅剩不足暖机的历史
logs = []
m2.log_callback = lambda s: logs.append(s)
m2.train(df_tiny)
assert m2.feature_mode == 'original', '历史不足应自动回退 original'
assert any('回退' in l for l in logs), '日志应有回退提示'
ok('A6 增强模式历史不足→自动回退 original')

# ===== A7: 权重文件被核验覆盖写后再读一致(原子性冒烟) =====
from prediction_records import load_records
recs = load_records()  # 真实只读
mw.update_all_weights(recs)
d1 = json.load(open(mw.WEIGHTS_FILE, encoding='utf-8'))
assert 'ssq' in d1 and 'dlt' in d1 and d1['dlt']['ensemble_sub_weights']
ok('A7 真实核验记录→权重全彩种更新可回读')

# ===== 新报告搬移 + 零污染 =====
retrain_dir = os.path.join(HERE, 'retrain_reports')
new_reports = [p for p in globmod.glob(os.path.join(retrain_dir, '*.json')) if p not in r0]
for p in new_reports:
    shutil.move(p, os.path.join(SANDBOX, os.path.basename(p)))
if os.path.isdir(retrain_dir) and not os.listdir(retrain_dir):
    os.rmdir(retrain_dir)

snap1, m1, r1 = snap()
bad = [f for f in snap0 if snap0[f] != snap1.get(f)]
assert not bad, f'真实文件被修改: {bad}'
assert not (set(m1) - set(m0)) and not (set(m0) - set(m1)), '模型目录被污染'
assert not [p for p in (set(m0) & set(m1)) if m0[p] != m1[p]], '模型文件变化'
assert not (set(r1) - set(r0)), f'报告残留: {set(r1)-set(r0)}'

print('\nADVERSARIAL WEIGHTS/RETRAIN ALL PASS (零污染)')
