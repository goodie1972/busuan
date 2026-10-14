# -*- coding: utf-8 -*-
"""
retrain_models_with_feedback 沙盒端到端测试（第6项验收）
- 写入全部重定向到 temp 沙盒：模型目录 / feature_analysis / model_weights.json
- retrain_reports 新产物事后移入沙盒；真实 prediction_history.json/csv 只读
- 测试前后对真实关键文件与 model/dlt 目录做快照，校验零污染
"""
import os, sys, json, hashlib, tempfile, time, shutil, glob as globmod

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

REAL_FILES = ['model_weights.json',
              os.path.join('feature_analysis', 'selected_features.json'),
              os.path.join('feature_analysis', 'feature_importance.json'),
              os.path.join('feature_analysis', 'drift_detection.json'),
              'prediction_history.json']

def snapshot():
    snap = {}
    for f in REAL_FILES:
        if os.path.exists(f):
            snap[f] = hashlib.md5(open(f, 'rb').read()).hexdigest()
    msnap = {}
    for root, _, files in os.walk(os.path.join('model', 'dlt')):
        for fn in files:
            p = os.path.join(root, fn)
            msnap[p] = os.path.getsize(p)
    reps = {p: os.path.getsize(p) for p in globmod.glob(os.path.join('retrain_reports', '*.json'))}
    return snap, msnap, reps

snap0, msnap0, reps0 = snapshot()

SANDBOX = tempfile.mkdtemp(prefix='retrain_e2e_')
print(f"沙盒目录: {SANDBOX}")

# ===== 重定向所有写入 =====
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

# ===== 运行重训练（真实代码路径: LotteryMLModels.train）=====
from retrain_models_with_feedback import retrain_lottery_models

t0 = time.time()
ok = retrain_lottery_models('dlt', use_gpu=False, model_types=['random_forest'],
                            recent_periods=300, feature_mode='enhanced')
dt = time.time() - t0
print(f"重训练返回: {ok}, 耗时 {dt:.1f}s")

# ===== 校验沙盒产物 =====
info_paths = globmod.glob(os.path.join(SANDBOX, 'model', 'dlt', '**', 'model_info.json'), recursive=True)
print(f"沙盒模型产物: {len(info_paths)} 个 model_info.json")
assert len(info_paths) >= 1, '模型未保存到沙盒'
info = json.load(open(info_paths[0], encoding='utf-8'))
print('模型信息: feature_mode=', info.get('feature_mode'), 'dim=', info.get('expected_feature_count'))
assert info.get('feature_mode') == 'enhanced', 'feature_mode 未持久化'
assert info.get('expected_feature_count', 0) > 100, '增强维度异常'

# 重训后权重刷新
assert os.path.exists(mw.WEIGHTS_FILE), '重训后权重文件未生成'
wdata = json.load(open(mw.WEIGHTS_FILE, encoding='utf-8'))
sub = (wdata.get('dlt') or {}).get('ensemble_sub_weights') or {}
print('重训后 dlt 权重:', {k: round(v, 3) for k, v in sub.items()})

# 特征选择/报告
fa_files = os.listdir(fa.FEATURE_ANALYSIS_DIR)
print(' kbh沙盒特征分析产物:', fa_files)
assert 'feature_importance.json' in fa_files or 'selected_features.json' in fa_files

# 新报告文件移入沙盒（retrain_dir 未重定向，允许生成后搬移）
retrain_dir = os.path.join(HERE, 'retrain_reports')
new_reports = [p for p in globmod.glob(os.path.join(retrain_dir, '*.json')) if p not in reps0]
for p in new_reports:
    shutil.move(p, os.path.join(SANDBOX, os.path.basename(p)))
    print('报告已移入沙盒:', os.path.basename(p))
if os.path.isdir(retrain_dir) and not os.listdir(retrain_dir):
    os.rmdir(retrain_dir)  # 本次新建的空目录删掉

# 预测可用性：加载沙盒模型 + 真实数据(只读)预测
m = ml_models.LotteryMLModels.__new__(ml_models.LotteryMLModels)
m2 = ml_models.LotteryMLModels('dlt', model_type='random_forest', log_callback=lambda s: None)
m2.models_dir = os.path.join(SANDBOX, 'model', 'dlt')
loaded = m2.load_models()
print('加载沙盒模型:', loaded, 'feature_mode=', m2.feature_mode)
assert loaded and m2.feature_mode == 'enhanced'
from backtest import _load_csv_data
df = _load_csv_data('dlt')
recent = df.sort_values('期数', ascending=False).head(m2.feature_window)
red, blue = m2.predict(recent)
print('预测(真实数据只读+沙盒模型):', red, blue)
assert red is not None and len(set(red)) == 5, '预测失败'

# ===== 零污染校验 =====
snap1, msnap1, reps1 = snapshot()
bad = [f for f in snap0 if snap0[f] != snap1.get(f)]
assert not bad, f'真实文件被修改: {bad}'
added = set(msnap1) - set(msnap0)
removed = set(msnap0) - set(msnap1)
changed = [p for p in (set(msnap0) & set(msnap1)) if msnap0[p] != msnap1[p]]
print('真实 model/dlt 变化: 新增', len(added), '删除', len(removed), '修改', len(changed))
assert not added and not removed and not changed, f'真实模型目录被污染: {added | removed | set(changed)}'
rep_added = set(reps1) - set(reps0)
rep_removed = set(reps0) - set(reps1)
assert not rep_added and not rep_removed, f'retrain_reports 未清理干净: {rep_added | rep_removed}'

print('\nRETRAIN E2E ALL PASS (沙盒隔离, 真实数据零污染)')
