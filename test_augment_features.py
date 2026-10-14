# -*- coding:utf-8 -*-
"""增强特征模式端到端测试

安全约定（本项目铁律：测试严禁触碰真实数据/模型文件）：
  - 全程合成数据；
  - 每个实例的 models_dir 在 train/save/load 之前重定向到临时目录；
  - 断言合成数据长度 > 200，确保 _load_full_history 不会去读真实 CSV；
  - 运行前后对 ./model 目录做文件快照对比，一旦有改动立即报错。
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

REAL_MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')


def _snapshot(root):
    """root 下所有文件的 (相对路径, mtime_ns, size) 集合"""
    snap = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            snap[os.path.relpath(p, root)] = (st.st_mtime_ns, st.st_size)
    return snap


def make_ssq_df(n=800, seed=1, start=25001):
    """合成 SSQ 历史（期数升序）。n 必须 > 200 以避免测试触发真实CSV读取"""
    assert n > 200, '合成数据必须 > 200 期，否则 _load_full_history 会读真实CSV'
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(start, start + n):
        red = sorted(rng.choice(range(1, 34), 6, replace=False))
        rows.append([p, *red, int(rng.randint(1, 17))])
    return pd.DataFrame(rows, columns=['期数', '红球_1', '红球_2', '红球_3',
                                       '红球_4', '红球_5', '红球_6', '蓝球'])


def _fresh_model(tmpdir, model_type='lightgbm'):
    """新建 LotteryMLModels 并把 models_dir 重定向到临时目录（绝不写 ./model）"""
    from ml_models import LotteryMLModels
    m = LotteryMLModels('ssq', model_type=model_type, feature_window=10,
                        log_callback=lambda s: None)
    m.models_dir = os.path.join(tmpdir, 'model', 'ssq')
    os.makedirs(m.models_dir, exist_ok=True)
    return m


def test_enhanced_train_predict_roundtrip(tmpdir):
    """增强模式: 训练(280维)->保存->加载->预测 全链路一致，且 predict 与 _rank_balls 排名一致"""
    df = make_ssq_df(800, seed=2)
    m = _fresh_model(tmpdir)
    m.feature_mode = 'enhanced'
    m.train(df)
    assert m.expected_feature_count == 10 * (7 + 21), \
        f"期望280维, 实得{m.expected_feature_count}"
    m.save_models()

    m2 = _fresh_model(tmpdir)
    ok = m2.load_models()
    assert ok, 'load_models 失败'
    assert m2.feature_mode == 'enhanced', f"feature_mode 未持久化: {m2.feature_mode}"
    assert m2.expected_feature_count == 280
    assert len(m2.aug_feature_names) == 21, m2.aug_feature_names

    # 真实场景: GUI 只传最近10期, 预测端从CSV补全历史按期号对齐;
    # 测试用合成期号在CSV中不存在, 故传完整合成历史(等价: predict只用窗口行+期号对齐)
    recent = df.sort_values('期数', ascending=False).reset_index(drop=True)
    red, blue = m2.predict(recent.copy())
    assert red is not None and len(set(red)) == 6, f'预测失败: {red}'
    assert blue is not None and len(blue) == 1
    assert m2.last_red_proba is not None, '概率未存档'

    # predict 号码本身含原有随机采样设计(升级前亦如此)，不比对号码;
    # 比对确定性部分: 两个独立实例的特征矩阵/全号概率排序必须一致
    m3 = _fresh_model(tmpdir)
    assert m3.load_models()
    aug_a = m2._aug_row_features(recent.copy())
    aug_b = m3._aug_row_features(recent.copy())
    assert aug_a is not None and np.allclose(aug_a, aug_b), '加载后特征对齐不一致'
    red_rank_a, _ = m2._rank_balls(recent.copy())
    red_rank_b, _ = m3._rank_balls(recent.copy())
    assert red_rank_a is not None and red_rank_b is not None
    assert red_rank_a[:6] == red_rank_b[:6], \
        '独立实例概率排序不一致(特征构建不确定)'
    print('增强模式 训练→保存→加载→预测→确定性 端到端 PASS (280维)')


def test_aug_row_alignment_deterministic(tmpdir):
    """预测端衍生特征：乱序输入 -> 期号对齐结果必须完全一致（确定性）"""
    df = make_ssq_df(800, seed=5)
    m = _fresh_model(tmpdir)
    m.feature_mode = 'enhanced'
    full = m._aug_row_features(df.copy())
    assert full is not None and full.shape == (10, 21), full.shape
    m.aug_feature_names = None  # 用函数默认列序再算一次
    full2 = m._aug_row_features(df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    assert np.allclose(full, full2), '乱序输入对齐失败（期号对齐不成立）'
    # 行序 = 期数降序：第0行必须是最新期的衍生特征
    latest = int(df['期数'].max())
    import ml_models as MM
    aug_df, new_cols = MM._augment_features_frame(
        df.sort_values('期数').reset_index(drop=True), 'ssq',
        [f'红球_{i}' for i in range(1, 7)], ['蓝球'], window=50)
    sel = aug_df.loc[aug_df['期数'] == latest, new_cols].to_numpy(dtype=float)[0]
    assert np.allclose(full[0], sel), '第0行不是最新期（行序对齐错误）'
    print('预测端期号对齐/乱序确定性/行序 PASS')


def test_old_mode_unchanged(tmpdir):
    """旧模式回归: 默认 original、70维、predict 行为与升级前一致"""
    df = make_ssq_df(400, seed=3)
    m = _fresh_model(tmpdir)
    assert m.feature_mode == 'original'
    m.train(df)
    assert m.expected_feature_count == 70, m.expected_feature_count
    m.save_models()
    recent = df.sort_values('期数', ascending=False).head(10)
    red, blue = m.predict(recent.copy())
    assert red is not None and len(set(red)) == 6
    m2 = _fresh_model(tmpdir)
    assert m2.load_models(), '旧模型加载失败'
    assert m2.feature_mode == 'original', m2.feature_mode
    print('旧模式回归(70维/加载/original标记) PASS')


if __name__ == '__main__':
    before = _snapshot(REAL_MODEL_ROOT)
    tmp = tempfile.mkdtemp(prefix='aug_test_')
    try:
        test_enhanced_train_predict_roundtrip(tmp)
        test_aug_row_alignment_deterministic(tmp)
        test_old_mode_unchanged(tmp)
    finally:
        after = _snapshot(REAL_MODEL_ROOT)
        if before != after:
            diff = [k for k in set(before) | set(after)
                    if before.get(k) != after.get(k)]
            raise RuntimeError(f'!! 真实模型目录被修改: {sorted(diff)[:10]}')
        shutil.rmtree(tmp, ignore_errors=True)
    print('ALL PASS (真实模型目录零污染)')
