# -*- coding:utf-8 -*-
"""复式/胆拖取注顺序测试。

存档路径原先把组合按 itertools.combinations 的字典序物化后取前 N 组，于是
"红球 1-16 选 6"永远先存档 [1,2,3,4,5,6]、[1,2,3,4,5,7]…… 号码整体压向小号段，
而模型概率完全没参与排序（2026-10-16 的存档红球均值只有 5.86，正常应接近 17）。
现在按"整注概率 = 注内各球概率乘积"降序取前 N 注。
"""
from math import comb

import pandas as pd
import pytest

import thread_utils
from ml_models import ball_proba_list, top_note_combos


def _score(combo, proba):
    p = 1.0
    for n in combo:
        p *= float((proba or {}).get(n - 1, 0.0))
    return p


def _proba(high_numbers, high, low):
    """给 1..33 赋两档概率，high_numbers 用高概率。"""
    return {n - 1: (high if n in high_numbers else low) for n in range(1, 34)}


def test_top_note_combos_orders_by_note_probability_not_lexicographic():
    red = list(range(1, 17))                      # 复式 16 个红球
    blue = [7]
    red_proba = _proba({11, 12, 13, 14, 15, 16}, high=0.20, low=0.01)
    blue_proba = {6: 0.5}

    notes = top_note_combos(red, blue, 6, 1, red_proba, blue_proba, count=3)

    assert len(notes) == 3
    first_red, first_blue = notes[0]
    assert first_blue == (7,)
    # 第一注必须是候选组合里概率乘积最大的那一组
    top_score = max(_score(c, red_proba) for c in _all_combos(red, 6))
    assert _score(first_red, red_proba) == pytest.approx(top_score)
    # 决不是字典序第一组 [1..6]
    assert list(first_red) != [1, 2, 3, 4, 5, 6]


def test_top_note_combos_is_optimal_and_deterministic():
    red = list(range(1, 12))
    blue = [1, 2, 3]
    red_proba = _proba(set(range(6, 12)), high=0.15, low=0.02)
    blue_proba = {0: 0.1, 1: 0.2, 2: 0.7}

    a = top_note_combos(red, blue, 6, 1, red_proba, blue_proba, count=8)
    b = top_note_combos(red, blue, 6, 1, red_proba, blue_proba, count=8)
    assert a == b, "同一批概率必须给出同一批注（可复现）"

    scores = [_score(r, red_proba) * _score(blue, blue_proba) for r, blue in a]
    assert scores == sorted(scores, reverse=True), "输出必须按整注概率降序"

    # 取前 8 注的分数下界 >= 任意未取注的上界（真正的前 N，而非字典序前 N）
    all_scores = sorted(
        (_score(r, red_proba) * _score((bl,), blue_proba)
         for r in _all_combos(red, 6) for bl in blue), reverse=True)
    assert scores[-1] == pytest.approx(all_scores[7])


def test_top_note_combos_without_probabilities_falls_back_to_combinations():
    red = list(range(1, 9))
    blue = [3, 4]
    notes = top_note_combos(red, blue, 6, 1, None, None, count=4)
    assert len(notes) == 4
    assert notes[0] == ((1, 2, 3, 4, 5, 6), (3,))
    assert len({(r, b) for r, b in notes}) == 4


def test_top_note_combos_caps_at_total_notes():
    red = [1, 2, 3, 4]
    blue = [5]
    notes = top_note_combos(red, blue, 4, 1, None, None, count=10)
    assert len(notes) == comb(4, 4) * 1 == 1


def _all_combos(items, k):
    from itertools import combinations
    return list(combinations(items, k))


def test_ball_proba_list_maps_numbers_via_zero_based_keys():
    assert ball_proba_list({5: 0.2, 6: 0.4}, [6, 7]) == [0.2, 0.4]
    # 概率里没出现的号码按 0 计（正是 _pad_ranked 补位号码的签名）
    assert ball_proba_list({5: 0.2}, [6, 9]) == [0.2, 0.0]


def test_ball_proba_list_returns_none_when_unusable():
    assert ball_proba_list([0.1, 0.2], [1]) is None      # 期望值模型给的是列表
    assert ball_proba_list(None, [1]) is None
    assert ball_proba_list({}, [1]) is None
    assert ball_proba_list({0: 0.1}, []) is None


def test_top_note_combos_supports_empty_side_for_dantuo():
    """胆拖里胆码已占满名额时，该侧只有一个空组合，仍要正常出注。"""
    notes = top_note_combos([], [3, 4], 0, 1, None, {2: 0.7}, count=2)
    assert notes == [((), (3,)), ((), (4,))]


def test_top_note_combos_tolerates_non_dict_proba():
    """期望值模型的 last_*_proba 是列表，不能因此报错或断档。"""
    notes = top_note_combos([1, 2, 3, 4], [9], 2, 1, [0.1, 0.2], [0.5], count=2)
    assert len(notes) == 2
    assert notes[0] == ((1, 2), (9,))


# ---------------------------------------------------------------- 线程集成
# 用假模型驱动 AutoPredictThread.run()，验证"一键预测"真正存档的那几注
# 是按概率挑出来的，而不是字典序前 N 注。

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeMLModel:
    """只提供线程用到的接口：号码数、复式/胆拖出号、逐球概率。"""

    def __init__(self, lottery_type='ssq', model_type='gbdt',
                 log_callback=None, use_gpu=False):
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.red_count, self.blue_count = 6, 1
        self.feature_window = 3
        # 概率明显偏向 20+ 的大号，小号只有边缘概率
        self.last_red_proba = {n - 1: (0.12 if n >= 20 else 0.004)
                               for n in range(1, 34)}
        self.last_blue_proba = {n - 1: (0.30 if n == 15 else 0.01)
                                 for n in range(1, 17)}

    def load_models(self):
        return True

    def predict_fushi(self, recent, red_count=0, blue_count=0):
        return [20, 21, 22, 23, 24, 25, 26, 27], [14, 15, 16]

    def predict_dantuo(self, recent, red_dan_count=0, red_tuo_count=0,
                       blue_dan_count=0, blue_tuo_count=0):
        return ([26, 27], [20, 21, 22, 23, 24, 25], [], [14, 15, 16])


def _run_thread(qapp, monkeypatch, mode, num_predictions=3):
    import scripts.data_analysis as da
    monkeypatch.setattr(thread_utils, 'LotteryMLModels', _FakeMLModel)
    monkeypatch.setattr(da, 'load_lottery_data',
                        lambda lt: pd.DataFrame({'期数': [26116, 26117, 26118]}))
    captured = {}
    thread = thread_utils.AutoPredictThread(
        'ssq', model_type='gbdt', num_predictions=num_predictions,
        skip_fetch=True, skip_train=True, predict_mode=mode)
    thread.predictions_signal.connect(lambda p: captured.setdefault('notes', p))
    thread.finished_signal.connect(
        lambda ok, text: captured.update(ok=ok, text=text))
    thread.log_signal.connect(lambda m: None)
    thread.step_signal.connect(lambda m: None)
    thread.run()                      # 不起事件循环，同步执行
    assert captured.get('ok'), captured.get('text')
    return captured['notes'], captured['text']


def test_auto_predict_thread_archives_highest_probability_notes(qapp, monkeypatch):
    notes, text = _run_thread(qapp, monkeypatch, '复式')

    assert len(notes) == 3, "每注都要入库：仍按 num_predictions 出注，不减少条数"
    # 蓝球 15 的概率最高 → 必须排在字典序第一的 14 之前
    assert (sorted(notes[0][0]), sorted(notes[0][1])) == (
        [20, 21, 22, 23, 24, 25], [15]), notes[0]
    balls = {n for red, _b, _rp, _bp in notes for n in red}
    assert all(n >= 20 for n in balls), f"仍出现被压到小号段的注: {sorted(balls)}"
    # 每注都带上了逐球概率（原先复式存的是 None，无法回溯补位号）
    assert all(rp is not None and bp is not None for _r, _b, rp, bp in notes)
    assert '84注' in text, "总注数 = C(8,6)×C(3,1) = 28×3"


def test_auto_predict_thread_dantuo_archives_by_probability(qapp, monkeypatch):
    notes, _text = _run_thread(qapp, monkeypatch, '胆拖', num_predictions=2)

    assert len(notes) == 2
    for red, blue, _rp, _bp in notes:
        assert set([26, 27]) <= set(red), "胆码必须固定在每一注里"
        assert len(red) == 6 and len(blue) == 1
        assert all(n >= 20 for n in red if n not in (26, 27))
