# -*- coding:utf-8 -*-
"""预测兜底路径安全性测试。

三处历史事故都在这里：
1. 2026-10-21 的 4 条大乐透存档出现后区 13（合法范围 1-12）——跨彩种复用模型实例时
   概率图的键越界，`_pad_ranked` 照单全收，非法号码被入库并参与权重回测。
2. 集成引擎 V2 在子模型名与权重表对不上时返回空概率图，调用方不检查，
   `_select_numbers`/`_pad_ranked` 就把 1..N 补齐 —— 存档里出现整批
   [1,2,3,4,5,6]、概率 0 的"模型根本没说过"的注（2026-10-16 红球均值只有 5.86）。
3. 补位一律从最小号开始，进一步把号码压向小号段。
"""
import numpy as np
import pytest

import ml_models
from ml_models import LotteryMLModels

from test_prediction_invariants import (  # noqa: E402
    _FakeClassProba, _increasing_probs, _make_df, _make_model,
)


def test_rank_balls_drops_out_of_range_balls():
    """概率图里越界的类别索引必须丢掉，绝不能变成合法号码之外的注。"""
    model = _make_model('dlt')
    # 蓝球只有 1-12；索引 12（=号码13）是跨彩种串号的真实形态
    red_map = {i: 0.01 for i in range(35)}
    blue_map = {i: 0.01 for i in range(14)}
    model._get_probability_maps = lambda recent: (red_map, blue_map)

    red_ranked, blue_ranked = model._rank_balls(_make_df(lottery='dlt'))

    assert blue_ranked is not None
    assert max(blue_ranked) <= 12, f"后区出现非法号码: {blue_ranked}"
    assert len(blue_ranked) == 12
    assert max(red_ranked) <= 35


def test_pad_ranked_refuses_to_invent_a_whole_ticket():
    """没有任何概率时不补位——造一注模型没说过的号比不预测更糟。"""
    assert LotteryMLModels._pad_ranked([], 6, 33) == []


def test_pad_ranked_stays_in_range_and_unique():
    padded = LotteryMLModels._pad_ranked([11, 12], 5, 12)
    assert len(padded) == 5
    assert len(set(padded)) == 5
    assert all(1 <= n <= 12 for n in padded), padded
    assert set([11, 12]) <= set(padded), "已排好序的号码必须原样保留"


def test_pad_ranked_does_not_always_take_the_lowest_numbers():
    """补位要铺开：历史上它永远从 1 开始，把整批注压进小号段。"""
    first = LotteryMLModels._pad_ranked([10], 6, 33)
    fillers = [n for n in first if n != 10]
    assert fillers != [1, 2, 3, 4, 5], f"补位仍是最小号段: {fillers}"
    assert max(fillers) - min(fillers) >= 10, f"补位没有铺开: {fillers}"


def test_pad_ranked_is_reproducible():
    a = LotteryMLModels._pad_ranked([7, 21], 8, 33)
    b = LotteryMLModels._pad_ranked([7, 21], 8, 33)
    assert a == b


def test_ensemble_falls_back_when_engine_returns_no_probabilities():
    """引擎给不出概率时回退旧算法，而不是把 1..6 当预测结果。"""
    model = _make_model('ssq')
    model.model_type = 'ensemble'
    model.models['red'] = {'gbdt': _FakeClassProba(_increasing_probs(33))}
    model.models['blue'] = {'gbdt': _FakeClassProba(_increasing_probs(16))}

    class _EmptyEngine:
        def __init__(self, data_dir='.'):
            pass

        def predict_ensemble(self, **kwargs):
            return {'red_proba': {}, 'blue_proba': {}, 'ensemble_id': 'e1'}

    model._ensemble_engine = _EmptyEngine()
    logs = []
    model.log = lambda msg: logs.append(str(msg))

    red, blue = model.predict(_make_df(), variation=0)

    assert red == [28, 29, 30, 31, 32, 33], f"空概率图没有回退: {red}"
    assert blue == [16]
    assert any('回退' in m for m in logs), logs


def test_predict_compound_without_probabilities_yields_no_fabricated_ticket():
    """概率不可用时复式宁可不出号，也不能出 [1..6] 这种模型没说过的注。"""
    model = _make_model('ssq')
    model._rank_balls = lambda recent: ([], [])

    red, blue = model.predict_compound(_make_df(), extra_red=2, extra_blue=1)

    assert not red and not blue, f"复式造出了伪号: {red} {blue}"


def test_predict_dantuo_without_probabilities_yields_no_fabricated_ticket():
    model = _make_model('ssq')
    model._rank_balls = lambda recent: ([], [])

    result = model.predict_dantuo(_make_df(), red_dan_count=2, red_tuo_count=6)

    assert not any(result), f"胆拖造出了伪号: {result}"
