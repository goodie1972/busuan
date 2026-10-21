# -*- coding: utf-8 -*-
"""预测链路回归测试：概率选号、时间顺序拆分、评估口径。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models import LotteryMLModels, WrappedMultilabelGBDTModel


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class _FakeMultilabelGBDT(WrappedMultilabelGBDTModel):
    """借用多标签包装器的类型判定，概率由测试直接给定。"""

    def __init__(self, probabilities):
        self._probs = np.asarray(probabilities, dtype=float)
        self.n_classes = len(self._probs)
        self.num_balls = 1
        self.ball_type = 'red'
        self.model = None

    def predict_proba(self, data):
        n = data.shape[0] if data.ndim == 2 else 1
        return np.tile(self._probs, (n, 1))

    def predict(self, data):
        top = np.argsort(self._probs)[-self.num_balls:][::-1]
        return sorted([int(i) + 1 for i in top])


class _FakeClassProba:
    """模拟带 classes_ 的 sklearn 分类器。"""

    def __init__(self, probabilities):
        self._probs = np.asarray(probabilities, dtype=float)
        self.classes_ = np.arange(len(self._probs))

    def predict_proba(self, data):
        n = data.shape[0] if data.ndim == 2 else 1
        return np.tile(self._probs, (n, 1))

    def predict(self, data):
        n = data.shape[0] if data.ndim == 2 else 1
        return np.tile(np.arange(len(self._probs)), (n, 1))


def _make_df(rows=15, lottery='ssq'):
    rng = np.random.default_rng(7)
    data = {'期数': np.arange(2024001, 2024001 + rows)}
    red_count = 6 if lottery == 'ssq' else 5
    blue_count = 1 if lottery == 'ssq' else 2
    for i in range(1, red_count + 1):
        data[f'红球_{i}'] = rng.integers(1, 34 if lottery == 'ssq' else 36, rows)
    if lottery == 'ssq':
        data['蓝球'] = rng.integers(1, 17, rows)
    else:
        for i in range(1, blue_count + 1):
            data[f'蓝球_{i}'] = rng.integers(1, 13, rows)
    return pd.DataFrame(data).sort_values('期数', ascending=False).reset_index(drop=True)


def _make_model(lottery='ssq'):
    model = LotteryMLModels(lottery_type=lottery, model_type='gbdt', feature_window=10)
    model.scalers['X'] = _IdentityScaler()
    model.expected_feature_count = 70
    return model


def _increasing_probs(n):
    return np.linspace(0.001, 0.9, n)


def test_predict_uses_probability_topk_without_double_offset():
    model = _make_model('ssq')
    model.models['red'] = _FakeMultilabelGBDT(_increasing_probs(33))
    model.models['blue'] = _FakeMultilabelGBDT(_increasing_probs(16))

    red, blue = model.predict(_make_df(), variation=0)

    assert red == [28, 29, 30, 31, 32, 33]
    assert blue == [16]
    assert model.last_red_proba is not None
    assert set(model.last_red_proba) == set(range(33))
    assert set(model.last_blue_proba) == set(range(16))
    assert model.last_red_conf == pytest.approx(np.mean(_increasing_probs(33)[27:33]))


def test_ensemble_predict_uses_probability_topk():
    model = _make_model('ssq')
    model.model_type = 'ensemble'
    model.models['red'] = {
        'a': _FakeClassProba(_increasing_probs(33)),
        'b': _FakeClassProba(_increasing_probs(33)),
    }
    model.models['blue'] = {
        'a': _FakeClassProba(_increasing_probs(16)),
        'b': _FakeClassProba(_increasing_probs(16)),
    }

    red, blue = model.predict(_make_df(), variation=0)

    assert red == [28, 29, 30, 31, 32, 33]
    assert blue == [16]


def test_rank_balls_probabilities_are_zero_based():
    model = _make_model('ssq')
    red_probs = _increasing_probs(33)
    model.models['red'] = _FakeMultilabelGBDT(red_probs)
    model.models['blue'] = _FakeMultilabelGBDT(_increasing_probs(16))

    red_ranked, blue_ranked = model._rank_balls(_make_df())

    assert red_ranked[:3] == [33, 32, 31]
    assert blue_ranked[:2] == [16, 15]
    assert set(model.last_red_proba) == set(range(33))
    assert model.last_red_proba[32] == pytest.approx(red_probs[32])


def test_variation_is_reproducible_and_stays_in_probability_pool():
    model = _make_model('ssq')
    model.models['red'] = _FakeMultilabelGBDT(_increasing_probs(33))
    model.models['blue'] = _FakeMultilabelGBDT(_increasing_probs(16))
    df = _make_df()

    baseline = [model.predict(df, variation=0)[0] for _ in range(3)]
    assert baseline[0] == baseline[1] == baseline[2] == [28, 29, 30, 31, 32, 33]

    varied = [model.predict(df, variation=1)[0] for _ in range(3)]
    assert varied[0] == varied[1] == varied[2]
    assert len(set(varied[0])) == 6
    # 12个候选池 = 单式6个 + 额外6个；概率递增时池内最小号码为22
    assert min(varied[0]) >= 22
    assert max(varied[0]) <= 33


def test_prepare_data_uses_chronological_split_and_train_only_scaler():
    rng = np.random.default_rng(11)
    rows = 80
    df = _make_df(rows=rows, lottery='ssq')
    for col in df.columns:
        if col == '期数':
            continue
        if col.startswith('红球_') or col.startswith('蓝球_') or col == '蓝球':
            df[col] = rng.normal(10, 5, rows).astype(int)
        else:
            df[col] = rng.normal(10, 5, rows)

    model = _make_model('ssq')
    X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = \
        model.prepare_data(df, test_size=0.25)

    n_samples = rows - model.feature_window
    n_test = max(1, int(n_samples * 0.25))
    n_train = n_samples - n_test

    assert X_train.shape[0] == n_train
    assert X_test.shape[0] == n_test
    assert y_red_train.shape == (n_train, 33)
    assert y_red_test.shape == (n_test, 33)
    assert model.data_split['method'] == 'chronological'
    assert model.data_split['train_samples'] == n_train
    assert np.allclose(model.scalers['X'].mean_, model._full_X_raw[:n_train].mean(axis=0))
    assert not np.allclose(model.scalers['X'].mean_, model._full_X_raw.mean(axis=0))


def test_evaluate_supports_all_multilabel_models():
    model = _make_model('ssq')
    model.models['red'] = _FakeMultilabelGBDT(_increasing_probs(33))
    model.models['blue'] = _FakeMultilabelGBDT(_increasing_probs(16))

    n = 5
    X_test = np.zeros((n, 70))
    y_red_test = np.zeros((n, 33), dtype=int)
    y_blue_test = np.zeros((n, 16), dtype=int)
    y_red_test[:, 27:33] = 1
    y_blue_test[:, 15] = 1

    metrics = model.evaluate(X_test, y_red_test, y_blue_test)

    assert metrics['samples'] == n
    assert metrics['red']['avg_hits'] == pytest.approx(6.0)
    assert metrics['blue']['avg_hits'] == pytest.approx(1.0)
    assert metrics['red']['exact_rate'] == pytest.approx(1.0)
