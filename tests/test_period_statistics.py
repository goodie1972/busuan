# -*- coding:utf-8 -*-
"""历史回测统计口径：按期聚合。

原口径按"注"统计中奖率。复式/胆拖存档一次就写入 C(8,6)×C(3,1)=84 注，
同一期的同一个结果被重复计了 84 次，注数越多的期对整体中奖率的权重越大，
看到的"中奖率"和"这期买不买得中"完全是两回事。

这里改为：期是样本单位，注数只作为该期的投入权重。
"""
import pytest

from prediction_records import format_verify_summary, summarize_by_period


def _note(period, red_hits=0, blue_hits=0, prize=0, lt='ssq',
          model='lightgbm', prize_name=None):
    return {
        'lottery_type': lt,
        'model_type': model,
        'latest_period': period - 1,
        'target_period': period,
        'red_hits': red_hits,
        'blue_hits': blue_hits,
        'prize_name': prize_name or ('未中奖' if prize == 0 else '六等奖'),
        'prize_amount': prize,
    }


def test_one_compound_period_counts_as_one_period():
    """复式一整票 84 注属于同一期，统计上只能算 1 期。"""
    records = [_note(26111) for _ in range(84)]
    summary = summarize_by_period(records)

    assert summary['overall']['periods'] == 1
    assert summary['overall']['notes'] == 84
    assert summary['overall']['stake'] == 84 * 2


def test_period_win_rate_uses_periods_as_denominator():
    """中奖率按期中：1 期里中 2 注 + 1 期全不中 = 50%，不是 2/85。"""
    records = [_note(26111, prize=5) , _note(26111, prize=5)]
    records += [_note(26111) for _ in range(82)]
    records.append(_note(26112))

    overall = summarize_by_period(records)['overall']

    assert overall['periods'] == 2
    assert overall['win_periods'] == 1
    assert overall['period_win_rate'] == pytest.approx(0.5)
    # 旧的按注口径仍然给出来，但要标注清楚，不能当期口径用
    assert overall['note_win_rate'] == pytest.approx(2 / 85)


def test_per_period_stake_and_return():
    """每期的投入 = 注数 × 单注价，回报 = 该期所有注的奖金之和。"""
    records = [_note(26111, prize=5) for _ in range(3)] + [_note(26111)]
    period = summarize_by_period(records)['periods_detail'][0]

    assert period['target_period'] == 26111
    assert period['notes'] == 4
    assert period['stake'] == 8
    assert period['prize'] == 15
    assert period['win_notes'] == 3
    assert period['is_win'] is True


def test_roi_aggregates_over_periods():
    records = [_note(26111, prize=10), _note(26111)]          # 投入4 回报10
    records += [_note(26112) for _ in range(5)]               # 投入10 回报0
    overall = summarize_by_period(records)['overall']

    assert overall['stake'] == 14
    assert overall['prize'] == 10
    assert overall['net'] == -4
    assert overall['roi'] == pytest.approx(-4 / 14)


def test_pending_records_excluded_from_period_stats():
    """未开奖的注不进期统计，但要单独报出来，不能悄悄丢。"""
    records = [_note(26111)]
    records.append({'lottery_type': 'ssq', 'model_type': 'lightgbm',
                    'latest_period': 26111, 'target_period': None,
                    'red_hits': None, 'blue_hits': None,
                    'prize_name': None, 'prize_amount': None})
    summary = summarize_by_period(records)

    assert summary['overall']['periods'] == 1
    assert summary['pending_notes'] == 1


def test_different_lotteries_same_period_number_are_separate_periods():
    records = [_note(26111, lt='ssq'), _note(26111, lt='dlt')]
    summary = summarize_by_period(records)

    assert summary['overall']['periods'] == 2
    assert {p['lottery_type'] for p in summary['periods_detail']} == {'ssq', 'dlt'}


def test_model_and_draw_persistence_are_not_changed():
    """约束：统计口径调整不得影响存档本身——每条注都还在，权重要按注吃。"""
    records = [_note(26111, model='gbdt') for _ in range(6)]
    summary = summarize_by_period(records)

    assert summary['overall']['notes'] == len(records)
    assert summary['periods_detail'][0]['notes'] == 6
    # 命中均值按注算，作为该期预测质量的权重信息保留
    assert summary['periods_detail'][0]['red_hits_avg'] == 0


def test_format_verify_summary_reports_period_view():
    records = [_note(26111, prize=5), _note(26111)]
    report = {
        'total': 3, 'verified': 2, 'pending': 1,
        'win_count': 2, 'prize_total_fixed': 10,
        'prize_counts': {'六等奖': 2}, 'records': records,
    }
    text = format_verify_summary(report)

    assert '期' in text
    assert '1 期' in text
    assert '中奖期' in text
    assert '26111' in text


def test_format_verify_summary_handles_no_records():
    report = {'total': 0, 'verified': 0, 'pending': 0, 'win_count': 0,
              'prize_total_fixed': 0, 'prize_counts': {}, 'records': []}
    text = format_verify_summary(report)

    assert isinstance(text, str)
    assert text.strip()
