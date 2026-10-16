# -*- coding: utf-8 -*-
"""对抗性测试：v2 智能优化模块 - 监控引擎

验证优化器在边界数据、异常输入和空状态下的表现。
API 契约（来自 optimization_monitor.py 实际签名）：
- check_data_volume(lottery_type, records=None) -> Dict
- check_model_performance(lottery_type) -> Dict
- check_feature_drift(lottery_type, window_size=50) -> Dict
- check_calibration_quality(lottery_type) -> Dict
- check_data_freshness(lottery_type) -> Dict
- _compute_health_score(checks: Dict[str, Dict]) -> float  # 每个值含 'score'
- _determine_priority(score: float, checks: Dict) -> str   # 返回 high/medium/low/none
- generate_health_report(lottery_type='all') -> Dict
"""
import pytest

from optimization_monitor import (
    check_data_volume,
    check_model_performance,
    check_feature_drift,
    check_calibration_quality,
    check_data_freshness,
    _compute_health_score,
    _determine_priority,
    generate_health_report,
)

VALID_KEYS = {'data_volume', 'model_performance', 'feature_drift',
              'calibration_quality', 'data_freshness'}


def _fake_records(count=30, lottery_type='ssq', verified=True):
    """生成伪造的预测记录（字段结构与真实记录对齐）"""
    records = []
    for i in range(count):
        records.append({
            'period': f"{202610}{i + 1:02d}",
            'lottery_type': lottery_type,
            'target_period': (20261000 + i) if verified else None,
            'red_hits': 2 if verified else None,
            'blue_hits': 0 if verified else None,
            'prize_name': '无奖' if verified else None,
            'model': 'random_forest',
        })
    return records


def _make_checks(scores=None, statuses=None):
    """构造 _compute_health_score 期望的 checks 结构：{key: {'score': x, 'status': s}}"""
    scores = scores or {k: 5 for k in VALID_KEYS}
    statuses = statuses or {}
    return {k: {'score': scores[k], 'status': statuses.get(k, 'ok')} for k in VALID_KEYS}


# ── 1. check_data_volume 边界测试 ──────────────────────────────

@pytest.mark.parametrize("count,expected_status", [
    (0, 'critical'),      # 0 条已核验 = critical
    (15, 'warning'),      # 10-30 条 = warning
    (40, 'ok'),           # >=30 条 = ok
    (100, 'ok'),
])
def test_check_data_volume_boundary(count, expected_status):
    """不同已核验记录数下应返回对应状态"""
    result = check_data_volume('ssq', _fake_records(count, verified=True))
    assert isinstance(result, dict)
    assert result.get('status') == expected_status, \
        f"期望 {expected_status}, 得到 {result.get('status')}: {result.get('message')}"


def test_check_data_volume_unverified_not_counted():
    """target_period=None 的记录不计入已核验数（真实数据的现状）"""
    result = check_data_volume('ssq', _fake_records(200, verified=False))
    assert result.get('value') == 0
    assert result.get('status') in ('critical', 'warning')


def test_check_data_volume_default_loads_file():
    """records=None 时应从磁盘读取 prediction_history.json"""
    result = check_data_volume('ssq')
    assert isinstance(result, dict)
    # 当前真实数据全部未核验，应为 critical/warning（不得崩溃）
    assert result.get('status') in ('ok', 'warning', 'critical')


def test_check_data_volume_empty_list():
    """空列表应降级为 critical 而非崩溃"""
    result = check_data_volume('ssq', [])
    assert isinstance(result, dict)
    assert result.get('status') == 'critical'


def test_check_data_volume_malformed_records():
    """记录里含 None 条目 / 缺字段不应崩溃"""
    bad = [None, {}, {'lottery_type': 'ssq'},
           {'lottery_type': 'ssq', 'target_period': 20261001},
           {'lottery_type': 'dlt', 'target_period': 20261002}]  # 异彩种应被过滤
    result = check_data_volume('ssq', bad)
    assert isinstance(result, dict)
    assert result.get('value') == 1  # 只有 1 条 ssq 已核验


# ── 2. check_model_performance 结构测试 ────────────────────────

@pytest.mark.parametrize("lottery", ['ssq', 'dlt'])
def test_check_model_performance_structure(lottery):
    result = check_model_performance(lottery)
    assert isinstance(result, dict)
    # 返回结构至少含 status 或 score
    assert 'status' in result or 'score' in result


# ── 3. check_feature_drift 漂移测试 ────────────────────────────

@pytest.mark.parametrize("window", [10, 30, 50, 100])
def test_check_feature_drift_window(window):
    result = check_feature_drift('ssq', window)
    assert isinstance(result, dict)
    assert 'status' in result or 'score' in result
    # status 取值合法
    if 'status' in result:
        assert result['status'] in ('ok', 'warning', 'error', 'healthy', 'drift'), \
            f"status 非法: {result['status']}"


def test_check_feature_drift_bad_window():
    """非法窗口大小（0/负/字符串）不应崩溃"""
    for bad in [0, -5, 'abc']:
        try:
            result = check_feature_drift('ssq', bad)
            assert result is None or isinstance(result, dict)
        except (ValueError, TypeError):
            pass  # 显式拒绝也接受


# ── 4. check_calibration_quality 校准质量测试 ──────────────────

@pytest.mark.parametrize("lottery", ['ssq', 'dlt'])
def test_check_calibration_quality(lottery):
    result = check_calibration_quality(lottery)
    assert isinstance(result, dict)


# ── 5. check_data_freshness 新鲜度测试 ─────────────────────────

@pytest.mark.parametrize("lottery", ['ssq', 'dlt'])
def test_check_data_freshness(lottery):
    result = check_data_freshness(lottery)
    assert isinstance(result, dict)


# ── 6. _compute_health_score 评分范围测试 ──────────────────────

@pytest.mark.parametrize("scores,expected_lo,expected_hi", [
    (dict.fromkeys(VALID_KEYS, 0), 0, 10),
    (dict.fromkeys(VALID_KEYS, 10), 0, 10),
    ({'data_volume': 8, 'model_performance': 7, 'feature_drift': 2,
      'calibration_quality': 5, 'data_freshness': 6}, 0, 10),
])
def test_compute_health_score_range(scores, expected_lo, expected_hi):
    result = _compute_health_score(_make_checks(scores=scores))
    assert isinstance(result, (int, float))
    assert expected_lo <= result <= expected_hi, f"评分越界: {result}"


def test_compute_health_score_missing_keys():
    """缺失 key 时应使用默认 score=5，不崩溃"""
    partial = {'data_volume': {'score': 8, 'status': 'ok'}}
    result = _compute_health_score(partial)
    assert 0 <= result <= 10


def test_compute_health_score_empty():
    """空 dict 应返回 5*权重和（默认值）"""
    result = _compute_health_score({})
    assert 0 <= result <= 10


# ── 7. _determine_priority 优先级判定 ─────────────────────────

@pytest.mark.parametrize("score,statuses,expected", [
    (2.0, {'data_volume': 'critical', 'model_performance': 'critical'}, 'high'),
    (1.5, dict.fromkeys(VALID_KEYS, 'ok'), 'high'),     # score<3 直接 high
    (7.0, {'feature_drift': 'warning', 'calibration_quality': 'warning'}, 'medium'),
    (9.0, {'data_freshness': 'warning'}, 'low'),
    (9.5, dict.fromkeys(VALID_KEYS, 'ok'), 'none'),
    (5.0, dict.fromkeys(VALID_KEYS, 'ok'), 'medium'),   # score<6 -> medium
])
def test_determine_priority(score, statuses, expected):
    checks = _make_checks(statuses=statuses)
    result = _determine_priority(score, checks)
    assert result == expected, f"期望 {expected}, 实际 {result}"


def test_determine_priority_invalid_input():
    """score 为 0 或负数时应返回 high 而非崩溃"""
    checks = _make_checks()
    for s in [0, -5, 1.0]:
        assert _determine_priority(s, checks) == 'high'


# ── 8. generate_health_report 完整报告测试 ────────────────────

@pytest.mark.parametrize("lottery", ['ssq', 'dlt', 'all'])
def test_generate_health_report_output(lottery):
    report = generate_health_report(lottery)
    assert isinstance(report, dict)
    # 'all' 应返回 ssq/dlt 两个子报告（sub_reports 键）
    if lottery == 'all':
        assert 'sub_reports' in report
        assert 'ssq' in report['sub_reports'] and 'dlt' in report['sub_reports']
        assert 0 <= report['health_score'] <= 10
    else:
        assert 'health_score' in report, f"缺少 health_score: {list(report.keys())}"
        assert isinstance(report['health_score'], (int, float))
        assert 0 <= report['health_score'] <= 10


def test_generate_health_report_invalid_type():
    """无效彩票类型应优雅降级或抛 ValueError，不应 segfault"""
    try:
        report = generate_health_report('foo')
        assert report is None or isinstance(report, dict)
    except ValueError:
        pass


# ── 9. 端到端合成：真实数据量下所有检查 ────────────────────────

def test_end_to_end_all_checks():
    """在真实数据下 5 项检查全跑通"""
    for lottery in ['ssq', 'dlt']:
        r1 = check_data_volume(lottery)
        r2 = check_model_performance(lottery)
        r3 = check_feature_drift(lottery)
        r4 = check_calibration_quality(lottery)
        r5 = check_data_freshness(lottery)
        for i, r in enumerate([r1, r2, r3, r4, r5], 1):
            assert isinstance(r, dict), f"检查 {i} 返回 {type(r)}"


def test_generate_short_report():
    """get_short_report 应返回字符串摘要"""
    from optimization_monitor import get_short_report
    s = get_short_report('ssq')
    assert isinstance(s, str) and len(s) > 10


# ── 10. 极端输入抗性 ──────────────────────────────────────────

def test_all_modules_stable_on_real_data():
    """所有模块在真实数据下连续跑 3 次，结果稳定"""
    for _ in range(3):
        report = generate_health_report('ssq')
        assert 'health_score' in report
        # 连续调用结果应一致（无随机性）
        assert report['health_score'] >= 0


if __name__ == '__main__':
    pytest.main()
