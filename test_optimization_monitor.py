# -*- coding: utf-8 -*-
"""
optimization_monitor.py 单元测试
=================================
6 项测试覆盖 5 项检查 + 健康度评分 + 优先级判定
"""
import os
import sys
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_check_data_volume():
    """测试数据量检查"""
    from optimization_monitor import check_data_volume
    # 模拟记录
    records = [
        {'lottery_type': 'ssq', 'target_period': 2026305,
         'red_numbers': [1, 2, 3, 4, 5, 6]},
    ] * 35
    records.append({'lottery_type': 'dlt', 'target_period': 2026305})  # 不同彩种

    result = check_data_volume('ssq', records)
    assert result['status'] == 'ok', f'35 verified should be ok: {result}'
    assert result['score'] == 10
    assert result['value'] == 35

    # 偏少
    records_small = records[:15]
    result = check_data_volume('ssq', records_small)
    assert result['status'] == 'warning', f'15 should be warning: {result}'
    assert result['score'] == 5

    # 严重不足
    records_tiny = records[:5]
    result = check_data_volume('ssq', records_tiny)
    assert result['status'] == 'critical', f'5 should be critical: {result}'
    assert result['score'] == 0

    print('  [OK] check_data_volume: ok/warn/critical all correct')


def test_check_model_performance():
    """测试模型表现检查"""
    from optimization_monitor import check_model_performance
    result = check_model_performance('ssq')
    assert 'status' in result
    assert 'score' in result
    assert 'details' in result
    assert 'message' in result
    # 当前数据少，应为 critical 或 warning
    assert result['status'] in ('critical', 'warning'), f'unexpected: {result}'
    print(f'  [OK] check_model_performance: status={result["status"]}, '
          f'score={result["score"]}')


def test_check_feature_drift():
    """测试特征漂移检查"""
    from optimization_monitor import check_feature_drift
    result = check_feature_drift('dlt', window_size=50)
    assert 'status' in result
    assert 'score' in result
    assert 'max_kl' in result
    assert 'drifted_features' in result
    print(f'  [OK] check_feature_drift: status={result["status"]}, '
          f'max_kl={result["max_kl"]}')


def test_check_calibration_quality():
    """测试校准质量检查"""
    from optimization_monitor import check_calibration_quality
    result = check_calibration_quality('ssq')
    assert 'status' in result
    assert 'score' in result
    # 当前未校准
    assert result['status'] == 'warning'
    assert '尚未校准' in result['message']
    print(f'  [OK] check_calibration_quality: status={result["status"]}')


def test_check_data_freshness():
    """测试数据新鲜度检查"""
    from optimization_monitor import check_data_freshness
    result = check_data_freshness('ssq')
    assert 'status' in result
    assert 'score' in result
    assert 'last_period' in result
    # CSV 存在，应能读取
    assert result['last_period'] is not None
    print(f'  [OK] check_data_freshness: status={result["status"]}, '
          f'last_period={result["last_period"]}')


def test_generate_health_report():
    """测试完整健康报告生成"""
    from optimization_monitor import (
        generate_health_report, get_short_report,
        _compute_health_score, _determine_priority
    )

    # 单彩种
    report = generate_health_report('ssq')
    assert 'health_score' in report
    assert 'priority' in report
    assert 'checks' in report
    assert len(report['checks']) == 5
    assert 0 <= report['health_score'] <= 10

    # all
    report_all = generate_health_report('all')
    assert 'sub_reports' in report_all
    assert 'ssq' in report_all['sub_reports']
    assert 'dlt' in report_all['sub_reports']

    # 短报告
    short = get_short_report('ssq')
    assert 'SSQ' in short
    short_all = get_short_report('all')
    assert 'SSQ' in short_all and 'DLT' in short_all

    # 优先级逻辑
    checks_all_crit = {
        'a': {'status': 'critical', 'score': 0},
        'b': {'status': 'critical', 'score': 0},
    }
    assert _determine_priority(1.0, checks_all_crit) == 'high'

    checks_all_ok = {
        'a': {'status': 'ok', 'score': 10},
        'b': {'status': 'ok', 'score': 10},
    }
    assert _determine_priority(10.0, checks_all_ok) == 'none'

    print(f'  [OK] generate_health_report: score={report["health_score"]}, '
          f'priority={report["priority"]}')
    print(f'  [OK] get_short_report: {short}')


def test_edge_cases():
    """测试边缘情况"""
    from optimization_monitor import (
        check_data_volume, _estimate_days_since, _compute_health_score
    )

    # 空记录
    result = check_data_volume('ssq', [])
    assert result['status'] == 'critical'
    assert result['value'] == 0

    # target_period 为 None（未核验）
    records = [
        {'lottery_type': 'ssq', 'target_period': None},
        {'lottery_type': 'ssq', 'target_period': None},
    ]
    result = check_data_volume('ssq', records)
    assert result['value'] == 0, 'None target_period should not count'

    # 期号推算
    assert _estimate_days_since('2026001') is not None  # 2026 第1天
    assert _estimate_days_since('abc') is None
    assert _estimate_days_since('123') is None

    # 健康度评分边界：空 dict → 各项默认分 5
    empty_checks = {}
    score = _compute_health_score(empty_checks)
    assert score == 5.0, f'empty checks should default to 5.0, got {score}'

    full_checks = {
        'data_volume': {'score': 10},
        'model_performance': {'score': 10},
        'feature_drift': {'score': 10},
        'calibration_quality': {'score': 10},
        'data_freshness': {'score': 10},
    }
    score = _compute_health_score(full_checks)
    assert score == 10.0

    print('  [OK] edge_cases: empty/None/invalid all handled')


if __name__ == '__main__':
    print('=' * 60)
    print('optimization_monitor.py - Unit Tests')
    print('=' * 60)

    tests = [
        test_check_data_volume,
        test_check_model_performance,
        test_check_feature_drift,
        test_check_calibration_quality,
        test_check_data_freshness,
        test_generate_health_report,
        test_edge_cases,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f'  [FAIL] {test.__name__}: {e}')
            traceback.print_exc()

    print()
    print('=' * 60)
    result = f'[{passed}/{len(tests)}] tests passed'
    if passed == len(tests):
        print(f'[OK] {result} - ALL PASS')
    else:
        print(f'[FAIL] {result}')
        sys.exit(1)
