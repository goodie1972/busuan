# -*- coding: utf-8 -*-
"""
optimization_log.py 单元测试
==============================
5 项测试：写入/读取/统计/效果报告/清理
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 临时日志文件
_TMP_LOG = None


def setup_test_env():
    """设置测试环境：用临时文件"""
    global _TMP_LOG
    _TMP_LOG = tempfile.mktemp(suffix='.json')
    import optimization_log
    optimization_log._LOG_FILE = _TMP_LOG


def cleanup():
    """清理临时文件"""
    global _TMP_LOG
    if _TMP_LOG and os.path.exists(_TMP_LOG):
        os.remove(_TMP_LOG)


def test_log_and_read():
    """测试写入和读取"""
    setup_test_env()
    try:
        from optimization_log import (
            log_optimization_action, get_optimization_history
        )

        # 写入3条
        for i in range(3):
            log_optimization_action(
                action='update_weights',
                recommendation={'reason': f'test{i}', 'params': {},
                                'priority': 'medium'},
                result={'success': True, 'detail': f'detail{i}'},
                trigger='manual',
                lottery_type='ssq',
                health_before=5.0 + i,
                health_after=6.0 + i,
            )

        # 读取
        history = get_optimization_history(limit=10)
        assert len(history) == 3, f'expected 3, got {len(history)}'
        # 倒序（最新在前）
        assert 'test2' in history[0]['reason']
        assert 'test0' in history[2]['reason']

        print(f'  [OK] log_and_read: 3 entries written and read')
    finally:
        cleanup()


def test_filter_by_lottery():
    """测试按彩种过滤"""
    setup_test_env()
    try:
        from optimization_log import (
            log_optimization_action, get_optimization_history
        )

        log_optimization_action(
            action='noop', recommendation={'reason': 'ssq', 'params': {}},
            result={'success': True}, trigger='auto', lottery_type='ssq')
        log_optimization_action(
            action='noop', recommendation={'reason': 'dlt', 'params': {}},
            result={'success': True}, trigger='auto', lottery_type='dlt')

        ssq_only = get_optimization_history(limit=10, lottery_type='ssq')
        assert len(ssq_only) == 1
        assert ssq_only[0]['lottery_type'] == 'ssq'

        dlt_only = get_optimization_history(limit=10, lottery_type='dlt')
        assert len(dlt_only) == 1
        assert dlt_only[0]['lottery_type'] == 'dlt'

        print(f'  [OK] filter_by_lottery: ssq=1, dlt=1')
    finally:
        cleanup()


def test_stats():
    """测试统计功能"""
    setup_test_env()
    try:
        from optimization_log import log_optimization_action, get_optimization_stats

        # 写入成功和失败
        log_optimization_action(
            action='update_weights',
            recommendation={'reason': 'ok', 'params': {}, 'priority': 'high'},
            result={'success': True}, trigger='auto', lottery_type='ssq',
            health_before=5.0, health_after=7.0)
        log_optimization_action(
            action='retrain_model',
            recommendation={'reason': 'fail', 'params': {}, 'priority': 'high'},
            result={'success': False, 'detail': 'error'},
            trigger='manual', lottery_type='ssq',
            health_before=5.0, health_after=5.0)

        stats = get_optimization_stats()
        assert stats['total'] == 2
        assert stats['success_rate'] == 50.0  # 1/2
        assert 'update_weights' in stats['by_action']
        assert 'retrain_model' in stats['by_action']
        assert 'auto' in stats['by_trigger']
        assert 'manual' in stats['by_trigger']

        print(f'  [OK] stats: total={stats["total"]}, '
              f'success_rate={stats["success_rate"]}%')
    finally:
        cleanup()


def test_effect_report():
    """测试效果报告"""
    setup_test_env()
    try:
        from optimization_log import (
            log_optimization_action, generate_effect_report
        )

        # 模拟一系列优化
        for i in range(5):
            log_optimization_action(
                action='update_weights',
                recommendation={'reason': f'cycle{i}', 'params': {},
                                'priority': 'medium'},
                result={'success': True},
                trigger='auto', lottery_type='dlt',
                health_before=3.0 + i, health_after=3.5 + i)

        report = generate_effect_report('dlt', days=30)
        assert report['lottery_type'] == 'dlt'
        assert report['total_optimizations'] == 5
        assert report['success_rate'] == 100.0
        assert report['health_trend']['start'] == 3.0
        assert report['health_trend']['end'] == 7.5
        assert report['health_trend']['total_change'] == 4.5

        print(f'  [OK] effect_report: 5 optimizations, '
              f'health 3.0→7.5 (+4.5)')
    finally:
        cleanup()


def test_max_entries_and_clear():
    """测试最大条目限制和清理"""
    setup_test_env()
    try:
        from optimization_log import (
            log_optimization_action, get_optimization_history, clear_old_logs,
        )
        import optimization_log as ol
        # 临时降低限制
        ol._MAX_LOG_ENTRIES = 10

        # 写入15条
        for i in range(15):
            log_optimization_action(
                action='noop',
                recommendation={'reason': f'fill{i}', 'params': {}},
                result={'success': True}, trigger='auto', lottery_type='ssq')

        # 应只保留10条
        history = get_optimization_history(limit=100)
        assert len(history) == 10, f'expected 10, got {len(history)}'

        # 清理到5条
        deleted = clear_old_logs(keep=5)
        assert deleted == 5
        history = get_optimization_history(limit=100)
        assert len(history) == 5

        print(f'  [OK] max_entries: capped at 10, cleared 5, remaining 5')
    finally:
        cleanup()


if __name__ == '__main__':
    print('=' * 60)
    print('optimization_log.py - Unit Tests')
    print('=' * 60)

    tests = [
        test_log_and_read,
        test_filter_by_lottery,
        test_stats,
        test_effect_report,
        test_max_entries_and_clear,
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
