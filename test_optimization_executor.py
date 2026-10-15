# -*- coding: utf-8 -*-
"""
optimization_executor.py 单元测试
===================================
5 项测试：自动执行/手动确认/未知操作/完整计划/提取待确认
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 使用临时日志文件避免污染真实数据
import optimization_log
import tempfile
_TMP = tempfile.mktemp(suffix='.json')
optimization_log._LOG_FILE = _TMP


def cleanup():
    if os.path.exists(_TMP):
        os.remove(_TMP)


def test_auto_executable_weights():
    """测试自动执行权重更新（mock）"""
    try:
        from optimization_executor import execute_single_action

        rec = {
            'action': 'update_weights',
            'params': {'lottery_type': 'ssq'},
            'reason': 'test', 'priority': 'medium',
        }

        def mock_func(lt, **kw):
            return {'success': True, 'detail': 'weights updated'}

        with patch.dict('optimization_executor._ACTION_MAP',
                        {'update_weights': mock_func}):
            result = execute_single_action(rec, auto_confirm=False)

        assert result['success'] is True
        assert 'detail' in result
        print(f'  [OK] auto_weights: executed without confirmation')
    finally:
        cleanup()


def test_manual_confirm_retrain():
    """测试重训需要确认"""
    try:
        from optimization_executor import execute_single_action

        rec = {
            'action': 'retrain_model',
            'params': {'lottery_type': 'dlt', 'recent_periods': 50},
            'reason': 'drift detected', 'priority': 'high',
        }

        # 不自动确认 → 需确认
        result = execute_single_action(rec, auto_confirm=False)
        assert result.get('needs_confirm') is True
        assert not result.get('success', True)

        # 自动确认 → 执行（mock）
        def mock_retrain(lt, **kw):
            return {'success': True, 'detail': 'retrained'}

        with patch.dict('optimization_executor._ACTION_MAP',
                        {'retrain_model': mock_retrain}):
            result = execute_single_action(rec, auto_confirm=True)
        assert result['success'] is True
        print(f'  [OK] manual_retrain: needs confirm then executes')
    finally:
        cleanup()


def test_unknown_action():
    """测试未知操作"""
    from optimization_executor import execute_single_action

    rec = {
        'action': 'unknown_op',
        'params': {'lottery_type': 'ssq'},
        'reason': 'test', 'priority': 'low',
    }
    result = execute_single_action(rec)
    assert result['success'] is False
    assert '未知操作' in result.get('detail', '')
    print(f'  [OK] unknown_action: correctly rejected')


def test_execute_plan():
    """测试完整计划执行"""
    try:
        from optimization_executor import execute_plan

        plan = {
            'source': 'rules',
            'recommendations': [
                # 自动执行
                {'action': 'update_weights', 'params': {'lottery_type': 'ssq'},
                 'reason': 'decline', 'priority': 'medium',
                 'auto_executable': True},
                # 需确认
                {'action': 'retrain_model',
                 'params': {'lottery_type': 'ssq', 'recent_periods': 50},
                 'reason': 'drift', 'priority': 'high',
                 'auto_executable': False},
                # noop
                {'action': 'noop', 'params': {'lottery_type': 'ssq'},
                 'reason': 'healthy', 'priority': 'low',
                 'auto_executable': True},
            ]
        }

        with patch.dict('optimization_executor._ACTION_MAP',
                        {'update_weights': lambda lt, **kw: {'success': True, 'detail': 'ok'}}):
            result = execute_plan(plan, auto_confirm=False)

        # update_weights 应执行
        assert 'update_weights(ssq)' in result['executed']
        # retrain_model 应待确认
        assert len(result['pending']) == 1
        assert result['pending'][0]['action'] == 'retrain_model'
        # noop 应跳过
        assert 'noop(ssq)' in result['skipped']

        print(f'  [OK] execute_plan: {len(result["executed"])} executed, '
              f'{len(result["pending"])} pending, '
              f'{len(result["skipped"])} skipped')
    finally:
        cleanup()


def test_get_pending_actions():
    """测试提取待确认操作"""
    from optimization_executor import get_pending_actions

    plan = {
        'recommendations': [
            {'action': 'update_weights', 'params': {'lottery_type': 'ssq'},
             'reason': 'auto', 'priority': 'medium'},
            {'action': 'retrain_model', 'params': {'lottery_type': 'dlt'},
             'reason': 'manual', 'priority': 'high'},
            {'action': 'update_data', 'params': {'lottery_type': 'ssq'},
             'reason': 'stale', 'priority': 'high'},
        ]
    }

    pending = get_pending_actions(plan)
    assert len(pending) == 2  # retrain + update_data
    actions = [p['action'] for p in pending]
    assert 'retrain_model' in actions
    assert 'update_data' in actions
    assert 'update_weights' not in actions  # 不需要确认
    print(f'  [OK] get_pending: {len(pending)} pending actions')


if __name__ == '__main__':
    print('=' * 60)
    print('optimization_executor.py - Unit Tests')
    print('=' * 60)

    tests = [
        test_auto_executable_weights,
        test_manual_confirm_retrain,
        test_unknown_action,
        test_execute_plan,
        test_get_pending_actions,
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
    cleanup()

    print()
    print('=' * 60)
    result = f'[{passed}/{len(tests)}] tests passed'
    if passed == len(tests):
        print(f'[OK] {result} - ALL PASS')
    else:
        print(f'[FAIL] {result}')
        sys.exit(1)
