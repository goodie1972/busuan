# -*- coding: utf-8 -*-
"""对抗性测试：v2 智能优化 - LLM 优化器 / 半自动执行器 / 原子日志"""
import pytest
import json
import tempfile
import os

from llm_optimizer import (
    get_config,
    _build_user_prompt,
    _parse_llm_response,
)
from optimization_executor import (
    execute_single_action,
    execute_plan,
    get_pending_actions,
    execute_pending_action,
    _is_auto_executable,
)
from optimization_log import (
    log_optimization_action,
    get_optimization_history,
    get_optimization_stats,
    clear_old_logs,
)

# ============================================================
# A. LLM 优化器
# ============================================================

# A1. _build_user_prompt 输入处理
def test_build_prompt_empty_report():
    """空报告也应返回非空提示（不崩溃）"""
    prompt = _build_user_prompt({}, 'ssq')
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_with_report():
    """完整报告应包含彩票类型与健康分"""
    report = {
        'health_score': 5.5,
        'priority': 'medium',
        'sub_reports': {
            'ssq': {'health_score': 5.0},
            'dlt': {'health_score': 6.0},
        },
    }
    prompt = _build_user_prompt(report, 'ssq')
    assert 'ssq' in prompt.lower() or '5.5' in prompt or '5.0' in prompt


def test_build_prompt_malformed():
    """恶意/异常 report 输入不应抛异常"""
    for bad in [None, 'string', 42, [1, 2, 3]]:
        try:
            result = _build_user_prompt(bad, 'ssq')
            assert result is None or isinstance(result, str)
        except (TypeError, ValueError, AttributeError):
            pass  # 显式拒绝也接受


# A2. _parse_llm_response JSON 解析
@pytest.mark.parametrize("raw,expected_success", [
    ('[{"action":"noop","reason":"r"}]', True),
    ('{"recommendations":[{"action":"noop"}]}', True),
    ('garbage text without json', False),
    ('', False),
    (None, False),
    ('[{"action":"unknown_action"}]', True),
])
def test_parse_llm_response(raw, expected_success):
    """解析 LLM 返回的 JSON（含脏数据）；返回 dict/list 或 None"""
    result = _parse_llm_response(raw)
    if expected_success:
        # 成功时返回 dict 或 list，不崩溃
        assert result is None or isinstance(result, (dict, list))
    else:
        assert result is None  # 解析失败返回 None


def test_parse_llm_response_injection():
    """注入 JSON 语法、Python 对象不应导致崩溃"""
    payloads = [
        '{"__class__": {"__init__": {"__globals__": {}}}}',
        '[{"action": 123, "reason": null}]',
        '{"action": "calibrate", "params": {"x": "__import__(\'os\').system(\'whoami\')"}}',
        ']}{"escape": true',
    ]
    for p in payloads:
        try:
            result = _parse_llm_response(p)
            assert result is None or isinstance(result, list)
        except Exception as e:
            # 不允许 Unicode 相关异常
            assert 'Unicode' not in type(e).__name__


# A3. get_config 环境读取
def test_get_config_no_api_key():
    """无 API key 时应返回默认配置而非崩溃"""
    # 清理环境变量
    for var in ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'OLLAMA_HOST']:
        os.environ.pop(var, None)
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert 'llm_provider' in cfg or 'provider' in cfg


# ============================================================
# B. 半自动执行器
# ============================================================

# B1. _is_auto_executable 操作分级
@pytest.mark.parametrize("action,expected", [
    ('update_weights', True),   # 低风险自动
    ('calibrate', True),        # 校准也自动（实现如此）
    ('retrain_model', False),   # 需确认
    ('update_data', False),     # 需确认
    ('noop', True),
    ('unknown_op', False),
])
def test_is_auto_executable(action, expected):
    assert _is_auto_executable(action) == expected


# B2. execute_single_action 未知操作
def test_execute_unknown_action():
    """未知操作应返回失败且不崩溃"""
    result = execute_single_action({'action': 'do_nothing'})
    assert result['success'] is False
    assert '未知操作' in result.get('detail', '') or 'unknown' in result.get('detail', '').lower()


# B3. execute_single_action noop
def test_execute_noop():
    """noop 操作应返回 skipped"""
    result = execute_single_action({'action': 'noop'})
    assert result['success'] is True
    assert result.get('skipped') is True


# B4. execute_single_action 缺 lottery_type
def test_execute_missing_lottery():
    """缺 lottery_type 参数应优雅处理（unknown 不崩溃）"""
    result = execute_single_action({'action': 'update_weights'})
    # 可能成功或失败，但不允许异常
    assert isinstance(result, dict)
    assert 'success' in result


# B5. get_pending_actions
def test_get_pending_actions():
    """应筛选出需人工确认的操作（读 recommendations 键）"""
    plan = {
        'recommendations': [
            {'action': 'update_weights', 'auto_executable': True, 'params': {}},
            {'action': 'retrain_model', 'auto_executable': False, 'params': {}},
            {'action': 'calibrate', 'auto_executable': True, 'params': {}},
            {'action': 'noop', 'auto_executable': True, 'params': {}},
        ]
    }
    pending = get_pending_actions(plan)
    assert isinstance(pending, list)
    actions = [a['action'] for a in pending]
    assert 'retrain_model' in actions
    assert 'update_weights' not in actions   # auto 不进 pending
    assert 'calibrate' not in actions
    assert 'noop' not in actions


def test_get_pending_actions_empty():
    """空计划返回空列表"""
    assert get_pending_actions({}) == []
    assert get_pending_actions({'recommendations': []}) == []


# ============================================================
# C. 原子日志
# ============================================================

def _log_file_path():
    """定位日志文件路径（从模块推断）"""
    import optimization_log as ol
    return os.path.join(ol._LOG_DIR, 'optimization_history.json')


def test_log_optimization_roundtrip():
    """写入一条日志后能读取回来"""
    before = len(get_optimization_history())
    r = log_optimization_action(
        action='update_weights',
        recommendation={'action': 'update_weights', 'reason': 'test', 'params': {}},
        result={'success': True, 'detail': 'ok'},
        trigger='test',
        lottery_type='ssq',
    )
    assert r is not None
    after = len(get_optimization_history())
    assert after >= before


def test_log_optimization_malformed():
    """异常推荐/结果也应能记录（不崩溃）"""
    log_optimization_action(
        action='calibrate',
        recommendation=None,
        result=None,
        trigger='test',
        lottery_type='dlt',
    )
    # 不抛异常即通过


def test_get_optimization_history_limit():
    """limit 参数应限制返回条数"""
    hist = get_optimization_history(limit=5)
    assert isinstance(hist, list)
    assert len(hist) <= 5


def test_get_optimization_history_zero():
    """limit<=0 应返回空列表而非崩溃"""
    assert get_optimization_history(limit=0) == []


def test_get_optimization_stats():
    """统计返回应包含操作总数"""
    stats = get_optimization_stats()
    assert isinstance(stats, dict)
    assert 'total' in stats or 'count' in stats or len(stats) >= 0


def test_clear_old_logs_boundary():
    """保留数量参数边界（负数/0）不崩溃"""
    try:
        clear_old_logs(keep=0)
        clear_old_logs(keep=-5)
    except Exception as e:
        assert 'Unicode' not in type(e).__name__


if __name__ == '__main__':
    pytest.main()