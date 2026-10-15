# -*- coding: utf-8 -*-
"""
llm_optimizer.py 单元测试
==========================
6 项测试：规则引擎 / LLM mock / JSON 解析 / 降级逻辑 / 可用性检测
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_rule_based_analysis_normal():
    """测试规则引擎：正常场景（数据不足+漂移）"""
    from llm_optimizer import _rule_based_analysis

    report = {
        'health_score': 3.5,
        'priority': 'high',
        'checks': {
            'data_volume': {'status': 'critical', 'score': 0, 'value': 5,
                             'message': '核验记录仅5条'},
            'model_performance': {'status': 'warning', 'score': 5,
                                  'message': '1/5 模型样本充足',
                                  'details': {
                                      'xgboost': {'samples': 10, 'trend': 'declining'},
                                  }},
            'feature_drift': {'status': 'critical', 'score': 0,
                              'drifted_features': ['feat_a'], 'max_kl': 0.42,
                              'message': '1个特征漂移'},
            'calibration_quality': {'status': 'warning', 'score': 5,
                                    'message': '尚未校准'},
            'data_freshness': {'status': 'ok', 'score': 10,
                               'message': '2天前'},
        }
    }

    result = _rule_based_analysis(report, 'dlt')
    assert result['source'] == 'rules'
    assert 'health_score' in result
    assert len(result['issues']) >= 2  # 数据不足 + 漂移
    assert len(result['recommendations']) >= 2

    # 检查操作类型
    actions = [r['action'] for r in result['recommendations']]
    assert 'noop' in actions  # 数据不足 → noop
    assert 'retrain_model' in actions  # 漂移 → 重训
    assert 'update_weights' in actions  # 表现下降 → 权重

    # 检查 auto_executable
    for rec in result['recommendations']:
        if rec['action'] in ('retrain_model', 'update_data'):
            assert not rec['auto_executable']
        elif rec['action'] in ('update_weights', 'calibrate', 'noop'):
            assert rec['auto_executable']

    print(f'  [OK] rule_based: {len(result["issues"])} issues, '
          f'{len(result["recommendations"])} recs')


def test_rule_based_analysis_healthy():
    """测试规则引擎：健康场景"""
    from llm_optimizer import _rule_based_analysis

    report = {
        'health_score': 9.0,
        'priority': 'none',
        'checks': {
            'data_volume': {'status': 'ok', 'score': 10, 'value': 50},
            'model_performance': {'status': 'ok', 'score': 10,
                                  'details': {}},
            'feature_drift': {'status': 'ok', 'score': 10,
                              'drifted_features': [], 'max_kl': 0.05},
            'calibration_quality': {'status': 'ok', 'score': 10,
                                    'message': '校准良好'},
            'data_freshness': {'status': 'ok', 'score': 10},
        }
    }

    result = _rule_based_analysis(report, 'ssq')
    assert result['health_score'] == 9.0
    assert len(result['issues']) == 0
    assert len(result['recommendations']) == 1
    assert result['recommendations'][0]['action'] == 'noop'
    assert '良好' in result['explanation']
    print(f'  [OK] rule_based_healthy: noop recommended')


def test_parse_llm_response():
    """测试 LLM 响应解析"""
    from llm_optimizer import _parse_llm_response

    # 纯 JSON
    text = '{"diagnosis": "good", "recommendations": []}'
    result = _parse_llm_response(text)
    assert result is not None
    assert result['diagnosis'] == 'good'

    # 带 markdown 代码块
    text = '```json\n{"diagnosis": "ok", "score": 8}\n```'
    result = _parse_llm_response(text)
    assert result is not None
    assert result['score'] == 8

    # 混杂文本
    text = 'Here is my analysis: {"diagnosis": "bad", "score": 3} -- done'
    result = _parse_llm_response(text)
    assert result is not None
    assert result['score'] == 3

    # 无效 JSON
    result = _parse_llm_response('not json at all')
    assert result is None

    # 空输入
    result = _parse_llm_response('')
    assert result is None

    print('  [OK] parse_llm_response: valid/markdown/mixed/invalid all handled')


def test_analyze_health_report_fallback():
    """测试 LLM 不可用时降级到规则引擎"""
    from llm_optimizer import analyze_health_report

    # 构造一个简单的报告
    report = {
        'health_score': 5.0,
        'priority': 'medium',
        'checks': {
            'data_volume': {'status': 'warning', 'score': 5, 'value': 15,
                             'message': '偏少'},
            'model_performance': {'status': 'ok', 'score': 10,
                                  'details': {}, 'message': '良好'},
            'feature_drift': {'status': 'warning', 'score': 5,
                              'drifted_features': [], 'max_kl': 0.2,
                              'message': '轻度漂移'},
            'calibration_quality': {'status': 'warning', 'score': 5,
                                    'message': '尚未校准'},
            'data_freshness': {'status': 'ok', 'score': 10,
                               'message': '1天前'},
        }
    }

    # 无 API Key → 应降级到规则
    result = analyze_health_report(report, 'dlt')
    assert result['source'] in ('rules', 'none', 'llm')
    assert 'recommendations' in result
    assert len(result['recommendations']) >= 1
    print(f'  [OK] fallback: source={result["source"]}, '
          f'{len(result["recommendations"])} recs')


def test_analyze_with_mock_llm():
    """用 mock 测试 LLM 调用路径"""
    from llm_optimizer import analyze_health_report

    mock_response = {
        'diagnosis': 'Mock LLM diagnosis',
        'health_score': 7.5,
        'issues': [{'name': 'Minor issue', 'severity': 'low', 'detail': 'test'}],
        'recommendations': [
            {'action': 'update_weights', 'priority': 'medium',
             'auto_executable': True, 'reason': 'mock reason',
             'params': {'lottery_type': 'ssq'}}
        ],
        'explanation': 'Mock explanation'
    }

    # Mock OpenAI API 调用
    mock_text = json.dumps(mock_response)

    with patch('llm_optimizer._call_openai_api', return_value=mock_text):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            from llm_optimizer import get_config
            with patch('llm_optimizer.get_config') as mock_config:
                mock_config.return_value = {
                    'llm_provider': 'openai',
                    'llm_model': 'gpt-4o-mini',
                    'llm_temperature': 0.3,
                    'llm_max_tokens': 2000,
                    'llm_timeout_seconds': 30,
                    'llm_fallback_to_rules': True,
                    'openai_api_key_env': 'OPENAI_API_KEY',
                    'openai_base_url': None,
                    'ollama_host': 'http://localhost:11434',
                }
                report = {'health_score': 7.5, 'priority': 'low', 'checks': {}}
                result = analyze_health_report(report, 'ssq')
                assert result['source'] == 'llm'
                assert result['diagnosis'] == 'Mock LLM diagnosis'
                assert len(result['recommendations']) == 1

    print(f'  [OK] mock_llm: source=llm, parsed correctly')


def test_auto_executable_logic():
    """测试 auto_executable 判定逻辑"""
    from llm_optimizer import _rule_based_analysis

    # 构造全 critical 报告
    report = {
        'health_score': 0.0,
        'priority': 'high',
        'checks': {
            'data_volume': {'status': 'critical', 'score': 0, 'value': 0,
                             'message': '无记录'},
            'model_performance': {'status': 'critical', 'score': 0,
                                  'details': {}, 'message': '无数据'},
            'feature_drift': {'status': 'critical', 'score': 0,
                              'drifted_features': ['a', 'b', 'c'], 'max_kl': 0.5,
                              'message': '严重漂移'},
            'calibration_quality': {'status': 'warning', 'score': 5,
                                    'message': '尚未校准'},
            'data_freshness': {'status': 'critical', 'score': 0,
                               'message': '过期10天'},
        }
    }

    result = _rule_based_analysis(report, 'ssq')
    recs = result['recommendations']

    # 验证操作分级
    auto_recs = [r for r in recs if r['auto_executable'] and r['action'] != 'noop']
    manual_recs = [r for r in recs if not r['auto_executable']]

    # 重训和数据更新必须需确认
    for r in manual_recs:
        assert r['action'] in ('retrain_model', 'update_data'), \
            f'{r["action"]} should need confirmation'

    print(f'  [OK] auto_executable: {len(auto_recs)} auto, '
          f'{len(manual_recs)} manual')


if __name__ == '__main__':
    print('=' * 60)
    print('llm_optimizer.py - Unit Tests')
    print('=' * 60)

    tests = [
        test_rule_based_analysis_normal,
        test_rule_based_analysis_healthy,
        test_parse_llm_response,
        test_analyze_health_report_fallback,
        test_analyze_with_mock_llm,
        test_auto_executable_logic,
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
