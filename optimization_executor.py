# -*- coding: utf-8 -*-
"""
半自动执行器 (Optimization Executor)
=====================================
执行优化计划，低风险自动执行，高风险请求确认。

操作分级:
  - auto_executable=True:  update_weights, calibrate, noop（低风险）
  - auto_executable=False: retrain_model, update_data（高风险，需确认）
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from optimization_log import log_optimization_action

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _execute_update_weights(lottery_type: str) -> Dict:
    """执行权重更新"""
    try:
        from model_weights import update_weights
        result = update_weights(lottery_type)
        if result:
            models = result.get('models', {})
            before = {mt: models.get(mt, {}).get('weight', 0)
                      for mt in ['random_forest', 'xgboost', 'gbdt',
                                 'lightgbm', 'catboost']
                      if mt in models}
            return {
                'success': True,
                'before': before,
                'after': before,  # update_weights 返回的已经是新权重
                'detail': f'{lottery_type.upper()} 权重已更新，'
                          f'推荐模型: {result.get("recommended_model", "-")}',
            }
        return {'success': False, 'detail': 'update_weights 返回空'}
    except Exception as e:
        return {'success': False, 'detail': f'权重更新失败: {e}'}


def _execute_calibrate(lottery_type: str) -> Dict:
    """执行校准训练"""
    try:
        from model_calibration import calibrate_lottery, get_calibration_info
        before = get_calibration_info(lottery_type)
        success = calibrate_lottery(lottery_type)
        after = get_calibration_info(lottery_type)
        if success:
            return {
                'success': True,
                'before': {'red_calibrated': before.get('red_calibrated', False),
                           'blue_calibrated': before.get('blue_calibrated', False)},
                'after': {'red_calibrated': after.get('red_calibrated', False),
                          'blue_calibrated': after.get('blue_calibrated', False),
                          'red_samples': after.get('red_samples', 0),
                          'blue_samples': after.get('blue_samples', 0)},
                'detail': f'{lottery_type.upper()} 校准完成，'
                          f'红球{after.get("red_samples", 0)}样本/'
                          f'蓝球{after.get("blue_samples", 0)}样本',
            }
        return {'success': False, 'detail': '校准训练失败'}
    except Exception as e:
        return {'success': False, 'detail': f'校准失败: {e}'}


def _execute_retrain_model(lottery_type: str,
                           model_types: Optional[List[str]] = None,
                           recent_periods: Optional[int] = None) -> Dict:
    """执行模型重训（高风险）"""
    try:
        from retrain_models_with_feedback import retrain_lottery_models
        result = retrain_lottery_models(
            lottery_type, use_gpu=False,
            model_types=model_types,
            recent_periods=recent_periods)
        if result and result.get('success'):
            return {
                'success': True,
                'detail': f'{lottery_type.upper()} 模型重训完成',
                'models_trained': result.get('models_trained', []),
            }
        return {'success': False,
                'detail': f'重训失败: {result.get("error", "unknown")}'}
    except Exception as e:
        return {'success': False, 'detail': f'重训失败: {e}'}


def _execute_update_data(lottery_type: str) -> Dict:
    """执行数据更新（高风险，需要网络请求）"""
    try:
        # 调用 fetch 脚本
        import subprocess
        script = os.path.join(_BASE_DIR, 'main.py')
        result = subprocess.run(
            [os.sys.executable, script, 'fetch', lottery_type],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return {'success': True,
                     'detail': f'{lottery_type.upper()} 数据更新完成'}
        return {'success': False,
                'detail': f'数据更新失败: {result.stderr[:200]}'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'detail': '数据更新超时(120s)'}
    except Exception as e:
        return {'success': False, 'detail': f'数据更新失败: {e}'}


# 操作映射
_ACTION_MAP = {
    'update_weights': _execute_update_weights,
    'calibrate': _execute_calibrate,
    'retrain_model': _execute_retrain_model,
    'update_data': _execute_update_data,
}


def _is_auto_executable(action: str) -> bool:
    """判断操作是否可自动执行"""
    return action in ('update_weights', 'calibrate', 'noop')


def execute_single_action(recommendation: Dict,
                          auto_confirm: bool = False,
                          trigger: str = 'manual') -> Dict:
    """
    执行单个优化建议

    Args:
        recommendation: LLM/规则引擎生成的建议
        auto_confirm: 是否自动确认高风险操作
        trigger: 触发方式 (auto/manual/llm)

    Returns:
        执行结果 {success, detail, ...}
    """
    action = recommendation.get('action', 'noop')
    params = recommendation.get('params', {})
    lottery_type = params.get('lottery_type', 'unknown')

    # noop 不执行
    if action == 'noop':
        return {'success': True, 'detail': '无需操作',
                'skipped': True}

    # 未知操作检查（优先于需确认检查）
    handler = _ACTION_MAP.get(action)
    if handler is None:
        return {'success': False, 'detail': f'未知操作: {action}'}

    # 高风险操作检查
    if not _is_auto_executable(action) and not auto_confirm:
        return {'success': False, 'detail': '需用户确认',
                'needs_confirm': True, 'action': action,
                'recommendation': recommendation}

    # 执行
    logger.info(f'Executing {action} for {lottery_type}...')
    result = handler(lottery_type, **{
        k: v for k, v in params.items() if k != 'lottery_type'
        and k in ('model_types', 'recent_periods')})

    # 记录日志
    log_optimization_action(
        action=action,
        recommendation=recommendation,
        result=result,
        trigger=trigger,
        lottery_type=lottery_type,
    )

    return result


def execute_plan(plan: Dict, auto_confirm: bool = False,
                 trigger: str = 'manual') -> Dict:
    """
    执行完整优化计划

    Args:
        plan: analyze_health_report() 的返回值
        auto_confirm: 是否自动确认高风险操作（定时任务用）
        trigger: 触发方式

    Returns:
        {
            "executed": [已执行的操作列表],
            "skipped": [跳过的操作列表],
            "pending": [待确认的操作列表],
            "results": {action_key: result},
        }
    """
    recommendations = plan.get('recommendations', [])
    executed = []
    skipped = []
    pending = []
    results = {}

    for rec in recommendations:
        action = rec.get('action', 'noop')
        params = rec.get('params', {})
        lottery_type = params.get('lottery_type', 'unknown')
        action_key = f'{action}({lottery_type})'

        result = execute_single_action(rec, auto_confirm, trigger)

        if result.get('skipped'):
            skipped.append(action_key)
            results[action_key] = result
        elif result.get('needs_confirm'):
            pending.append({
                'action': action,
                'lottery_type': lottery_type,
                'reason': rec.get('reason', ''),
                'params': params,
            })
            results[action_key] = result
        else:
            executed.append(action_key)
            results[action_key] = result

    return {
        'executed': executed,
        'skipped': skipped,
        'pending': pending,
        'results': results,
        'plan_source': plan.get('source', 'unknown'),
    }


def get_pending_actions(plan: Dict) -> List[Dict]:
    """从计划中提取需要用户确认的操作"""
    pending = []
    for rec in plan.get('recommendations', []):
        action = rec.get('action', 'noop')
        if not _is_auto_executable(action) and action != 'noop':
            pending.append({
                'action': action,
                'lottery_type': rec.get('params', {}).get('lottery_type', ''),
                'reason': rec.get('reason', ''),
                'priority': rec.get('priority', 'medium'),
                'params': rec.get('params', {}),
            })
    return pending


def execute_pending_action(action: str, lottery_type: str,
                           params: Optional[Dict] = None) -> Dict:
    """执行用户确认的高风险操作"""
    params = params or {}
    rec = {
        'action': action,
        'params': {**params, 'lottery_type': lottery_type},
        'reason': '用户手动确认执行',
        'priority': 'high',
    }
    return execute_single_action(rec, auto_confirm=True, trigger='manual')


if __name__ == '__main__':
    # 快速验证
    from optimization_monitor import generate_health_report
    from llm_optimizer import analyze_health_report

    lt = 'dlt'
    report = generate_health_report(lt)
    plan = analyze_health_report(report, lt)
    result = execute_plan(plan, auto_confirm=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
