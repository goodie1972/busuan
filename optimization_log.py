# -*- coding: utf-8 -*-
"""
优化日志 (Optimization Log)
============================
记录每次优化操作，支持追溯和效果对比。

日志格式:
{
  "logs": [
    {
      "id": "opt_20261015_100000",
      "timestamp": "2026-10-15T10:00:00",
      "trigger": "auto",           # auto / manual / llm
      "lottery_type": "dlt",
      "action": "update_weights",
      "reason": "lightgbm 表现下降",
      "params": {"lottery_type": "dlt"},
      "result": {
        "success": true,
        "before": {"lightgbm_weight": 0.25},
        "after": {"lightgbm_weight": 0.18},
        "detail": "权重已从0.25降至0.18"
      },
      "health_before": 6.5,
      "health_after": 7.0
    }
  ]
}
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from json_utils import safe_json_load, safe_json_write

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, 'optimization_log.json')
_MAX_LOG_ENTRIES = 500


def _generate_log_id() -> str:
    """生成唯一日志ID: opt_YYYYMMDD_HHMMSS"""
    return f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_optimization_action(
    action: str,
    recommendation: Dict,
    result: Dict,
    trigger: str = 'manual',
    lottery_type: str = 'unknown',
    health_before: Optional[float] = None,
    health_after: Optional[float] = None,
) -> str:
    """
    记录一条优化操作日志

    Args:
        action: 操作类型 (update_weights/calibrate/retrain_model/update_data/noop)
        recommendation: LLM/规则引擎生成的建议
        result: 执行结果 {success, before, after, detail}
        trigger: 触发方式 (auto/manual/llm)
        lottery_type: 彩票类型
        health_before: 操作前健康度
        health_after: 操作后健康度

    Returns:
        log_id: 日志ID
    """
    log_id = _generate_log_id()
    # 安全处理 None 参数（对抗脏数据）
    rec = recommendation if isinstance(recommendation, dict) else {}
    res = result if isinstance(result, dict) else {'success': False, 'detail': '空结果'}
    entry = {
        'id': log_id,
        'timestamp': datetime.now().isoformat(),
        'trigger': trigger,
        'lottery_type': lottery_type,
        'action': action,
        'reason': rec.get('reason', ''),
        'params': rec.get('params', {}),
        'priority': rec.get('priority', 'unknown'),
        'result': {
            'success': res.get('success', False),
            'detail': res.get('detail', ''),
        },
        'health_before': health_before,
        'health_after': health_after,
    }

    # 保存 before/after 数据（如果有）
    if 'before' in res:
        entry['result']['before'] = res['before']
    if 'after' in res:
        entry['result']['after'] = res['after']

    # 读取现有日志
    data = safe_json_load(_LOG_FILE, {'logs': []})
    if not isinstance(data, dict):
        data = {'logs': []}
    logs = data.get('logs', [])
    if not isinstance(logs, list):
        logs = []

    # 添加新条目
    logs.append(entry)

    # 限制最大条目数（清理旧条目）
    if len(logs) > _MAX_LOG_ENTRIES:
        logs = logs[-_MAX_LOG_ENTRIES:]

    data['logs'] = logs
    safe_json_write(_LOG_FILE, data)

    return log_id


def get_optimization_history(limit: int = 20,
                             lottery_type: Optional[str] = None) -> List[Dict]:
    """
    获取最近的优化操作历史

    Args:
        limit: 最大返回条数
        lottery_type: 可选，按彩种过滤

    Returns:
        日志列表（最新在前）
    """
    data = safe_json_load(_LOG_FILE, {'logs': []})
    logs = data.get('logs', []) if isinstance(data, dict) else []

    # 过滤彩种
    if lottery_type:
        logs = [l for l in logs if l.get('lottery_type') == lottery_type]

    # 倒序（最新在前）
    logs = list(reversed(logs))

    # 限制数量
    return logs[:limit]


def get_optimization_stats() -> Dict:
    """获取优化统计数据"""
    data = safe_json_load(_LOG_FILE, {'logs': []})
    logs = data.get('logs', []) if isinstance(data, dict) else []

    total = len(logs)
    if total == 0:
        return {'total': 0, 'success_rate': 0, 'by_action': {},
                'by_trigger': {}, 'recent': []}

    success_count = sum(1 for l in logs
                        if l.get('result', {}).get('success', False))
    success_rate = round(success_count / total * 100, 1)

    # 按操作类型统计
    by_action = {}
    by_trigger = {}
    for l in logs:
        action = l.get('action', 'unknown')
        by_action[action] = by_action.get(action, 0) + 1
        trigger = l.get('trigger', 'unknown')
        by_trigger[trigger] = by_trigger.get(trigger, 0) + 1

    # 健康度改善统计
    improvements = []
    for l in logs:
        before = l.get('health_before')
        after = l.get('health_after')
        if before is not None and after is not None:
            improvements.append(after - before)

    avg_improvement = round(sum(improvements) / len(improvements), 2) if improvements else 0.0

    return {
        'total': total,
        'success_rate': success_rate,
        'avg_improvement': avg_improvement,
        'by_action': by_action,
        'by_trigger': by_trigger,
        'recent': get_optimization_history(limit=5),
    }


def generate_effect_report(lottery_type: str, days: int = 30) -> Dict:
    """
    生成优化效果报告（对比优化前后命中率变化）

    Args:
        lottery_type: 彩种
        days: 回溯天数

    Returns:
        效果报告
    """
    data = safe_json_load(_LOG_FILE, {'logs': []})
    logs = data.get('logs', []) if isinstance(data, dict) else []

    # 过滤彩种和时间范围
    cutoff = datetime.now() - __import__('datetime').timedelta(days=days)
    filtered = []
    for l in logs:
        if l.get('lottery_type') != lottery_type:
            continue
        try:
            ts = datetime.fromisoformat(l.get('timestamp', ''))
            if ts >= cutoff:
                filtered.append(l)
        except (ValueError, TypeError):
            continue

    total = len(filtered)
    if total == 0:
        return {
            'lottery_type': lottery_type,
            'days': days,
            'total_optimizations': 0,
            'message': '该时间段内无优化记录'
        }

    success_count = sum(1 for l in filtered
                        if l.get('result', {}).get('success', False))
    health_changes = []
    for l in filtered:
        before = l.get('health_before')
        after = l.get('health_after')
        if before is not None and after is not None:
            health_changes.append({
                'date': l.get('timestamp', '')[:10],
                'action': l.get('action', ''),
                'before': before,
                'after': after,
                'change': round(after - before, 2),
            })

    # 健康度趋势
    if health_changes:
        first_h = health_changes[0]['before']
        last_h = health_changes[-1]['after']
        total_change = round(last_h - first_h, 2)
    else:
        first_h = last_h = total_change = 0

    # 按操作类型分组
    by_action = {}
    for l in filtered:
        action = l.get('action', 'unknown')
        by_action[action] = by_action.get(action, 0) + 1

    return {
        'lottery_type': lottery_type,
        'days': days,
        'total_optimizations': total,
        'success_rate': round(success_count / total * 100, 1) if total else 0,
        'health_trend': {
            'start': first_h,
            'end': last_h,
            'total_change': total_change,
            'details': health_changes,
        },
        'by_action': by_action,
    }


def clear_old_logs(keep: int = 100) -> int:
    """清理旧日志，保留最近 N 条，返回删除数量"""
    data = safe_json_load(_LOG_FILE, {'logs': []})
    logs = data.get('logs', []) if isinstance(data, dict) else []
    if len(logs) <= keep:
        return 0
    deleted = len(logs) - keep
    data['logs'] = logs[-keep:]
    safe_json_write(_LOG_FILE, data)
    return deleted


if __name__ == '__main__':
    # 快速功能验证
    print('=== Optimization Log Test ===')

    # 写入测试
    log_id = log_optimization_action(
        action='update_weights',
        recommendation={'reason': 'test', 'params': {'lottery_type': 'ssq'},
                        'priority': 'medium'},
        result={'success': True, 'before': {'w': 0.2}, 'after': {'w': 0.15},
                'detail': 'weight reduced'},
        trigger='manual',
        lottery_type='ssq',
        health_before=5.0,
        health_after=6.0,
    )
    print(f'Written log: {log_id}')

    # 读取历史
    history = get_optimization_history(limit=5)
    print(f'History: {len(history)} entries')

    # 统计
    stats = get_optimization_stats()
    print(f'Stats: total={stats["total"]}, success_rate={stats["success_rate"]}%')

    # 效果报告
    effect = generate_effect_report('ssq', days=30)
    print(f'Effect: {effect["total_optimizations"]} optimizations')
