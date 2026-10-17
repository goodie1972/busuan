# -*- coding:utf-8 -*-
"""
统计洞察提取模块 (Statistics Insights)
=====================================

从历史数据和高级统计分析中提取选号洞察，供"预测改进中心"底部统计洞察区展示。

功能：
1. 热号/冷号洞察：基于近 N 期频率统计
2. 遗漏洞察：长期未出号码的遗漏值
3. 趋势洞察：号码出现频率的变化趋势（上升/下降）
4. 综合选号建议：融合以上三种洞察生成推荐号码

数据来源：
- scripts/data_analysis.py 的 calculate_statistics()
- scripts/advanced_statistics.py 的 calculate_advanced_statistics()
- 数据库 lottery_draws 表（最新开奖记录）
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter

from db_access import get_lottery_draws, get_latest_period

# 号码规则
RULES = {
    'ssq': {'red_range': range(1, 34), 'red_count': 6,
            'blue_range': range(1, 17), 'blue_count': 1},
    'dlt': {'red_range': range(1, 36), 'red_count': 5,
            'blue_range': range(1, 13), 'blue_count': 2},
}


def _load_draws_as_records(lottery_type: str, limit: int = 100) -> List[Dict]:
    """
    从数据库加载开奖记录，转换为统计格式

    Returns:
        [{"period": 26118, "red": [1,5,...], "blue": [16]}, ...]
    """
    draws = get_lottery_draws(lottery_type, limit=limit)
    records = []
    for d in draws:
        records.append({
            'period': d['period'],
            'red': d['red_numbers'],
            'blue': d['blue_numbers'],
        })
    return records


def _calculate_frequency(records: List[Dict], ball_type: str = 'red') -> Dict[int, int]:
    """计算号码频率"""
    counter = Counter()
    for r in records:
        balls = r.get(ball_type, [])
        for b in balls:
            counter[int(b)] += 1
    return dict(counter)


def _calculate_gaps(records: List[Dict], lottery_type: str,
                    ball_type: str = 'red') -> Dict[int, int]:
    """计算遗漏值（当前距上次出现的期数）"""
    rule = RULES[lottery_type]
    ball_range = rule['red_range'] if ball_type == 'red' else rule['blue_range']
    gaps = {num: len(records) for num in ball_range}  # 默认=全部期未出现

    for i, r in enumerate(records):
        balls = r.get(ball_type, [])
        for b in balls:
            b = int(b)
            if b in gaps and gaps[b] == len(records):
                gaps[b] = i  # 第一次出现位置（=遗漏期数）
    return gaps


def _calculate_trend(records: List[Dict], ball_type: str = 'red',
                     window: int = 10) -> Dict[int, str]:
    """
    计算趋势（近期 vs 历史频率比较）

    Returns:
        {号码: 'up'/'down'/'stable'}
    """
    if len(records) < window * 2:
        return {}

    recent = records[:window]
    older = records[window:window * 2]

    recent_freq = _calculate_frequency(recent, ball_type)
    older_freq = _calculate_frequency(older, ball_type)

    trends = {}
    all_nums = set(recent_freq.keys()) | set(older_freq.keys())
    for num in all_nums:
        r_count = recent_freq.get(num, 0)
        o_count = older_freq.get(num, 0)
        if o_count == 0 and r_count > 0:
            trends[num] = 'up'
        elif r_count == 0 and o_count > 0:
            trends[num] = 'down'
        elif r_count > o_count * 1.3:
            trends[num] = 'up'
        elif r_count < o_count * 0.7:
            trends[num] = 'down'
        else:
            trends[num] = 'stable'

    return trends


def get_hot_cold_insight(lottery_type: str, top_n: int = 5,
                         window: int = 50) -> Dict:
    """
    获取热号/冷号洞察

    Returns:
        {
            'red_hot': [号码, ...],
            'red_cold': [号码, ...],
            'blue_hot': [号码, ...],
            'blue_cold': [号码, ...],
            'summary': "近50期热号: 红[1,5,...] 蓝[16]; 冷号: 红[33,...] 蓝[1]",
        }
    """
    records = _load_draws_as_records(lottery_type, limit=window)
    if not records:
        return {'red_hot': [], 'red_cold': [], 'blue_hot': [], 'blue_cold': [],
                'summary': '无数据'}

    red_freq = _calculate_frequency(records, 'red')
    blue_freq = _calculate_frequency(records, 'blue')

    # 热号（频率最高）
    red_hot = sorted(red_freq.keys(), key=lambda x: red_freq[x], reverse=True)[:top_n]
    blue_hot = sorted(blue_freq.keys(), key=lambda x: blue_freq[x], reverse=True)[:3]

    # 冷号（频率最低或未出现）
    rule = RULES[lottery_type]
    red_cold = sorted(rule['red_range'],
                      key=lambda x: red_freq.get(x, 0))[:top_n]
    blue_cold = sorted(rule['blue_range'],
                      key=lambda x: blue_freq.get(x, 0))[:3]

    summary = (f"近{window}期热号: 红{red_hot} 蓝{blue_hot}; "
               f"冷号: 红{red_cold} 蓝{blue_cold}")

    return {
        'red_hot': red_hot,
        'red_cold': red_cold,
        'blue_hot': blue_hot,
        'blue_cold': blue_cold,
        'summary': summary,
    }


def get_missing_insight(lottery_type: str, window: int = 100) -> Dict:
    """
    获取遗漏洞察（长期未出的号码）

    Returns:
        {
            'red_max_missing': [号码, ...],  # 遗漏最久的红球
            'blue_max_missing': [号码, ...],
            'red_gaps': {号码: 遗漏期数},
            'blue_gaps': {号码: 遗漏期数},
            'summary': "红球遗漏最久: 33(50期); 蓝球: 1(30期)",
        }
    """
    records = _load_draws_as_records(lottery_type, limit=window)
    if not records:
        return {'red_max_missing': [], 'blue_max_missing': [],
                'red_gaps': {}, 'blue_gaps': {}, 'summary': '无数据'}

    red_gaps = _calculate_gaps(records, lottery_type, 'red')
    blue_gaps = _calculate_gaps(records, lottery_type, 'blue')

    # 遗漏最久的号码（Top 5）
    red_sorted = sorted(red_gaps.items(), key=lambda x: x[1], reverse=True)[:5]
    blue_sorted = sorted(blue_gaps.items(), key=lambda x: x[1], reverse=True)[:3]

    red_missing = [num for num, gap in red_sorted if gap > 0]
    blue_missing = [num for num, gap in blue_sorted if gap > 0]

    summary_parts = []
    if red_missing:
        summary_parts.append(f"红球遗漏最久: {red_missing[0]}({red_gaps[red_missing[0]]}期)")
    if blue_missing:
        summary_parts.append(f"蓝球: {blue_missing[0]}({blue_gaps[blue_missing[0]]}期)")

    return {
        'red_max_missing': red_missing,
        'blue_max_missing': blue_missing,
        'red_gaps': red_gaps,
        'blue_gaps': blue_gaps,
        'summary': "; ".join(summary_parts) if summary_parts else "所有号码近期均有出现",
    }


def get_trend_insight(lottery_type: str, window: int = 10) -> Dict:
    """
    获取趋势洞察

    Returns:
        {
            'red_up': [号码, ...],    # 频率上升
            'red_down': [号码, ...],  # 频率下降
            'blue_up': [号码, ...],
            'blue_down': [号码, ...],
            'summary': "红球上升: [1,5]; 下降: [33]; 蓝球上升: [16]",
        }
    """
    records = _load_draws_as_records(lottery_type, limit=window * 2)
    if len(records) < window * 2:
        return {'red_up': [], 'red_down': [], 'blue_up': [], 'blue_down': [],
                'summary': '数据不足'}

    red_trends = _calculate_trend(records, 'red', window)
    blue_trends = _calculate_trend(records, 'blue', window)

    red_up = [n for n, t in red_trends.items() if t == 'up']
    red_down = [n for n, t in red_trends.items() if t == 'down']
    blue_up = [n for n, t in blue_trends.items() if t == 'up']
    blue_down = [n for n, t in blue_trends.items() if t == 'down']

    summary_parts = []
    if red_up:
        summary_parts.append(f"红球上升: {sorted(red_up)[:5]}")
    if red_down:
        summary_parts.append(f"下降: {sorted(red_down)[:5]}")
    if blue_up:
        summary_parts.append(f"蓝球上升: {sorted(blue_up)[:3]}")

    return {
        'red_up': sorted(red_up),
        'red_down': sorted(red_down),
        'blue_up': sorted(blue_up),
        'blue_down': sorted(blue_down),
        'summary': "; ".join(summary_parts) if summary_parts else "无明显趋势变化",
    }


def get_combined_suggestion(lottery_type: str, window: int = 50) -> Dict:
    """
    综合三种洞察，生成选号建议

    策略：
    - 热号：近期高频号（权重 0.4）
    - 遗漏：长期未出号（权重 0.3，遗漏回补预期）
    - 趋势上升：频率上升号（权重 0.3）
    
    Returns:
        {
            'red_suggested': [号码, ...],
            'blue_suggested': [号码, ...],
            'reasoning': "综合热号+遗漏+趋势分析",
            'details': {各洞察子项}
        }
    """
    hot_cold = get_hot_cold_insight(lottery_type, window=window)
    missing = get_missing_insight(lottery_type, window=window)
    trend = get_trend_insight(lottery_type, window=10)

    rule = RULES[lottery_type]

    # 红球综合评分
    red_scores = {}
    # 热号加权
    for num in hot_cold.get('red_hot', []):
        red_scores[num] = red_scores.get(num, 0) + 0.4
    # 遗漏加权
    for num in missing.get('red_max_missing', []):
        red_scores[num] = red_scores.get(num, 0) + 0.3
    # 趋势上升加权
    for num in trend.get('red_up', []):
        red_scores[num] = red_scores.get(num, 0) + 0.3

    # 蓝球综合评分
    blue_scores = {}
    for num in hot_cold.get('blue_hot', []):
        blue_scores[num] = blue_scores.get(num, 0) + 0.4
    for num in missing.get('blue_max_missing', []):
        blue_scores[num] = blue_scores.get(num, 0) + 0.3
    for num in trend.get('blue_up', []):
        blue_scores[num] = blue_scores.get(num, 0) + 0.3

    # 按得分排序取前 N
    red_suggested = sorted(red_scores.keys(), key=lambda x: red_scores[x], reverse=True)
    # 不足则补充热号
    if len(red_suggested) < rule['red_count']:
        for num in hot_cold.get('red_hot', []):
            if num not in red_suggested:
                red_suggested.append(num)
            if len(red_suggested) >= rule['red_count']:
                break
    red_suggested = sorted(red_suggested[:rule['red_count']])

    blue_suggested = sorted(blue_scores.keys(),
                           key=lambda x: blue_scores[x], reverse=True)
    if len(blue_suggested) < rule['blue_count']:
        for num in hot_cold.get('blue_hot', []):
            if num not in blue_suggested:
                blue_suggested.append(num)
            if len(blue_suggested) >= rule['blue_count']:
                break
    blue_suggested = sorted(blue_suggested[:rule['blue_count']])

    return {
        'red_suggested': red_suggested,
        'blue_suggested': blue_suggested,
        'reasoning': '综合热号(0.4) + 遗漏(0.3) + 趋势(0.3) 加权',
        'details': {
            'hot_cold': hot_cold['summary'],
            'missing': missing['summary'],
            'trend': trend['summary'],
        }
    }


def get_insight_cards(lottery_type: str) -> str:
    """
    生成洞察卡片文本（供 UI 底部统计洞察区展示）

    Returns:
        str: 多行文本，包含热冷号/遗漏/趋势/综合建议
    """
    lines = []
    lines.append(f"=== {lottery_type.upper()} 统计洞察 ===")
    lines.append("")

    # 1. 热冷号
    hc = get_hot_cold_insight(lottery_type)
    lines.append("[热冷号]")
    lines.append(hc['summary'])
    lines.append("")

    # 2. 遗漏
    mi = get_missing_insight(lottery_type)
    lines.append("[遗漏分析]")
    lines.append(mi['summary'])
    lines.append("")

    # 3. 趋势
    tr = get_trend_insight(lottery_type)
    lines.append("[趋势分析]")
    lines.append(tr['summary'])
    lines.append("")

    # 4. 综合建议
    cs = get_combined_suggestion(lottery_type)
    lines.append("[综合选号建议]")
    lines.append(f"红球: {cs['red_suggested']}")
    lines.append(f"蓝球: {cs['blue_suggested']}")
    lines.append(f"策略: {cs['reasoning']}")

    return "\n".join(lines)


if __name__ == "__main__":
    # 命令行测试
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'ssq'

    print(f"Testing statistics_insights for {target}...")
    print()

    print(get_insight_cards(target))
