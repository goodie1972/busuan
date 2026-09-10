# -*- coding:utf-8 -*-
"""
预测记录存档与核验模块

每次在 GUI / 命令行生成预测时，自动把号码保存到 prediction_history.json；
之后数据更新出新开奖期后，可核验"当时预测的号码 vs 对应期开奖"的命中与奖级，
实现与用户真实购买对应的回测（而非模型事后重放）。

核验依据：预测时记录当时已知的最新期数 latest_period，
目标期 = 数据中第一个 期数 > latest_period 的开奖期。
"""
import json
import os
from datetime import datetime

from backtest import SSQ_PRIZE_TABLE, DLT_PRIZE_TABLE, _load_csv_data

# 预测记录文件（UTF-8，与数据/模型分离）
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'prediction_history.json')


def load_records():
    """读取全部预测记录（空/损坏时返回空列表）"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_records(records):
    """写回预测记录"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def add_prediction_record(lottery_type, model_type, latest_period,
                          red_numbers, blue_numbers):
    """
    保存一条预测记录（失败不抛异常，避免影响预测主流程）

    Args:
        lottery_type: 'ssq' / 'dlt'
        model_type: 模型标识（lstm-crf / gbdt / ensemble ...）
        latest_period: 预测时已知的最新开奖期数
        red_numbers: 红球号码列表
        blue_numbers: 蓝球号码列表

    Returns:
        int: 存档后的累计记录条数；存档失败返回 None
    """
    try:
        record = {
            'predict_time': datetime.now().isoformat(timespec='seconds'),
            'lottery_type': lottery_type,
            'model_type': str(model_type),
            'latest_period': int(latest_period),
            'red_numbers': [int(x) for x in red_numbers],
            'blue_numbers': [int(x) for x in blue_numbers],
            # 核验后填充
            'target_period': None,
            'red_hits': None,
            'blue_hits': None,
            'prize_name': None,
            'prize_amount': None,
        }
        records = load_records()
        records.append(record)
        save_records(records)
        return len(records)
    except Exception:
        return None


def verify_records(log_callback=None):
    """
    核验全部预测记录：为每条寻找目标开奖期并比对命中与奖级。

    Returns:
        dict 核验报告：
        {
            'total': 总记录数,
            'pending': 待开奖条数,
            'verified': 已核验条数,
            'win_count': 中奖注数,
            'prize_total_fixed': 固定奖奖金合计（元）,
            'prize_counts': {奖级: 次数},
            'records': [核验后的记录, ...],
        }
        数据缺失时返回 None
    """
    log = log_callback or (lambda msg: None)
    records = load_records()
    if not records:
        log("暂无预测记录，先去'预测'页生成号码后自动存档。")
        return {
            'total': 0, 'pending': 0, 'verified': 0,
            'win_count': 0, 'prize_total_fixed': 0,
            'prize_counts': {}, 'records': [],
        }

    # 加载各彩票数据（轻量读取，避免 data_analysis 的 PyQt5/torch DLL 冲突）
    dfs = {}
    for lt in ('ssq', 'dlt'):
        df = _load_csv_data(lt)
        if df is not None and not df.empty:
            df = df.sort_values('期数', ascending=True).reset_index(drop=True)
            dfs[lt] = df

    red_cols = {}
    blue_cols = {}
    for lt, df in dfs.items():
        red_cols[lt] = [c for c in df.columns if c.startswith('红球_')]
        blue_cols[lt] = [c for c in df.columns
                         if c.startswith('蓝球_') or c == '蓝球']

    changed = False
    for rec in records:
        lt = rec.get('lottery_type')
        if lt not in dfs:
            continue  # 数据文件缺失，保持待核验
        df = dfs[lt]
        # 目标期：第一个 期数 > latest_period 的开奖期
        mask = df['期数'] > rec['latest_period']
        if not mask.any():
            continue  # 还没开出新期，保持待核验
        target_idx = df.index[mask][0]
        if rec.get('target_period') == int(df.loc[target_idx, '期数']):
            continue  # 已核验过
        # 计算命中
        actual_red = set(int(df.loc[target_idx, c]) for c in red_cols[lt])
        actual_blue = set(int(df.loc[target_idx, c]) for c in blue_cols[lt])
        red_hits = len(set(rec['red_numbers']) & actual_red)
        blue_hits = len(set(rec['blue_numbers']) & actual_blue)
        prize_name, prize_amount = _lookup_prize(lt, red_hits, blue_hits)
        rec['target_period'] = int(df.loc[target_idx, '期数'])
        rec['red_hits'] = red_hits
        rec['blue_hits'] = blue_hits
        rec['prize_name'] = prize_name
        rec['prize_amount'] = prize_amount
        changed = True
        log(f"核验 {lt} 第{rec['target_period']}期: 红{red_hits}蓝{blue_hits}"
            f" -> {prize_name}")

    if changed:
        save_records(records)

    # 汇总
    verified = [r for r in records if r.get('target_period') is not None]
    pending = [r for r in records if r.get('target_period') is None]
    prize_counts = {}
    win_count = 0
    prize_total = 0.0
    for r in verified:
        name = r.get('prize_name')
        if name and name not in ('未中奖',):
            prize_counts[name] = prize_counts.get(name, 0) + 1
            win_count += 1
            amount = r.get('prize_amount')
            if amount:
                prize_total += amount

    report = {
        'total': len(records),
        'pending': len(pending),
        'verified': len(verified),
        'win_count': win_count,
        'prize_total_fixed': prize_total,
        'prize_counts': prize_counts,
        'records': records,
    }
    return report


def _lookup_prize(lottery_type, red_hits, blue_hits):
    """按命中数查奖级表，返回 (奖级名, 固定奖金或 None)"""
    table = SSQ_PRIZE_TABLE if lottery_type == 'ssq' else DLT_PRIZE_TABLE
    key = (int(red_hits), int(blue_hits))
    if key in table:
        return table[key]
    return ('未中奖', 0)


def format_verify_summary(report):
    """生成核验摘要文本（GUI/命令行共用）"""
    if report is None:
        return "核验失败：数据加载异常"
    lines = []
    lines.append("=" * 46)
    lines.append("预测记录核验摘要")
    lines.append("=" * 46)
    if report['total'] == 0:
        lines.append("暂无预测记录。")
        return "\n".join(lines)
    lines.append(f"总记录: {report['total']} 注")
    lines.append(f"已开奖核验: {report['verified']} 注  "
                 f"待开奖: {report['pending']} 注")
    if report['verified']:
        lines.append(f"中奖: {report['win_count']} 注  "
                     f"命中率: {100.0 * report['win_count'] / report['verified']:.2f}%")
        lines.append(f"固定奖奖金合计: {report['prize_total_fixed']:.0f} 元  "
                     f"投注成本: {report['verified'] * 2.0:.0f} 元  "
                     f"净收益: {report['prize_total_fixed'] - report['verified'] * 2.0:.0f} 元")
        if report['prize_counts']:
            lines.append("奖级分布: " + ", ".join(
                f"{k} {v}次" for k, v in report['prize_counts'].items()))
        else:
            lines.append("奖级分布: 未中任何奖级")
        lines.append("最近核验明细（后 5 条）:")
        verified = [r for r in report['records'] if r.get('target_period') is not None]
        for r in verified[-5:]:
            lines.append(
                f"  {r['predict_time'][:10]} 第{r['target_period']}期  "
                f"红{r['red_hits']}蓝{r['blue_hits']} -> {r['prize_name']}")
    else:
        lines.append("所有记录均待开奖（数据尚未更新到对应期），请先更新数据。")
    lines.append("=" * 46)
    return "\n".join(lines)