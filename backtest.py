# -*- coding:utf-8 -*-
"""
回测模块：使用训练好的模型在历史开奖数据上逐期滚动回测，
统计命中情况、奖级分布与收益，并与随机选号基线（蒙特卡洛模拟）对比，
客观评估模型相对随机选号的表现。

用法（通过 main.py 或直接运行）:
    python main.py backtest ssq --model ensemble
    python main.py backtest dlt --model lightgbm --periods 500
    python backtest.py ssq ensemble --output backtest/ssq_ensemble.json

说明：
    - 回测为"逐期滚动"：每期只用截至当期的历史数据预测下一期，与实际开奖对比
    - 一、二等奖为浮动奖金，仅统计命中次数，金额按 0 计入（报告中有注明）
    - 模型训练时见过全部历史数据，回测存在轻微"训练集泄漏"，结果偏乐观，
      应作为相对评估（与随机基线对比）而非对未来收益的承诺
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime

# 必须在导入 torch 前设置：numpy/scipy 与 torch 的 OpenMP 运行时冲突
# 会导致 WinError 1114 DLL 初始化失败（直接运行 backtest.py 时不经过 main.py）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# 兼容直接运行与通过 main.py 运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from ml_models import LotteryMLModels
from model_utils import name_path

# 每注成本（元）
COST_PER_BET = 2.0

# 双色球奖级表：(红球命中数, 蓝球命中数) -> (奖级, 固定奖金元)；奖金 None 表示浮动奖
SSQ_PRIZE_TABLE = {
    (6, 1): ("一等奖", None),
    (6, 0): ("二等奖", None),
    (5, 1): ("三等奖", 3000),
    (5, 0): ("四等奖", 200),
    (4, 1): ("四等奖", 200),
    (4, 0): ("五等奖", 10),
    (3, 1): ("五等奖", 10),
    (2, 1): ("六等奖", 5),
    (1, 1): ("六等奖", 5),
    (0, 1): ("六等奖", 5),
}

# 大乐透奖级表
DLT_PRIZE_TABLE = {
    (5, 2): ("一等奖", None),
    (5, 1): ("二等奖", None),
    (5, 0): ("三等奖", 10000),
    (4, 2): ("四等奖", 3000),
    (4, 1): ("五等奖", 300),
    (3, 2): ("六等奖", 200),
    (4, 0): ("七等奖", 100),
    (3, 1): ("八等奖", 15),
    (2, 2): ("八等奖", 15),
    (3, 0): ("九等奖", 5),
    (1, 2): ("九等奖", 5),
    (2, 1): ("九等奖", 5),
    (0, 2): ("九等奖", 5),
}

# 彩票规则配置：号码池与奖级表
CONFIG = {
    'ssq': {
        'red_pool': 33, 'red_pick': 6, 'blue_pool': 16, 'blue_pick': 1,
        'prize_table': SSQ_PRIZE_TABLE,
    },
    'dlt': {
        'red_pool': 35, 'red_pick': 5, 'blue_pool': 12, 'blue_pick': 2,
        'prize_table': DLT_PRIZE_TABLE,
    },
}


def _load_csv_data(lottery_type):
    """
    轻量加载彩票历史数据（编码兜底）。
    注意：不能复用 scripts/data_analysis.load_lottery_data —— 该模块顶部 import 了
    PyQt5.QtGui/QtCore，而 PyQt5 的 DLL 与 torch 的 c10.dll 初始化冲突
    （WinError 1114），命令行回测在 torch 之后加载它会崩溃。
    """
    if lottery_type == 'dlt':
        file_path = './scripts/dlt/dlt_history.csv'
    else:
        file_path = './scripts/ssq/ssq_history.csv'
    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except FileNotFoundError:
            return None
    return None


def _extract_ball_cols(df):
    """从 DataFrame 列名中提取红球/蓝球列（SSQ 蓝球列为'蓝球'，DLT 为'蓝球_1'/'蓝球_2'）"""
    red_cols = [c for c in df.columns if c.startswith('红球_')]
    blue_cols = [c for c in df.columns if c.startswith('蓝球_') or c == '蓝球']
    return red_cols, blue_cols


def _numbers_from_row(row, red_cols, blue_cols):
    """取一行中的实际开奖号码（返回两个 set）"""
    red = set(int(row[c]) for c in red_cols)
    blue = set(int(row[c]) for c in blue_cols)
    return red, blue


def _lookup_prize(lottery_type, red_hits, blue_hits):
    """查奖级表，返回 (奖级名, 固定奖金或None)"""
    table = CONFIG[lottery_type]['prize_table']
    key = (red_hits, blue_hits)
    if key in table:
        return table[key]
    return (None, 0)


def _random_baseline(lottery_type, actual_red, actual_blue, num_sim=100000, seed=42):
    """
    随机选号基线：随机生成 num_sim 注，与同一期实际开奖号码比对。
    随机选号与随机开奖独立同分布，等价于随机选号对任意一期的理论表现。
    """
    cfg = CONFIG[lottery_type]
    rng = random.Random(seed)
    red_pool = range(1, cfg['red_pool'] + 1)
    blue_pool = range(1, cfg['blue_pool'] + 1)

    red_hit_dist = {k: 0 for k in range(cfg['red_pick'] + 1)}
    blue_hit_dist = {k: 0 for k in range(cfg['blue_pick'] + 1)}
    prize_counts = {}
    prize_total = 0.0
    any_prize = 0

    for _ in range(num_sim):
        r = set(rng.sample(red_pool, cfg['red_pick']))
        b = set(rng.sample(blue_pool, cfg['blue_pick']))
        red_hits = len(r & actual_red)
        blue_hits = len(b & actual_blue)
        red_hit_dist[red_hits] += 1
        blue_hit_dist[blue_hits] += 1
        prize_name, amount = _lookup_prize(lottery_type, red_hits, blue_hits)
        if prize_name is not None:
            any_prize += 1
            prize_counts[prize_name] = prize_counts.get(prize_name, 0) + 1
            if amount:
                prize_total += amount

    return {
        'simulations': num_sim,
        'any_prize_rate': any_prize / num_sim,
        'expected_prize_per_bet': prize_total / num_sim,
        'red_hit_distribution': {str(k): v / num_sim for k, v in red_hit_dist.items()},
        'blue_hit_distribution': {str(k): v / num_sim for k, v in blue_hit_dist.items()},
        'prize_counts': prize_counts,
    }


def run_backtest(lottery_type, model_type, periods=None, output_path=None,
                 seed=42, log_callback=None):
    """
    执行回测。

    参数:
        lottery_type: 'ssq' 或 'dlt'
        model_type: 模型类型（与训练一致）
        periods: 仅回测最近 N 期；None 表示全部可回测期数
        output_path: 报告 JSON 保存路径；None 时保存到 backtest/ 目录
        seed: 随机基线种子
        log_callback: 日志回调；None 时打印到控制台
    返回:
        报告 dict；失败返回 None
    """
    log = log_callback or (lambda msg: print(msg))
    cfg = CONFIG[lottery_type]
    lottery_name = name_path[lottery_type]['name']
    log(f"开始回测: {lottery_name} / 模型 {model_type}")

    df = _load_csv_data(lottery_type)
    if df is None or df.empty:
        log("错误: 数据加载失败")
        return None

    df = df.sort_values('期数', ascending=True).reset_index(drop=True)
    total = len(df)
    red_cols, blue_cols = _extract_ball_cols(df)
    if not red_cols or not blue_cols:
        log(f"错误: 数据列名不完整，列名: {list(df.columns)}")
        return None
    log(f"数据 {total} 条，红球列 {len(red_cols)} 个，蓝球列 {len(blue_cols)} 个")

    # 加载模型（静音内部日志）
    model_obj = LotteryMLModels(lottery_type=lottery_type, model_type=model_type,
                                log_callback=lambda msg: None)
    if not model_obj.load_models():
        log(f"错误: 模型加载失败: model/{lottery_type}/{model_type}")
        return None
    feature_window = int(getattr(model_obj, 'feature_window', 10) or 10)
    log(f"模型加载成功，特征窗口 {feature_window} 期")

    # 回测范围：从第 feature_window+1 期开始（需足够历史做特征）
    start = feature_window
    if periods is not None:
        start = max(start, total - periods)
    end = total - 1
    if start > end:
        log("错误: 数据量不足以回测（少于特征窗口+1 期）")
        return None
    log(f"回测期数范围: 第 {start + 1} ~ {end + 1} 期，共 {end - start + 1} 期")

    # 逐期滚动回测
    results = []
    n_total = end - start + 1
    for idx, t in enumerate(range(start, end + 1)):
        # 用截至 t-1 期的最近 feature_window 期做特征，预测第 t 期
        hist = df.iloc[max(0, t - feature_window):t]
        try:
            pred_red, pred_blue = model_obj.predict(hist)
        except Exception as e:
            log(f"第 {t + 1} 期(期数 {df.iloc[t]['期数']})预测出错: {e}")
            continue
        if pred_red is None:
            continue
        actual_red, actual_blue = _numbers_from_row(df.iloc[t], red_cols, blue_cols)
        red_hits = len(set(pred_red) & actual_red)
        blue_hits = len(set(pred_blue) & actual_blue)
        prize_name, amount = _lookup_prize(lottery_type, red_hits, blue_hits)
        results.append({
            'period': int(df.iloc[t]['期数']),
            'pred_red': sorted(int(x) for x in pred_red),
            'pred_blue': sorted(int(x) for x in pred_blue),
            'actual_red': sorted(actual_red),
            'actual_blue': sorted(actual_blue),
            'red_hits': red_hits,
            'blue_hits': blue_hits,
            'prize_name': prize_name,
            'prize_amount': amount,
        })
        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            pct = 100.0 * (idx + 1) / n_total
            log(f"回测进度: {idx + 1}/{n_total} 期 ({pct:.0f}%)")

    if not results:
        log("错误: 没有任何一期预测成功")
        return None
    log(f"回测完成，成功预测 {len(results)} 期")

    # 统计汇总
    n = len(results)
    cost_total = n * COST_PER_BET
    prize_total = sum(r['prize_amount'] or 0 for r in results)
    red_hit_dist = {k: 0 for k in range(cfg['red_pick'] + 1)}
    blue_hit_dist = {k: 0 for k in range(cfg['blue_pick'] + 1)}
    prize_counts = {}
    any_prize = 0
    for r in results:
        red_hit_dist[r['red_hits']] += 1
        blue_hit_dist[r['blue_hits']] += 1
        if r['prize_name'] is not None:
            any_prize += 1
            prize_counts[r['prize_name']] = prize_counts.get(r['prize_name'], 0) + 1

    # 随机基线（用最后一期真实号码）
    last_red, last_blue = _numbers_from_row(df.iloc[end], red_cols, blue_cols)
    baseline = _random_baseline(lottery_type, last_red, last_blue, seed=seed)

    report = {
        'meta': {
            'lottery_type': lottery_type,
            'lottery_name': lottery_name,
            'model_type': model_type,
            'feature_window': feature_window,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cost_per_bet': COST_PER_BET,
        },
        'periods': {
            'total_in_file': total,
            'backtested': n,
            'start_period': results[0]['period'],
            'end_period': results[-1]['period'],
        },
        'pnl': {
            'cost_total': round(cost_total, 2),
            'prize_total_fixed': round(prize_total, 2),
            'net': round(prize_total - cost_total, 2),
            'roi': round((prize_total - cost_total) / cost_total * 100, 2),
            'note': '一、二等奖为浮动奖金，金额按0计入',
        },
        'model_performance': {
            'any_prize_rate': round(any_prize / n, 6),
            'expected_prize_per_bet': round(prize_total / n, 4),
            'red_hit_distribution': {str(k): v for k, v in red_hit_dist.items()},
            'blue_hit_distribution': {str(k): v for k, v in blue_hit_dist.items()},
            'prize_counts': prize_counts,
        },
        'random_baseline': baseline,
        'comparison': {
            'any_prize_rate': {
                'model': round(any_prize / n, 6),
                'random': round(baseline['any_prize_rate'], 6),
            },
            'expected_prize_per_bet': {
                'model': round(prize_total / n, 4),
                'random': round(baseline['expected_prize_per_bet'], 4),
            },
            'note': '每注期望均不含浮动奖(一、二等奖)',
        },
        'notes': [
            '回测存在训练集泄漏(模型训练时见过全部历史)，结果偏乐观',
            '应作为相对评估(与随机基线对比)，而非对未来收益的承诺',
            '彩票为独立随机事件，长期期望收益为负',
        ],
    }

    # 保存报告
    if output_path is None:
        os.makedirs('backtest', exist_ok=True)
        output_path = os.path.join(
            'backtest', f"{lottery_type}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告已保存: {output_path}")

    _print_summary(report, log)
    return report


def format_summary_text(report):
    """生成回测摘要文本（GUI 与命令行共用）"""
    meta = report['meta']
    per = report['periods']
    pnl = report['pnl']
    mp = report['model_performance']
    rb = report['random_baseline']
    cmp_ = report['comparison']

    lines = []
    lines.append("=" * 46)
    lines.append("回测报告摘要")
    lines.append("=" * 46)
    lines.append(f"彩票: {meta['lottery_name']}  模型: {meta['model_type']}  "
                 f"特征窗口: {meta['feature_window']}期")
    lines.append(f"回测期数: {per['backtested']} 期  "
                 f"({per['start_period']} ~ {per['end_period']})")
    lines.append(f"投注成本: {pnl['cost_total']:.0f} 元  "
                 f"奖金(固定奖): {pnl['prize_total_fixed']:.0f} 元  "
                 f"净收益: {pnl['net']:.0f} 元  ROI: {pnl['roi']:.2f}%")
    lines.append("")
    lines.append("红球命中分布:")
    for k in sorted(mp['red_hit_distribution'], key=int):
        v = mp['red_hit_distribution'][k]
        bar = '#' * int(v / max(1, per['backtested']) * 50)
        lines.append(f"  命中{k}个红球: {v:5d} 期  {bar}")
    lines.append("蓝球命中分布:")
    for k in sorted(mp['blue_hit_distribution'], key=int):
        v = mp['blue_hit_distribution'][k]
        bar = '#' * int(v / max(1, per['backtested']) * 50)
        lines.append(f"  命中{k}个蓝球: {v:5d} 期  {bar}")
    lines.append(f"任意奖命中率: {mp['any_prize_rate'] * 100:.2f}%")
    if mp['prize_counts']:
        lines.append("奖级分布: " + ", ".join(f"{k} {v}次" for k, v in mp['prize_counts'].items()))
    else:
        lines.append("奖级分布: 未中任何奖级")
    lines.append("")
    lines.append(f"随机选号基线 ({rb['simulations']} 注蒙特卡洛模拟):")
    lines.append(f"  任意奖命中率: {rb['any_prize_rate'] * 100:.2f}%")
    lines.append(f"  每注期望奖金(固定奖): {rb['expected_prize_per_bet']:.4f} 元")
    lines.append("")
    lines.append("对比 (每注期望，均不含浮动奖):")
    lines.append(f"  模型: {cmp_['expected_prize_per_bet']['model']:.4f} 元  "
                 f"随机: {cmp_['expected_prize_per_bet']['random']:.4f} 元  "
                 f"成本: {meta['cost_per_bet']:.0f} 元")
    lines.append(f"  模型任意奖命中率 {cmp_['any_prize_rate']['model'] * 100:.2f}%  vs  "
                 f"随机 {cmp_['any_prize_rate']['random'] * 100:.2f}%")
    lines.append("")
    lines.append("注意: 回测存在训练集泄漏，结果偏乐观；彩票为独立随机事件，长期期望为负")
    lines.append("=" * 46)
    return "\n".join(lines)


def _print_summary(report, log):
    """打印回测摘要到控制台"""
    for line in format_summary_text(report).splitlines():
        log(line)


def main():
    """直接运行入口: python backtest.py ssq ensemble [--periods 500] [--output path]"""
    parser = argparse.ArgumentParser(description='彩票模型历史回测')
    parser.add_argument('lottery_type', choices=['ssq', 'dlt'], help='彩票类型')
    parser.add_argument('--model', default='ensemble',
                        choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble', 'expected_value'],
                        help='模型类型')
    parser.add_argument('--periods', type=int, default=None, help='仅回测最近N期(默认全部)')
    parser.add_argument('--output', default=None, help='报告JSON保存路径')
    parser.add_argument('--seed', type=int, default=42, help='随机基线种子')
    args = parser.parse_args()

    # Windows 控制台 UTF-8，避免中文乱码
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    result = run_backtest(args.lottery_type, args.model, periods=args.periods,
                          output_path=args.output, seed=args.seed)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
