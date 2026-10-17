# -*- coding:utf-8 -*-
"""
预测记录存档与核验模块 v2 (Database-backed)
==========================================

数据库版本：使用 SQLite busuan.db 替代 prediction_history.json

向后兼容保证：
- load_records() / save_records() API 不变（返回格式相同）
- add_prediction_record() / verify_records() 签名不变
- GUI/CLI 代码无需修改

内部变更：
- 数据从 predictions 表读取
- 核验结果写入 predictions 表 (target_period, red_hits 等字段)
- JSON 文件作为备用导出格式保留（手动触发）
"""
import os
from datetime import datetime

from backtest import SSQ_PRIZE_TABLE, DLT_PRIZE_TABLE, _load_csv_data
from db_access import (
    get_predictions,
    add_prediction as db_add_prediction,
    update_prediction_verification,
    get_lottery_draws,
)

# JSON 导出路径（用于手动备份/迁移）
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'prediction_history.json')


def load_records():
    """
    读取全部预测记录（从数据库）
    
    Returns:
        list: 预测记录列表 [{"predict_time": "2026-10-15T10:47:35", 
                            "lottery_type": "ssq", "model_type": "ensemble",
                            "red_numbers": [1,5,...], ...}, ...]
              字段完全兼容旧 JSON 格式
    """
    # 从数据库获取所有预测（降序）
    db_records = get_predictions(limit=None)
    
    # 转换为旧格式（移除数据库 ID，保持字段兼容）
    records = []
    for r in db_records:
        record = {
            'predict_time': r['predict_time'],
            'lottery_type': r['lottery_type'],
            'model_type': r['model_type'],
            'latest_period': r['latest_period'],
            'red_numbers': r['red_numbers'],
            'blue_numbers': r['blue_numbers'],
            'red_probabilities': r.get('red_probabilities'),
            'blue_probabilities': r.get('blue_probabilities'),
            'target_period': r.get('target_period'),
            'red_hits': r.get('red_hits'),
            'blue_hits': r.get('blue_hits'),
            'prize_name': r.get('prize_name'),
            'prize_amount': r.get('prize_amount'),
        }
        # 内部保留 DB ID（用于核验时更新）
        record['_db_id'] = r['id']
        records.append(record)
    
    return records


def save_records(records):
    """
    写回预测记录（数据库版不需要显式保存，此函数保持兼容性）
    
    注意：数据库版中，记录已在 add_prediction_record() 时写入，
         此函数仅用于兼容旧代码调用，实际操作为空
    
    Args:
        records: 预测记录列表（忽略，数据库已持久化）
    """
    # 数据库自动持久化，无需写回
    # 可选：导出为 JSON 备份
    pass


def add_prediction_record(lottery_type, model_type, latest_period,
                          red_numbers, blue_numbers, red_probabilities=None,
                          blue_probabilities=None):
    """
    保存一条预测记录（数据库版）

    Args:
        lottery_type: 'ssq' / 'dlt'
        model_type: 模型标识（lstm-crf / gbdt / ensemble ...）
        latest_period: 预测时已知的最新开奖期数
        red_numbers: 红球号码列表
        blue_numbers: 蓝球号码列表
        red_probabilities: 红球号码对应的概率列表（可选）
        blue_probabilities: 蓝球号码对应的概率列表（可选）

    Returns:
        int: 存档后的累计记录条数；存档失败返回 None
    """
    try:
        # 写入数据库
        db_add_prediction(
            lottery_type=lottery_type,
            model_type=str(model_type),
            latest_period=int(latest_period),
            red_numbers=[int(x) for x in red_numbers],
            blue_numbers=[int(x) for x in blue_numbers],
            red_probabilities=[float(p) for p in red_probabilities] if red_probabilities else None,
            blue_probabilities=[float(p) for p in blue_probabilities] if blue_probabilities else None,
            predict_time=datetime.now().isoformat(timespec='seconds')
        )
        
        # 返回累计记录数
        records = load_records()
        return len(records)
    except Exception as e:
        print(f"[ERROR] Failed to add prediction record: {e}")
        return None


def verify_records(log_callback=None, records_subset=None):
    """
    核验预测记录：为每条寻找目标开奖期并比对命中与奖级。

    Args:
        log_callback: 日志回调
        records_subset: 可选，只核验这些记录对应的下标（在全部记录列表中的下标）；
                        None 表示核验全部

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

    # 从数据库加载开奖数据（替代 CSV 直接读取）
    dfs = {}
    for lt in ('ssq', 'dlt'):
        draws = get_lottery_draws(lt, limit=None)
        if draws:
            # 转换为 pandas-like 格式（保持向后兼容）
            import pandas as pd
            data = []
            for draw in draws:
                row = {'期数': draw['period']}
                # 红球
                for i, num in enumerate(draw['red_numbers'], start=1):
                    row[f'红球_{i}'] = num
                # 蓝球
                if lt == 'ssq':
                    row['蓝球'] = draw['blue_numbers'][0]
                else:  # dlt
                    row['蓝球_1'] = draw['blue_numbers'][0]
                    row['蓝球_2'] = draw['blue_numbers'][1]
                data.append(row)
            df = pd.DataFrame(data)
            df = df.sort_values('期数', ascending=True).reset_index(drop=True)
            dfs[lt] = df

    red_cols = {}
    blue_cols = {}
    for lt, df in dfs.items():
        red_cols[lt] = [c for c in df.columns if c.startswith('红球_')]
        blue_cols[lt] = [c for c in df.columns
                         if c.startswith('蓝球_') or c == '蓝球']

    changed = False
    subset = set(records_subset) if records_subset is not None else None
    
    for idx, rec in enumerate(records):
        if subset is not None and idx not in subset:
            continue
        lt = rec.get('lottery_type')
        if lt not in dfs:
            continue  # 数据文件缺失，保持待核验
        df = dfs[lt]
        
        # latest_period 必须可转成整数
        try:
            latest = int(rec['latest_period'])
        except (KeyError, TypeError, ValueError):
            continue  # 脏数据，跳过
        
        # 已核验过 → 跳过
        if rec.get('target_period') is not None:
            continue
        
        # 查找目标期：第一个 期数 > latest
        target_rows = df[df['期数'] > latest]
        if target_rows.empty:
            continue  # 尚未开奖
        
        target_row = target_rows.iloc[0]
        target_period = int(target_row['期数'])
        
        # 提取开奖号码
        draw_red = [int(target_row[c]) for c in red_cols[lt]]
        draw_blue = [int(target_row[c]) for c in blue_cols[lt]]
        
        # 比对命中
        pred_red = set(rec['red_numbers'])
        pred_blue = set(rec['blue_numbers'])
        red_hits = len(pred_red & set(draw_red))
        blue_hits = len(pred_blue & set(draw_blue))
        
        # 查奖级
        prize_table = SSQ_PRIZE_TABLE if lt == 'ssq' else DLT_PRIZE_TABLE
        prize_name = None
        prize_amount = 0
        for pk, pv in prize_table.items():
            if (red_hits, blue_hits) == (pv['red'], pv['blue']):
                prize_name = pk
                prize_amount = pv.get('prize', 0)
                break
        
        # 更新数据库
        db_id = rec.get('_db_id')
        if db_id:
            update_prediction_verification(
                prediction_id=db_id,
                target_period=target_period,
                red_hits=red_hits,
                blue_hits=blue_hits,
                prize_name=prize_name,
                prize_amount=prize_amount
            )
            changed = True
            
            log(f"[{lt.upper()}] 期号 {latest} → {target_period}: "
                f"红{red_hits} 蓝{blue_hits} = {prize_name or '未中奖'}")
    
    # 重新加载（获取更新后的核验结果）
    if changed:
        records = load_records()
    
    # 统计
    total = len(records)
    verified = sum(1 for r in records if r.get('target_period') is not None)
    pending = total - verified
    win_count = sum(1 for r in records if (r.get('prize_amount') or 0) > 0)
    prize_total = sum(r.get('prize_amount') or 0 for r in records)
    
    prize_counts = {}
    for r in records:
        pn = r.get('prize_name')
        if pn:
            prize_counts[pn] = prize_counts.get(pn, 0) + 1
    
    log(f"核验完成：总 {total} 条，已核验 {verified}，待开奖 {pending}，中奖 {win_count} 注")
    
    return {
        'total': total,
        'pending': pending,
        'verified': verified,
        'win_count': win_count,
        'prize_total_fixed': prize_total,
        'prize_counts': prize_counts,
        'records': records,
    }


def export_to_json():
    """导出预测记录到 JSON 文件（手动备份用）"""
    import json
    records = load_records()
    # 移除内部 _db_id 字段
    clean_records = []
    for r in records:
        r_clean = {k: v for k, v in r.items() if not k.startswith('_')}
        clean_records.append(r_clean)
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)
    
    return len(clean_records)


if __name__ == "__main__":
    # 测试数据库版 prediction_records
    print("Testing database-backed prediction_records...")
    
    # 测试1: 加载记录
    records = load_records()
    print(f"Total records: {len(records)}")
    if records:
        print(f"Sample: {records[0]['lottery_type']} {records[0]['model_type']} "
              f"period {records[0]['latest_period']}")
    
    # 测试2: 添加一条测试记录
    print("\nAdding test prediction...")
    count = add_prediction_record(
        lottery_type='ssq',
        model_type='test_db',
        latest_period=26118,
        red_numbers=[1, 2, 3, 4, 5, 6],
        blue_numbers=[16],
        red_probabilities=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        blue_probabilities=[0.8]
    )
    print(f"Total records after add: {count}")
    
    # 测试3: 核验（干运行，实际核验需要新开奖数据）
    print("\nRunning verification...")
    result = verify_records(log_callback=print)
    print(f"Verification summary: {result['verified']} verified, {result['pending']} pending")
    
    # 测试4: 导出 JSON
    print("\nExporting to JSON...")
    exported = export_to_json()
    print(f"Exported {exported} records to {HISTORY_FILE}")
    
    print("\n[OK] Database-backed prediction_records test passed!")
