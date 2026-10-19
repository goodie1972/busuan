#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一次性修复脚本：更新所有 prize_name=None 的已核验记录

使用场景：
- 历史数据中有 prize_name=None 的已核验记录
- 需要将其统一更新为"未中奖"或正确的奖级名称

运行方法：
    python fix_null_prize_names.py
"""

from db_access import get_predictions, get_connection
from backtest import SSQ_PRIZE_TABLE, DLT_PRIZE_TABLE


def fix_null_prize_names(dry_run=False):
    """
    修复所有 prize_name=None 的已核验记录
    
    Args:
        dry_run: True=只统计不修改，False=实际修改数据库
    """
    print('=' * 60)
    print('修复 prize_name=None 的已核验记录')
    print('=' * 60)
    
    # 获取所有 prize_name=None 的已核验记录
    all_preds = get_predictions(limit=None)
    none_prize = [
        p for p in all_preds 
        if p.get('target_period') is not None and p.get('prize_name') is None
    ]
    
    print(f'\n找到 {len(none_prize)} 条需要修复的记录')
    
    if len(none_prize) == 0:
        print('OK - 无需修复')
        return {'total': 0, 'updated': 0}
    
    # 按彩票类型统计
    from collections import Counter
    by_type = Counter(p['lottery_type'] for p in none_prize)
    print(f'  SSQ: {by_type["ssq"]} 条')
    print(f'  DLT: {by_type["dlt"]} 条')
    
    if dry_run:
        print('\n[DRY RUN] 只统计不修改')
        return {'total': len(none_prize), 'updated': 0}
    
    # 开始修复
    print('\n开始修复...')
    with get_connection() as conn:
        cursor = conn.cursor()
        updated = 0
        
        for p in none_prize:
            pred_id = p['id']
            lt = p['lottery_type']
            red_hits = p.get('red_hits', 0)
            blue_hits = p.get('blue_hits', 0)
            
            # 查奖级
            prize_table = SSQ_PRIZE_TABLE if lt == 'ssq' else DLT_PRIZE_TABLE
            prize_name = '未中奖'
            prize_amount = 0
            
            if (red_hits, blue_hits) in prize_table:
                prize_info = prize_table[(red_hits, blue_hits)]
                prize_name = prize_info[0]
                prize_amount = prize_info[1] if prize_info[1] is not None else 0
            
            # 更新数据库
            cursor.execute('''
                UPDATE predictions 
                SET prize_name = ?, prize_amount = ?
                WHERE id = ?
            ''', (prize_name, prize_amount, pred_id))
            
            if cursor.rowcount > 0:
                updated += 1
        
        conn.commit()
    
    print(f'✓ 成功更新 {updated}/{len(none_prize)} 条记录')
    
    # 验证修复结果
    print('\n' + '=' * 60)
    print('验证修复结果:')
    all_preds = get_predictions(limit=None)
    none_prize_after = [
        p for p in all_preds 
        if p.get('target_period') is not None and p.get('prize_name') is None
    ]
    not_win = [p for p in all_preds if p.get('prize_name') == '未中奖']
    
    print(f'修复后 prize_name=None: {len(none_prize_after)}')
    print(f'未中奖记录数: {len(not_win)}')
    
    if len(none_prize_after) == 0:
        print('OK - 修复完成！')
    else:
        print(f'WARNING - 仍有 {len(none_prize_after)} 条记录未修复')
    
    return {'total': len(none_prize), 'updated': updated}


if __name__ == '__main__':
    import sys
    
    dry_run = '--dry-run' in sys.argv
    result = fix_null_prize_names(dry_run=dry_run)
    
    print('\n' + '=' * 60)
    print('修复完成！')
    print(f'  总计: {result["total"]} 条')
    print(f'  已修复: {result["updated"]} 条')
    print('=' * 60)
