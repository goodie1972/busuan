# -*- coding:utf-8 -*-
"""
模型动态权重模块 v2 (Database-backed, API兼容)
===========================================

数据库版本：使用 SQLite busuan.db/model_weights_history 表持久化，
同时完全保留旧版函数接口，供 GUI/executor/ml_models 无需改动即可调用。

兼容 API：
- compute_model_weights(lottery_type, records=None) -> Dict
- update_weights(lottery_type, records=None) -> Dict            (持久化到DB)
- update_all_weights(records=None) -> Dict[str, Dict]           (两彩种)
- get_model_weights(lottery_type) -> Dict[str, float]           (供集成预测加权)
- get_latest_weights(lottery_type) -> Optional[Dict]
- get_performance_report(lottery_type) -> Dict                  (供UI展示)
- update_weights_auto(lottery_type) -> bool
"""
import json
import os
import time
import math
from typing import Dict, List, Optional, Tuple

from db_access import (
    get_predictions,
    add_model_weights as db_add_model_weights,
    get_latest_model_weights as db_get_latest_model_weights,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(PROJECT_DIR, 'model_weights.json')

# 参与加权的模型（ensemble 子模型同名集合）
WEIGHTABLE_MODELS = ['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost']

# EWMA 平滑系数与 softmax 温度
EWMA_ALPHA = 0.05
SOFTMAX_TAU = 0.15
MIN_SAMPLES = 10
MAX_RECORDS = 200


def _random_base(lottery_type: str) -> Dict:
    """随机基线：红球/蓝球期望命中数"""
    if lottery_type == 'ssq':
        return {'red': 6 * 6 / 33.0, 'blue': 1 * 1 / 16.0}
    return {'red': 5 * 5 / 35.0, 'blue': 2 * 2 / 12.0}


def _bet_excess(record: Dict, base: Dict) -> Optional[float]:
    """单注记录的超额命中得分；缺命中字段返回 None"""
    rh = record.get('red_hits')
    bh = record.get('blue_hits')
    if rh is None or bh is None:
        return None
    return (rh - base['red']) + (bh - base['blue'])


def compute_model_weights(lottery_type: str, records: Optional[List[Dict]] = None) -> Dict:
    """
    由核验记录计算该彩票各模型的前向表现与动态权重。
    
    Returns:
        {
          'lottery_type': ..., 'updated_at': ...,
          'window': {...},
          'random_baseline': {'red': x, 'blue': x},
          'models': { mt: {samples, avg_excess, ewma_excess, avg_red_hits,
                           weight, prize_total, roi, sufficient} },
          'ensemble_sub_weights': {mt: 归一化权重},
          'recommended_model': str|None,
        }
    """
    base = _random_base(lottery_type)
    
    # 从数据库加载已核验记录（若未传入）
    if records is None:
        records = get_predictions(lottery_type=lottery_type, verified=True, limit=None)
    
    # 筛选参与加权的模型记录，按期号升序
    verified = [r for r in records
                if r.get('lottery_type') == lottery_type
                and r.get('target_period') is not None
                and r.get('model_type') in WEIGHTABLE_MODELS]
    try:
        verified.sort(key=lambda r: (int(r.get('target_period') or 0),
                                     str(r.get('predict_time') or '')))
    except (TypeError, ValueError):
        pass
    
    stats = {}
    for mt in WEIGHTABLE_MODELS:
        recs = [r for r in verified if r.get('model_type') == mt][-MAX_RECORDS:]
        scores = [s for r in recs if (s := _bet_excess(r, base)) is not None]
        n = len(scores)
        # EWMA
        ewma = 0.0
        for s in scores:
            ewma = EWMA_ALPHA * s + (1 - EWMA_ALPHA) * ewma
        # 样本不足收缩向 0
        if 0 < n < MIN_SAMPLES:
            ewma *= n / MIN_SAMPLES
        red_hits = [r.get('red_hits') for r in recs if r.get('red_hits') is not None]
        hit_cnt = sum(1 for r in recs if r.get('prize_name') not in (None, '未中奖', ''))
        prize_total = sum(float(r.get('prize_amount') or 0) for r in recs)
        cost = 2.0 * n
        stats[mt] = {
            'samples': n,
            'ewma_excess': round(float(ewma), 5) if n else 0.0,
            'avg_excess': round(float(sum(scores) / n), 5) if n else 0.0,
            'avg_red_hits': round(float(sum(red_hits) / len(red_hits)), 4) if red_hits else 0.0,
            'hit_rate': round(hit_cnt / n, 4) if n else 0.0,
            'prize_total': round(prize_total, 2),
            'roi': round((prize_total - cost) / cost, 4) if n else 0.0,
            'sufficient': n >= MIN_SAMPLES,
        }
    
    # softmax(EWMA超额/tau) → 权重
    logits = [stats[mt]['ewma_excess'] / SOFTMAX_TAU for mt in WEIGHTABLE_MODELS]
    mx = max(logits) if logits else 0.0
    exps = [math.exp(x - mx) for x in logits]
    total = sum(exps) or 1.0
    for mt, e in zip(WEIGHTABLE_MODELS, exps):
        stats[mt]['weight'] = round(e / total, 4)
    
    enough = [mt for mt in WEIGHTABLE_MODELS if stats[mt]['samples'] >= MIN_SAMPLES]
    recommended = None
    if enough:
        recommended = max(enough, key=lambda m: stats[m]['ewma_excess'])
    
    return {
        'lottery_type': lottery_type,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'window': {'max_records': MAX_RECORDS, 'alpha': EWMA_ALPHA,
                   'tau': SOFTMAX_TAU, 'min_samples': MIN_SAMPLES},
        'random_baseline': {'red': round(base['red'], 4), 'blue': round(base['blue'], 4)},
        'models': stats,
        'ensemble_sub_weights': {mt: stats[mt]['weight'] for mt in WEIGHTABLE_MODELS},
        'recommended_model': recommended,
    }


def update_weights(lottery_type: str, records: Optional[List[Dict]] = None) -> Dict:
    """重算并持久化指定彩票的权重（写入数据库），返回权重字典"""
    result = compute_model_weights(lottery_type, records=records)
    try:
        db_add_model_weights(
            lottery_type=lottery_type,
            window=result['window'],
            weights=result['models'],
            ensemble_sub_weights=result['ensemble_sub_weights'],
            recommended_model=result['recommended_model'],
            avg_red_hits=sum(m['avg_red_hits'] for m in result['models'].values()) / len(WEIGHTABLE_MODELS) if WEIGHTABLE_MODELS else 0.0,
            hit_rate=sum(m['hit_rate'] for m in result['models'].values()) / len(WEIGHTABLE_MODELS) if WEIGHTABLE_MODELS else 0.0,
        )
    except Exception as e:
        print(f"[WARN] 权重持久化到数据库失败: {e}")
    return result


def update_all_weights(records: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    """重算并持久化双色球/大乐透两彩种权重"""
    if records is None:
        from prediction_records import load_records
        try:
            records = load_records()
        except Exception:
            records = None
    return {lt: update_weights(lt, records=records) for lt in ('ssq', 'dlt')}


def get_latest_weights(lottery_type: str) -> Optional[Dict]:
    """获取最新权重（从数据库），无记录返回 None"""
    return db_get_latest_model_weights(lottery_type)


def get_model_weights(lottery_type: str) -> Dict[str, float]:
    """
    读取已保存的子模型权重（供 ml_models 集成预测加权）。
    数据库无记录时返回 {}（调用方回退等权）。
    """
    dbw = db_get_latest_model_weights(lottery_type)
    if not dbw:
        return {}
    sub = dbw.get('ensemble_sub_weights') or {}
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = {}
    return {k: float(v) for k, v in sub.items()
            if isinstance(v, (int, float)) and 0.0 < float(v) <= 1.0}


def get_performance_report(lottery_type: str) -> Dict:
    """读取完整表现报告（供UI展示）；无记录时即时计算"""
    dbw = db_get_latest_model_weights(lottery_type)
    if not dbw:
        return compute_model_weights(lottery_type)
    # DB 的 weights 列实际存的是 models 信息，转换为 GUI 期望的结构 {'models': ...}
    report = dict(dbw)
    if 'weights' in report and isinstance(report['weights'], dict):
        report['models'] = report['weights']
    return report


def update_weights_auto(lottery_type: str) -> bool:
    """自动更新权重并持久化到数据库（兼容旧接口）"""
    try:
        update_weights(lottery_type)
        return True
    except Exception as e:
        print(f"[ERROR] 更新 {lottery_type} 权重失败: {e}")
        return False


def export_to_json():
    """导出权重到 JSON 文件（手动备份用）"""
    data = {}
    for lt in ['ssq', 'dlt']:
        weights = get_latest_weights(lt)
        if weights:
            data[lt] = weights
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(data)


if __name__ == "__main__":
    # 命令行: python model_weights.py [ssq|dlt|all]
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if target == 'all':
        results = update_all_weights()
    else:
        results = {target: update_weights(target)}
    for lt, res in results.items():
        print(f"\n=== {lt.upper()} 前向表现与动态权重 ===")
        print(f"随机基线: 红{res['random_baseline']['red']:.2f} "
              f"蓝{res['random_baseline']['blue']:.2f}")
        for mt in WEIGHTABLE_MODELS:
            s = res['models'][mt]
            flag = '' if s['sufficient'] else '(样本不足,中性)'
            print(f"  {mt:<14} 注数{s['samples']:>4} 红均{s['avg_red_hits']:.2f} "
                  f"中奖率{s['hit_rate']*100:5.1f}% EWMA超额{s['ewma_excess']:+.4f} "
                  f"权重{s['weight']:.3f} {flag}")
        print(f"  推荐模型: {res['recommended_model']}")
