# -*- coding:utf-8 -*-
"""
模型动态权重模块（"预测→核验→反哺"闭环的结果端）

思路（对应2026-10-08确认的优化方案第3条"集成权重动态调整"）：
- 用已核验的预测记录（target_period 非空）计算每个模型类型最近 K 注的
  "超额命中" EWMA：每注得分 = (红球命中 - 随机基线) + (蓝球命中 - 随机基线)。
  预测先于开奖生成，核验后于开奖更新 —— 这是模型的真实前向表现，无泄漏。
- 权重 = softmax(EWMA超额 / tau)。样本不足的模型按中性(0)参与，不奖不罚。
- 集成模型(ensemble)内部子模型加权：ensemble 的子模型与独立的同名模型
  (random_forest/xgboost/gbdt/lightgbm/catboost) 是同一算法同一流程训练的，
  故用同名独立模型的前向表现作为子模型权重代理。
- 诚实声明：这不能提高彩票中奖概率（开奖独立随机），它只是让"近期表现
  好的模型"在集成投票/选模建议中获得更高话语权，属于自适应模型管理。

权重文件: model_weights.json（与 prediction_history.json 同目录）
"""
import json
import os
import time
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(PROJECT_DIR, 'model_weights.json')

# 参与加权的模型（ensemble 子模型同名集合；ziwei/meihua/lstm-crf 等不参与投票加权）
WEIGHTABLE_MODELS = ['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost']

# EWMA 平滑系数与 softmax 温度
EWMA_ALPHA = 0.05      # 每注更新权重，半衰期约14注
SOFTMAX_TAU = 0.15     # 每注超额差0.15球 ≈ e倍权重差
MIN_SAMPLES = 10       # 低于此样本量按中性参与
MAX_RECORDS = 200      # 每个模型最多取最近N注已核验记录

# 随机基线：n选m 的彩票，随机选 k 个号码的期望命中数 = k*m/总数
# SSQ: 红 6/33 选6 → 6*6/33; 蓝 1/16 选1 → 1*1/16
# DLT: 红 5/35 选5 → 5*5/35; 蓝 2/12 选2 → 2*2/12
_RANDOM_BASE = {
    'ssq': {'red': 6 * 6 / 33.0, 'blue': 1 * 1 / 16.0},
    'dlt': {'red': 5 * 5 / 35.0, 'blue': 2 * 2 / 12.0},
}


def _load_weights_file() -> Dict:
    """读取权重文件（不存在/损坏返回空结构）"""
    try:
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def _save_weights_file(data: Dict) -> bool:
    """原子写权重文件"""
    try:
        tmp = WEIGHTS_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, WEIGHTS_FILE)
        return True
    except OSError:
        return False


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

    Args:
        lottery_type: 'ssq' 或 'dlt'
        records: 预测记录列表（None则从 prediction_records 加载）

    Returns:
        {
          'lottery_type': ..., 'updated_at': ...,
          'models': {
             model_type: {'samples': n, 'avg_excess': x, 'ewma_excess': x,
                          'avg_red_hits': x, 'hit_rate': x, 'weight': w,
                          'prize_total': p, 'roi': r, 'sufficient': bool}
          },
          'ensemble_sub_weights': {子模型名: 归一化权重},
          'recommended_model': 权重最高的样本充足模型或 None,
        }
    """
    if records is None:
        from prediction_records import load_records
        records = load_records()

    base = _RANDOM_BASE.get(lottery_type, _RANDOM_BASE['ssq'])
    # 已核验记录按期号升序（predict_time 兜底），仅取真实 ML 模型
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
        # EWMA（按时间顺序递推；初始为0=中性先验）
        ewma = 0.0
        for s in scores:
            ewma = EWMA_ALPHA * s + (1 - EWMA_ALPHA) * ewma
        # 样本不足：EWMA 向 0 收缩（收缩量=未见样本的隐式先验质量）
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

    # softmax(EWMA超额/tau) → 归一化权重（全部模型参与；中性样本得基准权重）
    logits = [stats[mt]['ewma_excess'] / SOFTMAX_TAU for mt in WEIGHTABLE_MODELS]
    mx = max(logits) if logits else 0.0
    exps = [__import__('math').exp(x - mx) for x in logits]
    total = sum(exps) or 1.0
    for mt, e in zip(WEIGHTABLE_MODELS, exps):
        stats[mt]['weight'] = round(e / total, 4)

    # 子集内归一（供 ensemble 子模型加权用：只对模型文件中实际存在的子模型再归一由调用方处理）
    enough = [mt for mt in WEIGHTABLE_MODELS if stats[mt]['samples'] >= MIN_SAMPLES]
    recommended = None
    if enough:
        recommended = max(enough, key=lambda m: stats[m]['ewma_excess'])

    return {
        'lottery_type': lottery_type,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'window': {'max_records': MAX_RECORDS, 'alpha': EWMA_ALPHA, 'tau': SOFTMAX_TAU,
                   'min_samples': MIN_SAMPLES},
        'random_baseline': {'red': round(base['red'], 4), 'blue': round(base['blue'], 4)},
        'models': stats,
        'ensemble_sub_weights': {mt: stats[mt]['weight'] for mt in WEIGHTABLE_MODELS},
        'recommended_model': recommended,
    }


def update_weights(lottery_type: str, records: Optional[List[Dict]] = None) -> Dict:
    """重算并持久化指定彩票的权重，返回权重字典"""
    result = compute_model_weights(lottery_type, records=records)
    data = _load_weights_file()
    data[lottery_type] = result
    _save_weights_file(data)
    return result


def update_all_weights(records: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    """重算并提交/大乐透两彩种权重"""
    if records is None:
        from prediction_records import load_records
        records = load_records()
    return {lt: update_weights(lt, records=records) for lt in ('ssq', 'dlt')}


def get_model_weights(lottery_type: str) -> Dict[str, float]:
    """
    读取已保存的子模型权重（供 ml_models 集成预测加权）。
    文件不存在时返回 {}（调用方应回退等权，保持旧行为）。
    """
    data = _load_weights_file()
    entry = data.get(lottery_type) or {}
    sub = entry.get('ensemble_sub_weights') or {}
    # 仅保留 0<权重<=1 的合法项
    return {k: float(v) for k, v in sub.items()
            if isinstance(v, (int, float)) and 0.0 < float(v) <= 1.0}


def get_performance_report(lottery_type: str) -> Dict:
    """读取完整表现报告（供UI展示）；无文件时即时计算"""
    data = _load_weights_file()
    entry = data.get(lottery_type)
    if not entry:
        return compute_model_weights(lottery_type)
    return entry


if __name__ == '__main__':
    # 命令行: python model_weights.py [ssq|dlt|all]
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if target == 'all':
        results = update_all_weights()
    else:
        results = {target: update_weights(target)}
    for lt, res in results.items():
        print(f"\n=== {lt.upper()} 前向表现与动态权重 ===")
        print(f"随机基线: 红{res['random_baseline']['red']:.2f}球 "
              f"蓝{res['random_baseline']['blue']:.2f}球")
        models = res['models']
        for mt in WEIGHTABLE_MODELS:
            s = models[mt]
            flag = '' if s['sufficient'] else '(样本不足,中性)'
            print(f"  {mt:<14} 注数{s['samples']:>4}  红均{s['avg_red_hits']:.2f} "
                  f"中奖率{s['hit_rate']*100:5.1f}%  EWMA超额{s['ewma_excess']:+.4f} "
                  f"权重{s['weight']:.3f} {flag}")
        print(f"  推荐模型: {res['recommended_model']}")
