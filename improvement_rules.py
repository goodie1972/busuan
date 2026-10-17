# -*- coding:utf-8 -*-
"""
预测改进规则引擎 (Improvement Rules Engine)
==========================================

数据库版半自动优化建议引擎。
替代旧的"健康面板"复杂逻辑，专注一个目标：**提高优化预测概率**。

设计原则：
1. 单一职责：每条规则产生可执行的优化建议（update_weights / analyze_features /
   calibrate / retrain_model / update_data / noop）
2. 半自动：低风险操作标记 auto_executable=True（权重重算、特征分析、校准），
   高风险操作标记 auto_executable=False（重训练模型需用户确认）
3. 基于数据库真实数据：核验记录数、命中率、校准样本、特征漂移分、数据新鲜度
4. 与 statistics_insights 联动：把高级统计的分析结论也转成建议

规则清单（按优先级）：
- R1 数据新鲜度：CSV/DB 最新期 vs 官方 → 建议 update_data
- R2 样本积累：已核验但无概率记录 → 提示积累校准样本
- R3 校准就绪：校准样本 >= 30 且未校准/过期 → 建议 calibrate
- R4 权重重算：有新核验记录且权重更新时间较旧 → 建议 update_weights
- R5 特征漂移：特征漂移分超阈值 → 建议 analyze_features + retrain_model(手动)
- R6 模型表现：近期命中率 < 历史 → 建议 retrain_model（手动）
- R7 数据量不足：核验记录太少无法支撑优化 → 提示继续积累
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from db_access import (
    get_verification_summary,
    get_latest_period,
    get_latest_model_weights,
    get_optimization_history,
)
from model_weights import compute_model_weights, WEIGHTABLE_MODELS

# 阈值配置
MIN_CALIB_SAMPLES = 30        # 校准所需最小样本数
MAX_WEIGHTS_AGE_HOURS = 24    # 权重最大新鲜时长（小时）
DRIFT_THRESHOLD = 1.0        # 特征漂移分阈值（与 feature_analysis 的 is_drifting 阈值一致）
MIN_VERIFIED_TOTAL = 20       # 最小已核验记录数（低于则提示积累）
DATA_STALE_DAYS = 3           # 数据新鲜度天数阈值


def _is_config_json_record(r: dict) -> bool:
    """判断是否为配置型记录（weights 字段是 model 详情映射）"""
    return isinstance(r.get('weights', {}), dict) and len(r.get('weights', {})) > 0


def run_rules(lottery_type: str) -> List[Dict]:
    """
    执行所有规则，生成优化建议列表
    
    Args:
        lottery_type: 'ssq' / 'dlt'
    
    Returns:
        list of dict: 每条建议 {action, reason, params, auto_executable, priority}
    """
    suggestions = []
    
    # ===== R7 数据量检查（先做，数据不足则后续规则大多无意义）=====
    summary = get_verification_summary(lottery_type)
    verified = summary.get('verified', 0)
    with_prob = summary.get('with_probabilities', 0)
    pending = summary.get('pending', 0)
    total = summary.get('total', 0)
    
    if verified < MIN_VERIFIED_TOTAL:
        suggestions.append({
            'action': 'noop',
            'reason': f'已核验记录仅 {verified} 条，不足 {MIN_VERIFIED_TOTAL} 条，'
                      f'无法支撑有意义的优化。请持续生成预测并等待开奖核验（当前待核验 {pending} 条）。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': False,
            'priority': 'high',
        })
        return suggestions
    
    # ===== R1 数据新鲜度 =====
    # 从数据库最新期倒推是否陈旧（这里用最新的权重记录时间做参考）
    csv_period = _get_latest_csv_period(lottery_type)
    db_period = get_latest_period(lottery_type)
    if csv_period and db_period and csv_period != db_period:
        suggestions.append({
            'action': 'update_data',
            'reason': f'数据不一致：CSV 最新期 {csv_period}，数据库最新期 {db_period}。'
                      f'需要同步数据。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': False,
            'priority': 'high',
        })
    
    # ===== R2 校准样本积累 =====
    # 有概率的已核验记录数（校准样本的真正来源）
    calib_samples = _count_calibration_samples(lottery_type)
    
    # ===== R3 校准就绪 =====
    calib_meta = _get_calibration_meta(lottery_type)
    last_calib_time = calib_meta.get('last_train_time')
    need_calib = False
    if calib_samples >= MIN_CALIB_SAMPLES:
        # 有足够样本，但未校准或校准很久没更新
        if last_calib_time is None:
            need_calib = True
            reason = f'已有 {calib_samples} 条带概率的核验记录，校准模型尚未训练。'
        else:
            # 检查是否有新的核验记录在最近校准之后
            new_verified = summary.get('verified', 0)
            reason = f'校准模型上次训练于 {last_calib_time[:10]}，当前样本 {calib_samples} 条。'
            need_calib = True  # 定期重校准
    else:
        reason = f'校准样本不足：当前 {calib_samples} 条，需要 {MIN_CALIB_SAMPLES} 条。'
        need_calib = False
    
    if need_calib:
        suggestions.append({
            'action': 'calibrate',
            'reason': reason,
            'params': {'lottery_type': lottery_type},
            'auto_executable': True,     # 校准低风险，可自动执行
            'priority': 'medium' if calib_samples >= MIN_CALIB_SAMPLES else 'low',
        })
    elif calib_samples < MIN_CALIB_SAMPLES and calib_samples > 0:
        suggestions.append({
            'action': 'noop',
            'reason': f'校准样本积累中：{calib_samples}/{MIN_CALIB_SAMPLES} 条带概率核验记录。'
                      f'样本足够后将自动发起校准。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': False,
            'priority': 'low',
        })
    
    # ===== R4 权重重算 =====
    weights_age = _get_weights_age(lottery_type)
    if weights_age is not None and weights_age > MAX_WEIGHTS_AGE_HOURS:
        suggestions.append({
            'action': 'update_weights',
            'reason': f'模型权重已 {weights_age:.1f} 小时未更新，有新核验记录应重算。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': True,     # 权重重算低风险
            'priority': 'medium',
        })
    
    # ===== R5 特征漂移 =====
    drift = _check_feature_drift(lottery_type)
    if drift is not None and drift > DRIFT_THRESHOLD:
        suggestions.append({
            'action': 'analyze_features',
            'reason': f'检测到特征漂移（分数 {drift:.2f} > 阈值 {DRIFT_THRESHOLD}），'
                      f'建议先分析特征再决定是否重训。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': True,     # 占位：分析特征可自动
            'priority': 'medium',
        })
        # 重训练高风险 → 手动
        suggestions.append({
            'action': 'retrain_model',
            'reason': f'特征漂移明显（分数 {drift:.2f}），建议用最近窗口数据重训模型 '
                      f'以抗漂移。此操作耗时较长，需手动确认。',
            'params': {'lottery_type': lottery_type,
                       'model_types': list(WEIGHTABLE_MODELS),
                       'recent_periods': 50},
            'auto_executable': False,
            'priority': 'high',
        })
    
    # ===== R6 模型表现下降 =====
    perf = _check_performance_drop(lottery_type)
    if perf is not None and perf:
        suggestions.append({
            'action': 'retrain_model',
            'reason': f'近期模型表现较历史下降（近窗红球均命中 {perf:.2f}），'
                      f'建议重训模型。',
            'params': {'lottery_type': lottery_type,
                       'model_types': list(WEIGHTABLE_MODELS),
                       'recent_periods': 0},  # 全量重训
            'auto_executable': False,
            'priority': 'medium',
        })
    
    # 如果没有建议，给一个 noop 说明
    if not suggestions:
        suggestions.append({
            'action': 'noop',
            'reason': f'系统运行正常：核验记录 {verified} 条，权重新鲜。无需操作。',
            'params': {'lottery_type': lottery_type},
            'auto_executable': False,
            'priority': 'none',
        })
    
    return suggestions


def _count_calibration_samples(lottery_type: str) -> int:
    """统计带概率且已核验的记录数（校准样本来源）"""
    from db_access import execute_query
    rows = execute_query("""
        SELECT COUNT(*) as cnt FROM predictions
        WHERE lottery_type = ? AND verified_at IS NOT NULL
          AND red_probabilities IS NOT NULL
    """, (lottery_type,))
    return rows[0]['cnt'] if rows else 0


def _get_calibration_meta(lottery_type: str) -> Dict:
    """从优化日志中查找最近一次校准记录"""
    rows = get_optimization_history(lottery_type=lottery_type,
                                    action_type='calibrate',
                                    limit=1)
    if rows:
        return {'last_train_time': rows[0]['timestamp']}
    return {}


def _get_weights_age(lottery_type: str) -> Optional[float]:
    """计算权重文件年龄（小时）"""
    weights = get_latest_model_weights(lottery_type)
    if not weights:
        return None
    try:
        updated = datetime.fromisoformat(weights['updated_at'])
        age_hours = (datetime.now() - updated).total_seconds() / 3600
        return age_hours
    except Exception:
        return None


def _get_latest_csv_period(lottery_type: str) -> Optional[int]:
    """从 CSV 读取最新期号（仅用于判断数据不同步）"""
    import pandas as pd
    csv_path = {
        'ssq': 'scripts/ssq/ssq_history.csv',
        'dlt': 'scripts/dlt/dlt_history.csv',
    }.get(lottery_type)
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, encoding='gbk')
        return int(df.iloc[0]['期数'])
    except Exception:
        return None


def _check_feature_drift(lottery_type: str) -> Optional[float]:
    """特征漂移分（0-1），无法计算返回 None"""
    try:
        from feature_analysis import detect_feature_drift
        result = detect_feature_drift(lottery_type, window_size=50)
        # 返回漂移分数（0-1），取所有特征的最大漂移分
        return result.get('max_drift_score', 0.0) if isinstance(result, dict) else None
    except Exception:
        # 回退：不报阻断
        return None


def _check_performance_drop(lottery_type: str) -> Optional[float]:
    """检查近期表现 vs 历史表现是否下降，返回近期 avg_red_hits（若下降）"""
    from db_access import execute_query
    try:
        # 历史（全部已核验）平均红球命中
        hist = execute_query("""
            SELECT AVG(red_hits) as avg_rh FROM predictions
            WHERE lottery_type = ? AND verified_at IS NOT NULL
        """, (lottery_type,))
        hist_avg = hist[0]['avg_rh'] if hist and hist[0]['avg_rh'] is not None else 0.0
        
        # 近30天
        recent = execute_query("""
            SELECT AVG(red_hits) as avg_rh FROM predictions
            WHERE lottery_type = ? AND verified_at IS NOT NULL
              AND verified_at >= datetime('now', '-30 days')
        """, (lottery_type,))
        recent_avg = recent[0]['avg_rh'] if recent and recent[0]['avg_rh'] is not None else 0.0
        
        if hist_avg > 0 and recent_avg < hist_avg * 0.8:
            return recent_avg
        return None
    except Exception:
        return None


def get_improvement_plan(lottery_type: str = 'ssq') -> Dict:
    """
    生成改进计划的完整结果
    
    Returns:
        dict: {
            'lottery_type': str,
            'timestamp': str,
            'verified_count': int,
            'pending_count': int,
            'suggestions': [ {action, reason, params, auto_executable, priority}, ...],
            'auto_exec_list': [suggestion, ...],      # 可自动执行的列表
            'manual_confirm_list': [suggestion, ...], # 需手动确认的列表
        }
    """
    suggestions = run_rules(lottery_type)
    
    auto_exec_list = [s for s in suggestions if s.get('auto_executable')]
    manual_confirm_list = [s for s in suggestions if not s.get('auto_executable')]
    
    return {
        'lottery_type': lottery_type,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'verified_count': get_verification_summary(lottery_type).get('verified', 0),
        'pending_count': get_verification_summary(lottery_type).get('pending', 0),
        'suggestions': suggestions,
        'auto_exec_list': auto_exec_list,
        'manual_confirm_list': manual_confirm_list,
    }


if __name__ == "__main__":
    # 测试规则引擎
    print("Testing improvement rules engine...")
    
    for lt in ['ssq', 'dlt']:
        print(f"\n=== {lt.upper()} ===")
        plan = get_improvement_plan(lt)
        print(f"Verified: {plan['verified_count']}, Pending: {plan['pending_count']}")
        print(f"Auto-executable: {len(plan['auto_exec_list'])}")
        print(f"Manual-confirm: {len(plan['manual_confirm_list'])}")
        
        for s in plan['suggestions']:
            marker = '[AUTO]' if s['auto_executable'] else '[MANUAL]'
            print(f"  {marker} {s['action']}: {s['reason'][:60]}...")
    
    print("\n[OK] Improvement rules engine test passed!")
