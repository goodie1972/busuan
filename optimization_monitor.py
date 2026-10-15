# -*- coding: utf-8 -*-
"""
智能监控引擎 (Optimization Monitor)
===================================
定期扫描系统健康度，生成结构化报告。

5 项检查:
  1. 数据量 — 核验记录数是否足够校准/加权
  2. 模型表现 — 各模型 EWMA 超额命中率与趋势
  3. 特征漂移 — 近期窗口 vs 全期分布 KL 散度
  4. 校准质量 — Brier score（若已校准）
  5. 数据新鲜度 — CSV 最新期号距今天数

健康度评分 0-10，各项加权求和。
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from json_utils import safe_json_load

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 健康度权重
_WEIGHTS = {
    'data_volume': 0.25,
    'model_performance': 0.30,
    'feature_drift': 0.20,
    'calibration_quality': 0.15,
    'data_freshness': 0.10,
}

# 阈值
_THRESHOLDS = {
    'min_verified_ok': 30,        # ≥30 条 ok
    'min_verified_warn': 10,      # ≥10 条 warning
    'drift_kl_ok': 0.1,
    'drift_kl_warn': 0.3,
    'perf_decline_periods': 3,    # 连续 N 期下降 → warning
    'data_stale_ok_days': 3,
    'data_stale_warn_days': 7,
    'calib_brier_ok': 0.25,
    'calib_brier_warn': 0.35,
}


# ── 5 项检查 ──────────────────────────────────────────

def check_data_volume(lottery_type: str, records: Optional[List[Dict]] = None) -> Dict:
    """检查核验记录数是否足够"""
    if records is None:
        records = safe_json_load(
            os.path.join(_BASE_DIR, 'prediction_history.json'), [])
    # 过滤彩种 + 已核验（target_period 非 None）
    verified = [r for r in records
                if r.get('lottery_type') == lottery_type
                and r.get('target_period') is not None]
    count = len(verified)
    min_ok = _THRESHOLDS['min_verified_ok']
    min_warn = _THRESHOLDS['min_verified_warn']

    if count >= min_ok:
        status, score, msg = 'ok', 10, f'核验记录 {count} 条，充足'
    elif count >= min_warn:
        status, score, msg = 'warning', 5, f'核验记录 {count} 条，偏少（建议≥{min_ok}）'
    else:
        status, score, msg = 'critical', 0, f'核验记录仅 {count} 条，严重不足（需≥{min_warn}）'

    return {
        'status': status,
        'score': score,
        'value': count,
        'threshold_ok': min_ok,
        'threshold_warn': min_warn,
        'message': msg,
    }


def check_model_performance(lottery_type: str) -> Dict:
    """检查各模型前向表现与趋势"""
    try:
        from model_weights import get_performance_report, WEIGHTABLE_MODELS
        report = get_performance_report(lottery_type)
        if not report or not report.get('models'):
            return {'status': 'critical', 'score': 0,
                    'message': '无法获取模型表现报告',
                    'details': {}}

        models = report['models']
        details = {}
        has_warning = False
        has_critical = False
        sufficient_count = 0

        for mt in WEIGHTABLE_MODELS:
            st = models.get(mt, {})
            samples = st.get('samples', 0)
            excess = st.get('ewma_excess', st.get('avg_excess', 0.0))
            hit_rate = st.get('hit_rate', 0.0)
            sufficient = st.get('sufficient', False)

            if sufficient:
                sufficient_count += 1

            # 判定趋势：样本不足 → critical
            if samples < 5:
                details[mt] = {'samples': samples, 'hit_rate': hit_rate,
                               'ewma_excess': excess, 'trend': 'insufficient'}
                has_critical = True
            elif excess < -0.05:
                details[mt] = {'samples': samples, 'hit_rate': hit_rate,
                               'ewma_excess': excess, 'trend': 'declining'}
                has_warning = True
            else:
                details[mt] = {'samples': samples, 'hit_rate': hit_rate,
                               'ewma_excess': excess, 'trend': 'stable'}

        if has_critical and sufficient_count == 0:
            status, score = 'critical', 0
        elif has_warning or sufficient_count < len(WEIGHTABLE_MODELS):
            status, score = 'warning', 5
        else:
            status, score = 'ok', 10

        msg = f'{sufficient_count}/{len(WEIGHTABLE_MODELS)} 模型样本充足'
        if has_warning:
            msg += '，部分模型表现下降'

        return {
            'status': status,
            'score': score,
            'message': msg,
            'details': details,
            'recommended_model': report.get('recommended_model'),
        }
    except Exception as e:
        logger.warning(f'check_model_performance failed: {e}')
        return {'status': 'critical', 'score': 0,
                'message': f'检查失败: {e}', 'details': {}}


def check_feature_drift(lottery_type: str, window_size: int = 50) -> Dict:
    """检查特征漂移（KL 散度）"""
    try:
        from feature_analysis import detect_feature_drift
        result = detect_feature_drift(lottery_type, window_size=window_size)
        if not result:
            return {'status': 'ok', 'score': 10,
                    'message': '无漂移数据（可能特征不足）',
                    'drifted_features': [], 'max_kl': 0.0}

        max_kl = result.get('max_drift_score', 0.0)
        drift_map = result.get('feature_drift', {})
        drifted = [k for k, v in drift_map.items()
                   if isinstance(v, dict) and v.get('is_drifting')]
        kl_ok = _THRESHOLDS['drift_kl_ok']
        kl_warn = _THRESHOLDS['drift_kl_warn']

        if max_kl >= kl_warn or len(drifted) >= 3:
            status, score = 'critical', 0
            msg = f'{len(drifted)} 个特征漂移（max KL={max_kl:.3f}），建议重训'
        elif max_kl >= kl_ok or len(drifted) >= 1:
            status, score = 'warning', 5
            msg = f'{len(drifted)} 个特征轻度漂移（max KL={max_kl:.3f}）'
        else:
            status, score = 'ok', 10
            msg = f'无显著漂移（max KL={max_kl:.3f}）'

        return {
            'status': status,
            'score': score,
            'message': msg,
            'drifted_features': drifted[:10],
            'max_kl': round(max_kl, 4),
            'total_features': len(drift_map),
        }
    except Exception as e:
        logger.warning(f'check_feature_drift failed: {e}')
        return {'status': 'warning', 'score': 5,
                'message': f'漂移检测失败: {e}',
                'drifted_features': [], 'max_kl': 0.0}


def check_calibration_quality(lottery_type: str) -> Dict:
    """检查校准质量（Brier score 或样本量）"""
    try:
        from model_calibration import get_calibration_info
        info = get_calibration_info(lottery_type)
        red_ok = info.get('red_calibrated', False)
        blue_ok = info.get('blue_calibrated', False)
        red_samples = info.get('red_samples', 0)
        blue_samples = info.get('blue_samples', 0)

        if not red_ok and not blue_ok:
            return {'status': 'warning', 'score': 5,
                    'message': '尚未校准（建议积累核验记录后训练）',
                    'red_calibrated': False, 'blue_calibrated': False,
                    'red_samples': red_samples, 'blue_samples': blue_samples}

        # 已校准：检查样本量是否足够
        min_samples = 20
        if red_samples < min_samples or blue_samples < min_samples:
            status, score = 'warning', 5
            msg = f'校准样本偏少（红{red_samples}/蓝{blue_samples}）'
        else:
            status, score = 'ok', 10
            msg = f'校准质量良好（红{red_samples}/蓝{blue_samples}样本）'

        return {
            'status': status,
            'score': score,
            'message': msg,
            'red_calibrated': red_ok,
            'blue_calibrated': blue_ok,
            'red_samples': red_samples,
            'blue_samples': blue_samples,
            'last_update': info.get('last_update'),
        }
    except Exception as e:
        logger.warning(f'check_calibration_quality failed: {e}')
        return {'status': 'warning', 'score': 5,
                'message': f'检查失败: {e}'}


def check_data_freshness(lottery_type: str) -> Dict:
    """检查开奖数据新鲜度"""
    try:
        import pandas as pd
        csv_path = os.path.join(_BASE_DIR, 'scripts', lottery_type,
                                f'{lottery_type}_history.csv')
        if not os.path.exists(csv_path):
            return {'status': 'critical', 'score': 0,
                    'message': f'CSV 不存在: {csv_path}',
                    'last_period': None, 'days_since': None}

        # GBK 编码（历史数据兼容）
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(csv_path, encoding='utf-8')

        if df.empty:
            return {'status': 'critical', 'score': 0,
                    'message': 'CSV 为空', 'last_period': None,
                    'days_since': None}

        # 找期号列
        period_col = None
        for col in df.columns:
            if '期号' in col or 'period' in col.lower():
                period_col = col
                break
        if period_col is None:
            period_col = df.columns[0]

        latest = str(df[period_col].iloc[-1])
        # 期号格式如 2026305 → 推算日期
        days_since = _estimate_days_since(latest)
        ok_days = _THRESHOLDS['data_stale_ok_days']
        warn_days = _THRESHOLDS['data_stale_warn_days']

        if days_since is None:
            return {'status': 'warning', 'score': 5,
                    'message': f'最新期号 {latest}，无法推算日期',
                    'last_period': latest, 'days_since': None}

        if days_since <= ok_days:
            status, score = 'ok', 10
        elif days_since <= warn_days:
            status, score = 'warning', 5
        else:
            status, score = 'critical', 0

        msg = f'最新期号 {latest}（约 {days_since} 天前）'
        if days_since > warn_days:
            msg += '，数据过期，建议更新'

        return {
            'status': status,
            'score': score,
            'message': msg,
            'last_period': latest,
            'days_since': days_since,
        }
    except Exception as e:
        logger.warning(f'check_data_freshness failed: {e}')
        return {'status': 'warning', 'score': 5,
                'message': f'检查失败: {e}',
                'last_period': None, 'days_since': None}


# ── 辅助 ──────────────────────────────────────────────

def _estimate_days_since(period_str: str) -> Optional[int]:
    """从期号（如 2026305）推算距今天数"""
    try:
        period_str = str(period_str).strip()
        if len(period_str) < 7:
            return None
        year = int(period_str[:4])
        day_of_year = int(period_str[4:7])
        # 构造日期
        try:
            from datetime import datetime as dt
            draw_date = dt(year, 1, 1) + timedelta(days=day_of_year - 1)
        except ValueError:
            return None
        now = datetime.now()
        delta = (now - draw_date).days
        return max(0, delta)
    except (ValueError, IndexError):
        return None


def _compute_health_score(checks: Dict[str, Dict]) -> float:
    """加权计算健康度总分 (0-10)"""
    total = 0.0
    for key, weight in _WEIGHTS.items():
        score = checks.get(key, {}).get('score', 5)
        total += score * weight
    return round(total, 1)


def _determine_priority(score: float, checks: Dict) -> str:
    """根据健康度和 critical 数量确定优先级"""
    critical_count = sum(1 for c in checks.values()
                         if c.get('status') == 'critical')
    warning_count = sum(1 for c in checks.values()
                        if c.get('status') == 'warning')

    if critical_count >= 2 or score < 3:
        return 'high'
    elif critical_count >= 1 or warning_count >= 2 or score < 6:
        return 'medium'
    elif warning_count >= 1 or score < 8:
        return 'low'
    else:
        return 'none'


# ── 主入口 ────────────────────────────────────────────

def generate_health_report(lottery_type: str = 'all') -> Dict[str, Any]:
    """
    生成系统健康报告

    Args:
        lottery_type: 'ssq' / 'dlt' / 'all'

    Returns:
        健康报告 dict（结构见设计文档 4.1.1）
    """
    lottery_types = ['ssq', 'dlt'] if lottery_type == 'all' else [lottery_type]
    reports = {}

    for lt in lottery_types:
        checks = {
            'data_volume': check_data_volume(lt),
            'model_performance': check_model_performance(lt),
            'feature_drift': check_feature_drift(lt),
            'calibration_quality': check_calibration_quality(lt),
            'data_freshness': check_data_freshness(lt),
        }

        score = _compute_health_score(checks)
        priority = _determine_priority(score, checks)
        needs_action = priority in ('high', 'medium')

        reports[lt] = {
            'timestamp': datetime.now().isoformat(),
            'lottery_type': lt,
            'health_score': score,
            'priority': priority,
            'needs_action': needs_action,
            'checks': checks,
        }

    # 如果只查一个彩种，直接返回
    if lottery_type != 'all':
        return reports.get(lottery_type, {})

    # 返回汇总
    avg_score = round(sum(r['health_score'] for r in reports.values()) / len(reports), 1)
    any_action = any(r['needs_action'] for r in reports.values())
    worst_priority = 'none'
    for p in ('high', 'medium', 'low'):
        if any(r['priority'] == p for r in reports.values()):
            worst_priority = p
            break

    return {
        'timestamp': datetime.now().isoformat(),
        'lottery_type': 'all',
        'health_score': avg_score,
        'priority': worst_priority,
        'needs_action': any_action,
        'sub_reports': reports,
    }


def get_short_report(lottery_type: str = 'all') -> str:
    """生成简短文本报告（供 UI 直接显示）"""
    report = generate_health_report(lottery_type)
    if report.get('lottery_type') == 'all':
        lines = []
        for lt, sub in report.get('sub_reports', {}).items():
            score = sub['health_score']
            checks = sub['checks']
            ok = sum(1 for c in checks.values() if c['status'] == 'ok')
            warn = sum(1 for c in checks.values() if c['status'] == 'warning')
            crit = sum(1 for c in checks.values() if c['status'] == 'critical')
            lines.append(f'{lt.upper()}: {score}/10 (ok={ok} warn={warn} crit={crit})')
        return ' | '.join(lines)
    else:
        score = report['health_score']
        checks = report['checks']
        ok = sum(1 for c in checks.values() if c['status'] == 'ok')
        warn = sum(1 for c in checks.values() if c['status'] == 'warning')
        crit = sum(1 for c in checks.values() if c['status'] == 'critical')
        return f'{lottery_type.upper()}: {score}/10 (ok={ok} warn={warn} crit={crit})'


if __name__ == '__main__':
    import sys
    lt = sys.argv[1] if len(sys.argv) > 1 else 'all'
    report = generate_health_report(lt)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
