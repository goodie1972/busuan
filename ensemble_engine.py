# -*- coding: utf-8 -*-
"""
集成预测引擎 V2 - 完整算法实现
=========================================

核心功能：
1. 每次集成预测调用5个子模型，记录中间预测结果（概率分布）
2. 按权重汇总所有模型的概率分布，选出top号码作为最终结果
3. 每次开奖后对中间值做确认，更新每个模型的权重
4. 最近10次的每个模型的表现做加权平均，作为下一次预测的权重基础
5. 连续命中、趋势递增、加速因子加成，全方位评估模型表现

算法公式：
  基础超额命中率: excess = (red_hits/5 - baseline_red) + (blue_hits/2 - baseline_blue)
  连续命中加成: streak_bonus = 0.05 * streak^2  (二次增长)
  趋势加成: trend_bonus = 0.1 * max(0, trend)  (只奖励上升趋势)
  加速因子: 最近2次 > 前2次时 accel = 1.2
  综合得分: score = (excess + streak_bonus + trend_bonus) * accel
  EWMA平滑: ewma = 0.3 * score + 0.7 * ewma_prev
  最近10次加权平均: recent_avg = Σ(0.9^i * ewma_i) / Σ(0.9^i)
  Softmax归一化: weight_i = exp(recent_avg_i / 0.1) / Σ exp(recent_avg_j / 0.1)
"""

import json
import os
import math
import uuid
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ============ 配置参数 ============
ALPHA = 0.3              # EWMA平滑系数
TAU = 0.1                # Softmax温度参数
MIN_SAMPLES = 5          # 最小样本数（不足时收缩向0）
RECENT_WINDOW = 10       # 最近N次加权平均的窗口大小
EMA_WINDOW = 20          # EWMA 累计历史窗口（20期）
DECAY_FACTOR = 0.9       # 近期权重衰减因子
STREAK_BONUS_BASE = 0.05 # 连续命中加成基数
TREND_BONUS_RATE = 0.1   # 趋势加成系数
ACCEL_FACTOR = 1.2       # 加速因子

# ============ 随机基线 ============
def random_baseline(lottery_type: str) -> Dict:
    """计算随机选号的理论命中概率"""
    if lottery_type == 'ssq':
        red_total, red_pick = 33, 6
        blue_total, blue_pick = 16, 1
    else:  # dlt
        red_total, red_pick = 35, 5
        blue_total, blue_pick = 12, 2
    
    from math import comb
    red_baseline = comb(red_pick, red_pick) / comb(red_total, red_pick)
    blue_baseline = comb(blue_pick, blue_pick) / comb(blue_total, blue_pick)
    
    return {
        'red': red_baseline, 'blue': blue_baseline,
        'red_total': red_total, 'blue_total': blue_total,
        'red_pick': red_pick, 'blue_pick': blue_pick
    }


def compute_excess(red_hits: int, blue_hits: int,
                   red_pick: int, blue_pick: int,
                   baseline: Dict) -> float:
    """计算超额命中率"""
    red_rate = red_hits / red_pick if red_pick > 0 else 0
    blue_rate = blue_hits / blue_pick if blue_pick > 0 else 0
    excess = (red_rate - baseline['red']) + (blue_rate - baseline['blue'])
    return excess


def compute_streak_bonus(streak: int) -> float:
    """连续命中加成 - 二次增长
    streak=1: 0.05, streak=2: 0.20, streak=3: 0.45, streak=4: 0.80
    """
    if streak <= 0:
        return 0.0
    return STREAK_BONUS_BASE * (streak ** 2)


def compute_trend_bonus(hit_history: List[Tuple[int, int]], window: int = 3) -> float:
    """趋势加成 - 识别命中数递增趋势"""
    if len(hit_history) < 2:
        return 0.0
    
    recent = hit_history[-window:]
    older = hit_history[-(window*2):-window] if len(hit_history) >= window*2 else []
    
    if not older:
        return 0.0
    
    recent_avg = sum(r + b for r, b in recent) / len(recent)
    older_avg = sum(r + b for r, b in older) / len(older)
    trend = (recent_avg - older_avg) / max(len(recent), 1)
    
    # 只奖励上升趋势
    return max(0.0, TREND_BONUS_RATE * trend)


def compute_accel_factor(hit_history: List[Tuple[int, int]]) -> float:
    """加速因子 - 最近表现优于历史时加速"""
    if len(hit_history) < 4:
        return 1.0
    
    recent_2 = hit_history[-2:]
    older_2 = hit_history[-4:-2]
    
    recent_sum = sum(r + b for r, b in recent_2)
    older_sum = sum(r + b for r, b in older_2)
    
    if recent_sum > older_sum:
        return ACCEL_FACTOR
    return 1.0


def compute_model_score(hit_history: List[Tuple[int, int]],
                        red_pick: int, blue_pick: int,
                        baseline: Dict) -> float:
    """计算单个模型的综合得分"""
    if not hit_history:
        return 0.0

    # 最近一次命中数
    latest_red, latest_blue = hit_history[-1]

    # 基础超额命中率
    red_rate = latest_red / red_pick if red_pick > 0 else 0
    blue_rate = latest_blue / blue_pick if blue_pick > 0 else 0
    excess_base = (red_rate - baseline['red']) + (blue_rate - baseline['blue'])

    # 连续命中加成（连续超越基线且不下降的次数，最多MAX_STREAK）
    # 只奖励真正有表现的期数，避免全是1命中时的虚假长streak
    MAX_STREAK = 10  # 连续加成上限
    streak = 0
    for i in range(len(hit_history) - 1, -1, -1):
        r, b = hit_history[i]
        r_rate = r / red_pick if red_pick > 0 else 0
        b_rate = b / blue_pick if blue_pick > 0 else 0
        excess = (r_rate - baseline['red']) + (b_rate - baseline['blue'])
        if excess > 0:  # 超过基线才算
            streak += 1
        else:
            break
        if streak >= MAX_STREAK:
            break
    streak_bonus = compute_streak_bonus(streak)

    # 趋势加成
    trend_bonus = compute_trend_bonus(hit_history)

    # 加速因子
    accel = compute_accel_factor(hit_history)

    # 综合得分
    score = (excess_base + streak_bonus + trend_bonus) * accel

    return score


def compute_ewma(scores: List[float], alpha: float = ALPHA) -> float:
    """计算指数加权移动平均"""
    if not scores:
        return 0.0
    ewma = 0.0
    for s in scores:
        ewma = alpha * s + (1 - alpha) * ewma
    return ewma


def compute_recent_weighted_avg(ewma_values: List[float],
                                 window: int = RECENT_WINDOW,
                                 decay: float = DECAY_FACTOR) -> float:
    """最近N次的加权平均（衰减权重，最近的权重最高）"""
    if not ewma_values:
        return 0.0
    recent = ewma_values[-window:]
    weights = [decay ** i for i in range(len(recent))]
    weights = weights[::-1]  # 最近的权重最高
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(w * v for w, v in zip(weights, recent)) / total_weight


def softmax(logits: List[float], tau: float = TAU) -> List[float]:
    """Softmax归一化"""
    if not logits:
        return []
    mx = max(logits)
    exps = [math.exp((x - mx) / tau) for x in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


# ============ 中间预测记录管理 ============

class EnsemblePredictionRecorder:
    """记录每次集成预测的中间结果"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, 'ensemble_predictions.json')
        self.records = self._load()
    
    def _load(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _save(self):
        import tempfile
        import os
        # Write to temp file first, then atomic rename
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', 
                                        dir=os.path.dirname(self.file_path) or '.',
                                        delete=False) as tmp:
            json.dump(self.records, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, self.file_path)
    
    def record_prediction(self, lottery_type: str, target_period: int,
                         sub_predictions: Dict[str, Dict],
                         final_red: List[int], final_blue: List[int],
                         weights: Dict[str, float]) -> str:
        """记录一次集成预测的中间结果"""
        record = {
            'ensemble_id': str(uuid.uuid4())[:8],
            'predict_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'lottery_type': lottery_type,
            'target_period': target_period,
            'sub_predictions': sub_predictions,
            'final_red': final_red,
            'final_blue': final_blue,
            'weights_used': weights.copy()
        }
        self.records.append(record)
        self._save()
        return record['ensemble_id']
    
    def mark_verified(self, ensemble_id: str, draw_red: List[int],
                     draw_blue: List[int]) -> Dict:
        """标记已核验，计算每个子模型的表现"""
        record = next((r for r in self.records if r.get('ensemble_id') == ensemble_id), None)
        if not record:
            return None
        
        red_pick = 5 if record['lottery_type'] == 'dlt' else 6
        blue_pick = 2 if record['lottery_type'] == 'dlt' else 1
        
        sub_performance = {}
        for model_name, probs in record.get('sub_predictions', {}).items():
            red_probs = probs.get('red_probs', {})
            blue_probs = probs.get('blue_probs', {})
            
            sorted_red = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)[:red_pick]
            sorted_blue = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)[:blue_pick]
            
            model_red = [k for k, v in sorted_red]
            model_blue = [k for k, v in sorted_blue]
            
            red_hits = len(set(model_red) & set(draw_red))
            blue_hits = len(set(model_blue) & set(draw_blue))
            
            sub_performance[model_name] = {
                'red_hits': red_hits,
                'blue_hits': blue_hits,
                'model_red': model_red,
                'model_blue': model_blue
            }
        
        record['verified'] = True
        record['draw_red'] = draw_red
        record['draw_blue'] = draw_blue
        record['sub_performance'] = sub_performance
        self._save()
        
        return sub_performance
    
    def get_model_hit_history(self, lottery_type: str, model_name: str) -> List[Tuple[int, int]]:
        """获取指定模型的历史命中数序列 [(red_hits, blue_hits), ...]
        
        优先从 ensemble_predictions.json 读取（含中间值），
        兼容旧的 prediction_records.py 数据
        去重：同一 target_period 只保留最新的一条
        """
        history = []
        for r in self.records:
            if (r.get('lottery_type') == lottery_type
                and r.get('verified')
                and r.get('sub_performance', {}).get(model_name)):
                perf = r['sub_performance'][model_name]
                history.append((perf['red_hits'], perf['blue_hits']))
        
        # 兼容旧的 prediction_history.json 数据
        # 这些记录有 target_period 和 red_hits/blue_hits，但可能没有 has_drawn 字段
        try:
            from prediction_records import load_records
            records = load_records()
            # 去重：同一 target_period 只保留最后一条
            seen_periods = set()
            deduped = []
            for r in reversed(records):  # 从最新往回遍历
                if (r.get('lottery_type') == lottery_type
                    and r.get('model_type') == model_name
                    and r.get('red_hits') is not None
                    and r.get('blue_hits') is not None
                    and r.get('target_period') is not None):
                    period = r['target_period']
                    if period not in seen_periods:
                        seen_periods.add(period)
                        deduped.append((r['red_hits'], r['blue_hits']))
            # 恢复时间顺序
            history.extend(reversed(deduped))
        except:
            pass
        
        return history


# ============ 权重计算 ============

class EnsembleWeightCalculator:
    """计算集成模型的动态权重 - EMA 累计历史权重版"""

    def __init__(self, recorder: EnsemblePredictionRecorder):
        self.recorder = recorder

    def _get_ewma_history(self, lottery_type: str, model_name: str) -> List[float]:
        """获取指定模型的 EWMA 历史序列（从 model_weights.json 读取）"""
        try:
            with open('model_weights.json', 'r', encoding='utf-8') as f:
                all_weights = json.load(f)
            if lottery_type in all_weights:
                models = all_weights[lottery_type].get('models', {})
                if model_name in models:
                    ewma_hist = models[model_name].get('ewma_history', [])
                    if ewma_hist:
                        return ewma_hist
        except:
            pass
        return []

    def _save_ewma_history(self, lottery_type: str, model_name: str, ewma_history: List[float]):
        """保存 EWMA 历史序列到 model_weights.json"""
        try:
            with open('model_weights.json', 'r', encoding='utf-8') as f:
                all_weights = json.load(f)
        except:
            all_weights = {}

        if lottery_type not in all_weights:
            all_weights[lottery_type] = {'models': {}}
        if 'models' not in all_weights[lottery_type]:
            all_weights[lottery_type]['models'] = {}
        if model_name not in all_weights[lottery_type]['models']:
            all_weights[lottery_type]['models'][model_name] = {}

        all_weights[lottery_type]['models'][model_name]['ewma_history'] = ewma_history[-EMA_WINDOW:]
        with open('model_weights.json', 'w', encoding='utf-8') as f:
            json.dump(all_weights, f, ensure_ascii=False, indent=2)

    def compute_weights(self, lottery_type: str) -> Dict[str, Dict]:
        """计算指定彩票类型的模型权重

        算法：
        1. 获取每个模型的历史命中序列
        2. 计算每期的综合得分（超额命中率 + 连续命中加成 + 趋势加成 + 加速因子）
        3. 用 EWMA 平滑（alpha=0.3），从头开始累计历史（重新计算，不依赖旧存储）
        4. 保存 EWMA 历史到 model_weights.json
        5. 最近 EMA_WINDOW 次的 EWMA 做加权平均（decay=0.9）
        6. Softmax 归一化得到最终权重
        """
        models = ['gbdt', 'lightgbm', 'random_forest', 'xgboost', 'catboost']
        baseline = random_baseline(lottery_type)
        red_pick = baseline['red_pick']
        blue_pick = baseline['blue_pick']

        stats = {}
        recent_avgs = []

        for model_name in models:
            hit_history = self.recorder.get_model_hit_history(lottery_type, model_name)
            n = len(hit_history)

            # 计算每期的综合得分序列
            scores = []
            for i in range(n):
                period_history = hit_history[:i+1]
                score = compute_model_score(period_history, red_pick, blue_pick, baseline)
                scores.append(score)

            # 从头重新计算 EWMA（避免旧历史累积错误）
            ewma_history = []
            for score in scores:
                if ewma_history:
                    new_ewma = ALPHA * score + (1 - ALPHA) * ewma_history[-1]
                else:
                    new_ewma = score
                ewma_history.append(new_ewma)

            # 保存 EWMA 历史
            self._save_ewma_history(lottery_type, model_name, ewma_history)

            # 样本不足时收缩
            current_ewma = ewma_history[-1] if ewma_history else 0.0
            if 0 < n < MIN_SAMPLES:
                current_ewma *= n / MIN_SAMPLES

            # 最近 EMA_WINDOW 次的 EWMA 做加权平均
            recent_avg = compute_recent_weighted_avg(ewma_history, EMA_WINDOW, DECAY_FACTOR)

            stats[model_name] = {
                'samples': n,
                'score': round(scores[-1], 5) if scores else 0.0,
                'ewma': round(current_ewma, 5),
                'ewma_history': ewma_history[-EMA_WINDOW:],
                'recent_avg': round(recent_avg, 5),
                'hit_history': hit_history[-RECENT_WINDOW:]
            }
            recent_avgs.append(recent_avg)

        # Softmax 归一化
        weights = softmax(recent_avgs, TAU)

        result = {
            'lottery_type': lottery_type,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'baseline': baseline,
            'models': {m: {**stats[m], 'weight': round(w, 4)}
                      for m, w in zip(models, weights)},
            'weights': {m: round(w, 4) for m, w in zip(models, weights)},
            'recommended_model': max(models, key=lambda m: stats[m]['recent_avg'])
        }

        return result

    def save_weights(self, lottery_type: str, weights: Dict):
        """保存权重到 model_weights.json"""
        file_path = 'model_weights.json'
        all_weights = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_weights = json.load(f)
            except:
                pass

        all_weights[lottery_type] = weights
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_weights, f, ensure_ascii=False, indent=2)


# ============ 集成预测引擎 ============

class EnsembleEngine:
    """集成预测引擎 V2 - 完整算法"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.recorder = EnsemblePredictionRecorder(data_dir)
        self.weight_calculator = EnsembleWeightCalculator(self.recorder)
    
    def predict_ensemble(self, lottery_type: str, target_period: int,
                        red_sub_models: Dict[str, object],
                        blue_sub_models: Dict[str, object],
                        X_scaled: np.ndarray,
                        red_range: int, blue_range: int,
                        red_pick: int, blue_pick: int) -> Dict:
        """执行集成预测

        流程：
        1. 获取当前权重
        2. 每个子模型独立预测概率分布（红球用红球模型，蓝球用蓝球模型）
        3. 记录中间预测结果
        4. 加权汇总概率分布
        5. 取top号码作为最终结果
        """
        # 获取当前权重
        weights_result = self.weight_calculator.compute_weights(lottery_type)
        weights = weights_result['weights']

        # 每个子模型独立预测
        sub_predictions = {}
        for model_name in weights.keys():
            try:
                red_probs = {}
                blue_probs = {}
                
                # 红球预测
                red_model = red_sub_models.get(model_name)
                if red_model and hasattr(red_model, 'predict_proba'):
                    proba = red_model.predict_proba(X_scaled)
                    if isinstance(proba, np.ndarray):
                        if proba.ndim == 1:
                            red_probs = {i: float(proba[i]) for i in range(min(len(proba), red_range))}
                        else:
                            red_probs = {i: float(proba[0, i]) for i in range(min(proba.shape[1], red_range))}
                
                # 蓝球预测
                blue_model = blue_sub_models.get(model_name)
                if blue_model and hasattr(blue_model, 'predict_proba'):
                    proba = blue_model.predict_proba(X_scaled)
                    if isinstance(proba, np.ndarray):
                        if proba.ndim == 1:
                            blue_probs = {i: float(proba[i]) for i in range(min(len(proba), blue_range))}
                        else:
                            blue_probs = {i: float(proba[0, i]) for i in range(min(proba.shape[1], blue_range))}
                
                sub_predictions[model_name] = {
                    'red_probs': red_probs,
                    'blue_probs': blue_probs
                }
            except Exception as e:
                sub_predictions[model_name] = {
                    'red_probs': {},
                    'blue_probs': {},
                    'error': str(e)
                }

        # 加权汇总概率分布
        final_red_scores = defaultdict(float)
        final_blue_scores = defaultdict(float)

        for model_name, preds in sub_predictions.items():
            w = weights.get(model_name, 0.2)
            for num, prob in preds.get('red_probs', {}).items():
                final_red_scores[num] += w * prob
            for num, prob in preds.get('blue_probs', {}).items():
                final_blue_scores[num] += w * prob

        # 排序取top
        sorted_red = sorted(final_red_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_blue = sorted(final_blue_scores.items(), key=lambda x: x[1], reverse=True)

        final_red = [k + 1 for k, v in sorted_red[:red_pick]]
        final_blue = [k + 1 for k, v in sorted_blue[:blue_pick]]

        # 记录中间预测结果
        ensemble_id = self.recorder.record_prediction(
            lottery_type=lottery_type,
            target_period=target_period,
            sub_predictions=sub_predictions,
            final_red=final_red,
            final_blue=final_blue,
            weights=weights
        )
        
        # 归一化概率分布
        red_total = sum(final_red_scores.values()) if final_red_scores else 1.0
        blue_total = sum(final_blue_scores.values()) if final_blue_scores else 1.0
        
        red_proba = {k: v / red_total for k, v in final_red_scores.items()} if final_red_scores else {}
        blue_proba = {k: v / blue_total for k, v in final_blue_scores.items()} if final_blue_scores else {}
        
        return {
            'ensemble_id': ensemble_id,
            'red_numbers': final_red,
            'blue_numbers': final_blue,
            'red_proba': red_proba,
            'blue_proba': blue_proba,
            'sub_predictions': sub_predictions,
            'weights': weights,
            'recommended_model': weights_result['recommended_model']
        }
    
    def verify_ensemble(self, ensemble_id: str, draw_red: List[int],
                       draw_blue: List[int]) -> Dict:
        """核验集成预测，更新模型权重
        
        流程：
        1. 标记已核验，计算每个子模型的表现
        2. 更新EWMA
        3. 计算最近10次加权平均
        4. Softmax -> 新权重
        """
        # 标记核验
        sub_performance = self.recorder.mark_verified(ensemble_id, draw_red, draw_blue)
        if not sub_performance:
            return {'error': '记录不存在'}
        
        # 获取彩票类型
        record = next((r for r in self.recorder.records
                      if r.get('ensemble_id') == ensemble_id), None)
        if not record:
            return {'error': '记录不存在'}
        
        lottery_type = record['lottery_type']
        
        # 重新计算权重
        weights_result = self.weight_calculator.compute_weights(lottery_type)
        self.weight_calculator.save_weights(lottery_type, weights_result)
        
        return {
            'lottery_type': lottery_type,
            'sub_performance': sub_performance,
            'new_weights': weights_result['weights'],
            'recommended_model': weights_result['recommended_model']
        }


# ============ 测试 ============

if __name__ == '__main__':
    engine = EnsembleEngine(data_dir='.')
    
    print("========== 集成预测引擎 V2 测试 ==========")
    
    # 场景1: 连续两次命中3个
    print("\n场景1: 连续两次命中3个红球")
    for i in range(2):
        engine.recorder.records.append({
            'ensemble_id': f'test{i}',
            'lottery_type': 'dlt',
            'target_period': 26119 + i,
            'verified': True,
            'draw_red': [4, 6, 10, 24, 30],
            'draw_blue': [2, 9],
            'sub_performance': {
                'gbdt': {'red_hits': 3, 'blue_hits': 1},
                'lightgbm': {'red_hits': 2, 'blue_hits': 0},
                'random_forest': {'red_hits': 1, 'blue_hits': 1},
                'xgboost': {'red_hits': 2, 'blue_hits': 0},
                'catboost': {'red_hits': 1, 'blue_hits': 0}
            }
        })
    engine.recorder._save()
    
    weights = engine.weight_calculator.compute_weights('dlt')
    print(f"  GBDT权重: {weights['weights']['gbdt']:.4f}")
    print(f"  LightGBM权重: {weights['weights']['lightgbm']:.4f}")
    
    # 场景2: 连续两次命中4个
    print("\n场景2: 连续两次命中4个红球")
    engine.recorder.records.clear()
    for i in range(2):
        engine.recorder.records.append({
            'ensemble_id': f'test2_{i}',
            'lottery_type': 'dlt',
            'target_period': 26120 + i,
            'verified': True,
            'draw_red': [4, 6, 10, 24, 30],
            'draw_blue': [2, 9],
            'sub_performance': {
                'gbdt': {'red_hits': 4, 'blue_hits': 1},
                'lightgbm': {'red_hits': 2, 'blue_hits': 0},
                'random_forest': {'red_hits': 1, 'blue_hits': 1},
                'xgboost': {'red_hits': 2, 'blue_hits': 0},
                'catboost': {'red_hits': 1, 'blue_hits': 0}
            }
        })
    engine.recorder._save()
    
    weights2 = engine.weight_calculator.compute_weights('dlt')
    print(f"  GBDT权重: {weights2['weights']['gbdt']:.4f}")
    print(f"  LightGBM权重: {weights2['weights']['lightgbm']:.4f}")
    print(f"  权重提升: {(weights2['weights']['gbdt'] - weights['weights']['gbdt']) / weights['weights']['gbdt'] * 100:.1f}%")
    
    # 场景3: 连续三次命中
    print("\n场景3: 连续三次命中3个红球")
    engine.recorder.records.clear()
    for i in range(3):
        engine.recorder.records.append({
            'ensemble_id': f'test3_{i}',
            'lottery_type': 'dlt',
            'target_period': 26121 + i,
            'verified': True,
            'draw_red': [4, 6, 10, 24, 30],
            'draw_blue': [2, 9],
            'sub_performance': {
                'gbdt': {'red_hits': 3, 'blue_hits': 1},
                'lightgbm': {'red_hits': 2, 'blue_hits': 0},
                'random_forest': {'red_hits': 1, 'blue_hits': 1},
                'xgboost': {'red_hits': 2, 'blue_hits': 0},
                'catboost': {'red_hits': 1, 'blue_hits': 0}
            }
        })
    engine.recorder._save()
    
    weights3 = engine.weight_calculator.compute_weights('dlt')
    print(f"  GBDT权重: {weights3['weights']['gbdt']:.4f}")
    print(f"  权重提升(vs场景1): {(weights3['weights']['gbdt'] - weights['weights']['gbdt']) / weights['weights']['gbdt'] * 100:.1f}%")
    
    # 场景4: 递增命中 1->2->3
    print("\n场景4: 递增命中 1->2->3")
    engine.recorder.records.clear()
    hits = [(1, 0), (2, 0), (3, 1)]
    for i, (rh, bh) in enumerate(hits):
        engine.recorder.records.append({
            'ensemble_id': f'test4_{i}',
            'lottery_type': 'dlt',
            'target_period': 26122 + i,
            'verified': True,
            'draw_red': [4, 6, 10, 24, 30],
            'draw_blue': [2, 9],
            'sub_performance': {
                'gbdt': {'red_hits': rh, 'blue_hits': bh},
                'lightgbm': {'red_hits': 2, 'blue_hits': 0},
                'random_forest': {'red_hits': 1, 'blue_hits': 1},
                'xgboost': {'red_hits': 2, 'blue_hits': 0},
                'catboost': {'red_hits': 1, 'blue_hits': 0}
            }
        })
    engine.recorder._save()
    
    weights4 = engine.weight_calculator.compute_weights('dlt')
    print(f"  GBDT权重: {weights4['weights']['gbdt']:.4f}")
    print(f"  推荐模型: {weights4['recommended_model']}")
    
    print("\n========== 测试完成 ==========")