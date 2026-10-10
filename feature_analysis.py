# -*- coding:utf-8 -*-
"""
特征分析模块
用于分析特征重要性、检测特征漂移和生成特征选择建议
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

from backtest import _load_csv_data, SSQ_PRIZE_TABLE, DLT_PRIZE_TABLE
from prediction_records import load_records
import prediction_utils

# 特征分析配置
FEATURE_ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_analysis')
FEATURE_SELECTION_FILE = os.path.join(FEATURE_ANALYSIS_DIR, 'selected_features.json')
FEATURE_IMPORTANCE_FILE = os.path.join(FEATURE_ANALYSIS_DIR, 'feature_importance.json')
DRIFT_DETECTION_FILE = os.path.join(FEATURE_ANALYSIS_DIR, 'drift_detection.json')

# 确保目录存在
if not os.path.exists(FEATURE_ANALYSIS_DIR):
    os.makedirs(FEATURE_ANALYSIS_DIR)

def ensure_feature_dir():
    """确保特征分析目录存在"""
    if not os.path.exists(FEATURE_ANALYSIS_DIR):
        os.makedirs(FEATURE_ANALYSIS_DIR)

def get_feature_names(lottery_type: str) -> List[str]:
    """
    获取标准特征名称列表
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        
    Returns:
        特征名称列表
    """
    if lottery_type == 'ssq':
        # 双色球特征
        base_features = [
            # 遗漏特征
            '红球_遗漏_1', '红球_遗漏_2', '红球_遗漏_3', '红球_遗漏_4', '红球_遗漏_5', '红球_遗漏_6',
            '蓝球_遗漏_1',
            # 冷热号特征（最近10期出现次数）
            '红球_热度_1', '红球_热度_2', '红球_热度_3', '红球_热度_4', '红球_热度_5', '红球_热度_6',
            '蓝球_热度_1',
            # 和值特征
            '和值',
            # 跨度特征
            '跨度',
            # 奇偶比
            '奇偶比',
            # 大小比
            '大小比',
            # 连号特征
            '连号个数',
            # AC值
            'AC值',
        ]
    else:  # dlt
        # 大乐透特征
        base_features = [
            # 前区遗漏特征
            '红球_遗漏_1', '红球_遗漏_2', '红球_遗漏_3', '红球_遗漏_4', '红球_遗漏_5',
            # 后区遗漏特征
            '蓝球_遗漏_1', '蓝球_遗漏_2',
            # 前区热度特征（最近10期出现次数）
            '红球_热度_1', '红球_热度_2', '红球_热度_3', '红球_热度_4', '红球_热度_5',
            # 后区热度特征
            '蓝球_热度_1', '蓝球_热度_2',
            # 前区和值
            '前区和值',
            # 后区和值
            '后区和值',
            # 前区跨度
            '前区跨度',
            # 前区奇偶比
            '前区奇偶比',
            # 前区大小比
            '前区大小比',
            # 前区连号个数
            '前区连号个数',
        ]
    
    return base_features

def extract_features_from_data(df: pd.DataFrame, lottery_type: str) -> pd.DataFrame:
    """
    从原始开奖数据中提取特征
    
    Args:
        df: 原始开奖数据DataFrame
        lottery_type: 'ssq' 或 'dlt'
        
    Returns:
        包含特征的DataFrame
    """
    # 确保数据按期数升序排序
    df = df.sort_values('期数', ascending=True).reset_index(drop=True)
    
    if lottery_type == 'ssq':
        return _extract_ssq_features(df)
    else:
        return _extract_dlt_features(df)

def _red_occ(arr, n):
    """构建号码出现计数矩阵 oc[i, num-1] = 第 i 期号码 num 的出现次数。
    原版热度用 list.count（重复号码按次数计），遗漏用 in（存在即1），
    故用计数矩阵：遗漏处取 oc>0（bool），热度处直接求和（count）。"""
    m = arr.shape[0]
    oc = np.zeros((m, n), dtype=np.int32)
    r = np.arange(m)
    for ci in range(arr.shape[1]):
        v = arr[:, ci]
        ok = (v >= 1) & (v <= n)
        np.add.at(oc, (r[ok], v[ok] - 1), 1)
    return oc


def _last_before(b, m):
    """对布尔序列 b（长度 m），返回每位置 p 处严格小于 p 的最后一个 True 下标；无则为 -1。"""
    a = np.where(b, np.arange(m, dtype=float), np.nan)
    last = pd.Series(a).ffill().shift(1).fillna(-1).astype(int)
    return last.values


def _generic_miss(arr, n, k):
    """
    通用遗漏：对每个号码 num（1..n），计算其自上一次出现到当前期的间隔（独立于本期号码），
    再按 (num-1)%k+1 折叠到 k 个输出列（与原实现一致的"最后写入者获胜"语义）。
    """
    m = arr.shape[0]
    oc = _red_occ(arr, n)
    p = np.arange(m)
    out = np.zeros((m, k), dtype=int)
    for j in range(1, k + 1):
        winning = max(num for num in range(1, n + 1) if (num - 1) % k == j - 1)
        last = _last_before(oc[:, winning - 1] > 0, m)
        out[:, j - 1] = np.where(last >= 0, p - last, p + 1)
    return out


def _generic_hot(arr, n, k, window=10):
    """
    通用热度：每个号码在最近 window 期[不含本期]内的出现次数。
    原版窗口 = range(max(0,p-window), p) = [p-window, p-1]（含 p-1，不含本期 p），
    按 (num-1)%k+1 折叠到 k 列（"最后写入者获胜"，与原版一致）。
    """
    m = arr.shape[0]
    oc = _red_occ(arr, n)
    out = np.zeros((m, k), dtype=int)
    for j in range(1, k + 1):
        winning = max(num for num in range(1, n + 1) if (num - 1) % k == j - 1)
        out[:, j - 1] = _window_count(oc[:, winning - 1], window)
    return out


def _window_count(b, window):
    """每位置 p 处 [max(0,p-window), p-1] 区间内 b 的求和（原版窗口 range(p-window, p)，含 p-1）。"""
    m = len(b)
    b = np.asarray(b, dtype=int)
    cs = np.concatenate([[0], np.cumsum(b)])  # cs[i] = sum b[0..i-1]
    lo = np.clip(np.arange(m) - window, 0, m)
    return cs[np.arange(m)] - cs[lo]


def _same_value_miss(bv, n):
    """
    同值遗漏：SSQ 蓝球专用——对第 p 期出现的那个号码，计算它上一次出现到本期的间隔。
    无历史出现则为 p+1。
    """
    m = len(bv)
    out = np.zeros(m, dtype=int)
    for v in range(1, n + 1):
        pos = np.where(bv == v)[0]
        for t in range(len(pos)):
            p = pos[t]
            out[p] = (p + 1) if t == 0 else (p - pos[t - 1])
    return out


def _ensure_ball_columns(feature_df, cols):
    """确保球号列存在，缺失时按原实现的"列名变体"回退逻辑补 0。"""
    for col in cols:
        if col not in feature_df.columns:
            possible_cols = [c for c in feature_df.columns if col in c]
            if possible_cols:
                feature_df = feature_df.rename(columns={possible_cols[0]: col})
            else:
                feature_df[col] = 0
    return feature_df


def _region_miss_names(cols):
    return [f'{c.rsplit("_", 1)[0]}_遗漏_{c.rsplit("_", 1)[1]}' for c in cols]


def _region_hot_names(cols):
    return [f'{c.rsplit("_", 1)[0]}_热度_{c.rsplit("_", 1)[1]}' for c in cols]


def _extract_ssq_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取双色球特征（向量化实现，数值结果与原逐行实现完全一致，性能提升数千倍）。"""
    feature_df = df.copy()
    red_cols = [f'红球_{i}' for i in range(1, 7)]
    blue_col = '蓝球'
    feature_df = _ensure_ball_columns(feature_df, red_cols + [blue_col])
    m = len(feature_df)
    red = feature_df[red_cols].to_numpy(dtype=int)
    blue = feature_df[blue_col].to_numpy(dtype=int).ravel()

    # 红球遗漏（号码1-33，按 (num-1)%6 折叠到6列）
    feature_df[_region_miss_names(red_cols)] = _generic_miss(red, 33, 6)
    # 蓝球遗漏（当前蓝球号码的同值遗漏）
    feature_df['蓝球_遗漏_1'] = _same_value_miss(blue, 16)
    # 红球热度（最近10期[不含本期]）
    feature_df[_region_hot_names(red_cols)] = _generic_hot(red, 33, 6)
    # 蓝球热度（原版 (num-1)%1 恒为 col1，最后写入者=num16 => 16 的窗口计数）
    feature_df['蓝球_热度_1'] = _window_count((blue == 16).astype(int), 10)

    # 和值
    if '和值' not in feature_df.columns:
        feature_df['和值'] = 0
    feature_df['和值'] = feature_df[red_cols].sum(axis=1)

    # 跨度
    if '跨度' not in feature_df.columns:
        feature_df['跨度'] = 0
    red_values = feature_df[red_cols].values
    feature_df['跨度'] = np.max(red_values, axis=1) - np.min(red_values, axis=1)

    # 奇偶比
    if '奇偶比' not in feature_df.columns:
        feature_df['奇偶比'] = 0
    red_int = red_values.astype(int)
    feature_df['奇偶比'] = ((red_int % 2 == 1).sum(axis=1) / 6.0)

    # 大小比（大于16为大）
    if '大小比' not in feature_df.columns:
        feature_df['大小比'] = 0
    feature_df['大小比'] = ((red_int > 16).sum(axis=1) / 6.0)

    # 连号个数
    if '连号个数' not in feature_df.columns:
        feature_df['连号个数'] = 0
    srt = np.sort(red_int, axis=1)
    feature_df['连号个数'] = (np.diff(srt, axis=1) == 1).sum(axis=1)

    # AC值（排序后相邻差值之和）
    if 'AC值' not in feature_df.columns:
        feature_df['AC值'] = 0
    feature_df['AC值'] = np.abs(np.diff(srt, axis=1)).sum(axis=1)

    return feature_df


def _extract_dlt_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取大乐透特征（向量化实现，数值结果与原逐行实现完全一致，性能提升数千倍）。"""
    feature_df = df.copy()
    red_cols = [f'红球_{i}' for i in range(1, 6)]  # 前区1-35
    blue_cols = [f'蓝球_{i}' for i in range(1, 3)]  # 后区1-12
    feature_df = _ensure_ball_columns(feature_df, red_cols + blue_cols)
    front = feature_df[red_cols].to_numpy(dtype=int)
    back = feature_df[blue_cols].to_numpy(dtype=int)

    # 前区遗漏（号码1-35，按 (num-1)%5 折叠到5列）
    feature_df[_region_miss_names(red_cols)] = _generic_miss(front, 35, 5)
    # 后区遗漏（号码1-12，按 (num-1)%2 折叠到2列）
    feature_df[_region_miss_names(blue_cols)] = _generic_miss(back, 12, 2)
    # 前区热度
    feature_df[_region_hot_names(red_cols)] = _generic_hot(front, 35, 5)
    # 后区热度（原版 quirk：window_blue 只追加 蓝球_2 单列（循环变量 i 残留为 2），
    # 此处用单列输入精确复现，保持数值逐位一致）
    feature_df[_region_hot_names(blue_cols)] = _generic_hot(back[:, [1]], 12, 2)

    # 前区和值
    if '前区和值' not in feature_df.columns:
        feature_df['前区和值'] = 0
    feature_df['前区和值'] = feature_df[red_cols].sum(axis=1)

    # 后区和值
    if '后区和值' not in feature_df.columns:
        feature_df['后区和值'] = 0
    feature_df['后区和值'] = feature_df[blue_cols].sum(axis=1)

    # 前区跨度
    if '前区跨度' not in feature_df.columns:
        feature_df['前区跨度'] = 0
    red_values = feature_df[red_cols].values
    feature_df['前区跨度'] = np.max(red_values, axis=1) - np.min(red_values, axis=1)

    # 前区奇偶比
    if '前区奇偶比' not in feature_df.columns:
        feature_df['前区奇偶比'] = 0
    red_int = red_values.astype(int)
    feature_df['前区奇偶比'] = ((red_int % 2 == 1).sum(axis=1) / 5.0)

    # 前区大小比（大于17为大）
    if '前区大小比' not in feature_df.columns:
        feature_df['前区大小比'] = 0
    feature_df['前区大小比'] = ((red_int > 17).sum(axis=1) / 5.0)

    # 前区连号个数
    if '前区连号个数' not in feature_df.columns:
        feature_df['前区连号个数'] = 0
    srt = np.sort(red_int, axis=1)
    feature_df['前区连号个数'] = (np.diff(srt, axis=1) == 1).sum(axis=1)

    return feature_df

def analyze_feature_importance(lottery_type: str, model_type: str = 'ensemble') -> Dict[str, float]:
    """
    分析特征重要性
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        model_type: 模型类型，用于加载对应的模型
        
    Returns:
        特征重要性字典 {特征名: 重要性分数}
    """
    try:
        # 加载数据并提取特征
        df = _load_csv_data(lottery_type)
        if df is None or df.empty:
            return {}
        
        feature_df = extract_features_from_data(df, lottery_type)
        
        # 获取特征列名
        feature_names = get_feature_names(lottery_type)
        # 过滤掉不存在的列
        available_features = [f for f in feature_names if f in feature_df.columns]
        
        if not available_features:
            return {}
        
        # 准备特征矩阵（使用最近的数据）
        # 这里我们使用一个简单的方法：计算特征与中奖号码的相关性作为重要性度量
        # 在实际应用中，应该使用训练好的模型的特征重要性
        
        # 为了演示，我们使用互信息或简单的统计方法
        # 这里我们计算每个特征与中奖号码出现的关系
        
        importance_dict = {}
        
        if lottery_type == 'ssq':
            red_cols = [f'红球_{i}' for i in range(1, 7)]
            blue_col = '蓝球'
        else:
            red_cols = [f'红球_{i}' for i in range(1, 6)]
            blue_cols = [f'蓝球_{i}' for i in range(1, 3)]
        
        # 确保列存在
        for col_list in [red_cols, blue_cols] if lottery_type == 'dlt' else [red_cols + [blue_col]]:
            for col in col_list:
                if col not in feature_df.columns:
                    feature_df[col] = 0
        
        # 计算每个特征的重要性（基于与中奖号码的相关性）
        for feature in available_features:
            if feature not in feature_df.columns:
                continue
                
            # 计算特征值与中奖号码出现的关联度
            feature_vals = feature_df[feature].values
            
            # 对于每个位置，计算特征值与该位置号码的平均绝对差异的倒数作为重要性
            # 差异越小，特征越重要（因为特征值能更好地预测号码）
            total_relevance = 0
            count = 0
            
            if lottery_type == 'ssq':
                # 红球位置
                for i in range(1, 7):
                    col = f'红球_{i}'
                    if col in feature_df.columns:
                        nums = feature_df[col].values
                        # 计算特征值与号码值的相关性（使用绝对差异的倒数）
                        diffs = np.abs(feature_vals - nums)
                        # 避免除以零
                        diffs[diffs == 0] = 1e-10
                        relevance = np.mean(1.0 / (diffs + 1))  # 加1是为了平滑
                        total_relevance += relevance
                        count += 1
                
                # 蓝球位置
                if blue_col in feature_df.columns:
                    nums = feature_df[blue_col].values
                    diffs = np.abs(feature_vals - nums)
                    diffs[diffs == 0] = 1e-10
                    relevance = np.mean(1.0 / (diffs + 1))
                    total_relevance += relevance
                    count += 1
            else:  # dlt
                # 前区位置
                for i in range(1, 6):
                    col = f'红球_{i}'
                    if col in feature_df.columns:
                        nums = feature_df[col].values
                        diffs = np.abs(feature_vals - nums)
                        diffs[diffs == 0] = 1e-10
                        relevance = np.mean(1.0 / (diffs + 1))
                        total_relevance += relevance
                        count += 1
                
                # 后区位置
                for i in range(1, 3):
                    col = f'蓝球_{i}'
                    if col in feature_df.columns:
                        nums = feature_df[col].values
                        diffs = np.abs(feature_vals - nums)
                        diffs[diffs == 0] = 1e-10
                        relevance = np.mean(1.0 / (diffs + 1))
                        total_relevance += relevance
                        count += 1
            
            if count > 0:
                importance_dict[feature] = total_relevance / count
            else:
                importance_dict[feature] = 0.0
        
        # 归一化重要性分数
        if importance_dict:
            max_imp = max(importance_dict.values()) if importance_dict.values() else 1
            if max_imp > 0:
                for feat in importance_dict:
                    importance_dict[feat] /= max_imp
        
        return importance_dict
        
    except Exception as e:
        print(f"特征重要性分析出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def detect_feature_drift(lottery_type: str, window_size: int = 50) -> Dict[str, Any]:
    """
    检测特征漂移
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        window_size: 用于漂移检测的窗口大小
        
    Returns:
        漂移检测结果字典
    """
    try:
        # 加载数据并提取特征
        df = _load_csv_data(lottery_type)
        if df is None or len(df) < window_size * 2:
            return {'drift_detected': False, 'reason': '数据不足'}
        
        feature_df = extract_features_from_data(df, lottery_type)
        feature_names = get_feature_names(lottery_type)
        available_features = [f for f in feature_names if f in feature_df.columns]
        
        if not available_features:
            return {'drift_detected': False, 'reason': '没有可用特征'}
        
        # 将数据分成两半：旧数据和新数据
        split_point = len(feature_df) // 2
        old_data = feature_df.iloc[:split_point]
        new_data = feature_df.iloc[split_point:]
        
        drift_results = {}
        drift_detected = False
        max_drift_score = 0
        
        for feature in available_features:
            if feature not in feature_df.columns:
                continue
                
            old_vals = old_data[feature].values
            new_vals = new_data[feature].values
            
            if len(old_vals) == 0 or len(new_vals) == 0:
                continue
            
            # 使用Kolmogorov-Smirnov检测分布漂移（简化版）
            # 这里我们使用均值和标准差的变化来检测漂移
            old_mean, old_std = np.mean(old_vals), np.std(old_vals)
            new_mean, new_std = np.mean(new_vals), np.std(new_vals)
            
            # 计算漂移分数（归一化的变化量）
            mean_drift = abs(new_mean - old_mean) / (old_std + 1e-10)
            std_drift = abs(new_std - old_std) / (old_std + 1e-10)
            drift_score = (mean_drift + std_drift) / 2
            
            drift_results[feature] = {
                'drift_score': float(drift_score),
                'old_mean': float(old_mean),
                'new_mean': float(new_mean),
                'old_std': float(old_std),
                'new_std': float(new_std),
                'is_drifting': drift_score > 1.0  # 阈值可调
            }
            
            if drift_score > max_drift_score:
                max_drift_score = drift_score
                
            if drift_score > 1.0:
                drift_detected = True
        
        result = {
            'drift_detected': drift_detected,
            'max_drift_score': float(max_drift_score),
            'feature_drift': drift_results,
            'analysis_window': window_size,
            'total_samples': len(feature_df),
            'old_samples': len(old_data),
            'new_samples': len(new_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"特征漂移检测出错: {e}")
        import traceback
        traceback.print_exc()
        return {'drift_detected': False, 'error': str(e)}

def save_feature_selection(selected_features: Dict[str, List[str]]) -> bool:
    """
    保存特征选择结果
    
    Args:
        selected_features: {'ssq': [特征列表], 'dlt': [特征列表]}
        
    Returns:
        保存是否成功
    """
    try:
        ensure_feature_dir()
        data = {
            'selected_features': selected_features,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        with open(FEATURE_SELECTION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存特征选择失败: {e}")
        return False

def load_feature_selection() -> Dict[str, List[str]]:
    """
    加载特征选择结果
    
    Returns:
        特征选择字典，如果不存在则返回默认特征
    """
    try:
        if os.path.exists(FEATURE_SELECTION_FILE):
            with open(FEATURE_SELECTION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('selected_features', {})
        return {}
    except Exception as e:
        print(f"加载特征选择失败: {e}")
        return {}

def save_feature_importance(importance_data: Dict[str, Dict[str, float]]) -> bool:
    """
    保存特征重要性结果
    
    Args:
        importance_data: {'ssq': {特征: 重要性}, 'dlt': {特征: 重要性}}
        
    Returns:
        保存是否成功
    """
    try:
        ensure_feature_dir()
        data = {
            'feature_importance': importance_data,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        with open(FEATURE_IMPORTANCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存特征重要性失败: {e}")
        return False

def load_feature_importance() -> Dict[str, Dict[str, float]]:
    """
    加载特征重要性结果
    
    Returns:
        特征重要性字典
    """
    try:
        if os.path.exists(FEATURE_IMPORTANCE_FILE):
            with open(FEATURE_IMPORTANCE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('feature_importance', {})
        return {}
    except Exception as e:
        print(f"加载特征重要性失败: {e}")
        return {}

def save_drift_detection(drift_data: Dict[str, Any]) -> bool:
    """
    保存特征漂移检测结果
    
    Args:
        drift_data: 漂移检测结果字典
        
    Returns:
        保存是否成功
    """
    try:
        ensure_feature_dir()
        # numpy 标量（np.bool_/np.int64 等）无法被 json 序列化，递归转为原生 Python 类型
        def _to_py(obj):
            if isinstance(obj, dict):
                return {k: _to_py(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_py(v) for v in obj]
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            if isinstance(obj, (int, np.integer)):
                return int(obj)
            if isinstance(obj, (float, np.floating)):
                return float(obj)
            return obj
        data = {
            'drift_detection': _to_py(drift_data),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        with open(DRIFT_DETECTION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存漂移检测失败: {e}")
        return False

def load_drift_detection() -> Dict[str, Any]:
    """
    加载特征漂移检测结果
    
    Returns:
        漂移检测结果字典
    """
    try:
        if os.path.exists(DRIFT_DETECTION_FILE):
            with open(DRIFT_DETECTION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('drift_detection', {})
        return {}
    except Exception as e:
        print(f"加载漂移检测失败: {e}")
        return {}

def select_top_features(lottery_type: str, top_k: int = 20) -> List[str]:
    """
    根据特征重要性选择顶部特征
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        top_k: 选择的特征数量
        
    Returns:
        选中的特征名称列表
    """
    # 加载特征重要性
    importance_data = load_feature_importance()
    if lottery_type in importance_data:
        importance_dict = importance_data[lottery_type]
        # 按重要性排序并选择前k个
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:top_k]]
        return selected
    
    # 如果没有保存的重要性，返回默认特征的前k个
    all_features = get_feature_names(lottery_type)
    return all_features[:min(top_k, len(all_features))]

def run_feature_analysis(lottery_type: str) -> Dict[str, Any]:
    """
    运行完整的特征分析流程
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        
    Returns:
        特征分析结果字典
    """
    try:
        print(f"开始对 {lottery_type} 进行特征分析...")
        
        # 1. 特征重要性分析
        print("  正在分析特征重要性...")
        importance = analyze_feature_importance(lottery_type)
        
        # 2. 特征漂移检测
        print("  正在检测特征漂移...")
        drift = detect_feature_drift(lottery_type)
        
        # 3. 特征选择（基于重要性）
        print("  正在进行特征选择...")
        selected_features = select_top_features(lottery_type, top_k=20)
        
        # 4. 保存结果
        print("  正在保存分析结果...")
        if importance:
            save_feature_importance({lottery_type: importance})
        if selected_features:
            current_selection = load_feature_selection()
            current_selection[lottery_type] = selected_features
            save_feature_selection(current_selection)
        if drift:
            save_drift_detection({lottery_type: drift})
        
        result = {
            'lottery_type': lottery_type,
            'feature_importance': importance,
            'feature_drift': drift,
            'selected_features': selected_features,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        print(f"{lottery_type} 特征分析完成！")
        return result
        
    except Exception as e:
        print(f"特征分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'lottery_type': lottery_type,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        lt = sys.argv[1]
        if lt in ['ssq', 'dlt']:
            print(f"开始对 {lt} 进行特征分析...")
            result = run_feature_analysis(lt)
            print(f"分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print("用法: python feature_analysis.py [ssq|dlt]")
    else:
        print("用法: python feature_analysis.py [ssq|dlt]")