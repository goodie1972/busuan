# -*- coding:utf-8 -*-
"""
模型校准模块
使用历史预测和开奖数据对模型输出概率进行校准，使其与实际频率一致
"""
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
import joblib

from prediction_records import load_records
from backtest import _load_csv_data

# 校准模型存储目录
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_calibration')

def ensure_calibration_dir():
    """确保校准目录存在"""
    if not os.path.exists(CALIBRATION_DIR):
        os.makedirs(CALIBRATION_DIR)

def get_calibration_file(lottery_type: str, ball_type: str) -> str:
    """获取校准模型文件路径"""
    ensure_calibration_dir()
    return os.path.join(CALIBRATION_DIR, f'{lottery_type}_{ball_type}_isotonic.pkl')

def collect_calibration_data(lottery_type: str) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
    """
    从历史记录中收集校准数据
    
    Returns:
        red_data: {号码: [概率列表]} 红球概率和实际命中(0/1)数据
        blue_data: {号码: [概率列表]} 蓝球概率和实际命中(0/1)数据
    """
    records = load_records()
    if not records:
        return {}, {}
    
    # 加载开奖数据
    df = _load_csv_data(lottery_type)
    if df is None or df.empty:
        return {}, {}
    
    # 构建期号到开奖号码的映射
    draw_map = {}
    if lottery_type == 'ssq':
        red_cols = [f'红球_{i}' for i in range(1, 7)]
        blue_col = '蓝球'
    else:  # dlt
        red_cols = [f'红球_{i}' for i in range(1, 6)]
        blue_cols = [f'蓝球_{i}' for i in range(1, 3)]
    
    for _, row in df.iterrows():
        period = int(row['期数'])
        red_nums = [int(row[col]) for col in red_cols]
        if lottery_type == 'ssq':
            blue_nums = [int(row[blue_col])]
        else:
            blue_nums = [int(row[col]) for col in blue_cols]
        draw_map[period] = {
            'red': set(red_nums),
            'blue': set(blue_nums)
        }
    
    # 收集预测概率和实际结果
    red_data = {}  # {号码: [概率列表]}
    blue_data = {}  # {号码: [概率列表]}
    
    for record in records:
        # 只统计对应彩种的记录（两彩种期号可能重复，混入会污染校准样本）
        if record.get('lottery_type') and record.get('lottery_type') != lottery_type:
            continue
        # 跳过未核验的记录
        if record.get('target_period') is None:
            continue
            
        target_period = record['target_period']
        if target_period not in draw_map:
            continue
            
        actual_red = draw_map[target_period]['red']
        actual_blue = draw_map[target_period]['blue']
        
        # 处理红球
        red_nums = record['red_numbers']
        red_probas = record.get('red_probabilities')
        if red_nums and red_probas and len(red_nums) == len(red_probas):
            for i, num in enumerate(red_nums):
                if num not in red_data:
                    red_data[num] = {'probas': [], 'hits': []}
                red_data[num]['probas'].append(float(red_probas[i]))
                red_data[num]['hits'].append(1 if num in actual_red else 0)
        
        # 处理蓝球
        blue_nums = record['blue_numbers']
        blue_probas = record.get('blue_probabilities')
        if blue_nums and blue_probas and len(blue_nums) == len(blue_probas):
            for i, num in enumerate(blue_nums):
                if num not in blue_data:
                    blue_data[num] = {'probas': [], 'hits': []}
                blue_data[num]['probas'].append(float(blue_probas[i]))
                blue_data[num]['hits'].append(1 if num in actual_blue else 0)
    
    # 转换为所需格式
    red_result = {}
    blue_result = {}
    for num, data in red_data.items():
        red_result[num] = list(zip(data['probas'], data['hits']))
    for num, data in blue_data.items():
        blue_result[num] = list(zip(data['probas'], data['hits']))
        
    return red_result, blue_result

def train_calibration_model(lottery_type: str, ball_type: str, 
                          prob_hits_list: List[Tuple[float, int]]) -> Optional[IsotonicRegression]:
    """
    训练等回归校准模型
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        ball_type: 'red' 或 'blue'
        prob_hits_list: [(概率, 实际命中(0/1)), ...] 列表
        
    Returns:
        训练好的IsotonicRegression模型，如果数据不足则返回None
    """
    if len(prob_hits_list) < 10:  # 需要足够的数据
        return None
    
    # 按概率排序
    prob_hits_list.sort(key=lambda x: x[0])
    probabilities = [x[0] for x in prob_hits_list]
    actuals = [x[1] for x in prob_hits_list]
    
    # 创建并训练等回归模型
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(probabilities, actuals)
    
    return iso_reg

def calibrate_lottery(lottery_type: str) -> bool:
    """
    对指定彩票类型进行模型校准
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        
    Returns:
        校准是否成功
    """
    try:
        # 收集校准数据
        red_data, blue_data = collect_calibration_data(lottery_type)
        
        if not red_data and not blue_data:
            print(f"警告: {lottery_type} 没有足够的校准数据")
            return False
        
        # 训练红球校准模型
        if red_data:
            # 展平数据: [(prob, hit), ...]
            red_flat = []
            for num, prob_hits in red_data.items():
                red_flat.extend(prob_hits)
            
            red_model = train_calibration_model(lottery_type, 'red', red_flat)
            if red_model:
                red_path = get_calibration_file(lottery_type, 'red')
                joblib.dump(red_model, red_path)
                print(f"{lottery_type} 红球校准模型已保存: {red_path}")
            else:
                print(f"{lottery_type} 红球校准模型训练失败（数据不足）")
        
        # 训练蓝球校准模型
        if blue_data:
            # 展平数据: [(prob, hit), ...]
            blue_flat = []
            for num, prob_hits in blue_data.items():
                blue_flat.extend(prob_hits)
            
            blue_model = train_calibration_model(lottery_type, 'blue', blue_flat)
            if blue_model:
                blue_path = get_calibration_file(lottery_type, 'blue')
                joblib.dump(blue_model, blue_path)
                print(f"{lottery_type} 蓝球校准模型已保存: {blue_path}")
            else:
                print(f"{lottery_type} 蓝球校准模型训练失败（数据不足）")
        
        return True
        
    except Exception as e:
        print(f"校准过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_calibration_model(lottery_type: str, ball_type: str) -> Optional[IsotonicRegression]:
    """
    加载校准模型
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        ball_type: 'red' 或 'blue'
        
    Returns:
        加载的IsotonicRegression模型，如果不存在则返回None
    """
    try:
        model_path = get_calibration_file(lottery_type, ball_type)
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception:
        return None

def calibrate_probability(lottery_type: str, ball_type: str, probability: float) -> float:
    """
    使用校准模型校准单个概率值
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        ball_type: 'red' 或 'blue'
        probability: 原始概率值 (0-1)
        
    Returns:
        校准后的概率值 (0-1)
    """
    model = load_calibration_model(lottery_type, ball_type)
    if model is None:
        return probability  # 如果没有校准模型，返回原始概率
    
    try:
        # 确保概率在有效范围内
        probability = max(0.0, min(1.0, probability))
        calibrated = model.predict([probability])[0]
        # 确保输出也在[0,1]范围内
        return max(0.0, min(1.0, float(calibrated)))
    except Exception:
        return probability

def get_calibration_info(lottery_type: str) -> Dict:
    """
    获取校准信息
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        
    Returns:
        包含校准状态和统计信息的字典
    """
    info = {
        'lottery_type': lottery_type,
        'red_calibrated': False,
        'blue_calibrated': False,
        'red_samples': 0,
        'blue_samples': 0,
        'last_update': None
    }
    
    try:
        # 检查红球校准模型
        red_model = load_calibration_model(lottery_type, 'red')
        if red_model is not None:
            info['red_calibrated'] = True
            # 尝试获取模型信息
            red_path = get_calibration_file(lottery_type, 'red')
            if os.path.exists(red_path):
                stat = os.stat(red_path)
                info['last_update'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # 检查蓝球校准模型
        blue_model = load_calibration_model(lottery_type, 'blue')
        if blue_model is not None:
            info['blue_calibrated'] = True
            # 更新最后更新时间
            blue_path = get_calibration_file(lottery_type, 'blue')
            if os.path.exists(blue_path):
                stat = os.stat(blue_path)
                update_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
                if info['last_update'] is None or update_time > info['last_update']:
                    info['last_update'] = update_time
        
        # 收集样本数量（简化版）
        red_data, blue_data = collect_calibration_data(lottery_type)
        info['red_samples'] = sum(len(v) for v in red_data.values()) if red_data else 0
        info['blue_samples'] = sum(len(v) for v in blue_data.values()) if blue_data else 0
        
    except Exception as e:
        print(f"获取校准信息时出错: {e}")
    
    return info

if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        lt = sys.argv[1]
        if lt in ['ssq', 'dlt']:
            print(f"开始对 {lt} 进行模型校准...")
            success = calibrate_lottery(lt)
            if success:
                print(f"{lt} 校准完成！")
                info = get_calibration_info(lt)
                print(f"校准信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
            else:
                print(f"{lt} 校准失败")
        else:
            print("用法: python model_calibration.py [ssq|dlt]")
    else:
        print("用法: python model_calibration.py [ssq|dlt]")