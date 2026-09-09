# -*- coding:utf-8 -*-
"""
Data Processing Module for Lottery Data
Author: Yang Zhao
"""
import pandas as pd
import matplotlib.pyplot as plt
from scripts.data_analysis import (
    load_lottery_data, check_data_quality, calculate_statistics, 
    create_enhanced_features, plot_frequency_distribution, 
    plot_hot_cold_numbers, plot_gap_statistics, plot_patterns, 
    plot_trend_analysis
)

def process_analysis_data(lottery_type):
    """
    加载和处理分析所需的数据
    
    Args:
        lottery_type: 彩票类型 ('ssq' 或 'dlt')
        
    Returns:
        tuple: (current_df, current_stats, enhanced_df) 处理后的数据框和统计信息
    """
    # 加载数据
    current_df = load_lottery_data(lottery_type)
    
    # 检查数据质量
    cleaned_df, report = check_data_quality(current_df)
    current_df = cleaned_df
    
    # 计算统计信息
    current_stats = calculate_statistics(current_df, lottery_type)
    
    # 创建增强特征
    enhanced_df = create_enhanced_features(current_df, lottery_type)
    
    return current_df, current_stats, enhanced_df, report

def get_trend_features(enhanced_df):
    """
    从增强特征数据集中获取趋势特征列表
    
    Args:
        enhanced_df: 包含增强特征的数据框
        
    Returns:
        list: 可用于趋势分析的特征列表
    """
    if enhanced_df is None:
        return []
        
    numeric_cols = enhanced_df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols 
                   if not col.startswith('红球_') and 
                      not col.startswith('蓝球_') and 
                      col != '期数']
    return feature_cols

def prepare_recent_trend_data(enhanced_df, n_periods=10):
    """
    准备最近n期的趋势数据
    
    Args:
        enhanced_df: 增强特征数据框
        n_periods: 要分析的最近期数
        
    Returns:
        DataFrame: 含有趋势相关列的最近n期数据
    """
    if enhanced_df is None:
        return None
        
    # 获取最近n期数据
    recent_df = enhanced_df.head(n_periods).copy()
    recent_df = recent_df.sort_values(by='期数')
    
    # 确保所需的列存在，如果不存在则计算
    if '红球和' not in recent_df:
        red_ball_cols = [col for col in recent_df.columns if col.startswith('红球_')]
        recent_df['红球和'] = recent_df[red_ball_cols].sum(axis=1)
    
    if '红球最大差值' not in recent_df:
        red_ball_cols = [col for col in recent_df.columns if col.startswith('红球_')]
        recent_df['红球最大差值'] = recent_df[red_ball_cols].max(axis=1) - recent_df[red_ball_cols].min(axis=1)
    
    return recent_df

def format_quality_report(report):
    """
    格式化数据质量报告
    
    Args:
        report: 数据质量检查报告
        
    Returns:
        str: 格式化的报告文本
    """
    quality_report = f"数据质量报告:\n"
    quality_report += f"总行数: {report['原始数据行数']}\n"
    quality_report += f"清洗后行数: {report['清洗后数据行数']}\n"
    quality_report += f"红球无效行数: {report['红球无效数据行数']}\n"
    quality_report += f"蓝球无效行数: {report['蓝球无效数据行数']}\n"
    
    return quality_report

def format_frequency_stats(current_stats):
    """
    格式化频率统计信息
    
    Args:
        current_stats: 统计信息字典
        
    Returns:
        str: 格式化的统计信息
    """
    stats_text = "红球出现频率前5名:\n"
    red_freq = pd.Series(current_stats["红球出现频率"])
    for num, freq in red_freq.sort_values(ascending=False).head(5).items():
        stats_text += f"  {num}号: {freq}次\n"
        
    stats_text += "\n蓝球出现频率前5名:\n"
    blue_freq = pd.Series(current_stats["蓝球出现频率"])
    for num, freq in blue_freq.sort_values(ascending=False).head(5).items():
        stats_text += f"  {num}号: {freq}次\n"
        
    return stats_text

def format_hot_cold_stats(current_stats):
    """
    格式化热冷号码统计信息
    
    Args:
        current_stats: 统计信息字典
        
    Returns:
        str: 格式化的统计信息
    """
    stats_text = "红球热门号码:\n"
    for num, freq in current_stats["红球热门号码"].items():
        stats_text += f"  {num}号: {freq}次\n"
    
    stats_text += "\n红球冷门号码:\n"
    for num, freq in current_stats["红球冷门号码"].items():
        stats_text += f"  {num}号: {freq}次\n"
        
    stats_text += "\n蓝球热门号码:\n"
    for num, freq in current_stats["蓝球热门号码"].items():
        stats_text += f"  {num}号: {freq}次\n"
    
    stats_text += "\n蓝球冷门号码:\n"
    for num, freq in current_stats["蓝球冷门号码"].items():
        stats_text += f"  {num}号: {freq}次\n"
        
    return stats_text

def format_gap_stats(current_stats):
    """
    格式化遗漏值统计信息
    
    Args:
        current_stats: 统计信息字典
        
    Returns:
        str: 格式化的统计信息
    """
    stats_text = "红球最大遗漏值:\n"
    red_gaps = pd.Series(current_stats["红球遗漏值"])
    for num, gap in red_gaps.sort_values(ascending=False).head(5).items():
        stats_text += f"  {num}号: {gap}期\n"
        
    stats_text += "\n蓝球最大遗漏值:\n"
    blue_gaps = pd.Series(current_stats["蓝球遗漏值"])
    for num, gap in blue_gaps.sort_values(ascending=False).head(5).items():
        stats_text += f"  {num}号: {gap}期\n"
        
    return stats_text

def format_pattern_stats(current_stats):
    """
    格式化号码模式统计信息
    
    Args:
        current_stats: 统计信息字典
        
    Returns:
        str: 格式化的统计信息
    """
    red_odd = current_stats["红球奇偶比"]["奇数"]
    red_even = current_stats["红球奇偶比"]["偶数"]
    red_total = red_odd + red_even
    
    blue_odd = current_stats["蓝球奇偶比"]["奇数"]
    blue_even = current_stats["蓝球奇偶比"]["偶数"]
    blue_total = blue_odd + blue_even
    
    stats_text = "奇偶统计:\n"
    stats_text += f"  红球奇数比例: {red_odd/red_total:.2%}\n"
    stats_text += f"  红球偶数比例: {red_even/red_total:.2%}\n"
    stats_text += f"  蓝球奇数比例: {blue_odd/blue_total:.2%}\n"
    stats_text += f"  蓝球偶数比例: {blue_even/blue_total:.2%}\n"
    
    red_small = current_stats["红球大小比"]["小号"]
    red_big = current_stats["红球大小比"]["大号"]
    
    blue_small = current_stats["蓝球大小比"]["小号"]
    blue_big = current_stats["蓝球大小比"]["大号"]
    
    stats_text += "\n大小区间统计:\n"
    stats_text += f"  红球小号比例: {red_small/red_total:.2%}\n"
    stats_text += f"  红球大号比例: {red_big/red_total:.2%}\n"
    stats_text += f"  蓝球小号比例: {blue_small/blue_total:.2%}\n"
    stats_text += f"  蓝球大号比例: {blue_big/blue_total:.2%}\n"
    
    stats_text += f"\n连号出现次数: {current_stats['连号出现次数']}\n"
    stats_text += f"连号出现比例: {current_stats['连号出现比例']:.2%}\n"
    
    return stats_text

def format_trend_stats(recent_df):
    """
    格式化趋势统计信息
    
    Args:
        recent_df: 最近几期的数据
        
    Returns:
        str: 格式化的统计信息
    """
    if recent_df is None or len(recent_df) == 0:
        return "无趋势数据"
        
    stats_text = "最近10期趋势指标平均值:\n"
    stats_text += f"  红球和值: {recent_df['红球和'].mean():.2f}\n"
    stats_text += f"  红球最大差值: {recent_df['红球最大差值'].mean():.2f}\n"
    if '红球平均值' in recent_df:
        stats_text += f"  红球平均值: {recent_df['红球平均值'].mean():.2f}\n"
    if '蓝球和' in recent_df:
        stats_text += f"  蓝球和值: {recent_df['蓝球和'].mean():.2f}\n"
    
    stats_text += "\n最新一期指标值:\n"
    latest = recent_df.iloc[-1]
    stats_text += f"  红球和值: {latest['红球和']}\n"
    stats_text += f"  红球最大差值: {latest['红球最大差值']}\n"
    if '红球平均值' in latest:
        stats_text += f"  红球平均值: {latest['红球平均值']:.2f}\n"
    if '蓝球和' in latest:
        stats_text += f"  蓝球和值: {latest['蓝球和']}\n"
        
    return stats_text 