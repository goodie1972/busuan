# -*- coding:utf-8 -*-
"""
Data Analysis Module for Lottery Data
Author: Yang Zhao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from io import BytesIO
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import functools
import time
import logging


try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass

# 简单缓存装饰器，用于避免重复计算
def memoize(expire_seconds=300):
    """
    函数结果缓存装饰器，带过期时间
    
    Args:
        expire_seconds: 缓存过期时间（秒）
    """
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存是否存在且未过期
            current_time = time.time()
            if key in cache and current_time - cache[key]['timestamp'] < expire_seconds:
                logging.debug(f"使用缓存结果: {func.__name__}")
                return cache[key]['result']
            
            # 计算新结果
            result = func(*args, **kwargs)
            
            # 更新缓存
            cache[key] = {'result': result, 'timestamp': current_time}
            return result
        return wrapper
    return decorator

# load_lottery_data 的结果缓存：key=彩票类型，value=(文件mtime, DataFrame)
# 数据更新脚本在子进程中重写 CSV 后，文件 mtime 变化会自动使缓存失效，
# 避免 GUI 内"更新数据后立即训练/预测"读到更新前的旧数据
_load_lottery_data_cache = {}

def load_lottery_data(lottery_type):
    """加载彩票历史数据（按文件修改时间缓存，数据更新后自动失效）"""
    if lottery_type == 'dlt':
        file_path = './scripts/dlt/dlt_history.csv'
    else:  # ssq
        file_path = './scripts/ssq/ssq_history.csv'

    # 以文件 mtime 作为缓存依据
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = None

    cached = _load_lottery_data_cache.get(lottery_type)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    if not os.path.exists(file_path):
        logging.error(f"找不到数据文件: {file_path}")
        return None

    try:
        # 尝试不同编码读取数据
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']

        for encoding in encodings:
            try:
                logging.info(f"尝试使用 {encoding} 编码读取数据...")
                df = pd.read_csv(file_path, encoding=encoding)
                logging.info(f"成功使用 {encoding} 读取数据，共 {len(df)} 条记录")
                _load_lottery_data_cache[lottery_type] = (mtime, df)
                return df
            except UnicodeDecodeError:
                continue

        logging.error("所有编码尝试均失败")
        return None
    except Exception as e:
        logging.error(f"加载数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

@memoize(expire_seconds=600)
def check_data_quality(df):
    """
    检查数据质量，处理缺失值和异常值
    
    Args:
        df: 彩票历史数据DataFrame
    
    Returns:
        清洗后的DataFrame和质量报告dict
    """
    original_rows = len(df)
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    
    # 检查数据范围 (针对不同的彩票类型)
    if '红球_6' in df.columns:  # SSQ (6个红球)
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球')]
        invalid_red = df[df[red_ball_cols].apply(lambda x: (x < 1) | (x > 33)).any(axis=1)]
        invalid_blue = df[df[blue_ball_cols].apply(lambda x: (x < 1) | (x > 16)).any(axis=1)]
    else:  # DLT
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球_')]
        invalid_red = df[df[red_ball_cols].apply(lambda x: (x < 1) | (x > 35)).any(axis=1)]
        invalid_blue = df[df[blue_ball_cols].apply(lambda x: (x < 1) | (x > 12)).any(axis=1)]
    
    # 删除无效数据
    df = df.dropna()
    df = df[~df.index.isin(invalid_red.index)]
    df = df[~df.index.isin(invalid_blue.index)]
    
    # 重新索引
    df = df.reset_index(drop=True)
    
    # 生成报告
    report = {
        "原始数据行数": original_rows,
        "清洗后数据行数": len(df),
        "缺失值": missing_values.to_dict(),
        "红球无效数据行数": len(invalid_red),
        "蓝球无效数据行数": len(invalid_blue)
    }
    
    return df, report

@memoize(expire_seconds=600)
def calculate_statistics(df, lottery_type, num_periods=50):
    """
    计算各种统计数据
    
    Args:
        df: 历史数据DataFrame
        lottery_type: 'ssq' 或 'dlt'
        num_periods: 计算最近多少期的数据
    
    Returns:
        包含各种统计数据的字典
    """
    stats = {}
    
    # 识别红蓝球列
    if lottery_type == 'ssq':
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球')]
        red_range = range(1, 34)
        blue_range = range(1, 17)
    else:  # dlt
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球_')]
        red_range = range(1, 36)
        blue_range = range(1, 13)
    
    # 所有红球合并成一列
    all_red_balls = pd.Series(df[red_ball_cols].values.flatten())
    all_blue_balls = pd.Series(df[blue_ball_cols].values.flatten())
    
    # 计算每个号码的出现频率
    red_freq = all_red_balls.value_counts().sort_index()
    blue_freq = all_blue_balls.value_counts().sort_index()
    
    # 计算热门号码 (出现频率最高的前5个)
    # 使用排序和切片代替nlargest
    red_hot = red_freq.sort_values(ascending=False).head(5)
    blue_hot = blue_freq.sort_values(ascending=False).head(3)
    
    # 计算冷门号码 (出现频率最低的前5个)
    # 使用排序和切片代替nsmallest
    red_cold = red_freq.sort_values().head(5)
    blue_cold = blue_freq.sort_values().head(3)
    
    # 计算遗漏值 (最近一期距离上次出现的期数)
    latest_draw = df.iloc[0]  # 假设数据按期数降序排列
    red_gaps = {}
    blue_gaps = {}
    
    for num in red_range:
        for i, row in df.iterrows():
            if num in row[red_ball_cols].values:
                red_gaps[num] = i
                break
            if i == len(df) - 1:
                red_gaps[num] = i + 1
    
    for num in blue_range:
        for i, row in df.iterrows():
            if num in row[blue_ball_cols].values:
                blue_gaps[num] = i
                break
            if i == len(df) - 1:
                blue_gaps[num] = i + 1
    
    # 计算连号情况 (相邻的号码同时出现的次数)
    consecutive_count = 0
    for _, row in df.iterrows():
        red_numbers = sorted(row[red_ball_cols].values)
        for i in range(len(red_numbers) - 1):
            if red_numbers[i] + 1 == red_numbers[i + 1]:
                consecutive_count += 1
                break
    
    # 奇偶比例
    red_odd_even = {"奇数": 0, "偶数": 0}
    for num in all_red_balls:
        if num % 2 == 0:
            red_odd_even["偶数"] += 1
        else:
            red_odd_even["奇数"] += 1
    
    blue_odd_even = {"奇数": 0, "偶数": 0}
    for num in all_blue_balls:
        if num % 2 == 0:
            blue_odd_even["偶数"] += 1
        else:
            blue_odd_even["奇数"] += 1
    
    # 大小比例 (以中间值为界)
    red_mid = (red_range[-1] + red_range[0]) / 2
    blue_mid = (blue_range[-1] + blue_range[0]) / 2
    
    red_size = {"小号": 0, "大号": 0}
    for num in all_red_balls:
        if num > red_mid:
            red_size["大号"] += 1
        else:
            red_size["小号"] += 1
    
    blue_size = {"小号": 0, "大号": 0}
    for num in all_blue_balls:
        if num > blue_mid:
            blue_size["大号"] += 1
        else:
            blue_size["小号"] += 1
    
    # 组装统计结果
    stats = {
        "红球出现频率": red_freq.to_dict(),
        "蓝球出现频率": blue_freq.to_dict(),
        "红球热门号码": red_hot.to_dict(),
        "蓝球热门号码": blue_hot.to_dict(),
        "红球冷门号码": red_cold.to_dict(),
        "蓝球冷门号码": blue_cold.to_dict(),
        "红球遗漏值": red_gaps,
        "蓝球遗漏值": blue_gaps,
        "连号出现次数": consecutive_count,
        "连号出现比例": consecutive_count / len(df),
        "红球奇偶比": red_odd_even,
        "蓝球奇偶比": blue_odd_even,
        "红球大小比": red_size,
        "蓝球大小比": blue_size
    }
    
    return stats

@memoize(expire_seconds=600)
def create_enhanced_features(df, lottery_type):
    """
    创建增强特征，用于模型训练
    
    Args:
        df: 彩票历史数据DataFrame
        lottery_type: 'ssq' 或 'dlt'
    
    Returns:
        含增强特征的DataFrame
    """
    # 复制一份数据，避免修改原始数据
    enhanced_df = df.copy()
    
    # 识别红蓝球列
    if lottery_type == 'ssq':
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球')]
    else:  # dlt
        red_ball_cols = [col for col in df.columns if col.startswith('红球_')]
        blue_ball_cols = [col for col in df.columns if col.startswith('蓝球_')]
    
    # 1. 添加红球之和、平均值、方差
    enhanced_df['红球和'] = df[red_ball_cols].sum(axis=1)
    enhanced_df['红球平均值'] = df[red_ball_cols].mean(axis=1)
    enhanced_df['红球方差'] = df[red_ball_cols].var(axis=1)
    
    # 2. 添加蓝球之和、平均值 (如果有多个蓝球)
    enhanced_df['蓝球和'] = df[blue_ball_cols].sum(axis=1)
    if len(blue_ball_cols) > 1:
        enhanced_df['蓝球平均值'] = df[blue_ball_cols].mean(axis=1)
        enhanced_df['蓝球方差'] = df[blue_ball_cols].var(axis=1)
    
    # 3. 添加奇偶属性
    for col in red_ball_cols + blue_ball_cols:
        enhanced_df[f'{col}_奇偶'] = df[col] % 2
    
    # 4. 添加大小区间属性 (红球1-33分为1-17和18-33, 蓝球根据范围)
    if lottery_type == 'ssq':
        red_mid = 17
        blue_mid = 8
    else:  # dlt
        red_mid = 18
        blue_mid = 6
    
    for col in red_ball_cols:
        enhanced_df[f'{col}_区间'] = (df[col] > red_mid).astype(int)
    
    for col in blue_ball_cols:
        enhanced_df[f'{col}_区间'] = (df[col] > blue_mid).astype(int)
    
    # 5. 添加相邻期次的差值
    for col in red_ball_cols + blue_ball_cols:
        enhanced_df[f'{col}_差值'] = df[col].diff().fillna(0)
    
    # 6. 添加连号特征 (每组号码中是否有连续号码)
    def has_consecutive(row, cols):
        numbers = sorted(row[cols].values)
        for i in range(len(numbers) - 1):
            if numbers[i] + 1 == numbers[i + 1]:
                return 1
        return 0
    
    enhanced_df['有连号'] = df.apply(lambda row: has_consecutive(row, red_ball_cols), axis=1)
    
    # 7. 添加号码距离特征 (最大号-最小号)
    enhanced_df['红球跨度'] = df[red_ball_cols].max(axis=1) - df[red_ball_cols].min(axis=1)
    if len(blue_ball_cols) > 1:
        enhanced_df['蓝球跨度'] = df[blue_ball_cols].max(axis=1) - df[blue_ball_cols].min(axis=1)
    
    # 8. 添加期数信息 (假设期数按时间顺序)
    if '期数' in df.columns:
        enhanced_df['期数_年'] = df['期数'].astype(str).str[:4].astype(int)
        if df['期数'].astype(str).str.len().max() >= 7:  # 有足够位数表示月份
            enhanced_df['期数_月'] = df['期数'].astype(str).str[4:6].astype(int)
    
    return enhanced_df

@memoize(expire_seconds=300)
def plot_frequency_distribution(data, lottery_type, width=10, height=6, dpi=100):
    """
    绘制号码出现频率分布图，并返回QPixmap对象
    
    Args:
        data: 统计数据dict
        lottery_type: 'ssq' 或 'dlt'
        width, height: 图像尺寸
        dpi: 图像分辨率
    
    Returns:
        QPixmap对象
    """
    # 设置更好的图表风格
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
    
    # 红球频率
    red_freq = pd.Series(data["红球出现频率"])
    x = list(red_freq.index)
    y = list(red_freq.values)
    
    # 计算理论平均频率
    if lottery_type == 'ssq':
        total_draws = sum(y) / 6  # 每期6个红球
        expected_red_freq = total_draws / 33  # 1-33的平均概率
    else:  # dlt
        total_draws = sum(y) / 5  # 每期5个红球
        expected_red_freq = total_draws / 35  # 1-35的平均概率
    
    bars = ax1.bar(x, y, color='#FF5555', alpha=0.7)
    ax1.axhline(y=expected_red_freq, color='black', linestyle='--', label='期望频率')
    
    # 突出显示超过期望值的频率
    for i, bar in enumerate(bars):
        if y[i] > expected_red_freq:
            bar.set_color('#FF0000')
            bar.set_alpha(0.9)
    
    ax1.set_xlabel('红球号码', fontsize=12)
    ax1.set_ylabel('出现次数', fontsize=12)
    ax1.set_title('红球号码出现频率分布', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 蓝球频率
    blue_freq = pd.Series(data["蓝球出现频率"])
    x = list(blue_freq.index)
    y = list(blue_freq.values)
    
    # 计算理论平均频率
    if lottery_type == 'ssq':
        total_draws = sum(y)  # 每期1个蓝球
        expected_blue_freq = total_draws / 16  # 1-16的平均概率
    else:  # dlt
        total_draws = sum(y) / 2  # 每期2个蓝球
        expected_blue_freq = total_draws / 12  # 1-12的平均概率
    
    bars = ax2.bar(x, y, color='#5555FF', alpha=0.7)
    ax2.axhline(y=expected_blue_freq, color='black', linestyle='--', label='期望频率')
    
    # 突出显示超过期望值的频率
    for i, bar in enumerate(bars):
        if y[i] > expected_blue_freq:
            bar.set_color('#0000FF')
            bar.set_alpha(0.9)
    
    ax2.set_xlabel('蓝球号码', fontsize=12)
    ax2.set_ylabel('出现次数', fontsize=12)
    ax2.set_title('蓝球号码出现频率分布', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # 将matplotlib图形转换为QPixmap
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    
    return pixmap

@memoize(expire_seconds=300)
def plot_hot_cold_numbers(data, width=10, height=6, dpi=100):
    """
    绘制热门与冷门号码对比图，并返回QPixmap对象
    
    Args:
        data: 统计数据dict
        width, height: 图像尺寸
        dpi: 图像分辨率
    
    Returns:
        QPixmap对象
    """
    # 设置更好的图表风格
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
    
    # 热门冷门红球
    red_hot = pd.Series(data["红球热门号码"])
    red_cold = pd.Series(data["红球冷门号码"])
    
    df_red = pd.DataFrame({
        '热门号码': red_hot, 
        '冷门号码': pd.Series(red_cold.values, index=red_cold.index)
    })
    
    df_red.plot(kind='bar', ax=ax1, color=['#FF2222', '#FFAAAA'])
    ax1.set_xlabel('红球号码', fontsize=12)
    ax1.set_ylabel('出现次数', fontsize=12)
    ax1.set_title('红球热门与冷门号码对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 热门冷门蓝球
    blue_hot = pd.Series(data["蓝球热门号码"])
    blue_cold = pd.Series(data["蓝球冷门号码"])
    
    df_blue = pd.DataFrame({
        '热门号码': blue_hot, 
        '冷门号码': pd.Series(blue_cold.values, index=blue_cold.index)
    })
    
    df_blue.plot(kind='bar', ax=ax2, color=['#2222FF', '#AAAAFF'])
    ax2.set_xlabel('蓝球号码', fontsize=12)
    ax2.set_ylabel('出现次数', fontsize=12)
    ax2.set_title('蓝球热门与冷门号码对比', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # 将matplotlib图形转换为QPixmap
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    
    return pixmap

@memoize(expire_seconds=300)
def plot_gap_statistics(data, lottery_type, width=10, height=6, dpi=100):
    """
    绘制遗漏值统计图，并返回QPixmap对象
    
    Args:
        data: 统计数据dict
        lottery_type: 'ssq' 或 'dlt'
        width, height: 图像尺寸
        dpi: 图像分辨率
    
    Returns:
        QPixmap对象
    """
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
    
    # 红球遗漏值
    red_gaps = pd.Series(data["红球遗漏值"])
    x = list(red_gaps.index)
    y = list(red_gaps.values)
    
    # 创建渐变色，遗漏值越大颜色越深
    norm = plt.Normalize(min(y), max(y))
    colors = plt.cm.Reds(norm(y))
    
    ax1.bar(x, y, color=colors)
    ax1.set_xlabel('红球号码', fontsize=12)
    ax1.set_ylabel('遗漏期数', fontsize=12)
    ax1.set_title('红球当前遗漏统计', fontsize=14, fontweight='bold')
    
    # 给较大遗漏值标记数字
    threshold = np.percentile(y, 75)  # 标记前25%的遗漏值
    for i, value in enumerate(y):
        if value >= threshold:
            ax1.text(x[i], value + 0.5, str(value), ha='center', fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    
    # 蓝球遗漏值
    blue_gaps = pd.Series(data["蓝球遗漏值"])
    x = list(blue_gaps.index)
    y = list(blue_gaps.values)
    
    # 创建渐变色，遗漏值越大颜色越深
    norm = plt.Normalize(min(y), max(y))
    colors = plt.cm.Blues(norm(y))
    
    ax2.bar(x, y, color=colors)
    ax2.set_xlabel('蓝球号码', fontsize=12)
    ax2.set_ylabel('遗漏期数', fontsize=12)
    ax2.set_title('蓝球当前遗漏统计', fontsize=14, fontweight='bold')
    
    # 给较大遗漏值标记数字
    threshold = np.percentile(y, 75)  # 标记前25%的遗漏值
    for i, value in enumerate(y):
        if value >= threshold:
            ax2.text(x[i], value + 0.5, str(value), ha='center', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # 将matplotlib图形转换为QPixmap
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    
    return pixmap

@memoize(expire_seconds=300)
def plot_patterns(data, width=10, height=8, dpi=100):
    """
    绘制号码模式分析图，包括奇偶比、大小比，并返回QPixmap对象
    
    Args:
        data: 统计数据dict
        width, height: 图像尺寸
        dpi: 图像分辨率
    
    Returns:
        QPixmap对象
    """
    # 设置更好的图表风格
    plt.style.use('ggplot')
    
    fig, axs = plt.subplots(2, 2, figsize=(width, height), dpi=dpi)
    
    # 红球奇偶比
    red_odd_even = data["红球奇偶比"]
    axs[0, 0].pie([red_odd_even["奇数"], red_odd_even["偶数"]], 
                  labels=["奇数", "偶数"], 
                  colors=['#FF5555', '#FFAAAA'],
                  autopct='%1.1f%%',
                  startangle=90,
                  explode=(0.05, 0),
                  shadow=True)
    axs[0, 0].set_title('红球奇偶比例', fontsize=14, fontweight='bold')
    
    # 蓝球奇偶比
    blue_odd_even = data["蓝球奇偶比"]
    axs[0, 1].pie([blue_odd_even["奇数"], blue_odd_even["偶数"]], 
                  labels=["奇数", "偶数"], 
                  colors=['#5555FF', '#AAAAFF'],
                  autopct='%1.1f%%',
                  startangle=90,
                  explode=(0.05, 0),
                  shadow=True)
    axs[0, 1].set_title('蓝球奇偶比例', fontsize=14, fontweight='bold')
    
    # 红球大小比
    red_size = data["红球大小比"]
    axs[1, 0].pie([red_size["小号"], red_size["大号"]], 
                 labels=["小号", "大号"], 
                 colors=['#FFAA55', '#FF7722'],
                 autopct='%1.1f%%',
                 startangle=90,
                 explode=(0.05, 0),
                 shadow=True)
    axs[1, 0].set_title('红球大小比例', fontsize=14, fontweight='bold')
    
    # 蓝球大小比
    blue_size = data["蓝球大小比"]
    axs[1, 1].pie([blue_size["小号"], blue_size["大号"]], 
                  labels=["小号", "大号"], 
                  colors=['#55AAFF', '#2277FF'],
                  autopct='%1.1f%%',
                  startangle=90,
                  explode=(0.05, 0),
                  shadow=True)
    axs[1, 1].set_title('蓝球大小比例', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    # 将matplotlib图形转换为QPixmap
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    
    return pixmap

@memoize(expire_seconds=300)
def plot_trend_analysis(data, width=12, height=6, dpi=100):
    """
    绘制号码趋势分析图，并返回QPixmap对象
    
    Args:
        data: 统计数据dict
        width, height: 图像尺寸
        dpi: 图像分辨率
    
    Returns:
        QPixmap对象
    """
    plt.style.use('ggplot')
    
    if not data["趋势数据"] or len(data["趋势数据"]["期数"]) < 2:
        # 如果没有足够的趋势数据，创建一个提示图
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.text(0.5, 0.5, '没有足够的趋势数据可供分析', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 将matplotlib图形转换为QPixmap
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        return pixmap
    
    fig, axs = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
    
    # 复制数据防止修改原始数据
    trend_data = data["趋势数据"].copy()
    
    # 获取日期和指标
    issues = trend_data["期数"]
    red_sum = trend_data["红球和值"]
    red_max_diff = trend_data["红球最大差值"]
    
    # 绘制红球和值趋势
    axs[0].plot(issues, red_sum, 'r-', marker='o', markersize=5, linewidth=2)
    axs[0].set_title('红球和值趋势', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('和值', fontsize=12)
    axs[0].set_xlabel('期数', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    
    # 计算移动平均线
    if len(red_sum) >= 5:
        window = min(5, len(red_sum) - 1)
        red_sum_ma = pd.Series(red_sum).rolling(window=window).mean().values
        axs[0].plot(issues, red_sum_ma, 'b--', linewidth=1.5, label=f'{window}期移动平均')
        axs[0].legend(loc='best')
    
    # 绘制红球最大差值趋势
    axs[1].plot(issues, red_max_diff, 'g-', marker='s', markersize=5, linewidth=2)
    axs[1].set_title('红球最大差值趋势', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('最大差值', fontsize=12)
    axs[1].set_xlabel('期数', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    
    # 计算移动平均线
    if len(red_max_diff) >= 5:
        window = min(5, len(red_max_diff) - 1)
        red_max_diff_ma = pd.Series(red_max_diff).rolling(window=window).mean().values
        axs[1].plot(issues, red_max_diff_ma, 'b--', linewidth=1.5, label=f'{window}期移动平均')
        axs[1].legend(loc='best')
    
    fig.tight_layout()
    
    # 将matplotlib图形转换为QPixmap
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    
    return pixmap 