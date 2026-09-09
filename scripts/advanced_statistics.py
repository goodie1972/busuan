# -*- coding:utf-8 -*-
"""
Advanced Statistics Module for Lottery Analysis
Author: Yang Zhao
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from io import BytesIO
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import functools
import time
import logging
import traceback
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('advanced_statistics')

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"设置中文字体支持时出错: {str(e)}")
    pass

# 添加自定义的runs_test函数
def runs_test(x):
    """
    实现一个简单的游程检验(Wald–Wolfowitz runs test)
    
    参数:
        x: 布尔数组，表示数据是否超过中位数
        
    返回:
        (runs, p-value): 游程数和对应的p值
    """
    # 计算游程数
    runs = 1 + np.sum(x[1:] != x[:-1])
    
    # 计算上升和下降的个数
    n1 = np.sum(x)
    n2 = len(x) - n1
    
    # 计算期望游程数和标准差
    r_mean = 2 * n1 * n2 / (n1 + n2) + 1
    r_var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    r_std = np.sqrt(r_var)
    
    # 计算Z统计量
    z = (runs - r_mean) / r_std
    
    # 计算p值（双尾检验）
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return runs, p_value

def memoize(expire_seconds=300):
    """函数结果缓存装饰器，带过期时间"""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            if key in cache and current_time - cache[key]['timestamp'] < expire_seconds:
                logging.debug(f"使用缓存结果: {func.__name__}")
                return cache[key]['result']
            
            result = func(*args, **kwargs)
            cache[key] = {'result': result, 'timestamp': current_time}
            return result
        return wrapper
    return decorator

@memoize(expire_seconds=600)
def calculate_advanced_statistics(df, lottery_type):
    """
    计算高级统计指标
    
    Args:
        df: 彩票数据DataFrame
        lottery_type: 彩票类型 ('dlt' 或 'ssq')
    
    Returns:
        dict: 包含各种统计指标的字典
    """
    stats_dict = {}
    
    try:
        logger.info(f"开始计算{lottery_type}的高级统计指标...")
        
        # 确定红球和蓝球的列名
        if lottery_type == 'dlt':
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
            red_range = (1, 35)
            blue_range = (1, 12)
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
            # 确保正确获取蓝球列
            blue_cols = []
            for col in df.columns:
                if col.startswith('蓝球_') or col.startswith('蓝球'):
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                        break
            
            red_range = (1, 33)
            blue_range = (1, 16)
        
        logger.info(f"红球列: {red_cols}")
        logger.info(f"蓝球列: {blue_cols}")
        
        # 检查是否找到了所有需要的列
        if not red_cols:
            logger.error(f"未找到红球列，可用列: {list(df.columns)}")
            return {"error": "未找到红球列"}
            
        if not blue_cols:
            logger.error(f"未找到蓝球列，可用列: {list(df.columns)}")
            return {"error": "未找到蓝球列"}
        
        # 计算红球统计指标
        for col in red_cols:
            try:
                numbers = df[col].values
                stats_dict[f'{col}_stats'] = calculate_column_statistics(numbers)
                logger.info(f"计算{col}统计指标完成")
            except Exception as e:
                logger.error(f"计算{col}统计指标时出错: {str(e)}")
                logger.error(traceback.format_exc())
                stats_dict[f'{col}_stats'] = {'error': str(e)}
        
        # 计算蓝球统计指标
        for col in blue_cols:
            try:
                numbers = df[col].values
                stats_dict[f'{col}_stats'] = calculate_column_statistics(numbers)
                logger.info(f"计算{col}统计指标完成")
            except Exception as e:
                logger.error(f"计算{col}统计指标时出错: {str(e)}")
                logger.error(traceback.format_exc())
                stats_dict[f'{col}_stats'] = {'error': str(e)}
        
        # 计算相关性矩阵
        try:
            # 红球间的相关性
            if len(red_cols) > 1:
                red_corr = df[red_cols].corr()
                if 'correlations' not in stats_dict:
                    stats_dict['correlations'] = {}
                stats_dict['correlations']['red'] = red_corr
                logger.info("计算红球相关性矩阵完成")
            
            # 蓝球间的相关性
            if len(blue_cols) > 1:
                blue_corr = df[blue_cols].corr()
                if 'correlations' not in stats_dict:
                    stats_dict['correlations'] = {}
                stats_dict['correlations']['blue'] = blue_corr
                logger.info("计算蓝球相关性矩阵完成")
            
            # 如果蓝球只有一列，创建一个1x1的相关性矩阵
            elif len(blue_cols) == 1:
                if 'correlations' not in stats_dict:
                    stats_dict['correlations'] = {}
                stats_dict['correlations']['blue'] = pd.DataFrame(
                    [[1.0]], index=blue_cols, columns=blue_cols
                )
                logger.info("蓝球只有一列，创建1x1相关性矩阵")
        except Exception as e:
            logger.error(f"计算相关性矩阵时出错: {str(e)}")
            logger.error(traceback.format_exc())
            if 'correlations' not in stats_dict:
                stats_dict['correlations'] = {}
            stats_dict['correlations']['error'] = str(e)
        
        # 添加周期性分析（使用简化的FFT方法）
        try:
            for col in red_cols + blue_cols:
                # 使用FFT计算周期性
                data = df[col].values
                data_normalized = (data - np.mean(data)) / np.std(data) if np.std(data) != 0 else data - np.mean(data)
                fft_values = np.abs(np.fft.rfft(data_normalized))
                frequencies = np.fft.rfftfreq(len(data_normalized))
                
                # 滤除0频率分量（直流分量）
                mask = frequencies > 0
                filtered_fft = fft_values[mask]
                filtered_freq = frequencies[mask]
                
                # 保存结果
                stats_dict[f'{col}_periodicity'] = {
                    'frequencies': filtered_freq.tolist(),
                    'amplitudes': filtered_fft.tolist()
                }
                logger.info(f"计算{col}周期性分析完成")
        except Exception as e:
            logger.error(f"计算周期性分析时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 提供空数据，避免绘图函数出错
            for col in red_cols + blue_cols:
                stats_dict[f'{col}_periodicity'] = {
                    'frequencies': [],
                    'amplitudes': []
                }
        
        logger.info(f"{lottery_type}高级统计指标计算完成")
        return stats_dict
        
    except Exception as e:
        logger.error(f"计算高级统计指标时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def calculate_column_statistics(numbers):
    """
    计算单个列的统计指标
    
    Args:
        numbers: 号码数组
    
    Returns:
        dict: 包含该列统计指标的字典
    """
    result = {}
    
    try:
        # 基本统计指标
        result['mean'] = np.mean(numbers)
        result['median'] = np.median(numbers)
        result['std'] = np.std(numbers)
        result['skewness'] = stats.skew(numbers)
        result['kurtosis'] = stats.kurtosis(numbers)
        
        # 众数
        mode_result = stats.mode(numbers)
        try:
            if hasattr(mode_result, 'mode'):
                # 新版scipy
                if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
                    result['mode'] = float(mode_result.mode[0])
                else:
                    result['mode'] = float(mode_result.mode) if hasattr(mode_result.mode, 'item') else float(numbers[0])
            else:
                # 旧版scipy
                if isinstance(mode_result[0], np.ndarray) and mode_result[0].size > 0:
                    result['mode'] = float(mode_result[0][0])
                else:
                    result['mode'] = float(mode_result[0]) if hasattr(mode_result[0], 'item') else float(numbers[0])
        except (IndexError, TypeError, AttributeError) as e:
            # 如果出现任何错误，使用第一个值作为备选
            result['mode'] = float(numbers[0]) if len(numbers) > 0 else 0.0
            logging.warning(f"计算众数时出错: {str(e)}，使用第一个值代替")
        
        # 四分位数
        q1 = np.percentile(numbers, 25)
        q3 = np.percentile(numbers, 75)
        result['q1'] = q1
        result['q3'] = q3
        result['iqr'] = q3 - q1
        
        # 范围和其他基本指标
        result['range'] = np.max(numbers) - np.min(numbers)
        result['variance'] = np.var(numbers)
        result['coefficient_of_variation'] = result['std'] / result['mean'] if result['mean'] != 0 else np.nan
        result['mad'] = np.mean(np.abs(numbers - result['median']))
        result['sem'] = result['std'] / np.sqrt(len(numbers))
        
   
        values, counts = np.unique(numbers, return_counts=True)
        probs = counts / len(numbers)
        result['entropy'] = -np.sum(probs * np.log2(probs))
        
     
        try:
            normality_test = stats.normaltest(numbers)
            result['normality_test'] = (normality_test.statistic, normality_test.pvalue)
        except Exception as e:
            logger.warning(f"正态性检验失败: {str(e)}")
            result['normality_test'] = (np.nan, np.nan)
        
   
        try:
          
            median = np.median(numbers)
            binary_seq = numbers > median
            runs_result = runs_test(binary_seq)
            result['runs_test'] = runs_result
        except Exception as e:
            logger.warning(f"游程检验失败: {str(e)}")
            result['runs_test'] = (np.nan, np.nan)
            
    
        try:
            if len(numbers) > 1:
               
                series = pd.Series(numbers)
                result['autocorr'] = series.autocorr(lag=1)
                if pd.isna(result['autocorr']):  
                    
                    n = len(numbers)
                    mean = np.mean(numbers)
                    c0 = np.sum((numbers - mean) ** 2) / n
                    c1 = np.sum((numbers[:-1] - mean) * (numbers[1:] - mean)) / (n - 1)
                    result['autocorr'] = c1 / c0 if c0 != 0 else 0
            else:
                result['autocorr'] = 0
        except Exception as e:
            logger.warning(f"自相关系数计算失败: {str(e)}")
            result['autocorr'] = 0
            
        return result
        
    except Exception as e:
        logger.error(f"计算列统计指标时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}

@memoize(expire_seconds=300)
def plot_advanced_statistics(df, lottery_type, width=15, height=15, dpi=100):
    """
    绘制高级统计图表
    
    Args:
        df: 彩票数据DataFrame
        lottery_type: 彩票类型 ('dlt' 或 'ssq')
        width: 图表宽度
        height: 图表高度
        dpi: 图表分辨率
    
    Returns:
        QPixmap: 包含统计图表的QPixmap对象
    """
 
    fig = plt.figure(figsize=(width, height))
    
   
    stats_dict = calculate_advanced_statistics(df, lottery_type)
    
  
    if lottery_type == 'dlt':
        red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
        blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
    else:  # ssq
        red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
        blue_cols = [col for col in df.columns if col.startswith('蓝球')][:1]
    
   
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1])
    

    ax1 = fig.add_subplot(gs[0, :])
    data_to_plot = [df[col] for col in red_cols + blue_cols]
    ax1.boxplot(data_to_plot, labels=red_cols + blue_cols, notch=True)
    ax1.set_title('号码分布箱线图')
    ax1.set_ylabel('号码值')
    
  
    ax2 = fig.add_subplot(gs[1, 0])
    skewness_data = [stats_dict[f'{col}_stats']['skewness'] for col in red_cols + blue_cols]
    kurtosis_data = [stats_dict[f'{col}_stats']['kurtosis'] for col in red_cols + blue_cols]
    x = np.arange(len(red_cols + blue_cols))
    width_bar = 0.35
    ax2.bar(x - width_bar/2, skewness_data, width_bar, label='偏度')
    ax2.bar(x + width_bar/2, kurtosis_data, width_bar, label='峰度')
    ax2.set_title('号码偏度和峰度对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(red_cols + blue_cols, rotation=45)
    ax2.legend()
    
  
    ax3 = fig.add_subplot(gs[1, 1])
    std_data = [stats_dict[f'{col}_stats']['std'] for col in red_cols + blue_cols]
    cv_data = [stats_dict[f'{col}_stats']['coefficient_of_variation'] for col in red_cols + blue_cols]
    ax3.bar(x - width_bar/2, std_data, width_bar, label='标准差')
    ax3.bar(x + width_bar/2, cv_data, width_bar, label='变异系数')
    ax3.set_title('号码标准差和变异系数对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(red_cols + blue_cols, rotation=45)
    ax3.legend()
    
   
    ax4 = fig.add_subplot(gs[2, 0])
    mad_data = [stats_dict[f'{col}_stats']['mad'] for col in red_cols + blue_cols]
    sem_data = [stats_dict[f'{col}_stats']['sem'] for col in red_cols + blue_cols]
    ax4.bar(x - width_bar/2, mad_data, width_bar, label='中位数绝对偏差')
    ax4.bar(x + width_bar/2, sem_data, width_bar, label='标准误')
    ax4.set_title('号码稳定性分析')
    ax4.set_xticks(x)
    ax4.set_xticklabels(red_cols + blue_cols, rotation=45)
    ax4.legend()
    
    
    ax5 = fig.add_subplot(gs[2, 1])
    entropy_data = [stats_dict[f'{col}_stats']['entropy'] for col in red_cols + blue_cols]
    autocorr_data = [stats_dict[f'{col}_stats']['autocorr'] for col in red_cols + blue_cols]
    ax5.bar(x - width_bar/2, entropy_data, width_bar, label='熵值')
    ax5.bar(x + width_bar/2, autocorr_data, width_bar, label='自相关系数')
    ax5.set_title('号码随机性分析')
    ax5.set_xticks(x)
    ax5.set_xticklabels(red_cols + blue_cols, rotation=45)
    ax5.legend()
    
  
    ax6 = fig.add_subplot(gs[3, 0])
    try:
        sns.heatmap(stats_dict['correlations']['red'], annot=True, cmap='coolwarm', center=0, ax=ax6)
        ax6.set_title('红球号码相关性')
    except Exception as e:
        ax6.text(0.5, 0.5, f"无法显示红球相关性:\n{str(e)}", 
                ha='center', va='center', fontsize=10, color='red',
                transform=ax6.transAxes)
        ax6.set_title('红球相关性 - 出错')
    
   
    ax7 = fig.add_subplot(gs[3, 1])
    try:
        if 'blue' in stats_dict['correlations']:
            sns.heatmap(stats_dict['correlations']['blue'], annot=True, cmap='coolwarm', center=0, ax=ax7)
            ax7.set_title('蓝球号码相关性')
        else:
            ax7.text(0.5, 0.5, "蓝球相关性数据不可用", 
                    ha='center', va='center', fontsize=10, color='blue',
                    transform=ax7.transAxes)
            ax7.set_title('蓝球相关性 - 无数据')
    except Exception as e:
        ax7.text(0.5, 0.5, f"无法显示蓝球相关性:\n{str(e)}", 
                ha='center', va='center', fontsize=10, color='red',
                transform=ax7.transAxes)
        ax7.set_title('蓝球相关性 - 出错')
    
 
    ax8 = fig.add_subplot(gs[4, :])
    try:
        # 显示红蓝球周期性
        for col in red_cols:
            if f'{col}_periodicity' in stats_dict and stats_dict[f'{col}_periodicity']['frequencies']:
                periodicity = stats_dict[f'{col}_periodicity']
                ax8.plot(periodicity['frequencies'], periodicity['amplitudes'], label=col)
        
        for col in blue_cols:
            if f'{col}_periodicity' in stats_dict and stats_dict[f'{col}_periodicity']['frequencies']:
                periodicity = stats_dict[f'{col}_periodicity']
                ax8.plot(periodicity['frequencies'], periodicity['amplitudes'], label=col, linestyle='--')
        
        ax8.set_title('号码周期性分析')
        ax8.set_xlabel('频率')
        ax8.set_ylabel('振幅')
        ax8.legend()
    except Exception as e:
        ax8.text(0.5, 0.5, f"无法显示周期性分析:\n{str(e)}", 
                ha='center', va='center', fontsize=10, color='red',
                transform=ax8.transAxes)
        ax8.set_title('周期性分析 - 出错')
    

    plt.tight_layout()
    

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    buf.close()
    
    return pixmap

@memoize(expire_seconds=300)
def plot_distribution_analysis(df, lottery_type, width=15, height=12, dpi=100):
    """
    绘制分布分析图表
    
    Args:
        df: 彩票数据DataFrame
        lottery_type: 彩票类型 ('dlt' 或 'ssq')
        width: 图表宽度
        height: 图表高度
        dpi: 图表分辨率
    
    Returns:
        QPixmap: 包含分布分析图表的QPixmap对象
    """

    fig = plt.figure(figsize=(width, height))
    
 
    if lottery_type == 'dlt':
        red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
        blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
    else:  # ssq
        red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
        blue_cols = [col for col in df.columns if col.startswith('蓝球')][:1]
    
  
    gs = fig.add_gridspec(3, 2)
    

    ax1 = fig.add_subplot(gs[0, 0])
    for col in red_cols:
        sns.kdeplot(data=df[col], label=col, ax=ax1)
    ax1.set_title('红球号码核密度估计')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    for col in blue_cols:
        sns.kdeplot(data=df[col], label=col, ax=ax2)
    ax2.set_title('蓝球号码核密度估计')
    ax2.legend()
    
 
    ax3 = fig.add_subplot(gs[1, 0])
    for col in red_cols:
        stats.probplot(df[col], dist="norm", plot=ax3)
    ax3.set_title('红球号码Q-Q图')
    
 
    ax4 = fig.add_subplot(gs[1, 1])
    for col in blue_cols:
        stats.probplot(df[col], dist="norm", plot=ax4)
    ax4.set_title('蓝球号码Q-Q图')
    
    ax5 = fig.add_subplot(gs[2, 0])
    for col in red_cols:
        plt.hist(df[col], bins=20, alpha=0.3, label=col)
    ax5.set_title('红球号码直方图')
    ax5.legend()
    
   
    ax6 = fig.add_subplot(gs[2, 1])
    for col in blue_cols:
        plt.hist(df[col], bins=20, alpha=0.3, label=col)
    ax6.set_title('蓝球号码直方图')
    ax6.legend()
    
 
    plt.tight_layout()
    
 
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    buf.close()
    
    return pixmap 