# -*- coding:utf-8 -*-
"""
基于期望值模型的彩票预测策略
基于博弈论中期望值概念
参考自：Winning the Ransomware Lottery: A Game-Theoretic Approach
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import os
import pickle
import time

class ExpectedValueLotteryModel:
    """基于期望值模型的彩票预测策略"""
    
    def __init__(self, lottery_type='ssq', log_callback=None, use_gpu=False, verbose=False):
        """
        初始化模型
        
        Args:
            lottery_type: 'ssq' 或 'dlt'
            log_callback: 日志回调函数
            use_gpu: 是否使用GPU (不影响此算法)
            verbose: 是否输出详细的计算过程日志
        """
        self.lottery_type = lottery_type
        self.log_callback = log_callback
        self.verbose = verbose
        
        # 选择基于彩票类型的球号范围
        if lottery_type == 'dlt':
            # 大乐透: 5个红球(1-35)，2个蓝球(1-12)
            self.red_range = 35
            self.blue_range = 12
            self.red_count = 5
            self.blue_count = 2
        else:  # ssq
            # 双色球: 6个红球(1-33)，1个蓝球(1-16)
            self.red_range = 33
            self.blue_range = 16
            self.red_count = 6
            self.blue_count = 1
        
        # 初始化概率缓存
        self.red_probabilities = None
        self.blue_probabilities = None
        self.red_combinations_value = None
        self.blue_combinations_value = None
        
        # 为方便访问，提供属性别名
        self.red_probs = {}
        self.blue_probs = {}
        
        # 模型文件路径
        self.models_dir = os.path.join(f'./model/{lottery_type}')
        os.makedirs(self.models_dir, exist_ok=True)
        self.red_probs_file = os.path.join(self.models_dir, 'ev_red_probabilities.pkl')
        self.blue_probs_file = os.path.join(self.models_dir, 'ev_blue_probabilities.pkl')
        self.red_values_file = os.path.join(self.models_dir, 'ev_red_values.pkl')
        self.blue_values_file = os.path.join(self.models_dir, 'ev_blue_values.pkl')
        
        # 初始化logger
        self.logger = logging.getLogger(f"ev_model_{lottery_type}")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log(self, message):
        """记录日志并发送到UI（如果有回调）"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    def train(self, df):
        """
        训练期望值模型
        
        Args:
            df: 包含历史开奖数据的DataFrame
        """
        self.log("训练期望值模型...")
        
        # 确保数据按期数排序（从旧到新）
        df = df.sort_values('期数', ascending=True).reset_index(drop=True)
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
            blue_cols = []
            for col in df.columns:
                if col.startswith('蓝球_') or col == '蓝球':
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                        break
        
        # 计算每个号码的出现频率（概率）
        self.log("计算号码出现概率...")
        self.red_probabilities = self._calculate_number_probabilities(df, red_cols, 'red')
        self.blue_probabilities = self._calculate_number_probabilities(df, blue_cols, 'blue')
        
        # 计算组合的价值
        self.log("计算组合价值...")
        self.red_combinations_value = self._calculate_combinations_value(df, red_cols, 'red')
        self.blue_combinations_value = self._calculate_combinations_value(df, blue_cols, 'blue')
        
        # 保存模型数据
        self._save_model_data()
        
        self.log("期望值模型训练完成")
    
    def _calculate_number_probabilities(self, df, ball_cols, ball_type):
        """
        计算每个号码的出现概率
        
        Args:
            df: 历史数据
            ball_cols: 球号列名
            ball_type: 'red'或'blue'
        
        Returns:
            每个号码的概率字典
        """
        ball_range = self.red_range if ball_type == 'red' else self.blue_range
        
        count_dict = defaultdict(int)
        
        total_drawings = len(df)
        
        for _, row in df.iterrows():
            for col in ball_cols:
                number = row[col]
                if pd.notna(number):
                    count_dict[int(number)] += 1
        
        probability_dict = {}
        for number in range(1, ball_range + 1):
            probability_dict[number] = count_dict[number] / (total_drawings * len(ball_cols))
            
        self.log(f"{ball_type}球概率分布:")
        sorted_probs = sorted(probability_dict.items(), key=lambda x: x[1], reverse=True)
        for number, prob in sorted_probs[:5]:
            self.log(f"  {number}号: {prob:.6f}")
        
        return probability_dict
    
    def _calculate_combinations_value(self, df, ball_cols, ball_type):
        """
        计算号码组合的价值
        分析历史数据中不同组合模式的价值
        
        Args:
            df: 历史数据
            ball_cols: 球号列名
            ball_type: 'red'或'blue'
        
        Returns:
            组合特征的价值评分字典
        """
        value_dict = {
            'even_odd_ratio': {},  # 奇偶比例价值
            'high_low_ratio': {},  # 大小比例价值
            'sum_range': {},       # 和值范围价值
            'span_range': {},      # 跨度范围价值
            'sequence_pattern': {} # 序列模式价值
        }
        
        for idx, row in df.iterrows():
            numbers = [int(row[col]) for col in ball_cols if pd.notna(row[col])]
            if not numbers:
                continue
                
            even_count = sum(1 for n in numbers if n % 2 == 0)
            odd_count = len(numbers) - even_count
            even_odd_key = f"{even_count}:{odd_count}"
            
            ball_range = self.red_range if ball_type == 'red' else self.blue_range
            mid_point = ball_range // 2
            high_count = sum(1 for n in numbers if n > mid_point)
            low_count = len(numbers) - high_count
            high_low_key = f"{high_count}:{low_count}"
            
            numbers_sum = sum(numbers)
            span = max(numbers) - min(numbers)
            
            sum_range_key = (numbers_sum // 10) * 10  # 按10分区间
            
            span_range_key = (span // 5) * 5  # 按5分区间
            
            sequence_count = sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1)
            sequence_key = sequence_count
            
            value_dict['even_odd_ratio'][even_odd_key] = value_dict['even_odd_ratio'].get(even_odd_key, 0) + 1
            value_dict['high_low_ratio'][high_low_key] = value_dict['high_low_ratio'].get(high_low_key, 0) + 1
            value_dict['sum_range'][sum_range_key] = value_dict['sum_range'].get(sum_range_key, 0) + 1
            value_dict['span_range'][span_range_key] = value_dict['span_range'].get(span_range_key, 0) + 1
            value_dict['sequence_pattern'][sequence_key] = value_dict['sequence_pattern'].get(sequence_key, 0) + 1
        
        total_drawings = len(df)
        for feature, feature_dict in value_dict.items():
            for key in feature_dict:
                feature_dict[key] /= total_drawings
        
        return value_dict
    
    def _save_model_data(self):
        """保存模型数据到文件"""
        self.log("保存期望值模型数据...")
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.red_probs_file), exist_ok=True)
            
            # 创建模型类型目录，保持与其他模型兼容
            model_type_dir = os.path.join(self.models_dir, 'expected_value')
            os.makedirs(model_type_dir, exist_ok=True)
            
            # 保存模型信息
            model_info = {
                'model_type': 'expected_value',
                'lottery_type': self.lottery_type,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(model_type_dir, 'model_info.json'), 'w') as f:
                import json
                json.dump(model_info, f, indent=4)
            
            # 保存模型数据
            with open(self.red_probs_file, 'wb') as f:
                pickle.dump(self.red_probabilities, f)
                
            with open(self.blue_probs_file, 'wb') as f:
                pickle.dump(self.blue_probabilities, f)
                
            with open(self.red_values_file, 'wb') as f:
                pickle.dump(self.red_combinations_value, f)
                
            with open(self.blue_values_file, 'wb') as f:
                pickle.dump(self.blue_combinations_value, f)
            
            # 更新属性别名
            self.red_probs = {num-1: prob for num, prob in self.red_probabilities.items()}
            self.blue_probs = {num-1: prob for num, prob in self.blue_probabilities.items()}
                
            self.log(f"模型数据保存成功: {self.models_dir}")
        except Exception as e:
            self.log(f"保存模型数据时出错: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
    
    def load(self):
        """加载模型数据"""
        self.log("加载期望值模型数据...")
        
        try:
            # 检查模型文件路径
            self.log(f"尝试从路径加载: {self.red_probs_file}")
            if not os.path.exists(self.red_probs_file):
                self.log(f"模型文件不存在: {self.red_probs_file}")
                return False
                
            with open(self.red_probs_file, 'rb') as f:
                self.red_probabilities = pickle.load(f)
                
            with open(self.blue_probs_file, 'rb') as f:
                self.blue_probabilities = pickle.load(f)
                
            with open(self.red_values_file, 'rb') as f:
                self.red_combinations_value = pickle.load(f)
                
            with open(self.blue_values_file, 'rb') as f:
                self.blue_combinations_value = pickle.load(f)
            
            # 检查加载的数据是否有效
            if (self.red_probabilities and self.blue_probabilities and 
                self.red_combinations_value and self.blue_combinations_value):
                self.log("模型数据加载成功")
                return True
            else:
                self.log("部分模型数据无效或为空")
                return False
                
        except Exception as e:
            self.log(f"模型数据加载失败: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            return False
    
    def predict(self, recent_data=None, num_predictions=1):
        """
        生成预测结果
        
        Args:
            recent_data: 最近的数据，用于调整预测（可选）
            num_predictions: 生成的预测数量
            
        Returns:
            tuple: (red_predictions, blue_predictions) 预测的红球和蓝球类别索引
        """
        if self.red_probabilities is None or self.blue_probabilities is None:
            if not self.load():
                self.log("错误：模型数据未加载，无法进行预测")
                return None, None
                
        # 更新属性别名，确保UI可以访问到概率
        if not self.red_probs and self.red_probabilities:
            self.red_probs = {num-1: prob for num, prob in self.red_probabilities.items()}
        if not self.blue_probs and self.blue_probabilities:
            self.blue_probs = {num-1: prob for num, prob in self.blue_probabilities.items()}
        
        if self.verbose:
            self.log(f"开始生成{num_predictions}组预测...")
        
        red_predictions = []
        blue_predictions = []
        
        for i in range(num_predictions):
            if self.verbose:
                self.log(f"生成第{i+1}组预测:")
                
            # 预测红球
            red_balls = self._predict_balls('red', recent_data)
            red_predictions.append(red_balls)
            
            # 预测蓝球
            blue_balls = self._predict_balls('blue', recent_data)
            blue_predictions.append(blue_balls)
            
            if self.verbose:
                # 将索引转换为实际号码（1-based）进行显示
                red_nums = [r+1 for r in red_balls]
                blue_nums = [b+1 for b in blue_balls]
                red_nums.sort()
                blue_nums.sort()
                self.log(f"  预测结果 - 红球: {red_nums}, 蓝球: {blue_nums}")
        
        return red_predictions, blue_predictions
    
    def _predict_balls(self, ball_type, recent_data=None):
        """
        基于期望值模型预测号码
        
        Args:
            ball_type: 'red'或'blue'
            recent_data: 最近的数据，用于调整预测（可选）
            
        Returns:
            预测的球号索引列表
        """
        if self.verbose:
            self.log(f"  计算{ball_type}球预测...")

        ball_range = self.red_range if ball_type == 'red' else self.blue_range
        ball_count = self.red_count if ball_type == 'red' else self.blue_count
        ball_probs = self.red_probabilities if ball_type == 'red' else self.blue_probabilities
        ball_values = self.red_combinations_value if ball_type == 'red' else self.blue_combinations_value
        
        if self.verbose:
            self.log(f"  {ball_type}球区间: 1-{ball_range}, 需选择{ball_count}个")

        base_probs = np.zeros(ball_range)
        for num, prob in ball_probs.items():
            if 1 <= num <= ball_range:
                base_probs[num-1] = prob
        
        # 调整因子1：基于组合价值对概率加权
        combination_weights = self._calculate_combination_weights(ball_type, recent_data)
        
        # 调整因子2：基于最近数据的时序影响
        recency_weights = np.ones(ball_range)
        if recent_data is not None:
            recency_weights = self._calculate_recency_weights(ball_type, recent_data)
        
        # 计算综合概率
        combined_probs = base_probs * combination_weights * recency_weights
        combined_probs = combined_probs / combined_probs.sum()
        
        if self.verbose:
            # 显示权重最高的几个号码
            indices = np.argsort(-combined_probs)[:10]  # 取概率最高的10个号码
            self.log(f"  {ball_type}球权重最高的号码:")
            for idx in indices:
                num = idx + 1  # 转为1-based显示
                self.log(f"    {num}号: 基础={base_probs[idx]:.4f}, 组合={combination_weights[idx]:.2f}, 最近={recency_weights[idx]:.2f}, 最终={combined_probs[idx]:.4f}")
        
        selected_indices = []
        temp_probs = combined_probs.copy()
        
        for i in range(ball_count):
            if self.verbose:
                self.log(f"  选择第{i+1}个{ball_type}球...")
                
            temperature = 1.2  # >1增加随机性，<1减少随机性
            adjusted_probs = np.power(temp_probs, 1/temperature)
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            idx = np.random.choice(ball_range, p=adjusted_probs)
            selected_indices.append(idx)
            
            if self.verbose:
                self.log(f"    选中: {idx+1}号 (权重={combined_probs[idx]:.4f})")
            
            # 设置已选号码的概率为0，确保不重复选择
            temp_probs[idx] = 0
            if temp_probs.sum() > 0:
                temp_probs = temp_probs / temp_probs.sum()
        
        if self.verbose:
            sorted_indices = sorted(selected_indices)
            self.log(f"  最终选择的{ball_type}球: {[i+1 for i in sorted_indices]}")
            
        return selected_indices
    
    def _calculate_combination_weights(self, ball_type, recent_data=None):
        """
        计算基于组合价值的权重
        
        Args:
            ball_type: 'red'或'blue'
            recent_data: 最近的数据（可选）
            
        Returns:
            每个号码的权重数组
        """
        ball_range = self.red_range if ball_type == 'red' else self.blue_range
        weights = np.ones(ball_range)
        
        # 这里我们可以实现更复杂的权重计算策略
        # 例如，基于奇偶比、大小比、和值等特征给每个号码赋予权重
        
        if recent_data is not None and len(recent_data) > 0:
            even_odd_pattern = self._analyze_even_odd_pattern(ball_type, recent_data)
            
            for i in range(ball_range):
                number = i + 1
                if number % 2 == 0 and even_odd_pattern.get('even_ratio', 0.5) < 0.4:
                    weights[i] *= 1.2
                elif number % 2 != 0 and even_odd_pattern.get('odd_ratio', 0.5) < 0.4:
                    weights[i] *= 1.2
        
        return weights
    
    def _calculate_recency_weights(self, ball_type, recent_data):
        """
        计算基于最近数据的时序权重
        
        Args:
            ball_type: 'red'或'blue'
            recent_data: 最近的开奖数据
            
        Returns:
            每个号码的权重数组
        """
        ball_range = self.red_range if ball_type == 'red' else self.blue_range
        weights = np.ones(ball_range)
        
        if recent_data is None or len(recent_data) == 0:
            return weights
            
        if ball_type == 'red':
            if self.lottery_type == 'dlt':
                ball_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            else:  # ssq
                ball_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
        else:  # blue
            if self.lottery_type == 'dlt':
                ball_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
            else:  # ssq
                ball_cols = []
                for col in recent_data.columns:
                    if col.startswith('蓝球_') or col == '蓝球':
                        ball_cols.append(col)
                        if len(ball_cols) >= 1:  # 双色球只有1个蓝球
                            break
        
        recent_count = {i+1: 0 for i in range(ball_range)}
        total_records = min(5, len(recent_data))  # 只看最近5期
        
        for idx, row in recent_data.head(total_records).iterrows():
            for col in ball_cols:
                if pd.notna(row[col]):
                    number = int(row[col])
                    if 1 <= number <= ball_range:
                        recent_count[number] += 1
        
        for i in range(ball_range):
            number = i + 1
            count = recent_count[number]
            if count >= 3:  # 频繁出现
                weights[i] *= 0.8  # 降低权重
            elif count == 0:  # 没有出现
                weights[i] *= 1.2  # 提高权重
        
        return weights
    
    def _analyze_even_odd_pattern(self, ball_type, recent_data):
        """分析最近数据中的奇偶模式"""
        if ball_type == 'red':
            if self.lottery_type == 'dlt':
                ball_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            else:  # ssq
                ball_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
        else:  # blue
            if self.lottery_type == 'dlt':
                ball_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
            else:  # ssq
                ball_cols = []
                for col in recent_data.columns:
                    if col.startswith('蓝球_') or col == '蓝球':
                        ball_cols.append(col)
                        if len(ball_cols) >= 1:
                            break
        
        even_count = 0
        odd_count = 0
        total_numbers = 0
        
        for idx, row in recent_data.head(5).iterrows():
            for col in ball_cols:
                if pd.notna(row[col]):
                    number = int(row[col])
                    if number % 2 == 0:
                        even_count += 1
                    else:
                        odd_count += 1
                    total_numbers += 1
        
        if total_numbers == 0:
            return {'even_ratio': 0.5, 'odd_ratio': 0.5}
            
        return {
            'even_ratio': even_count / total_numbers,
            'odd_ratio': odd_count / total_numbers
        }

if __name__ == "__main__":
    from scripts.data_analysis import load_lottery_data
    
    lottery_type = 'ssq'  # 或 'dlt'
    df = load_lottery_data(lottery_type)
    
    model = ExpectedValueLotteryModel(lottery_type=lottery_type)
    
    model.train(df)
    
    recent_data = df.sort_values('期数', ascending=False).head(10)
    red_predictions, blue_predictions = model.predict(recent_data, num_predictions=3)
    
    print(f"预测红球号码: {red_predictions}")
    print(f"预测蓝球号码: {blue_predictions}") 