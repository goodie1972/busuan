# -*- coding:utf-8 -*-
"""
Machine Learning Models for Lottery Prediction
Author: Yang Zhao

多种机器学习模型支持模块
包含XGBoost、随机森林等多种预测模型以及集成学习功能
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
from math import comb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging
import json

# 尝试导入可选的模型库
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 导入期望值模型
try:
    from expected_value_model import ExpectedValueLotteryModel
    EXPECTED_VALUE_MODEL_AVAILABLE = True
except ImportError:
    EXPECTED_VALUE_MODEL_AVAILABLE = False

# 定义支持的模型类型
MODEL_TYPES = {
    'random_forest': '随机森林',
    'xgboost': 'XGBoost',
    'gbdt': '梯度提升树',
    'ensemble': '集成模型',
    'ziwei': '紫微斗数',
    'meihua': '梅花易数'  # 已从预测模型下拉框移除，仅保留供独立标签页调用
}

# 如果可选库可用，添加到支持的模型中
if LIGHTGBM_AVAILABLE:
    MODEL_TYPES['lightgbm'] = 'LightGBM'
if CATBOOST_AVAILABLE:
    MODEL_TYPES['catboost'] = 'CatBoost'
if EXPECTED_VALUE_MODEL_AVAILABLE:
    MODEL_TYPES['expected_value'] = '期望值模型'

class WrappedXGBoostModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def __getattr__(self, name):
        # 把未定义的属性访问转发给底层原生模型（如 n_features_in_、feature_importances_）
        # 用 object.__getattribute__ 直接查实例 dict，防止 pickle 反序列化重建对象
        # （self.model 尚未恢复）时触发自引用递归
        try:
            model = object.__getattribute__(self, 'model')
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")
        return getattr(model, name)

    def predict(self, data):
        if not isinstance(data, xgb.DMatrix):
            data = xgb.DMatrix(data)
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

class WrappedGBDTModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def __getattr__(self, name):
        # 把未定义的属性访问转发给底层原生模型（如 n_features_in_、feature_importances_）
        # 用 object.__getattribute__ 直接查实例 dict，防止 pickle 反序列化重建对象
        # （self.model 尚未恢复）时触发自引用递归
        try:
            model = object.__getattribute__(self, 'model')
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")
        return getattr(model, name)

    def predict(self, data):
        raw_preds = self.model.predict_proba(data)
        return self.process_prediction(raw_preds)

class WrappedLightGBMModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def __getattr__(self, name):
        # 把未定义的属性访问转发给底层原生模型（如 n_features_in_、feature_importances_）
        # 用 object.__getattribute__ 直接查实例 dict，防止 pickle 反序列化重建对象
        # （self.model 尚未恢复）时触发自引用递归
        try:
            model = object.__getattribute__(self, 'model')
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")
        return getattr(model, name)

    def predict(self, data):
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

class WrappedCatBoostModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def __getattr__(self, name):
        # 把未定义的属性访问转发给底层原生模型（如 n_features_in_、feature_importances_）
        # 用 object.__getattribute__ 直接查实例 dict，防止 pickle 反序列化重建对象
        # （self.model 尚未恢复）时触发自引用递归
        try:
            model = object.__getattribute__(self, 'model')
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")
        return getattr(model, name)

    def predict(self, data):
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

def _class_proba_dict(model, X_scaled):
    """获取模型对单样本的逐类概率 dict {类别索引(号码-1): 概率}；无概率信息时返回 None"""
    try:
        # 递归解包（加载时可能出现包装器嵌套：wrapper.model = wrapper.model = Booster）
        base = model
        for _ in range(5):
            if not type(base).__name__.startswith('Wrapped'):
                break
            inner = getattr(base, 'model', None)
            if inner is None or inner is base:
                break
            base = inner
        # sklearn 系分类器：classes_ + predict_proba（概率列顺序=classes_）
        classes = getattr(model, 'classes_', None)
        if classes is not None and hasattr(model, 'predict_proba'):
            row = np.asarray(model.predict_proba(X_scaled))
            if row.ndim == 2:
                row = row[0]
                return {int(c): float(row[i]) for i, c in enumerate(classes) if i < len(row)}
        # lightgbm / xgboost Booster（无 classes_、predict 直接返回概率矩阵）
        # 注意: 必须用解包后的 base 调用，包装器 predict 会经 process_prediction 转成类别索引
        if 'Booster' in type(base).__name__:
            if 'XGB' in type(base).__name__:
                try:
                    import xgboost as xgb
                    row = np.asarray(base.predict(xgb.DMatrix(X_scaled)))
                except Exception:
                    row = np.asarray(base.predict(X_scaled))
            else:
                row = np.asarray(base.predict(X_scaled))
            if row.ndim == 2:
                row = row[0]
                return {i: float(row[i]) for i in range(len(row))}
    except Exception:
        pass
    return None


def _ensemble_proba_dict(models_dict, X_scaled):
    """集成模型：对每个子模型取逐类概率并平均，返回 {类别索引: 平均概率}"""
    sums = None
    count = 0
    for model in models_dict.values():
        d = _class_proba_dict(model, X_scaled)
        if d:
            if sums is None:
                sums = dict(d)
            else:
                for k, v in d.items():
                    sums[k] = sums.get(k, 0.0) + v
            count += 1
    if sums is None or count == 0:
        return None
    return {k: v / count for k, v in sums.items()}


def _mean_confidence(proba_map, numbers):
    """所选号码的模型概率均值（号码=类别索引+1；未出现的类别按0计）"""
    if not proba_map:
        return None
    vals = [proba_map.get(int(num) - 1, 0.0) for num in numbers]
    return float(np.mean(vals)) if vals else None


class LotteryMLModels:
    """彩票预测机器学习模型类"""

    def __init__(self, lottery_type='dlt', model_type='ensemble', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            model_type: 模型类型，可选值见MODEL_TYPES
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.feature_window = feature_window
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.log_callback = log_callback
        self.use_gpu = use_gpu
        
     
        self.raw_models = {}
        
   
        self.logger = logging.getLogger(f"ml_models_{lottery_type}")
        self.logger.setLevel(logging.INFO)
   
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
     
        if self.use_gpu:
            self.log(f"ML模型使用GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用'}")
        else:
            self.log("ML模型使用CPU")
        
    
        if lottery_type == 'dlt':
        
            self.red_range = 35
            self.blue_range = 12
            self.red_count = 5
            self.blue_count = 2
        else:  
     
            self.red_range = 33
            self.blue_range = 16
            self.red_count = 6
            self.blue_count = 1
        
   
        self.models_dir = os.path.join(f'./model/{lottery_type}')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def log(self, message):
        """记录日志并发送到UI（如果有回调）"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
    
        self.log("准备训练数据...")
        
       
        window_size = self.feature_window
        
   
        df = df.sort_values('期数').reset_index(drop=True)
        
   
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
        
        self.feature_cols = red_cols + blue_cols
        
     
        X_data = []
        y_red = []
        y_blue = []
        
        for i in range(len(df) - window_size):

            features = []
            for j in range(window_size):
                row_features = []
                for col in red_cols + blue_cols:
                    row_features.append(df.iloc[i + j][col])
                features.append(row_features)
            
            # 添加特征
            X_data.append(features)
            
            # 添加标签
            red_labels = []
            for col in red_cols:
                red_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_red.append(red_labels)
            
            # 添加蓝球标签
            blue_labels = []
            for col in blue_cols:
                blue_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_blue.append(blue_labels)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        y_red = np.array(y_red)
        y_blue = np.array(y_blue)
        
        # 记录数据形状
        self.log(f"特征形状: {X.shape}")
        self.log(f"红球标签形状: {y_red.shape}")
        self.log(f"蓝球标签形状: {y_blue.shape}")
        
        # 将3D特征展平为2D，以便用于传统ML模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 保存特征维度，用于预测时的兼容性检查
        self.expected_feature_count = X_reshaped.shape[1]
        self.log(f"特征维度: {self.expected_feature_count}")
        
        # 数据标准化，仅对输入特征进行处理，不对标签进行处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # 保存缩放器以便预测时使用
        self.scalers['X'] = scaler
        
        # 拆分训练集和测试集
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
            X_scaled, y_red, y_blue, test_size=test_size, random_state=42
        )
        
        # 对于ML模型，需要将多维标签展平为单维
        if y_red_train.ndim > 1:
            # 对于多个红球，随机选择一个位置进行预测
            red_pos = np.random.randint(0, y_red_train.shape[1])
            self.red_pos = red_pos
            y_red_class = y_red_train[:, red_pos]
        else:
            y_red_class = y_red_train
        
        # 对于蓝球标签，检查维度并确保安全访问
        if y_blue_train.shape[1] > 0:  # 确保数组有列可以访问
            # 对于多个蓝球，随机选择一个位置进行预测
            blue_pos = np.random.randint(0, y_blue_train.shape[1])
            self.blue_pos = blue_pos
            y_blue_class = y_blue_train[:, blue_pos]
        else:
            # 处理双色球可能没有蓝球的情况
            self.log("警告: 蓝球标签维度为0, 使用默认值0")
            y_blue_class = np.zeros(y_blue_train.shape[0], dtype=int)
            self.blue_pos = 0
        
        self.log(f"训练集特征形状: {X_train.shape}")
        
        return X_train, X_test, y_red_class, y_red_test[:, self.red_pos], y_blue_class, y_blue_test[:, self.blue_pos] if y_blue_test.shape[1] > 0 else np.zeros(y_blue_test.shape[0], dtype=int)
    
    def train_random_forest(self, X_train, y_train, ball_type, n_estimators=100):
        """训练随机森林模型"""
        from sklearn.ensemble import RandomForestClassifier
        
        self.log(f"训练{ball_type}球随机森林模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        self.log(f"随机森林参数: n_estimators={n_estimators}, max_depth=10")
        
        # 创建模型
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=1  # 启用详细输出
        )
        
        # 训练模型
        self.log(f"开始训练随机森林模型...")
        model.fit(X_train, y_train)
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            self.log(f"随机森林特征重要性 (前10个):")
            max_features = min(10, len(indices))
            for i in range(max_features):
                feature_idx = indices[i]
                feature_name = f"特征_{feature_idx}" if feature_idx >= len(self.feature_cols) else self.feature_cols[feature_idx]
                self.log(f"  {i+1}. {feature_name}: {importances[feature_idx]:.4f}")
        
        self.log(f"{ball_type}球随机森林模型训练完成")
        return model
    
    # 添加处理多维预测的静态方法，使其可序列化
    @staticmethod
    def process_multidim_prediction(raw_preds):
        """处理多维预测结果，返回类别索引"""
        if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
            # 获取前3个最可能的类别，然后随机选择一个，添加随机性
            # 这样可以避免始终返回相同的预测结果
            top_n = min(3, raw_preds.shape[1])
            if np.random.random() < 0.7:  # 70%的概率使用最高概率类别
                return np.argmax(raw_preds, axis=1)
            else:  # 30%的概率从前N个最可能的类别中随机选择
                top_indices = np.argsort(-raw_preds, axis=1)[:, :top_n]
                selected_indices = np.zeros(raw_preds.shape[0], dtype=int)
                for i in range(raw_preds.shape[0]):
                    selected_indices[i] = np.random.choice(top_indices[i])
                return selected_indices
        return raw_preds
    
    def train_xgboost(self, X_train, y_train, ball_type):
        """训练XGBoost模型"""
        import xgboost as xgb
        
        self.log(f"训练{ball_type}球XGBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数，如果使用GPU，则使用GPU实现
        params = {
            'objective': 'multi:softmax',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'verbosity': 0 
        }
        
        self.log(f"XGBoost参数: {params}")
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        if self.use_gpu and torch.cuda.is_available():
            params['tree_method'] = 'gpu_hist'  # 使用GPU加速
            self.log("XGBoost使用GPU加速训练")
        
        # 创建DMatrix数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # 设置评估列表，用于记录训练进度
        watchlist = [(dtrain, 'train')]
        
        # 使用正确的XGBoost回调函数API
        class XGBCallback(xgb.callback.TrainingCallback):
            def __init__(self, log_func):
                self.log_func = log_func
                self.iteration = 0
                
            def after_iteration(self, model, epoch, evals_log):
                if (self.iteration + 1) % 10 == 0 or self.iteration == 0:  # 每10次迭代输出一次
                    # 从evals_log获取最新的评估结果
                    metric_values = evals_log.get('train', {}).get('mlogloss', [])
                    if metric_values:
                        msg = f'XGBoost迭代 {self.iteration + 1:3d}: {metric_values[-1]:.6f}'
                        self.log_func(msg)
                self.iteration += 1
                return False
        
        # 创建回调函数，确保始终使用回调
        callbacks = [XGBCallback(self.log)]
        
        # 训练模型
        self.log(f"开始训练XGBoost模型...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=100, 
            evals=watchlist,
            callbacks=callbacks,
            verbose_eval=False  # 禁用内置的输出，使用我们的回调
        )
        
        # 输出特征重要性
        if model.feature_names:
            importance = model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            self.log(f"XGBoost特征重要性 (前10个):")
            for i, (feature, score) in enumerate(sorted_importance[:10]):
                self.log(f"  {i+1}. {feature}: {score:.4f}")
        
    
        self.raw_models[f'xgboost_{ball_type}'] = model
        
    
        wrapped_model = WrappedXGBoostModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球XGBoost模型训练完成")
        return wrapped_model
    
    def train_gbdt(self, X_train, y_train, ball_type):
        """训练梯度提升决策树模型"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.log(f"训练{ball_type}球GBDT模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
   
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'verbose': 0  # 减少自带的输出，使用我们自己的回调
        }
        
        self.log(f"GBDT参数: {params}")
        
  
        model = GradientBoostingClassifier(**params)
        
        self.log(f"开始训练GBDT模型...")
        
        n_estimators_per_batch = 10
        total_estimators = params['n_estimators']
        
        init_model = GradientBoostingClassifier(
            n_estimators=n_estimators_per_batch,
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=params['random_state'],
            warm_start=True  # 允许增量训练
        )
        
        init_model.fit(X_train, y_train)
        
        for i in range(n_estimators_per_batch, total_estimators, n_estimators_per_batch):
            self.log(f"GBDT训练进度: {i}/{total_estimators} 棵树已完成 ({i/total_estimators*100:.1f}%)")
            init_model.n_estimators = min(i + n_estimators_per_batch, total_estimators)
            init_model.fit(X_train, y_train)
        
        self.log(f"GBDT训练进度: {total_estimators}/{total_estimators} 棵树已完成 (100%)")
        
        model = init_model
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
            if len(self.feature_cols) == X_train.shape[1]:
                feature_names = self.feature_cols
                
            importance_data = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
            
            self.log(f"GBDT特征重要性 (前10个):")
            for i, (feature, importance) in enumerate(importance_data[:10]):
                self.log(f"  {i+1}. {feature}: {importance:.4f}")
        
        self.raw_models[f'gbdt_{ball_type}'] = model
        
        wrapped_model = WrappedGBDTModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球GBDT模型训练完成")
        return wrapped_model
    
    def train_lightgbm(self, X_train, y_train, ball_type):
        """训练LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM未安装，跳过此模型训练")
            return None
            
        import lightgbm as lgb
            
        self.log(f"训练{ball_type}球LightGBM模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        params = {
            'objective': 'multiclass',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'metric': 'multi_logloss',
            'verbose': -1,  # 减少自带的输出，使用我们自己的回调
            'pred_early_stop': True,  # 早停预测
            'predict_disable_shape_check': True  # 禁用形状检查
        }
        
        self.log(f"LightGBM参数: {params}")
        
        if self.use_gpu and torch.cuda.is_available():
            params['device'] = 'gpu'  # 使用GPU加速
            self.log("LightGBM使用GPU加速训练")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        def progress_callback(env):
            if env.iteration % 10 == 0 or env.iteration == env.end_iteration - 1:
                try:
                    eval_result = env.evaluation_result_list
                    if eval_result and len(eval_result) > 0:
                        metric_name = eval_result[0][1]
                        metric_value = eval_result[0][2]
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}: {metric_name}={metric_value:.6f}")
                    else:
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}")
                except Exception as e:
                    self.log(f"记录LightGBM进度时出错: {str(e)}")
            return False
        
        callbacks = [progress_callback]
        
        self.log(f"开始训练LightGBM模型...")
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=100,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        if hasattr(model, 'feature_importance'):
            try:
                importances = model.feature_importance(importance_type='gain')
                feature_names = model.feature_name()
                feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"LightGBM特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(feature_importance[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                self.log(f"获取LightGBM特征重要性时出错: {str(e)}")
        
        self.raw_models[f'lightgbm_{ball_type}'] = model
        
        wrapped_model = WrappedLightGBMModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球LightGBM模型训练完成")
        return wrapped_model
    
    def train_catboost(self, X_train, y_train, ball_type):
        """训练CatBoost模型"""
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost未安装，跳过此模型训练")
            return None
            
        import catboost as cb
            
        self.log(f"训练{ball_type}球CatBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置输出目录
        train_dir = f"catboost_info_{ball_type}"
        os.makedirs(train_dir, exist_ok=True)
        
        # 确定分类数
        if ball_type == 'red':
            classes_count = self.red_range
        else:  # blue
            classes_count = self.blue_range
        
        # 设置参数
        params = {
            'loss_function': 'MultiClass',
            'classes_count': classes_count,
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False,
            'train_dir': train_dir
        }
        
        # 如果GPU可用，启用GPU加速
        if self.use_gpu and hasattr(cb, 'CatBoostClassifier') and hasattr(cb.CatBoostClassifier, 'is_cuda_supported') and cb.CatBoostClassifier.is_cuda_supported():
            params['task_type'] = 'GPU'
            self.log("CatBoost使用GPU加速训练")
        
        self.log(f"CatBoost参数: {params}")
        
        # 创建自定义回调，用于日志和暂停支持
        # 使用正确的回调方式
        class LoggerCallback(object):
            def __init__(self, logger, progress_interval=10):
                self.logger = logger
                self.progress_interval = progress_interval
                self.iteration = 0
                
            def after_iteration(self, info):
                self.iteration += 1
                if self.iteration % self.progress_interval == 0:
                    self.logger(f"CatBoost训练进度: {self.iteration}/{info.params.iterations} 迭代已完成 ({self.iteration/info.params.iterations*100:.1f}%)")
                return False  # 返回False表示继续训练
        
        # 创建模型
        model = cb.CatBoostClassifier(**params)
        
        # 训练模型
        self.log(f"开始训练CatBoost模型...")
        
        try:
            # 检查是否支持回调
            if hasattr(cb, 'CallbackCustom') or hasattr(model, 'set_user_callback'):
                # 新版CatBoost
                callbacks = [LoggerCallback(self.log)]
                model.fit(X_train, y_train, callbacks=callbacks)
            else:
                # 不使用回调，直接训练
                model.fit(X_train, y_train)
                self.log(f"CatBoost训练完成，共{params['iterations']}迭代")
            
            # 特征重要性
            if hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
                feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
                
                importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"CatBoost特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(importance_data[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            
            self.raw_models[f'catboost_{ball_type}'] = model
            
            wrapped_model = WrappedCatBoostModel(model, self.process_multidim_prediction)
            
            self.log(f"{ball_type}球CatBoost模型训练完成")
            return wrapped_model
            
        except Exception as e:
            self.log(f"训练CatBoost模型时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def train_ensemble(self, X_train, y_train, ball_type):
        """训练集成模型（包含多种基础模型）"""
        self.log(f"训练{ball_type}球集成模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        ensemble_models = {}
        total_models = 5 if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE else 3
        current_model = 0
        
        self.log(f"集成模型将训练以下子模型: 随机森林, XGBoost, GBDT{', LightGBM' if LIGHTGBM_AVAILABLE else ''}{', CatBoost' if CATBOOST_AVAILABLE else ''}")
        
    
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - 随机森林")
        rf_model = self.train_random_forest(X_train, y_train, ball_type)
        if rf_model:
            ensemble_models['random_forest'] = rf_model
            self.log(f"集成模型进度: 随机森林模型添加完成 ({current_model}/{total_models})")
            
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - XGBoost")
        xgb_model = self.train_xgboost(X_train, y_train, ball_type)
        if xgb_model:
            ensemble_models['xgboost'] = xgb_model
            self.log(f"集成模型进度: XGBoost模型添加完成 ({current_model}/{total_models})")
            
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - GBDT")
        gbdt_model = self.train_gbdt(X_train, y_train, ball_type)
        if gbdt_model:
            ensemble_models['gbdt'] = gbdt_model
            self.log(f"集成模型进度: GBDT模型添加完成 ({current_model}/{total_models})")
            
        if LIGHTGBM_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - LightGBM")
            lgb_model = self.train_lightgbm(X_train, y_train, ball_type)
            if lgb_model:
                ensemble_models['lightgbm'] = lgb_model
                self.log(f"集成模型进度: LightGBM模型添加完成 ({current_model}/{total_models})")
                
        if CATBOOST_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - CatBoost")
            cb_model = self.train_catboost(X_train, y_train, ball_type)
            if cb_model:
                ensemble_models['catboost'] = cb_model
                self.log(f"集成模型进度: CatBoost模型添加完成 ({current_model}/{total_models})")
        
        self.log(f"集成模型完成，包含 {len(ensemble_models)} 个子模型")
        
        model_info = []
        for model_name in ensemble_models.keys():
            model_info.append(f"- {model_name}")
        
        self.log(f"集成模型包含以下子模型:\n" + "\n".join(model_info))
        
        return ensemble_models
    
    def train(self, df):
        """训练模型"""
        self.log("============ 开始训练模型 ============")
        self.log(f"彩票类型: {self.lottery_type.upper()}")
        self.log(f"选择的模型类型: {self.model_type}")
        
        training_start_time = time.time()
        
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            self.log("\n====== 使用期望值模型 ======")
            self.train_expected_value_model(df)
            total_time = time.time() - training_start_time
            self.log(f"\n训练完成，总耗时: {total_time:.2f}秒")
            return self.models
        
        # 紫微斗数和梅花易数：无需训练，基于传统算法即时生成
        if self.model_type in ['ziwei', 'meihua']:
            self.log(f"\n====== {MODEL_TYPES[self.model_type]}：传统易学算法，无需训练 ======")
            self.log(f"{MODEL_TYPES[self.model_type]}基于时间起卦/排盘，即时可用")
            # 设置虚拟模型占位
            self.models['red'] = 'ziwei_meihua_placeholder'
            self.models['blue'] = 'ziwei_meihua_placeholder'
            self.log(f"\n{MODEL_TYPES[self.model_type]}准备完成")
            return self.models
        
        self.log("\n====== 第1阶段: 数据准备 ======")
        data_prep_start = time.time()
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_data(df)
        data_prep_time = time.time() - data_prep_start
        self.log(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
        
        self.models = {}
        
        self.log("\n====== 第2阶段: 模型训练 ======")
        model_train_start = time.time()
        
        if self.model_type == 'random_forest':
            self.log("\n----- 使用随机森林模型 -----")
            self.models['red'] = self.train_random_forest(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_random_forest(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'xgboost':
            self.log("\n----- 使用XGBoost模型 -----")
            self.models['red'] = self.train_xgboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_xgboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'gbdt':
            self.log("\n----- 使用梯度提升决策树 -----")
            self.models['red'] = self.train_gbdt(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_gbdt(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.log("\n----- 使用LightGBM模型 -----")
            self.models['red'] = self.train_lightgbm(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_lightgbm(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.log("\n----- 使用CatBoost模型 -----")
            self.models['red'] = self.train_catboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_catboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'ensemble':
            self.log("\n----- 使用集成模型 -----")
            self.models['red'] = self.train_ensemble(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_ensemble(X_train, y_blue_train, 'blue')
        
        model_train_time = time.time() - model_train_start
        self.log(f"\n模型训练完成，耗时: {model_train_time:.2f}秒")
        
        self.log("\n====== 第3阶段: 模型评估 ======")
        if X_test is not None and y_red_test is not None and y_blue_test is not None:
            eval_start = time.time()
            self.evaluate(X_test, y_red_test, y_blue_test)
            eval_time = time.time() - eval_start
            self.log(f"模型评估完成，耗时: {eval_time:.2f}秒")
        
        self.log("\n====== 第4阶段: 模型保存 ======")
        save_start = time.time()
        self.save_models()
        save_time = time.time() - save_start
        self.log(f"模型保存完成，耗时: {save_time:.2f}秒")
        
        total_time = time.time() - training_start_time
        self.log(f"\n训练完成，总耗时: {total_time:.2f}秒")
        
        return self.models
        
    def train_expected_value_model(self, df):
        """训练期望值模型"""
        self.log(f"开始训练期望值模型...")
        
        self.models = {}
        
        red_probs_file = os.path.join(self.models_dir, 'ev_red_probabilities.pkl')
        if os.path.exists(red_probs_file):
            self.log("期望值模型文件已存在，尝试加载...")
            
            ev_model = ExpectedValueLotteryModel(
                lottery_type=self.lottery_type,
                log_callback=self.log,
                use_gpu=self.use_gpu
            )
                
            load_success = ev_model.load()
            if load_success:
                self.log("期望值模型加载成功")
                self.models['red'] = ev_model
                self.models['blue'] = ev_model
                # 保存到原始模型中以便序列化
                self.raw_models['expected_value_model'] = ev_model
                return
            else:
                self.log("期望值模型加载失败，将重新训练...")
        
 
        ev_model = ExpectedValueLotteryModel(
            lottery_type=self.lottery_type,
            log_callback=self.log,
            use_gpu=self.use_gpu
        )
        
       
        ev_model.train(df)
        
        self.models['red'] = ev_model
        self.models['blue'] = ev_model
        self.raw_models['expected_value_model'] = ev_model
        
        self.log("期望值模型训练完成")
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """评估模型性能"""
        self.log("评估模型性能...")
        
        if len(y_red_test.shape) == 1:
            y_red_test = y_red_test.reshape(-1, 1)
        if len(y_blue_test.shape) == 1:
            y_blue_test = y_blue_test.reshape(-1, 1)
            
        red_accuracy = 0
        blue_accuracy = 0
        
        if 'red' in self.models:
            # 处理不同类型的模型
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理，对每个子模型进行预测，然后进行投票
                red_votes = {}
                for model_name, model in self.models['red'].items():
                    self.log(f"评估{model_name}模型...")
                    try:
                        y_pred = model.predict(X_test)
                        
                        # 预测结果处理，确保能够进行投票
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            y_pred = np.argmax(y_pred, axis=1)
                        
                        # 记录每个样本的投票
                        for i, pred in enumerate(y_pred):
                            # catboost 等可能返回 (n,1) 形状，pred 为单元素数组，需转标量才能作 dict key
                            if isinstance(pred, np.ndarray):
                                pred = pred.item() if pred.size == 1 else int(np.argmax(pred))
                            else:
                                pred = int(pred)
                            if i not in red_votes:
                                red_votes[i] = {}
                            if pred not in red_votes[i]:
                                red_votes[i][pred] = 0
                            red_votes[i][pred] += 1
                    except Exception as e:
                        self.log(f"评估{model_name}模型时出错: {e}")
                
                # 根据投票确定最终预测结果
                y_pred_red = []
                for i in range(len(X_test)):
                    if i in red_votes and red_votes[i]:
                        # 找出得票最多的类别
                        pred_class = max(red_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_red.append(pred_class)
                    else:
                        # 如果没有投票，默认预测0
                        y_pred_red.append(0)
                
                y_pred_red = np.array(y_pred_red)
            else:
                # 单一模型
                y_pred_red = self.models['red'].predict(X_test)
            
            if len(y_pred_red.shape) > 1 and y_pred_red.shape[1] > 1:
                self.log(f"处理多维预测结果，形状: {y_pred_red.shape}")
                y_pred_red = np.argmax(y_pred_red, axis=1)
            
            y_red_test_flat = y_red_test.flatten()
            y_pred_red_flat = y_pred_red.flatten()
            
            self.log(f"红球预测形状: {y_pred_red_flat.shape}, 真实值形状: {y_red_test_flat.shape}")
            
            red_accuracy = np.mean(y_pred_red_flat == y_red_test_flat)
            self.log(f"红球预测准确率: {red_accuracy:.4f}")
        
        if 'blue' in self.models:
            # 处理不同类型的模型
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理，对每个子模型进行预测，然后进行投票
                blue_votes = {}
                for model_name, model in self.models['blue'].items():
                    self.log(f"评估{model_name}模型...")
                    try:
                        y_pred = model.predict(X_test)
                        
                        # 预测结果处理，确保能够进行投票
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            y_pred = np.argmax(y_pred, axis=1)
                        
                        # 记录每个样本的投票
                        for i, pred in enumerate(y_pred):
                            # catboost 等可能返回 (n,1) 形状，pred 为单元素数组，需转标量才能作 dict key
                            if isinstance(pred, np.ndarray):
                                pred = pred.item() if pred.size == 1 else int(np.argmax(pred))
                            else:
                                pred = int(pred)
                            if i not in blue_votes:
                                blue_votes[i] = {}
                            if pred not in blue_votes[i]:
                                blue_votes[i][pred] = 0
                            blue_votes[i][pred] += 1
                    except Exception as e:
                        self.log(f"评估{model_name}模型时出错: {e}")
                
                # 根据投票确定最终预测结果
                y_pred_blue = []
                for i in range(len(X_test)):
                    if i in blue_votes and blue_votes[i]:
                        # 找出得票最多的类别
                        pred_class = max(blue_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_blue.append(pred_class)
                    else:
                        # 如果没有投票，默认预测0
                        y_pred_blue.append(0)
                
                y_pred_blue = np.array(y_pred_blue)
            else:
                # 单一模型
                y_pred_blue = self.models['blue'].predict(X_test)
            
            if len(y_pred_blue.shape) > 1 and y_pred_blue.shape[1] > 1:
                self.log(f"处理多维预测结果，形状: {y_pred_blue.shape}")
                y_pred_blue = np.argmax(y_pred_blue, axis=1)
            
            y_blue_test_flat = y_blue_test.flatten()
            y_pred_blue_flat = y_pred_blue.flatten()
            
            # 记录预测和实际值的形状以便调试
            self.log(f"蓝球预测形状: {y_pred_blue_flat.shape}, 真实值形状: {y_blue_test_flat.shape}")
            
            # 计算准确率
            blue_accuracy = np.mean(y_pred_blue_flat == y_blue_test_flat)
            self.log(f"蓝球预测准确率: {blue_accuracy:.4f}")
        
        # 计算整体准确率
        overall_accuracy = (red_accuracy + blue_accuracy) / 2
        self.log(f"整体预测准确率: {overall_accuracy:.4f}")
        
        return red_accuracy, blue_accuracy
    
    def save_models(self):
        """保存模型和缩放器"""
        self.log("\n----- 保存模型和缩放器 -----")
        
        # 对于期望值模型，创建信息文件以确保目录存在
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            # 期望值模型在train_expected_value_model中已经保存
            self.log("期望值模型已在训练过程中自动保存")
            
            # 创建模型信息文件，确保目录存在
            model_dir = os.path.join(self.models_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            model_info = {
                'model_type': self.model_type,
                'lottery_type': self.lottery_type,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            info_path = os.path.join(model_dir, 'model_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            return True
        
        # 为其他模型创建保存目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型和缩放器
        for ball_type in ['red', 'blue']:
            if ball_type not in self.models:
                continue
                
            model = self.models[ball_type]
            
            # 对于不同类型的模型使用不同的保存方式
            if self.model_type == 'ensemble':
                # 为集成模型保存每个子模型
                for model_name, sub_model in model.items():
                    sub_model_path = os.path.join(model_dir, f'{ball_type}_{model_name}_model.pkl')
                    with open(sub_model_path, 'wb') as f:
                        pickle.dump(sub_model, f)
                    self.log(f"保存{ball_type}球{model_name}模型: {sub_model_path}")
            elif hasattr(model, 'predict'):
                # 对于sklearn或类似模型使用pickle保存
                model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"保存{ball_type}球模型: {model_path}")
            else:
                self.log(f"警告: {ball_type}球模型不支持序列化")
        
        # 保存特征缩放器
        if 'X' in self.scalers:
            # 保存X缩放器，这是在训练过程中创建的主要缩放器
            x_scaler_path = os.path.join(model_dir, 'X_scaler.pkl')
            with open(x_scaler_path, 'wb') as f:
                pickle.dump(self.scalers['X'], f)
            self.log(f"保存特征缩放器: {x_scaler_path}")
            
            # 将X缩放器复制到red和blue球的缩放器中
            for ball_type in ['red', 'blue']:
                self.scalers[ball_type] = self.scalers['X']
                scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers['X'], f)
                self.log(f"保存{ball_type}球特征缩放器: {scaler_path}")
        else:
            # 保存单独的球缩放器（如果存在）
            for ball_type in ['red', 'blue']:
                if ball_type in self.scalers:
                    scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[ball_type], f)
                    self.log(f"保存{ball_type}球特征缩放器: {scaler_path}")
                else:
                    self.log(f"警告: 没有找到{ball_type}球的特征缩放器")
        
        # 保存模型信息
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加特征数量信息
        expected_feature_count = getattr(self, 'expected_feature_count', None)
        if expected_feature_count:
            model_info['expected_feature_count'] = expected_feature_count
            self.log(f"保存特征数量信息: {expected_feature_count}")
        
        # 尝试从模型中获取特征数量信息
        if 'red' in self.models:
            model_obj = self.models['red']
            # 处理不同类型的模型
            if self.model_type == 'ensemble' and 'random_forest' in model_obj:
                model_obj = model_obj['random_forest']
                
            if hasattr(model_obj, 'n_features_in_'):
                model_info['n_features_in'] = model_obj.n_features_in_
                self.log(f"从模型中获取的特征数量: {model_obj.n_features_in_}")
            # 尝试获取更多的模型属性
            elif hasattr(model_obj, 'estimators_') and model_obj.estimators_:
                first_estimator = model_obj.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    model_info['n_features_in'] = first_estimator.n_features_in_
                    self.log(f"从模型第一个估计器中获取的特征数量: {first_estimator.n_features_in_}")
            # 最后尝试使用特征重要性的维度
            elif hasattr(model_obj, 'feature_importances_') and hasattr(model_obj.feature_importances_, 'shape'):
                model_info['n_features_in'] = model_obj.feature_importances_.shape[0]
                self.log(f"从模型特征重要性中获取的特征数量: {model_obj.feature_importances_.shape[0]}")
        
        # 如果没有从任何源获取到特征数量，默认保存为70
        if 'expected_feature_count' not in model_info and 'n_features_in' not in model_info:
            model_info['expected_feature_count'] = 70
            self.log(f"使用默认特征数量: 70")
            
        # 保存预测使用的预处理方法
        model_info['feature_data'] = {
            'window_size': self.feature_window,
            'red_count': self.red_count,
            'blue_count': self.blue_count,
            'red_range': self.red_range,
            'blue_range': self.blue_range
        }
        
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        self.log(f"保存模型信息: {info_path}")
        return True
        
    def load_models(self):
        """
        加载保存的模型和缩放器
        
        Returns:
            bool: 是否成功加载模型
        """
        self.log(f"尝试加载{self.lottery_type}的{MODEL_TYPES[self.model_type]}模型...")
        
        # 对于期望值模型，使用特殊的加载方法
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            self.log("尝试加载期望值模型...")
            ev_model = ExpectedValueLotteryModel(
                lottery_type=self.lottery_type,
                log_callback=self.log,
                use_gpu=self.use_gpu
            )
            
            load_success = ev_model.load()
            if load_success:
                self.models['red'] = ev_model
                self.models['blue'] = ev_model
                self.raw_models['expected_value_model'] = ev_model
                self.log("期望值模型加载成功")
                return True
            else:
                self.log("期望值模型加载失败")
                return False
        
        # 紫微斗数和梅花易数：无需加载模型文件，直接可用
        if self.model_type in ['ziwei', 'meihua']:
            self.log(f"{MODEL_TYPES[self.model_type]}无需加载模型文件，即时可用")
            # 设置虚拟模型占位，防止后续 predict 判断为未加载
            self.models['red'] = 'ziwei_meihua_placeholder'
            self.models['blue'] = 'ziwei_meihua_placeholder'
            return True
        
        # 检查模型目录是否存在
        model_dir = os.path.join(self.models_dir, self.model_type)
        if not os.path.exists(model_dir):
            self.log(f"模型目录不存在: {model_dir}")
            return False
        
        # 检查模型信息文件是否存在
        info_path = os.path.join(model_dir, 'model_info.json')
        if not os.path.exists(info_path):
            self.log(f"模型信息文件不存在: {info_path}")
            return False
        
        # 加载模型信息
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                
            self.feature_window = model_info.get('feature_window', self.feature_window)
            self.log(f"模型信息: 类型={model_info.get('model_type')}, 创建时间={model_info.get('created_at')}")
            
            # 加载特征数量信息
            if 'expected_feature_count' in model_info:
                self.expected_feature_count = model_info['expected_feature_count']
                self.log(f"已加载特征数量信息: {self.expected_feature_count}")
            elif 'n_features_in' in model_info:
                self.expected_feature_count = model_info['n_features_in']
                self.log(f"已加载特征数量信息(从n_features_in): {self.expected_feature_count}")
            
        except Exception as e:
            self.log(f"加载模型信息失败: {e}")
            return False
        
        # 加载X缩放器（通用特征缩放器）
        x_scaler_path = os.path.join(model_dir, 'X_scaler.pkl')
        if os.path.exists(x_scaler_path):
            try:
                with open(x_scaler_path, 'rb') as f:
                    self.scalers['X'] = pickle.load(f)
                self.log(f"加载通用特征缩放器成功")
                # 同时设置红蓝球的缩放器
                self.scalers['red'] = self.scalers['X']
                self.scalers['blue'] = self.scalers['X']
            except Exception as e:
                self.log(f"加载通用特征缩放器失败: {e}")
                # 创建一个默认的缩放器
                self.scalers['X'] = StandardScaler()
                self.scalers['red'] = StandardScaler()
                self.scalers['blue'] = StandardScaler()
                self.log("创建了默认特征缩放器作为替代")
        
        # 加载红球和蓝球模型
        models_loaded = True
        balls_loaded = 0  # 记录加载成功的球数量
        
        for ball_type in ['red', 'blue']:
            try:
                if self.model_type == 'ensemble':
                    # 对于集成模型，加载每个子模型
                    self.models[ball_type] = {}
                    ensemble_loaded = False
                    for model_name in ['random_forest', 'gbdt', 'xgboost', 'lightgbm']:
                        model_path = os.path.join(model_dir, f'{ball_type}_{model_name}_model.pkl')
                        if os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                self.models[ball_type][model_name] = pickle.load(f)
                            self.log(f"加载{ball_type}球{model_name}模型成功")
                            
                            # 尝试从模型中获取特征数量
                            if not hasattr(self, 'expected_feature_count'):
                                model_obj = self.models[ball_type][model_name]
                                if hasattr(model_obj, 'n_features_in_'):
                                    self.expected_feature_count = model_obj.n_features_in_
                                    self.log(f"从{model_name}模型中获取特征数量: {self.expected_feature_count}")
                            
                            ensemble_loaded = True
                        else:
                            self.log(f"警告: {ball_type}球{model_name}模型文件不存在")
                    if ensemble_loaded:
                        balls_loaded += 1
                    else:
                        models_loaded = False
                else:
                    # 对于其他模型，直接加载
                    model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                    if os.path.exists(model_path):
                        try:
                            with open(model_path, 'rb') as f:
                                model_obj = pickle.load(f)
                                
                            # 特别处理 LightGBM 和其他可能返回原始对象而不是包装后对象的模型
                            if self.model_type == 'lightgbm' and hasattr(model_obj, 'predict'):
                                # 对于 LightGBM 和其他带有 predict 方法的库，创建一个包装器
                                if 'lightgbm' in str(type(model_obj)).lower():
                                    self.log(f"检测到原始 LightGBM 模型，创建包装器")
                                    self.models[ball_type] = WrappedLightGBMModel(model_obj, self.process_multidim_prediction)
                                else:
                                    self.models[ball_type] = model_obj
                            else:
                                self.models[ball_type] = model_obj
                                
                            self.log(f"加载{ball_type}球模型成功")
                            balls_loaded += 1
                            
                            # 尝试从模型中获取特征数量
                            if not hasattr(self, 'expected_feature_count'):
                                model_obj_inner = self.models[ball_type]
                                # 如果是包装类，获取内部模型
                                if hasattr(model_obj_inner, 'model'):
                                    model_obj_inner = model_obj_inner.model
                                
                                if hasattr(model_obj_inner, 'n_features_in_'):
                                    self.expected_feature_count = model_obj_inner.n_features_in_
                                    self.log(f"从模型中获取特征数量: {self.expected_feature_count}")
                        except Exception as e:
                            self.log(f"加载{ball_type}球模型失败: {e}, 尝试创建包装器")
                            # 尝试创建适当的包装器
                            try:
                                with open(model_path, 'rb') as f:
                                    raw_model = pickle.load(f)
                                
                                if self.model_type == 'lightgbm':
                                    self.models[ball_type] = WrappedLightGBMModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'catboost':
                                    self.models[ball_type] = WrappedCatBoostModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'xgboost':
                                    self.models[ball_type] = WrappedXGBoostModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'gbdt':
                                    self.models[ball_type] = WrappedGBDTModel(raw_model, self.process_multidim_prediction)
                                else:
                                    # 创建一个通用包装器
                                    class GenericWrapper:
                                        def __init__(self, model, processor):
                                            self.model = model
                                            self.process_prediction = processor
                                        
                                        def predict(self, data):
                                            if hasattr(self.model, 'predict_proba'):
                                                raw_preds = self.model.predict_proba(data)
                                            else:
                                                raw_preds = self.model.predict(data)
                                            return self.process_prediction(raw_preds)
                                    
                                    self.models[ball_type] = GenericWrapper(raw_model, self.process_multidim_prediction)
                                
                                self.log(f"成功为{ball_type}球创建了{self.model_type}包装器")
                                balls_loaded += 1
                            except Exception as e2:
                                self.log(f"创建包装器也失败: {e2}")
                                models_loaded = False
                    else:
                        self.log(f"警告: {ball_type}球模型文件不存在: {model_path}")
                        models_loaded = False
                
                # 尝试加载特定的球特征缩放器(如果还没有通用缩放器的话)
                if ball_type not in self.scalers:
                    scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                    if os.path.exists(scaler_path):
                        try:
                            with open(scaler_path, 'rb') as f:
                                self.scalers[ball_type] = pickle.load(f)
                            self.log(f"加载{ball_type}球特征缩放器成功")
                        except Exception as e:
                            self.log(f"加载{ball_type}球特征缩放器失败: {e}")
                            self.scalers[ball_type] = StandardScaler()
                            self.log(f"创建了{ball_type}球默认特征缩放器作为替代")
                    else:
                        self.log(f"警告: {ball_type}球特征缩放器文件不存在")
                        # 如果没有缩放器，创建一个默认的缩放器
                        self.scalers[ball_type] = StandardScaler()
                        self.log(f"创建了{ball_type}球默认特征缩放器作为替代")
                        
            except Exception as e:
                self.log(f"加载{ball_type}球模型失败: {e}")
                models_loaded = False
        
        # 如果所有模型都成功加载，返回True
        if models_loaded:
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功")
            return True
        elif balls_loaded >= 2:  # 至少加载了红球和蓝球
            # 即使有警告，只要基础模型存在，我们也认为模型可用
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但有一些警告")
            return True
        elif 'red' in self.models and 'blue' in self.models:
            # 模型存在但可能有其他问题
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但可能存在兼容性问题")
            return True
        else:
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载失败，未找到必要的红球和蓝球模型")
            return False
        
    def predict(self, recent_data, variation=0):
        """
        生成预测结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            variation: 第几组预测（0-based），用于紫微/梅花等传统算法生成不同结果
            
        Returns:
            预测的红球和蓝球号码
        """
        # 每次预测前清空置信度和概率，仅在主预测路径末尾填充
        self.last_confidence = None
        self.last_red_conf = None
        self.last_blue_conf = None
        self.last_red_proba = None
        self.last_blue_proba = None
        # 检查是否使用期望值模型
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            # 检查模型是否已经加载
            if ('red' not in self.models or not isinstance(self.models['red'], ExpectedValueLotteryModel) or
                'blue' not in self.models or not isinstance(self.models['blue'], ExpectedValueLotteryModel)):
                # 尝试重新加载期望值模型
                self.log("期望值模型未加载或类型不正确，正在尝试重新加载...")
                ev_model = ExpectedValueLotteryModel(
                    lottery_type=self.lottery_type,
                    log_callback=self.log,
                    use_gpu=self.use_gpu
                )
                
                load_success = ev_model.load()
                if load_success:
                    self.models['red'] = ev_model
                    self.models['blue'] = ev_model
                    self.raw_models['expected_value_model'] = ev_model
                    self.log("期望值模型重新加载成功")
                else:
                    self.log("错误：期望值模型加载失败，请先训练模型")
                    return None, None
            
            # 使用期望值模型进行预测
            self.log("使用期望值模型进行预测...")
            red_preds, blue_preds = self.models['red'].predict(recent_data, num_predictions=1)
            # 期望值模型返回的是索引列表的列表，需要处理成号码
            if red_preds and blue_preds:
                red_numbers = [idx + 1 for idx in red_preds[0]]  # 索引转换为号码
                blue_numbers = [idx + 1 for idx in blue_preds[0]]
                
                # 确保红球和蓝球号码数量符合要求
                red_numbers = sorted(list(set(red_numbers)))[:self.red_count]
                blue_numbers = sorted(list(set(blue_numbers)))[:self.blue_count]
                
                # 如果数量不足，补充随机号码
                while len(red_numbers) < self.red_count:
                    new_num = np.random.randint(1, self.red_range + 1)
                    if new_num not in red_numbers:
                        red_numbers.append(new_num)
                red_numbers.sort()
                        
                while len(blue_numbers) < self.blue_count:
                    new_num = np.random.randint(1, self.blue_range + 1)
                    if new_num not in blue_numbers:
                        blue_numbers.append(new_num)
                blue_numbers.sort()
                
                # 提取概率信息用于后续校准
                ev_model = self.models['red']  # 红球和蓝球使用同一个模型实例
                if hasattr(ev_model, 'red_probabilities') and ev_model.red_probabilities is not None:
                    # 创建与预测号码对应的概率列表
                    self.last_red_proba = [ev_model.red_probabilities.get(num, 0.0) for num in red_numbers]
                    self.last_blue_proba = [ev_model.blue_probabilities.get(num, 0.0) for num in blue_numbers]
                else:
                    self.last_red_proba = None
                    self.last_blue_proba = None
                
                return red_numbers, blue_numbers
            self.log("期望值模型预测失败")
            return None, None
        
        # 紫微斗数预测 - 基于时间起卦的传统排盘算法
        if self.model_type == 'ziwei':
            return self._predict_ziwei(recent_data, variation)
        
        # 梅花易数预测 - 基于时间、数字起卦的传统算法
        if self.model_type == 'meihua':
            return self._predict_meihua(recent_data, variation)
        
        # 确保模型已加载
        if 'red' not in self.models or 'blue' not in self.models:
            self.log("模型未加载，无法预测")
            return None, None
        
        # 对于其他模型的处理保持不变
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            # 注意：SSQ的CSV蓝球列名为"蓝球"(无下划线)，必须与训练时prepare_data的
            # 兼容逻辑保持一致，否则blue_cols为空导致特征数与模型不匹配(70 vs 60)
            blue_cols = []
            for col in recent_data.columns:
                if col.startswith('蓝球_') or col == '蓝球':
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                        break
            
        # 确保数据按期数降序排列
        recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
        
        # 确保有足够的历史数据
        if len(recent_data) < self.feature_window:
            self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
            return None, None
        
        # 创建特征序列
        X_data = []
        
        # 使用滑动窗口创建序列数据
        features = []
        for j in range(self.feature_window):
            row_features = []
            for col in red_cols + blue_cols:
                row_features.append(recent_data.iloc[j][col])
            features.append(row_features)
            
        X_data.append(features)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        
        # 重塑特征以适合模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 获取模型详细信息以进行调试
        red_model_info = ""
        if 'red' in self.models:
            if self.model_type == 'ensemble':
                if 'random_forest' in self.models['red']:
                    red_model_info = f"特征数量: {getattr(self.models['red']['random_forest'], 'n_features_in_', '未知')}, 类型: {type(self.models['red']['random_forest']).__name__}"
            else:
                # 特征数量显示：优先模型自身 n_features_in_，无效(缺失/0/None)时回退到
                # model_info.json 中记录的 expected_feature_count，避免显示"未知"或0
                try:
                    n_feat = self.models['red'].n_features_in_
                    if n_feat is None or n_feat == 0:
                        raise AttributeError  # 无效值，走回退
                except AttributeError:
                    n_feat = getattr(self, 'expected_feature_count', '未知')
                red_model_info = f"特征数量: {n_feat}, 类型: {type(self.models['red']).__name__}"
        self.log(f"红球模型信息: {red_model_info}")
        
        # 检查特征数量是否与训练时的特征数量匹配
        # 从模型信息或模型本身获取预期的特征数量
        expected_features = getattr(self, 'expected_feature_count', 70)  # 默认70，与训练时保持一致
        
        # 也可以尝试从模型中获取
        if 'red' in self.models:
            model_obj = self.models['red']
            # 处理不同类型的模型
            if self.model_type == 'ensemble' and 'random_forest' in model_obj:
                model_obj = model_obj['random_forest']
                
            # catboost 等模型反序列化后 n_features_in_ 可能是 0，视为无效回退到 expected_feature_count
            if hasattr(model_obj, 'n_features_in_') and model_obj.n_features_in_:
                expected_features = model_obj.n_features_in_
                self.log(f"从模型中获取的特征数量: {expected_features}")
            elif hasattr(model_obj, 'feature_importances_') and hasattr(model_obj.feature_importances_, 'shape'):
                expected_features = model_obj.feature_importances_.shape[0]
                self.log(f"从模型特征重要性中获取的特征数量: {expected_features}")
            # 尝试获取更多的模型属性
            elif hasattr(model_obj, 'estimators_') and model_obj.estimators_:
                first_estimator = model_obj.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    expected_features = first_estimator.n_features_in_
                    self.log(f"从模型第一个估计器中获取的特征数量: {expected_features}")
        
        actual_features = X_reshaped.shape[1]
        
        if actual_features != expected_features:
            self.log(f"警告: 特征数量不匹配，预期{expected_features}个，实际{actual_features}个")
            # 根据情况补充或截断特征
            if actual_features < expected_features:
                # 如果特征不足，填充零
                padding = np.zeros((X_reshaped.shape[0], expected_features - actual_features))
                X_reshaped = np.concatenate([X_reshaped, padding], axis=1)
                self.log(f"已将特征填充至{X_reshaped.shape[1]}个")
            else:
                # 如果特征过多，截断
                X_reshaped = X_reshaped[:, :expected_features]
                self.log(f"已将特征截断至{X_reshaped.shape[1]}个")
        
        # 应用特征缩放
        try:
            if 'X' in self.scalers:
                X_scaled = self.scalers['X'].transform(X_reshaped)
                self.log("使用通用特征缩放器进行预测")
            else:
                # 回退到红球缩放器
                X_scaled = self.scalers['red'].transform(X_reshaped)
                self.log("使用红球特征缩放器进行预测")
        except Exception as e:
            self.log(f"应用特征缩放时出错: {e}")
            self.log("使用未缩放的特征进行预测")
            X_scaled = X_reshaped
        
        try:
            if self.model_type == 'ensemble':
                # 集成模型：self.models['red'] 是各子模型的 dict（没有predict方法），
                # 必须逐模型预测后投票，不能直接调用 self.models['red'].predict()
                red_votes = {}
                for name, model in self.models['red'].items():
                    preds = model.predict(X_scaled)[0]
                    if hasattr(preds, "__iter__"):
                        for pred in preds:
                            # 统一转标量：catboost 等可能输出单元素数组
                            if isinstance(pred, np.ndarray):
                                pred = pred.item() if pred.size == 1 else int(np.argmax(pred))
                            else:
                                pred = int(pred)
                            if pred not in red_votes:
                                red_votes[pred] = 0
                            red_votes[pred] += 1
                    else:
                        # 单一预测值
                        if preds not in red_votes:
                            red_votes[preds] = 0
                        red_votes[preds] += 1
                red_predictions = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)[:self.red_count]
                red_predictions = [int(p[0]) + 1 for p in red_predictions]  # +1 转回原始号码范围
            else:
                # 单一模型预测
                red_pred = self.models['red'].predict(X_scaled)[0]
                if hasattr(red_pred, "__iter__"):
                    red_predictions = [int(p) + 1 for p in red_pred]  # +1 转回原始号码范围
                else:
                    red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
            
            # 计算红球置信度（所选红球号码的模型概率均值）
            try:
                if self.model_type == 'ensemble':
                    proba_map = _ensemble_proba_dict(self.models['red'], X_scaled)
                else:
                    proba_map = _class_proba_dict(self.models['red'], X_scaled)
                self.last_red_conf = _mean_confidence(proba_map, red_predictions)
                self.last_red_proba = proba_map  # 存储完整概率映射用于校准
            except Exception:
                self.last_red_conf = None
                self.last_red_proba = None
            
            # 预测蓝球
            if self.model_type == 'ensemble':
                # 集成模型：self.models['blue'] 是各子模型的 dict，逐模型预测后投票
                blue_votes = {}
                for name, model in self.models['blue'].items():
                    preds = model.predict(X_scaled)[0]
                    if hasattr(preds, "__iter__"):
                        for pred in preds:
                            # 统一转标量：catboost 等可能输出单元素数组
                            if isinstance(pred, np.ndarray):
                                pred = pred.item() if pred.size == 1 else int(np.argmax(pred))
                            else:
                                pred = int(pred)
                            if pred not in blue_votes:
                                blue_votes[pred] = 0
                            blue_votes[pred] += 1
                    else:
                        # 单一预测值
                        if preds not in blue_votes:
                            blue_votes[preds] = 0
                        blue_votes[preds] += 1
                
                # 从得票前3的蓝球中随机选择，而不是总是选择得票最高的
                top_blue_votes = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)
                top_count = min(3, len(top_blue_votes))
                
                # 有60%概率使用票数最高的蓝球，40%概率从票数前3的蓝球中随机选择
                if np.random.random() < 0.6 or top_count == 1:
                    blue_predictions = [int(p[0]) + 1 for p in top_blue_votes[:self.blue_count]]  # +1 转回原始号码范围
                else:
                    # 随机选择前top_count个中的blue_count个
                    selected_indices = np.random.choice(top_count, size=min(self.blue_count, top_count), replace=False)
                    blue_predictions = [int(top_blue_votes[i][0]) + 1 for i in selected_indices]  # +1 转回原始号码范围
            else:
                # 单一模型预测
                blue_pred = self.models['blue'].predict(X_scaled)[0]
                if hasattr(blue_pred, "__iter__"):
                    # 随机设定阈值，增加随机性
                    if np.random.random() < 0.3 and len(blue_pred) > 1:
                        # 30%的概率，从前3个最高概率蓝球中随机选择
                        top_indices = np.argsort(blue_pred)[-3:] if len(blue_pred) >= 3 else np.argsort(blue_pred)
                        selected_idx = np.random.choice(top_indices)
                        blue_predictions = [int(selected_idx) + 1]  # +1 转回原始号码范围
                    else:
                        # 70%的概率，使用原始预测
                        blue_predictions = [int(p) + 1 for p in blue_pred]  # +1 转回原始号码范围
                else:
                    # 直接使用并添加随机性
                    if np.random.random() < 0.25:  # 25%概率使用随机蓝球而不是模型预测
                        # 根据彩票类型确定蓝球范围
                        blue_range = self.blue_range
                        blue_predictions = [np.random.randint(1, blue_range + 1)]
                    else:
                        blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
            
            # 计算蓝球置信度（所选蓝球号码的模型概率均值）
            try:
                if self.model_type == 'ensemble':
                    proba_map = _ensemble_proba_dict(self.models['blue'], X_scaled)
                else:
                    proba_map = _class_proba_dict(self.models['blue'], X_scaled)
                self.last_blue_conf = _mean_confidence(proba_map, blue_predictions)
                self.last_blue_proba = proba_map  # 存储完整概率映射用于校准
            except Exception:
                self.last_blue_conf = None
                self.last_blue_proba = None
        except Exception as e:
            self.log(f"预测过程中出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            # 出错时返回随机号码
            red_predictions = []
            blue_predictions = []
            
        # 生成完整号码集
        while len(red_predictions) < self.red_count:
            # 如果预测的红球数量不足，随机补充
            new_num = np.random.randint(1, self.red_range + 1)
            if new_num not in red_predictions:
                red_predictions.append(new_num)
                
        while len(blue_predictions) < self.blue_count:
            # 如果预测的蓝球数量不足，随机补充
            new_num = np.random.randint(1, self.blue_range + 1)
            if new_num not in blue_predictions:
                blue_predictions.append(new_num)
                
        # 确保号码不重复且按升序排列
        red_predictions = sorted(list(set(red_predictions)))[:self.red_count]
        blue_predictions = sorted(list(set(blue_predictions)))[:self.blue_count]
        
        # 增加随机性：有5%的概率完全随机生成一个蓝球号码
        if np.random.random() < 0.05:
            self.log("随机生成蓝球号码以增加多样性")
            blue_predictions = []
            for _ in range(self.blue_count):
                blue_predictions.append(np.random.randint(1, self.blue_range + 1))
            blue_predictions = sorted(list(set(blue_predictions)))
            
            # 如果随机生成后数量不足，继续随机补充
            while len(blue_predictions) < self.blue_count:
                new_num = np.random.randint(1, self.blue_range + 1)
                if new_num not in blue_predictions:
                    blue_predictions.append(new_num)
            blue_predictions = sorted(blue_predictions)[:self.blue_count]
            # 随机号码没有模型依据，置信度记为 0
            self.last_blue_conf = 0.0
        
        # 组装整体置信度：按号码个数加权红/蓝
        if self.last_red_conf is not None or self.last_blue_conf is not None:
            red_c = self.last_red_conf if self.last_red_conf is not None else 0.0
            blue_c = self.last_blue_conf if self.last_blue_conf is not None else 0.0
            total = self.red_count + self.blue_count
            self.last_confidence = {
                'red': red_c,
                'blue': blue_c,
                'overall': (red_c * self.red_count + blue_c * self.blue_count) / total,
            }
        
        return red_predictions, blue_predictions

    def predict_with_confidence(self, recent_data):
        """
        生成预测并附带置信度（所选号码的模型概率均值）。
        
        Returns:
            (red_numbers, blue_numbers, confidence) 或 (None, None, None)
            confidence: {'red': float, 'blue': float, 'overall': float} 或 None
        """
        result = self.predict(recent_data)
        if result is None or result[0] is None:
            return None, None, None
        return result[0], result[1], getattr(self, 'last_confidence', None)

    def _rank_balls(self, recent_data):
        """
        构建特征并按模型概率对全部号码排序（ensemble 取各子模型平均概率）。
        复式/胆拖共用的底层逻辑。

        Returns:
            (red_ranked, blue_ranked): 按概率从高到低排列的号码列表(1起) 或 (None, None)
        """
        if self.model_type == 'expected_value':
            self.log("期望值模型不支持该预测方式，请使用 ensemble/gbdt/lightgbm/catboost")
            return None, None

        try:
            # ---- 特征构建（与 predict 保持一致）----
            if self.lottery_type == 'dlt':
                red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
                blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
            else:  # ssq
                red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
                # SSQ 蓝球列名为"蓝球"(无下划线)，必须与 predict/训练兼容
                blue_cols = []
                for col in recent_data.columns:
                    if col.startswith('蓝球_') or col == '蓝球':
                        blue_cols.append(col)
                        if len(blue_cols) >= 1:
                            break

            recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
            if len(recent_data) < self.feature_window:
                self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
                return None, None

            # 滑动窗口创建序列特征
            features = []
            for j in range(self.feature_window):
                row_features = []
                for col in red_cols + blue_cols:
                    row_features.append(recent_data.iloc[j][col])
                features.append(row_features)
            X = np.array([features])
            X_reshaped = X.reshape(X.shape[0], -1)

            # 特征缩放（与 predict 相同的回退逻辑）
            try:
                if 'X' in self.scalers:
                    X_scaled = self.scalers['X'].transform(X_reshaped)
                else:
                    X_scaled = self.scalers['red'].transform(X_reshaped)
            except Exception:
                X_scaled = X_reshaped

            # ---- 获取全号码概率排序 ----
            if self.model_type == 'ensemble':
                red_proba = _ensemble_proba_dict(self.models['red'], X_scaled)
                blue_proba = _ensemble_proba_dict(self.models['blue'], X_scaled)
            else:
                red_proba = _class_proba_dict(self.models['red'], X_scaled)
                blue_proba = _class_proba_dict(self.models['blue'], X_scaled)

            if not red_proba or not blue_proba:
                self.log("模型无概率输出。xgboost(softmax)等模型不支持，请改用 ensemble/gbdt/lightgbm/catboost")
                return None, None

            # 存储概率映射用于后续校准
            self.last_red_proba = red_proba
            self.last_blue_proba = blue_proba

            red_ranked = sorted(red_proba.items(), key=lambda x: x[1], reverse=True)
            blue_ranked = sorted(blue_proba.items(), key=lambda x: x[1], reverse=True)
            red_ranked = [int(p) + 1 for p, _ in red_ranked]
            blue_ranked = [int(p) + 1 for p, _ in blue_ranked]
            return red_ranked, blue_ranked
        except Exception as e:
            self.log(f"号码概率排序出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None, None

    @staticmethod
    def _pad_ranked(ranked, need, ball_range):
        """概率排序号码不足 need 个时随机补充，保证不重复。"""
        ranked = list(ranked)
        while len(ranked) < need:
            new_num = np.random.randint(1, ball_range + 1)
            if new_num not in ranked:
                ranked.append(new_num)
        return ranked

    def predict_compound(self, recent_data, extra_red=0, extra_blue=0):
        """
        预测复式号码：按模型概率排序（ensemble 为各子模型平均概率），
        返回 top (red_count+extra_red) 红球 + top (blue_count+extra_blue) 蓝球，
        而非单式的固定数量号码。确定性输出，不做随机扰动。

        Args:
            recent_data: 历史数据 DataFrame（与 predict 相同）
            extra_red: 红球额外数量，复式红球数 = red_count + extra_red（上限为号码范围）
            extra_blue: 蓝球额外数量，复式蓝球数 = blue_count + extra_blue

        Returns:
            (red_numbers, blue_numbers) 或 (None, None)
            复式注数 = C(红球数, red_count) × C(蓝球数, blue_count)，金额 = 注数 × 2 元
        """
        # 复式数量不能超过号码范围
        red_need = min(self.red_count + extra_red, self.red_range)
        blue_need = min(self.blue_count + extra_blue, self.blue_range)
        if red_need <= self.red_count and blue_need <= self.blue_count:
            self.log("复式数量未超出单式，退化为单式预测")
            return self.predict(recent_data)

        red_ranked, blue_ranked = self._rank_balls(recent_data)
        if red_ranked is None or blue_ranked is None:
            return None, None

        red_ranked = self._pad_ranked(red_ranked, red_need, self.red_range)
        blue_ranked = self._pad_ranked(blue_ranked, blue_need, self.blue_range)
        red_numbers = sorted(red_ranked[:red_need])
        blue_numbers = sorted(blue_ranked[:blue_need])
        n_notes = comb(red_need, self.red_count) * comb(blue_need, self.blue_count)
        self.log(f"复式预测: 红球{red_need}个 + 蓝球{blue_need}个，共 {n_notes} 注（{n_notes * 2} 元）")
        return red_numbers, blue_numbers

    def predict_dantuo(self, recent_data, red_dan_count=2, red_tuo_count=6,
                       blue_dan_count=1, blue_tuo_count=2):
        """
        预测胆拖号码：按模型概率排序，前 K 个作为胆码，接下来 N 个作为拖码。
        注数 = C(红拖个数, 红球单式数-红胆个数) × C(蓝拖个数, 蓝球单式数-蓝胆个数)。

        规则约束（自动截断）：
        - 红胆 1 ~ red_count-1（胆码必须少于单式个数）
        - 蓝胆 1 ~ blue_count-1（SSQ 蓝球单式只选 1 个，故 SSQ 不支持蓝胆）
        - 拖码个数 >= 单式数-胆数，且 胆+拖 <= 号码范围

        Args:
            recent_data: 历史数据 DataFrame
            red_dan_count: 红胆个数
            red_tuo_count: 红拖个数
            blue_dan_count: 蓝胆个数
            blue_tuo_count: 蓝拖个数

        Returns:
            (red_dan, red_tuo, blue_dan, blue_tuo) 四元组，元素均为号码列表 或 (None, None, None, None)
        """
        red_dan_count = max(1, min(red_dan_count, self.red_count - 1))
        blue_dan_count = max(1, min(blue_dan_count, self.blue_count - 1)) \
            if self.blue_count >= 2 else 0
        red_tuo_min = self.red_count - red_dan_count
        blue_tuo_min = self.blue_count - blue_dan_count if blue_dan_count >= 1 else self.blue_count
        red_tuo_count = max(red_tuo_min, min(red_tuo_count, self.red_range - red_dan_count))
        blue_tuo_count = max(blue_tuo_min, min(blue_tuo_count, self.blue_range - blue_dan_count))

        if blue_dan_count == 0 and blue_tuo_count > 0:
            # SSQ 蓝球无胆拖空间（单式只选1个蓝球），退化为复式蓝球
            self.log(f"{self.lottery_type} 蓝球单式只选1个，不支持蓝胆，蓝球按复式出号")
            red_ranked, blue_ranked = self._rank_balls(recent_data)
            if red_ranked is None or blue_ranked is None:
                return None, None, None, None
            red_total = red_dan_count + red_tuo_count
            blue_need = min(blue_tuo_count, self.blue_range)
            red_ranked = self._pad_ranked(red_ranked, red_total, self.red_range)
            blue_ranked = self._pad_ranked(blue_ranked, blue_need, self.blue_range)
            red_dan = sorted(red_ranked[:red_dan_count])
            red_tuo = sorted(red_ranked[red_dan_count:red_total])
            blue_tuo = sorted(blue_ranked[:blue_need])
            n_notes = comb(red_tuo_count, self.red_count - red_dan_count) * comb(blue_need, 1)
            self.log(
                f"胆拖预测: 红胆{red_dan_count}个+红拖{red_tuo_count}个 / "
                f"蓝拖{blue_need}个，共 {n_notes} 注（{n_notes * 2} 元）")
            return red_dan, red_tuo, [], blue_tuo

        red_ranked, blue_ranked = self._rank_balls(recent_data)
        if red_ranked is None or blue_ranked is None:
            return None, None, None, None

        red_total = red_dan_count + red_tuo_count
        blue_total = blue_dan_count + blue_tuo_count
        red_ranked = self._pad_ranked(red_ranked, red_total, self.red_range)
        blue_ranked = self._pad_ranked(blue_ranked, blue_total, self.blue_range)

        red_dan = sorted(red_ranked[:red_dan_count])
        red_tuo = sorted(red_ranked[red_dan_count:red_total])
        blue_dan = sorted(blue_ranked[:blue_dan_count]) if blue_dan_count >= 1 else []
        blue_tuo = sorted(blue_ranked[blue_dan_count:blue_total])

        n_notes = comb(red_tuo_count, self.red_count - red_dan_count) * \
            (comb(blue_tuo_count, self.blue_count - blue_dan_count) if blue_dan_count >= 1
             else comb(blue_tuo_count, self.blue_count))
        self.log(
            f"胆拖预测: 红胆{red_dan_count}个+红拖{red_tuo_count}个 / "
            f"蓝胆{blue_dan_count}个+蓝拖{blue_tuo_count}个，共 {n_notes} 注（{n_notes * 2} 元）")
        return red_dan, red_tuo, blue_dan, blue_tuo

    def _predict_ziwei(self, recent_data, variation=0):
        """
        紫微斗数预测 - 调用 astrology.ziwei 引擎
        三组取数视角：0=正财(财帛宫) 1=偏财(官禄+迁移+福德) 2=本命(命宫+身宫+大限)
        """
        import datetime
        from astrology.ziwei.engine import predict as ziwei_predict

        self.log(f"使用紫微斗数排盘预测... (第{variation+1}组)")

        now = datetime.datetime.now()
        latest_period = int(recent_data.iloc[0]['期数']) if len(recent_data) > 0 else 0

        birth_info = getattr(self, 'birth_info', None) or {
            'year': now.year, 'month': now.month, 'day': now.day,
            'hour': now.hour, 'minute': now.minute,
            'city': '北京', 'gender': '男'
        }

        red, blue, interp = ziwei_predict(
            lottery_type=self.lottery_type,
            red_range=self.red_range, blue_range=self.blue_range,
            red_count=self.red_count, blue_count=self.blue_count,
            birth_info=birth_info,
            target_period=latest_period,
            variation=variation,
        )

        self.log(f"紫微斗数预测: 红球 {red} 蓝球 {blue}")
        if not hasattr(self, '_astrology_interp'):
            self._astrology_interp = []
        if variation < 3:
            while len(self._astrology_interp) <= variation:
                self._astrology_interp.append('')
            self._astrology_interp[variation] = interp

        return red, blue

    def _predict_meihua(self, recent_data, variation=0):
        """
        梅花易数预测 - 调用 astrology.meihua 引擎
        三组取数路径：0=本卦主象 1=互卦过程 2=外应灵数
        """
        import datetime
        from astrology.meihua.engine import predict as meihua_predict

        self.log(f"使用梅花易数起卦预测... (第{variation+1}组)")

        now = datetime.datetime.now()
        latest_period = int(recent_data.iloc[0]['期数']) if len(recent_data) > 0 else 0

        input_info = getattr(self, 'meihua_info', None) or {
            'mode': 'time',
            'year': now.year, 'month': now.month, 'day': now.day,
            'hour': now.hour, 'minute': now.minute,
        }

        red, blue, interp = meihua_predict(
            lottery_type=self.lottery_type,
            red_range=self.red_range, blue_range=self.blue_range,
            red_count=self.red_count, blue_count=self.blue_count,
            input_info=input_info,
            target_period=latest_period,
            variation=variation,
        )

        self.log(f"梅花易数预测: 红球 {red} 蓝球 {blue}")
        if not hasattr(self, '_astrology_interp'):
            self._astrology_interp = []
        if variation < 3:
            while len(self._astrology_interp) <= variation:
                self._astrology_interp.append('')
            self._astrology_interp[variation] = interp

        return red, blue
# 使用示例
def demo():
    # 加载数据
    from scripts.data_analysis import load_lottery_data
    lottery_type = 'dlt'  # 或 'ssq'
    df = load_lottery_data(lottery_type)
    
    # 初始化模型
    model = LotteryMLModels(lottery_type=lottery_type, model_type='ensemble')
    
    # 训练模型
    model.train(df)
    
    # 预测下一期号码
    recent_data = df.sort_values('期数', ascending=False).head(10)
    red_predictions, blue_predictions = model.predict(recent_data)
    
    print(f"预测红球号码: {red_predictions}")
    print(f"预测蓝球号码: {blue_predictions}")

if __name__ == "__main__":
    demo() 