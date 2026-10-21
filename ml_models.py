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

# 复式/胆拖选号池上限：红球最多 16 个，蓝球最多 6 个
MAX_RED_SELECTION = 16
MAX_BLUE_SELECTION = 6


def _augment_features_frame(df, lottery_type, red_cols, blue_cols, window=50):
    """为期数升序的完整历史DataFrame追加时序衍生特征列（增强特征模式）。

    衍生特征（每行只用当期及之前数据，无未来泄漏）：
      - miss_<号码列>: 该槽位号码开出前已连续遗漏的期数（截断到miss_cut）
      - {sum,span,odd,big}_rmean/_rstd: 和值/跨度/奇数比/大号比 的前window期
        滚动均值与标准差（shift(1)不含当期自身）
      - {dup1,dup3,dup7}_rmean/_rstd: 红球与上1/3/7期重号率的滚动统计
    训练端(prepare_data)与预测端(predict/_rank_balls)必须同时启用保持列一致；
    增强模式要求历史长度 > window+feature_window。旧模型（model_info无
    feature_mode标记）预测不走此函数，行为与升级前完全一致。

    Args:
        df: 按'期数'升序的完整历史 DataFrame
        lottery_type: 'ssq' 或 'dlt'
        red_cols / blue_cols: 号码列名列表（按序）
        window: 滚动窗口长度

    Returns:
        (DataFrame副本, 新增列名列表)
    """
    out = df.copy().reset_index(drop=True)
    is_dlt = (lottery_type == 'dlt')
    big = 25 if is_dlt else 18          # 大号分界（DLT>=25, SSQ>=18）
    miss_cut = 30 if is_dlt else 28     # 遗漏截断上限
    n = len(out)
    new_cols = []

    # --- 遗漏值列：每个号码槽位一列，O(n) 单遍扫描 ---
    for c in list(red_cols) + list(blue_cols):
        vals = out[c].to_numpy(dtype=int)
        miss = np.empty(n, dtype=float)
        last_seen = {}
        for i, v in enumerate(vals):
            prev = last_seen.get(v, -1)
            miss[i] = float(i - prev - 1) if prev >= 0 else float(i)
            last_seen[v] = i
        col_name = f'miss_{c}'
        out[col_name] = np.minimum(miss, float(miss_cut))
        new_cols.append(col_name)

    # --- 每行聚合统计（向量化）---
    red_arr = out[list(red_cols)].to_numpy(dtype=int)
    sum_v = red_arr.sum(axis=1).astype(float)
    span_v = (red_arr.max(axis=1) - red_arr.min(axis=1)).astype(float)
    odd_v = (red_arr % 2 == 1).mean(axis=1)
    big_v = (red_arr >= big).mean(axis=1)
    # 重号动量：本期红球在上1/3/7期(不含本期)出现过的比例
    row_sets = [set(int(x) for x in row) for row in red_arr]
    dup1 = np.zeros(n)
    dup3 = np.zeros(n)
    dup7 = np.zeros(n)
    for i in range(1, n):
        cur = row_sets[i]
        if not cur:
            continue
        dup1[i] = len(cur & row_sets[i - 1]) / len(cur)
        if i >= 3:
            s3 = set().union(*row_sets[i - 3:i])
            dup3[i] = len(cur & s3) / len(cur)
        if i >= 7:
            s7 = set().union(*row_sets[i - 7:i])
            dup7[i] = len(cur & s7) / len(cur)
    stats = {'sum': sum_v, 'span': span_v, 'odd': odd_v, 'big': big_v,
             'dup1': dup1, 'dup3': dup3, 'dup7': dup7}

    # --- 滚动统计列（shift(1)：只用之前window期，不含当期）---
    for key, vals_s in stats.items():
        s = pd.Series(vals_s, dtype=float)
        rm = s.rolling(window, min_periods=window).mean().shift(1)
        rs = s.rolling(window, min_periods=window).std().shift(1)
        rstd_fill = float(rs.median()) if bool(rs.notna().any()) else 0.0
        out[key + '_rmean'] = rm.fillna(float(s.mean()))
        out[key + '_rstd'] = rs.fillna(rstd_fill)
        new_cols += [key + '_rmean', key + '_rstd']

    return out, new_cols


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

class WrappedMultilabelLightGBMModel:
    """
    多标签LightGBM包装器：包含多个二分类器，每个号码一个
    
    预测时返回每个号码被选中的概率，选出概率最高的N个作为预测结果
    """
    def __init__(self, models, ball_type, num_balls):
        """
        Args:
            models: list of LightGBM Booster，每个号码一个二分类器
            ball_type: 'red' 或 'blue'
            num_balls: 需要选出的号码个数（红球5/6个，蓝球1/2个）
        """
        self.models = models
        self.ball_type = ball_type
        self.num_balls = num_balls
        self.n_classes = len(models)
    
    def predict(self, data):
        """
        预测：返回概率最高的num_balls个号码
        
        Args:
            data: (n_samples, n_features) or (n_features,)
        
        Returns:
            如果是单样本：[号码1, 号码2, ...] (1-based, sorted)
            如果是多样本：[[...], [...], ...]
        """
        # 统一处理为2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        n_samples = data.shape[0]
        predictions = []
        
        for sample_idx in range(n_samples):
            sample = data[sample_idx:sample_idx+1]
            
            # 获取每个号码的预测概率
            probas = []
            for model in self.models:
                pred = model.predict(sample)[0]  # 返回正类概率
                probas.append(pred)
            
            probas = np.array(probas)
            
            # 选出概率最高的num_balls个
            top_indices = np.argsort(probas)[-self.num_balls:][::-1]
            top_numbers = sorted([int(idx) + 1 for idx in top_indices])  # 转为1-based
            
            predictions.append(top_numbers)
        
        return predictions[0] if single_sample else predictions
    
    def predict_proba(self, data):
        """
        返回所有号码的预测概率（用于复式/胆拖）
        
        Returns:
            (n_samples, n_classes) 每列是该号码被选中的概率
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = data.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        
        for sample_idx in range(n_samples):
            sample = data[sample_idx:sample_idx+1]
            
            for ball_idx, model in enumerate(self.models):
                pred = model.predict(sample)[0]
                all_probas[sample_idx, ball_idx] = pred
        
        return all_probas

class WrappedMultilabelRandomForestModel:
    """
    多标签RandomForest包装器（基于sklearn的MultiOutputClassifier）
    """
    def __init__(self, model, ball_type, num_balls):
        self.model = model  # MultiOutputClassifier实例
        self.ball_type = ball_type
        self.num_balls = num_balls
        self.n_classes = len(model.estimators_)
    
    def predict(self, data):
        """预测：返回概率最高的num_balls个号码"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # 获取概率
        probas_list = self.model.predict_proba(data)  # 返回list，每个元素是一个(n_samples, 2)数组
        
        predictions = []
        for sample_idx in range(data.shape[0]):
            # 提取每个号码的正类概率；训练集中某号码从未开出时只有一列
            probas = np.array([
                proba[sample_idx, 1] if proba.shape[1] > 1 else proba[sample_idx, 0]
                for proba in probas_list
            ])
            
            # 选出概率最高的num_balls个
            top_indices = np.argsort(probas)[-self.num_balls:][::-1]
            top_numbers = sorted([int(idx) + 1 for idx in top_indices])
            
            predictions.append(top_numbers)
        
        return predictions[0] if single_sample else predictions
    
    def predict_proba(self, data):
        """返回所有号码的预测概率"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        probas_list = self.model.predict_proba(data)
        
        # 转为 (n_samples, n_classes) 格式
        n_samples = data.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        
        for ball_idx, proba in enumerate(probas_list):
            all_probas[:, ball_idx] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        return all_probas


class WrappedMultilabelXGBoostModel:
    """
    多标签XGBoost包装器：为每个号码训练一个二分类XGBoost
    """
    def __init__(self, classifiers, ball_type, num_balls):
        """
        Args:
            classifiers: list of xgboost.Booster，每个对应一个号码
            ball_type: 'red' 或 'blue'
            num_balls: 需要选择的号码数量（红5蓝2）
        """
        self.classifiers = classifiers
        self.ball_type = ball_type
        self.num_balls = num_balls
        self.n_classes = len(classifiers)
    
    def predict(self, data):
        """预测：返回概率最高的num_balls个号码（1-based）"""
        import xgboost as xgb
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        dmatrix = xgb.DMatrix(data)
        predictions = []
        
        for sample_idx in range(data.shape[0]):
            # 获取每个号码的预测概率
            probas = []
            for clf in self.classifiers:
                # XGBoost返回(n_samples, 2)，第2列是正类概率
                proba_matrix = clf.predict(dmatrix)
                if proba_matrix.ndim == 1:
                    # 单类输出，需要转换
                    proba = proba_matrix[sample_idx]
                else:
                    proba = proba_matrix[sample_idx, 1] if proba_matrix.shape[1] > 1 else proba_matrix[sample_idx, 0]
                probas.append(proba)
            
            probas = np.array(probas)
            
            # 选出概率最高的num_balls个
            top_indices = np.argsort(probas)[-self.num_balls:][::-1]
            top_numbers = sorted([int(idx) + 1 for idx in top_indices])
            
            predictions.append(top_numbers)
        
        return predictions[0] if single_sample else predictions
    
    def predict_proba(self, data):
        """返回所有号码的预测概率"""
        import xgboost as xgb
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        dmatrix = xgb.DMatrix(data)
        n_samples = data.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        
        for ball_idx, clf in enumerate(self.classifiers):
            proba_matrix = clf.predict(dmatrix)
            if proba_matrix.ndim == 1:
                all_probas[:, ball_idx] = proba_matrix
            else:
                all_probas[:, ball_idx] = proba_matrix[:, 1] if proba_matrix.shape[1] > 1 else proba_matrix[:, 0]
        
        return all_probas


class WrappedMultilabelGBDTModel:
    """
    多标签GBDT包装器（基于sklearn的MultiOutputClassifier）
    """
    def __init__(self, model, ball_type, num_balls):
        self.model = model  # MultiOutputClassifier实例
        self.ball_type = ball_type
        self.num_balls = num_balls
        self.n_classes = len(model.estimators_)
    
    def predict(self, data):
        """预测：返回概率最高的num_balls个号码"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # 获取概率
        probas_list = self.model.predict_proba(data)
        
        predictions = []
        for sample_idx in range(data.shape[0]):
            # 提取每个号码的正类概率；训练集中某号码从未开出时只有一列
            probas = np.array([
                proba[sample_idx, 1] if proba.shape[1] > 1 else proba[sample_idx, 0]
                for proba in probas_list
            ])
            
            # 选出概率最高的num_balls个
            top_indices = np.argsort(probas)[-self.num_balls:][::-1]
            top_numbers = sorted([int(idx) + 1 for idx in top_indices])
            
            predictions.append(top_numbers)
        
        return predictions[0] if single_sample else predictions
    
    def predict_proba(self, data):
        """返回所有号码的预测概率"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        probas_list = self.model.predict_proba(data)
        
        # 转为 (n_samples, n_classes) 格式
        n_samples = data.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        
        for ball_idx, proba in enumerate(probas_list):
            all_probas[:, ball_idx] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        return all_probas


class WrappedMultilabelCatBoostModel:
    """
    多标签CatBoost包装器：为每个号码训练一个二分类CatBoost
    """
    def __init__(self, classifiers, ball_type, num_balls):
        """
        Args:
            classifiers: list of CatBoostClassifier，每个对应一个号码
            ball_type: 'red' 或 'blue'
            num_balls: 需要选择的号码数量（红5蓝2）
        """
        self.classifiers = classifiers
        self.ball_type = ball_type
        self.num_balls = num_balls
        self.n_classes = len(classifiers)
    
    def predict(self, data):
        """预测：返回概率最高的num_balls个号码（1-based）"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        predictions = []
        
        for sample_idx in range(data.shape[0]):
            sample_data = data[sample_idx:sample_idx+1]
            
            # 获取每个号码的预测概率
            probas = []
            for clf in self.classifiers:
                # CatBoost的predict_proba返回(n_samples, n_classes)，取正类概率
                proba_matrix = clf.predict_proba(sample_data)
                proba = proba_matrix[0, 1] if proba_matrix.shape[1] > 1 else proba_matrix[0, 0]
                probas.append(proba)
            
            probas = np.array(probas)
            
            # 选出概率最高的num_balls个
            top_indices = np.argsort(probas)[-self.num_balls:][::-1]
            top_numbers = sorted([int(idx) + 1 for idx in top_indices])
            
            predictions.append(top_numbers)
        
        return predictions[0] if single_sample else predictions
    
    def predict_proba(self, data):
        """返回所有号码的预测概率"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = data.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        
        for ball_idx, clf in enumerate(self.classifiers):
            proba_matrix = clf.predict_proba(data)
            all_probas[:, ball_idx] = (proba_matrix[:, 1] if proba_matrix.shape[1] > 1
                                       else proba_matrix[:, 0])
        
        return all_probas


def _unwrap_model(model, max_depth=5):
    """递归解包加载时可能出现的包装器嵌套（wrapper.model = wrapper.model = Booster）。"""
    base = model
    for _ in range(max_depth):
        if not type(base).__name__.startswith('Wrapped'):
            break
        inner = getattr(base, 'model', None)
        if inner is None or inner is base:
            break
        base = inner
    return base


def _proba_matrix(model, X_scaled, n_classes=None):
    """返回 (n_samples, n_classes) 的逐类概率矩阵，列索引为0-based号码索引。

    仅接受有概率输出的模型；无概率信息时返回 None。
    """
    try:
        X_scaled = np.asarray(X_scaled)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        # 多标签模型：predict_proba 已返回完整号码概率矩阵
        if isinstance(model, (WrappedMultilabelLightGBMModel, WrappedMultilabelRandomForestModel,
                              WrappedMultilabelXGBoostModel, WrappedMultilabelGBDTModel,
                              WrappedMultilabelCatBoostModel)):
            matrix = np.asarray(model.predict_proba(X_scaled))
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            return matrix

        # sklearn 系分类器：classes_ + predict_proba（概率列顺序=classes_）
        base = _unwrap_model(model)
        for candidate in (model, base):
            classes = getattr(candidate, 'classes_', None)
            if classes is None or not hasattr(candidate, 'predict_proba'):
                continue
            proba = np.asarray(candidate.predict_proba(X_scaled))
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            classes = [int(c) for c in np.asarray(classes).ravel()]
            if n_classes is None:
                n_classes = (max(classes) + 1) if classes else proba.shape[1]
            matrix = np.zeros((proba.shape[0], int(n_classes)))
            for i, cls in enumerate(classes):
                if 0 <= cls < matrix.shape[1] and i < proba.shape[1]:
                    matrix[:, cls] = proba[:, i]
            return matrix

        # lightgbm / xgboost Booster（无 classes_、predict 直接返回概率矩阵）
        # 注意: 必须用解包后的 base 调用，包装器 predict 会转成类别索引
        if 'Booster' in type(base).__name__:
            if 'XGB' in type(base).__name__:
                try:
                    import xgboost as xgb
                    proba = np.asarray(base.predict(xgb.DMatrix(X_scaled)))
                except Exception:
                    proba = np.asarray(base.predict(X_scaled))
            else:
                proba = np.asarray(base.predict(X_scaled))
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            return proba
    except Exception:
        pass
    return None


def _class_proba_dict(model, X_scaled):
    """获取模型对单样本的逐类概率 dict {类别索引(号码-1): 概率}；无概率信息时返回 None"""
    matrix = _proba_matrix(model, X_scaled)
    if matrix is None or matrix.shape[0] == 0:
        return None
    row = matrix[0]
    return {i: float(row[i]) for i in range(len(row))}


def _ensemble_proba_matrix(models_dict, X_scaled, weights=None, n_classes=None):
    """集成模型：对各子模型概率矩阵(加权)平均，返回列索引为0-based号码索引的矩阵。"""
    sums = None
    total_w = 0.0
    for name, model in models_dict.items():
        matrix = _proba_matrix(model, X_scaled, n_classes=n_classes)
        if matrix is None:
            continue
        if n_classes is not None:
            if matrix.shape[1] < n_classes:
                matrix = np.pad(matrix, ((0, 0), (0, n_classes - matrix.shape[1])))
            elif matrix.shape[1] > n_classes:
                matrix = matrix[:, :n_classes]
        w = 1.0
        if weights:
            try:
                w = float(weights.get(name, 1.0)) or 1.0
            except (TypeError, ValueError):
                w = 1.0
        if sums is None:
            sums = matrix * w
        else:
            if matrix.shape[1] != sums.shape[1]:
                cols = max(matrix.shape[1], sums.shape[1])
                sums = np.pad(sums, ((0, 0), (0, cols - sums.shape[1])))
                matrix = np.pad(matrix, ((0, 0), (0, cols - matrix.shape[1])))
            sums = sums + matrix * w
        total_w += w
    if sums is None or total_w <= 0:
        return None
    return sums / total_w


def _ensemble_proba_dict(models_dict, X_scaled, weights=None):
    """集成模型：对每个子模型取逐类概率并(加权)平均，返回 {类别索引: 平均概率}

    weights: 可选 {子模型名: 权重}，由 model_weights 模块基于前向核验表现计算；
    缺省或子模型名不在表中时按等权(1.0)，与旧行为一致。
    """
    matrix = _ensemble_proba_matrix(models_dict, X_scaled, weights=weights)
    if matrix is None or matrix.shape[0] == 0:
        return None
    row = matrix[0]
    return {i: float(row[i]) for i in range(len(row))}


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
        # 集成子模型动态权重（来自 model_weights 模块，按前向核验表现计算）
        # 默认空表=等权，与旧行为完全一致；load_models 成功后刷新
        self.sub_weights = {}
        # 特征模式: original=70维bitmap(旧行为, 与升级前模型完全兼容)
        # enhanced=+21维时序衍生特征(遗漏/滚动和值跨度/重号率)，需配套重训
        self.feature_mode = 'original'
        self.aug_feature_names = []
        self.feature_cols = []
        self.log_callback = log_callback
        self.use_gpu = use_gpu
        
     
        self.raw_models = {}
        self.eval_metrics = None
        self.data_split = None
        self._full_X_raw = None
        self._full_y_red = None
        self._full_y_blue = None
        self._full_history_cache = None
        self._full_history_mtime = None        
   
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
    
    def _load_full_history(self, df):
        """增强模式训练/预测入口的完整历史保障：
        调用方传的可能只是尾部窗口切片，若行数不足以暖机(<=200)则从本地CSV补全。
        已足够时原样返回，不重复IO；CSV按mtime缓存，避免每次预测重复读取。"""
        try:
            if df is not None and len(df) > 200:
                return df
            from backtest import _load_csv_data
            csv_path = f'./scripts/{self.lottery_type}/{self.lottery_type}_history.csv'
            try:
                mtime = os.path.getmtime(csv_path)
            except OSError:
                mtime = None
            full = self._full_history_cache
            if full is None or mtime != self._full_history_mtime:
                full = _load_csv_data(self.lottery_type)
                self._full_history_cache = full
                self._full_history_mtime = mtime
            if full is not None and len(full) > 200:
                self.log(f"完整历史补全: 输入切片{0 if df is None else len(df)}期，"
                         f"改用完整历史 {len(full)} 期")
                return full
        except Exception as e:
            self.log(f"完整历史补全失败(用原数据): {e}")
        return df

    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
    
        self.log("准备训练数据...")
        
       
        window_size = self.feature_window
        
   
        df = df.sort_values('期数').reset_index(drop=True)

        # 增强特征模式：在完整历史上追加时序衍生列并纳入特征；暖机不足自动回退
        if self.feature_mode == 'enhanced':
            _nred = 6 if self.lottery_type == 'ssq' else 5
            _red0 = [c for c in df.columns if c.startswith('红球_')][:_nred]
            _blue0 = [c for c in df.columns if c.startswith('蓝球_')][:self.blue_count]
            if not _blue0:
                _blue0 = [c for c in df.columns if c == '蓝球'][:1]
            try:
                df, _new_aug = _augment_features_frame(df, self.lottery_type,
                                                       _red0, _blue0, window=50)
            except Exception as _e:
                self.log(f"增强特征计算失败，回退 original 模式: {_e}")
                df, _new_aug = df.sort_values('期数').reset_index(drop=True), []
                self.feature_mode = 'original'
            if _new_aug and len(df) > 50 + self.feature_window:
                self.aug_feature_names = list(_new_aug)
                df = df.iloc[50:].reset_index(drop=True)
                self.log(f"增强特征模式: 新增 {len(_new_aug)} 列，暖机丢弃最早50期，剩 {len(df)} 期")
            elif _new_aug:
                self.log(f"警告: 历史({len(df)}期)不足以启用增强特征，回退 original 模式")
                df = df.drop(columns=_new_aug)
                self.feature_mode = 'original'
                self.aug_feature_names = []
        
   
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
        # 增强模式：把衍生列接在末尾（训练/预测两侧拼接顺序必须一致）
        _aug_in_df = []
        if self.feature_mode == 'enhanced' and self.aug_feature_names:
            _aug_in_df = [c for c in self.aug_feature_names if c in df.columns]
            self.feature_cols = self.feature_cols + _aug_in_df
        
     
        X_data = []
        y_red = []
        y_blue = []
        
        for i in range(len(df) - window_size):

            features = []
            for j in range(window_size):
                row_features = []
                for col in red_cols + blue_cols + _aug_in_df:
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

        # 多标签编码：每个号码一列，1=当期开出
        n_samples = len(y_red)
        y_red_multilabel = np.zeros((n_samples, self.red_range), dtype=int)
        y_blue_multilabel = np.zeros((n_samples, self.blue_range), dtype=int)
        for i in range(n_samples):
            for ball_idx in y_red[i]:
                ball_idx = int(ball_idx)
                if 0 <= ball_idx < self.red_range:
                    y_red_multilabel[i, ball_idx] = 1
            for ball_idx in y_blue[i]:
                ball_idx = int(ball_idx)
                if 0 <= ball_idx < self.blue_range:
                    y_blue_multilabel[i, ball_idx] = 1

        if n_samples < 2:
            self.log(f"样本不足({n_samples})，无法拆分训练/测试集")
            return None, None, None, None, None, None

        # 时间序列必须按时间顺序拆分：测试集取最近一段，不能随机打乱
        n_test = max(1, min(int(n_samples * test_size), n_samples - 1))
        n_train = n_samples - n_test
        X_train_raw = X_reshaped[:n_train]
        X_test_raw = X_reshaped[n_train:]

        # 缩放器只能在训练集上拟合，避免测试集信息泄漏到训练过程
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        self.scalers['X'] = scaler

        # 保留全量数据，评估结束后用全部历史重新拟合部署模型
        self._full_X_raw = X_reshaped
        self._full_y_red = y_red_multilabel
        self._full_y_blue = y_blue_multilabel
        self.data_split = {
            'method': 'chronological',
            'total_samples': int(n_samples),
            'train_samples': int(n_train),
            'test_samples': int(n_test),
            'test_size': float(test_size),
        }

        y_red_multilabel_train = y_red_multilabel[:n_train]
        y_red_multilabel_test = y_red_multilabel[n_train:]
        y_blue_multilabel_train = y_blue_multilabel[:n_train]
        y_blue_multilabel_test = y_blue_multilabel[n_train:]
        
        self.log(f"时间顺序拆分: 训练 {n_train} 条(较早), 测试 {n_test} 条(最近)")
        self.log(f"训练集特征形状: {X_train.shape}")
        self.log(f"多标签训练集: 红球{y_red_multilabel_train.shape}, 蓝球{y_blue_multilabel_train.shape}")
        
        return X_train, X_test, y_red_multilabel_train, y_red_multilabel_test, y_blue_multilabel_train, y_blue_multilabel_test
    
    def train_random_forest(self, X_train, y_train, ball_type, n_estimators=100):
        """
        训练随机森林多标签模型：使用MultiOutputClassifier为每个号码训练分类器
        
        Args:
            X_train: 特征 (n_samples, n_features)
            y_train: 多标签 (n_samples, n_classes) 每列表示该号码是否被选中
            ball_type: 'red' 或 'blue'
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        self.log(f"训练{ball_type}球随机森林多标签模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        self.log(f"随机森林参数: n_estimators={n_estimators}, max_depth=10")
        
        # 使用MultiOutputClassifier包装随机森林
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        # 训练模型
        self.log(f"开始训练随机森林多标签模型...")
        model.fit(X_train, y_train)
        
        self.log(f"{ball_type}球随机森林多标签模型训练完成，共{y_train.shape[1]}个分类器")
        
        # 包装模型
        wrapped_model = WrappedMultilabelRandomForestModel(model, ball_type, self.red_count if ball_type == 'red' else self.blue_count)
        
        return wrapped_model
    
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
        """
        训练XGBoost多标签模型：为每个号码训练一个二分类XGBoost
        
        Args:
            X_train: 特征 (n_samples, n_features)
            y_train: 多标签 (n_samples, n_classes) 每列表示该号码是否被选中
            ball_type: 'red' 或 'blue'
        """
        import xgboost as xgb
        
        self.log(f"训练{ball_type}球XGBoost多标签模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 基础参数
        base_params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        
        # GPU加速
        if self.use_gpu and torch.cuda.is_available():
            base_params['tree_method'] = 'gpu_hist'
            self.log("XGBoost使用GPU加速训练")
        
        self.log(f"XGBoost参数: {base_params}")
        
        # 为每个号码训练一个二分类器
        classifiers = []
        n_classes = y_train.shape[1]
        
        self.log(f"开始训练{n_classes}个XGBoost二分类器...")
        
        for ball_idx in range(n_classes):
            if (ball_idx + 1) % 10 == 0 or ball_idx == 0:
                self.log(f"训练第 {ball_idx+1}/{n_classes} 个分类器...")
            
            # 当前号码的标签
            y_ball = y_train[:, ball_idx]
            
            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_ball)
            
            # 训练
            model = xgb.train(
                base_params,
                dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
            
            classifiers.append(model)
        
        self.log(f"{ball_type}球XGBoost多标签模型训练完成，共{n_classes}个分类器")
        
        # 包装模型
        wrapped_model = WrappedMultilabelXGBoostModel(
            classifiers, 
            ball_type, 
            self.red_count if ball_type == 'red' else self.blue_count
        )
        
        return wrapped_model
    
    def train_gbdt(self, X_train, y_train, ball_type):
        """
        训练GBDT多标签模型：使用MultiOutputClassifier为每个号码训练分类器
        
        Args:
            X_train: 特征 (n_samples, n_features)
            y_train: 多标签 (n_samples, n_classes) 每列表示该号码是否被选中
            ball_type: 'red' 或 'blue'
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        self.log(f"训练{ball_type}球GBDT多标签模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'verbose': 0
        }
        
        self.log(f"GBDT参数: {params}")
        
        # 使用MultiOutputClassifier包装GBDT
        base_model = GradientBoostingClassifier(**params)
        model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        self.log(f"开始训练GBDT多标签模型...")
        model.fit(X_train, y_train)
        
        self.log(f"{ball_type}球GBDT多标签模型训练完成，共{y_train.shape[1]}个分类器")
        
        # 包装模型
        wrapped_model = WrappedMultilabelGBDTModel(
            model, 
            ball_type, 
            self.red_count if ball_type == 'red' else self.blue_count
        )
        
        return wrapped_model
    
    def train_lightgbm(self, X_train, y_train, ball_type):
        """
        训练LightGBM多标签模型：为每个号码训练一个二分类器
        
        Args:
            X_train: 特征 (n_samples, n_features)
            y_train: 多标签 (n_samples, n_classes) 每列表示该号码是否被选中
            ball_type: 'red' 或 'blue'
        
        Returns:
            包含所有二分类器的列表
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM未安装，跳过此模型训练")
            return None
            
        import lightgbm as lgb
            
        self.log(f"训练{ball_type}球LightGBM多标签模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        n_classes = self.red_range if ball_type == 'red' else self.blue_range
        models = []
        
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
        }
        
        if self.use_gpu and torch.cuda.is_available():
            base_params['device'] = 'gpu'
            self.log("LightGBM使用GPU加速训练")
        
        # 为每个号码训练一个二分类器
        for ball_num in range(n_classes):
            y_binary = y_train[:, ball_num]
            
            # 只在特定号码处打印进度
            if ball_num % 5 == 0 or ball_num == n_classes - 1:
                positive_rate = y_binary.sum() / len(y_binary)
                self.log(f"  训练号码{ball_num+1}的分类器 (正样本率: {positive_rate:.3f})")
            
            train_data = lgb.Dataset(X_train, label=y_binary)
            
            model = lgb.train(
                base_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=0)]  # 静默训练
            )
            
            models.append(model)
        
        self.log(f"{ball_type}球LightGBM多标签模型训练完成，共{n_classes}个分类器")
        
        # 保存原始模型
        self.raw_models[f'lightgbm_{ball_type}'] = models
        
        # 包装模型
        wrapped_model = WrappedMultilabelLightGBMModel(models, ball_type, self.red_count if ball_type == 'red' else self.blue_count)
        
        return wrapped_model
        
        self.log(f"{ball_type}球LightGBM模型训练完成")
        return wrapped_model
    
    def train_catboost(self, X_train, y_train, ball_type):
        """
        训练CatBoost多标签模型：为每个号码训练一个二分类CatBoost
        
        Args:
            X_train: 特征 (n_samples, n_features)
            y_train: 多标签 (n_samples, n_classes) 每列表示该号码是否被选中
            ball_type: 'red' 或 'blue'
        """
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost未安装，跳过此模型训练")
            return None
            
        import catboost as cb
            
        self.log(f"训练{ball_type}球CatBoost多标签模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置输出目录
        train_dir = f"catboost_info_{ball_type}"
        os.makedirs(train_dir, exist_ok=True)
        
        # 基础参数
        base_params = {
            'loss_function': 'Logloss',
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False,
            'train_dir': train_dir
        }
        
        # GPU加速
        if self.use_gpu and hasattr(cb, 'CatBoostClassifier') and hasattr(cb.CatBoostClassifier, 'is_cuda_supported') and cb.CatBoostClassifier.is_cuda_supported():
            base_params['task_type'] = 'GPU'
            self.log("CatBoost使用GPU加速训练")
        
        self.log(f"CatBoost参数: {base_params}")
        
        # 为每个号码训练一个二分类器
        classifiers = []
        n_classes = y_train.shape[1]
        
        self.log(f"开始训练{n_classes}个CatBoost二分类器...")
        
        try:
            for ball_idx in range(n_classes):
                if (ball_idx + 1) % 10 == 0 or ball_idx == 0:
                    self.log(f"训练第 {ball_idx+1}/{n_classes} 个分类器...")
                
                # 当前号码的标签
                y_ball = y_train[:, ball_idx]
                
                # 创建并训练模型
                model = cb.CatBoostClassifier(**base_params)
                model.fit(X_train, y_ball)
                
                classifiers.append(model)
            
            self.log(f"{ball_type}球CatBoost多标签模型训练完成，共{n_classes}个分类器")
            
            # 包装模型
            wrapped_model = WrappedMultilabelCatBoostModel(
                classifiers,
                ball_type,
                self.red_count if ball_type == 'red' else self.blue_count
            )
            
            return wrapped_model
            
        except Exception as e:
            self.log(f"训练CatBoost多标签模型时出错: {str(e)}")
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
    
    def _fit_models(self, X_train, y_red_train, y_blue_train):
        """按当前 model_type 训练红/蓝球模型，返回 {'red': ..., 'blue': ...}"""
        models = {}
        if self.model_type == 'random_forest':
            self.log("\n----- 使用随机森林模型 -----")
            models['red'] = self.train_random_forest(X_train, y_red_train, 'red')
            models['blue'] = self.train_random_forest(X_train, y_blue_train, 'blue')

        elif self.model_type == 'xgboost':
            self.log("\n----- 使用XGBoost模型 -----")
            models['red'] = self.train_xgboost(X_train, y_red_train, 'red')
            models['blue'] = self.train_xgboost(X_train, y_blue_train, 'blue')

        elif self.model_type == 'gbdt':
            self.log("\n----- 使用梯度提升决策树 -----")
            models['red'] = self.train_gbdt(X_train, y_red_train, 'red')
            models['blue'] = self.train_gbdt(X_train, y_blue_train, 'blue')

        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM 未安装，无法训练")
            self.log("\n----- 使用LightGBM模型 -----")
            models['red'] = self.train_lightgbm(X_train, y_red_train, 'red')
            models['blue'] = self.train_lightgbm(X_train, y_blue_train, 'blue')

        elif self.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ValueError("CatBoost 未安装，无法训练")
            self.log("\n----- 使用CatBoost模型 -----")
            models['red'] = self.train_catboost(X_train, y_red_train, 'red')
            models['blue'] = self.train_catboost(X_train, y_blue_train, 'blue')

        elif self.model_type == 'ensemble':
            self.log("\n----- 使用集成模型 -----")
            models['red'] = self.train_ensemble(X_train, y_red_train, 'red')
            models['blue'] = self.train_ensemble(X_train, y_blue_train, 'blue')

        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        return models

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
        # 增强特征模式需完整历史（重算遗漏/滚动统计）；original 模式零改变
        if self.feature_mode == 'enhanced':
            df = self._load_full_history(df)
        data_prep_start = time.time()
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_data(df)
        data_prep_time = time.time() - data_prep_start
        self.log(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
        
        self.models = {}
        self.eval_metrics = None
        
        self.log("\n====== 第2阶段: 时间留出集评估 ======")
        if X_test is not None and len(X_test) > 0:
            eval_start = time.time()
            eval_models = self._fit_models(X_train, y_red_train, y_blue_train)
            self.eval_metrics = self.evaluate(
                X_test, y_red_test, y_blue_test, models=eval_models)
            del eval_models
            self.log(f"评估完成，耗时: {time.time() - eval_start:.2f}秒")
        else:
            self.log("无可用测试集，跳过评估")

        self.log("\n====== 第3阶段: 全量数据重训 ======")
        if self._full_X_raw is not None:
            refit_start = time.time()
            full_scaler = StandardScaler()
            X_full = full_scaler.fit_transform(self._full_X_raw)
            self.scalers['X'] = full_scaler
            self.scalers['red'] = full_scaler
            self.scalers['blue'] = full_scaler
            self.models = self._fit_models(X_full, self._full_y_red, self._full_y_blue)
            self.log(f"全量重训完成（{len(X_full)} 条样本），耗时: {time.time() - refit_start:.2f}秒")
        else:
            self.models = self._fit_models(X_train, y_red_train, y_blue_train)

        self.log("\n====== 第4阶段: 模型保存 ======")
        save_start = time.time()
        self.save_models()
        save_time = time.time() - save_start
        self.log(f"模型保存完成，耗时: {save_time:.2f}秒")

        # 训练后刷新动态权重（ensemble 立即具备加权投票能力，与 load_models 行为一致）
        self._refresh_sub_weights()

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
    
    def evaluate(self, X_test, y_red_test, y_blue_test, models=None):
        """
        在时间顺序留出集上评估模型（支持全部多标签模型与集成模型）。

        Args:
            X_test: 测试特征（已用训练集缩放器变换）
            y_red_test: 红球多标签矩阵 (n_samples, red_range)
            y_blue_test: 蓝球多标签矩阵 (n_samples, blue_range)
            models: 待评估模型；默认使用 self.models（训练中传入留出集模型）
        """
        eval_models = self.models if models is None else models
        n_samples = len(X_test)
        if n_samples == 0:
            self.log("测试集为空，跳过评估")
            return {}

        self.log(f"评估模型性能（时间留出集 {n_samples} 期）...")

        if self.model_type == 'ensemble':
            red_matrix = _ensemble_proba_matrix(
                eval_models['red'], X_test,
                weights=getattr(self, 'sub_weights', None), n_classes=self.red_range)
            blue_matrix = _ensemble_proba_matrix(
                eval_models['blue'], X_test,
                weights=getattr(self, 'sub_weights', None), n_classes=self.blue_range)
        else:
            red_matrix = _proba_matrix(eval_models['red'], X_test, n_classes=self.red_range)
            blue_matrix = _proba_matrix(eval_models['blue'], X_test, n_classes=self.blue_range)

        if red_matrix is None or blue_matrix is None:
            self.log("模型不支持概率输出，无法评估")
            return {}

        red_matrix = self._pad_proba_matrix(red_matrix, self.red_range)
        blue_matrix = self._pad_proba_matrix(blue_matrix, self.blue_range)

        from collections import Counter

        def evaluate_ball(matrix, y_true, count, name):
            hits = []
            exact = 0
            for i in range(n_samples):
                pred = set(np.argsort(-matrix[i], kind='stable')[:count].tolist())
                true = set(np.flatnonzero(y_true[i]).tolist())
                hit = len(pred & true)
                hits.append(hit)
                exact += int(pred == true)
            distribution = {str(k): int(v) for k, v in sorted(Counter(hits).items())}
            avg_hits = float(np.mean(hits)) if hits else 0.0
            exact_rate = exact / n_samples
            self.log(
                f"{name}平均命中: {avg_hits:.2f}/{count}，"
                f"完全命中率: {exact_rate:.2%}，命中分布: {distribution}")
            return {
                'avg_hits': avg_hits,
                'exact_rate': float(exact_rate),
                'hit_distribution': distribution,
                'count': int(count),
            }

        metrics = {
            'samples': int(n_samples),
            'red': evaluate_ball(red_matrix, y_red_test, self.red_count, '红球'),
            'blue': evaluate_ball(blue_matrix, y_blue_test, self.blue_count, '蓝球'),
        }
        self.eval_metrics = metrics
        return metrics

    @staticmethod
    def _pad_proba_matrix(matrix, n_classes):
        """把概率矩阵列数补齐/截断到号码范围。"""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < n_classes:
            matrix = np.pad(matrix, ((0, 0), (0, n_classes - matrix.shape[1])))
        elif matrix.shape[1] > n_classes:
            matrix = matrix[:, :n_classes]
        return matrix
    
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
        
        # 标记是否是多标签模型
        if 'red' in self.models and isinstance(self.models['red'], (
            WrappedMultilabelLightGBMModel, 
            WrappedMultilabelRandomForestModel,
            WrappedMultilabelXGBoostModel,
            WrappedMultilabelGBDTModel,
            WrappedMultilabelCatBoostModel
        )):
            model_info['is_multilabel'] = True
        
        # 添加特征数量信息
        model_info['feature_mode'] = getattr(self, 'feature_mode', 'original')
        model_info['aug_feature_names'] = list(getattr(self, 'aug_feature_names', []))
        if getattr(self, 'eval_metrics', None):
            model_info['eval_metrics'] = self.eval_metrics
        if getattr(self, 'data_split', None):
            model_info['data_split'] = self.data_split
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

            # 恢复特征模式（旧模型无此字段=original，行为与升级前完全一致）
            self.feature_mode = model_info.get('feature_mode', 'original')
            self.aug_feature_names = list(model_info.get('aug_feature_names', []))
            self.eval_metrics = model_info.get('eval_metrics')
            self.data_split = model_info.get('data_split')
            if self.feature_mode == 'enhanced':
                self.log(f"模型为增强特征模式({len(self.aug_feature_names)}个衍生列)")

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
                            
                            # 如果已经是包装后的模型，直接使用
                            if isinstance(model_obj, (
                                WrappedMultilabelLightGBMModel, 
                                WrappedMultilabelRandomForestModel,
                                WrappedMultilabelXGBoostModel,
                                WrappedMultilabelGBDTModel,
                                WrappedMultilabelCatBoostModel,
                                WrappedLightGBMModel
                            )):
                                self.models[ball_type] = model_obj
                                self.log(f"{ball_type}球模型加载成功（已包装）")
                                
                            # 检查是否是多标签模型（未包装的原始模型）
                            elif model_info.get('is_multilabel', False):
                                # 多标签模型，需要包装
                                if self.model_type == 'lightgbm' and 'lightgbm' in str(type(model_obj)).lower():
                                    num_balls = self.red_count if ball_type == 'red' else self.blue_count
                                    self.models[ball_type] = WrappedMultilabelLightGBMModel(model_obj, ball_type, num_balls)
                                    self.log(f"使用多标签包装器")
                                else:
                                    self.models[ball_type] = model_obj
                                    
                            # 旧的单标签模型
                            elif self.model_type == 'lightgbm' and hasattr(model_obj, 'predict'):
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
            self._refresh_sub_weights()
            return True
        elif balls_loaded >= 2:  # 至少加载了红球和蓝球
            # 即使有警告，只要基础模型存在，我们也认为模型可用
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但有一些警告")
            self._refresh_sub_weights()
            return True
        elif 'red' in self.models and 'blue' in self.models:
            # 模型存在但可能有其他问题
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但可能存在兼容性问题")
            self._refresh_sub_weights()
            return True
        else:
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载失败，未找到必要的红球和蓝球模型")
            return False

    def _refresh_sub_weights(self):
        """从 model_weights 模块加载本彩种的子模型动态权重。
        无权重文件/异常时保持空表（等权，旧行为）。"""
        try:
            from model_weights import get_model_weights
            self.sub_weights = get_model_weights(self.lottery_type)
            if self.sub_weights and self.model_type == 'ensemble':
                named = [n for n in self.models.get('red', {}) if n in self.sub_weights]
                if named:
                    self.log("集成子模型已启用动态权重: " +
                             ", ".join(f"{n}={self.sub_weights[n]:.2f}" for n in named))
        except Exception:
            self.sub_weights = {}

    def _expected_feature_count(self):
        """训练时期望的展平特征维度；优先使用模型元数据，最后回退到70。"""
        expected = getattr(self, 'expected_feature_count', None)
        if expected:
            return int(expected)
        model_obj = self.models.get('red')
        if self.model_type == 'ensemble' and isinstance(model_obj, dict):
            model_obj = model_obj.get('random_forest') or next(iter(model_obj.values()), None)
        if model_obj is not None:
            n_feat = getattr(model_obj, 'n_features_in_', None)
            if n_feat:
                return int(n_feat)
            importances = getattr(model_obj, 'feature_importances_', None)
            if importances is not None and getattr(importances, 'shape', None):
                return int(importances.shape[0])
        return 70

    def _build_prediction_features(self, recent_data):
        """构建与训练一致的滑动窗口特征，并按训练缩放器标准化。失败返回 None。"""
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            blue_cols = []
            for col in recent_data.columns:
                if col.startswith('蓝球_') or col == '蓝球':
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:
                        break

        recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
        if len(recent_data) < self.feature_window:
            self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
            return None

        aug_mat = None
        if getattr(self, 'feature_mode', 'original') == 'enhanced':
            aug_mat = self._aug_row_features(recent_data)
            if aug_mat is None:
                self.log("错误: 增强模型无法计算衍生特征(历史不足或期号对不齐)，本次预测中止")
                return None

        features = []
        for j in range(self.feature_window):
            row_features = [recent_data.iloc[j][col] for col in red_cols + blue_cols]
            if aug_mat is not None:
                row_features.extend([float(x) for x in aug_mat[j]])
            features.append(row_features)

        X_reshaped = np.array([features]).reshape(1, -1)
        expected = self._expected_feature_count()
        actual = X_reshaped.shape[1]
        if actual != expected:
            self.log(f"警告: 特征数量不匹配，预期{expected}个，实际{actual}个")
            if actual < expected:
                X_reshaped = np.pad(X_reshaped, ((0, 0), (0, expected - actual)))
            else:
                X_reshaped = X_reshaped[:, :expected]

        try:
            scaler = self.scalers.get('X')
            if scaler is None:
                scaler = self.scalers.get('red')
            if scaler is None:
                return X_reshaped
            return scaler.transform(X_reshaped)
        except Exception as e:
            self.log(f"应用特征缩放时出错，使用未缩放特征: {e}")
            return X_reshaped

    def _get_probability_maps(self, recent_data):
        """返回红/蓝球逐号码概率 {0-based索引: 概率}，并写入 last_*_proba。"""
        X_scaled = self._build_prediction_features(recent_data)
        if X_scaled is None:
            return None

        if self.model_type == 'ensemble':
            red_matrix = _ensemble_proba_matrix(
                self.models['red'], X_scaled,
                weights=getattr(self, 'sub_weights', None), n_classes=self.red_range)
            blue_matrix = _ensemble_proba_matrix(
                self.models['blue'], X_scaled,
                weights=getattr(self, 'sub_weights', None), n_classes=self.blue_range)
        else:
            red_matrix = _proba_matrix(self.models['red'], X_scaled, n_classes=self.red_range)
            blue_matrix = _proba_matrix(self.models['blue'], X_scaled, n_classes=self.blue_range)

        if red_matrix is None or blue_matrix is None:
            self.log("模型无概率输出，无法按概率选号")
            return None

        red_matrix = self._pad_proba_matrix(red_matrix, self.red_range)
        blue_matrix = self._pad_proba_matrix(blue_matrix, self.blue_range)
        red_proba = {i: float(red_matrix[0, i]) for i in range(self.red_range)}
        blue_proba = {i: float(blue_matrix[0, i]) for i in range(self.blue_range)}
        self.last_red_proba = red_proba
        self.last_blue_proba = blue_proba
        return red_proba, blue_proba

    @staticmethod
    def _select_numbers(proba, count, ball_range, variation=0, seed_period=0):
        """按概率排序选号。

        variation=0 取概率最高的 count 个；variation>0 时在概率靠前的候选池中
        按种子(由 variation 和期号决定)无放回抽样，保证同一次预测可复现。
        """
        ranked = sorted(
            ((int(k), float(v)) for k, v in (proba or {}).items()
             if 0 <= int(k) < ball_range),
            key=lambda kv: (-kv[1], kv[0]))
        ranked_numbers = [k + 1 for k, _ in ranked]
        if len(ranked_numbers) < ball_range:
            ranked_numbers.extend(
                n for n in range(1, ball_range + 1) if n not in ranked_numbers)

        if variation <= 0 or count >= len(ranked_numbers):
            return sorted(ranked_numbers[:count])

        pool = ranked[:min(len(ranked), count + max(3, count))]
        pool_numbers = np.array([k + 1 for k, _ in pool], dtype=int)
        weights = np.array([max(0.0, p) for _, p in pool], dtype=float)
        if weights.sum() <= 0:
            weights = None
        else:
            weights = weights / weights.sum()

        rng = np.random.default_rng((int(variation) + 1) * 1000003 + int(seed_period))
        chosen = rng.choice(
            pool_numbers, size=min(count, len(pool_numbers)),
            replace=False, p=weights)
        selected = [int(n) for n in chosen]
        if len(selected) < count:
            remaining = [n for n in ranked_numbers if n not in selected]
            selected.extend(remaining[:count - len(selected)])
        return sorted(selected)

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
        
        # 统一走概率排序选号：模型输出的是 0-based 类别概率，
        # 这里只做一次 +1 转号码，避免不同包装器重复偏移。
        maps = self._get_probability_maps(recent_data)
        if maps is None:
            return None, None
        red_proba, blue_proba = maps

        latest_period = int(recent_data['期数'].max()) if len(recent_data) else 0
        red_predictions = self._select_numbers(
            red_proba, self.red_count, self.red_range,
            variation=variation, seed_period=latest_period)
        blue_predictions = self._select_numbers(
            blue_proba, self.blue_count, self.blue_range,
            variation=variation, seed_period=latest_period)

        self.last_red_conf = _mean_confidence(red_proba, red_predictions)
        self.last_blue_conf = _mean_confidence(blue_proba, blue_predictions)

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

    def _aug_row_features(self, recent_data):
        """预测侧：在完整历史上重算衍生列，返回按 recent_data 期号对齐的衍生特征矩阵。

        与训练侧 prepare_data 的增强分支严格对应：都在"按期数升序的完整历史"上调用
        _augment_features_frame，因此同一期号得到的衍生值完全一致。

        Args:
            recent_data: 最近若干期数据（期数顺序不限）

        Returns:
            形状 (feature_window, len(aug_feature_names)) 的 ndarray；
            无法计算时返回 None（调用方回退到 original 特征）
        """
        try:
            nred = 6 if self.lottery_type == 'ssq' else 5
            red_cols = [c for c in recent_data.columns if c.startswith('红球_')][:nred]
            blue_cols = [c for c in recent_data.columns
                         if c.startswith('蓝球_') or c == '蓝球'][:self.blue_count]
            if len(red_cols) < nred or not blue_cols:
                return None
            rec = recent_data.sort_values('期数', ascending=True).reset_index(drop=True)
            if len(rec) < self.feature_window:
                return None
            # 衍生特征需要 50+window 暖机历史：不足时从本地CSV补全（recent 本身足够则不重复IO）
            hist = self._load_full_history(rec)
            hist = hist.sort_values('期数', ascending=True).reset_index(drop=True)
            aug_df, new_cols = _augment_features_frame(
                hist, self.lottery_type, red_cols, blue_cols, window=50)
            if not new_cols:
                return None
            cols = [c for c in self.aug_feature_names if c in aug_df.columns] \
                if self.aug_feature_names else new_cols
            # 按期号精确对齐取行（预测侧行序=期数降序，与 predict/_rank_balls 一致）
            pos = {int(p): i for i, p in enumerate(aug_df['期数'])}
            periods = rec['期数'].astype(int).tolist()[-self.feature_window:][::-1]
            if any(p not in pos for p in periods):
                return None
            return np.array([aug_df[cols].iloc[pos[p]].to_numpy(dtype=float)
                             for p in periods], dtype=float)
        except Exception as e:
            self.log(f"预测端衍生特征计算失败: {e}")
            return None

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

        maps = self._get_probability_maps(recent_data)
        if maps is None:
            return None, None
        red_proba, blue_proba = maps

        red_ranked = [k + 1 for k, _ in sorted(
            red_proba.items(), key=lambda kv: (-kv[1], kv[0]))]
        blue_ranked = [k + 1 for k, _ in sorted(
            blue_proba.items(), key=lambda kv: (-kv[1], kv[0]))]
        return red_ranked, blue_ranked

    @staticmethod
    def _pad_ranked(ranked, need, ball_range):
        """概率排序号码不足 need 个时按号码升序补充，保证不重复且可复现。"""
        ranked = list(ranked)
        for number in range(1, ball_range + 1):
            if len(ranked) >= need:
                break
            if number not in ranked:
                ranked.append(number)
        return ranked

    def predict_compound(self, recent_data, extra_red=0, extra_blue=0):
        """
        预测复式号码：按模型概率排序（ensemble 为各子模型平均概率），
        返回 top (red_count+extra_red) 红球 + top (blue_count+extra_blue) 蓝球，
        而非单式的固定数量号码。确定性输出，不做随机扰动。

        Args:
            recent_data: 历史数据 DataFrame（与 predict 相同）
            extra_red: 红球额外数量，复式红球数 = red_count + extra_red（上限 MAX_RED_SELECTION）
            extra_blue: 蓝球额外数量，复式蓝球数 = blue_count + extra_blue（上限 MAX_BLUE_SELECTION）

        Returns:
            (red_numbers, blue_numbers) 或 (None, None)
            复式注数 = C(红球数, red_count) × C(蓝球数, blue_count)，金额 = 注数 × 2 元
        """
        # 号码池下限为单式数量，上限为用户规则（红16/蓝6）
        red_need = max(self.red_count,
                       min(self.red_count + max(0, int(extra_red)), MAX_RED_SELECTION))
        blue_need = max(self.blue_count,
                        min(self.blue_count + max(0, int(extra_blue)), MAX_BLUE_SELECTION))
        if red_need > self.red_range:
            red_need = self.red_range
        if blue_need > self.blue_range:
            blue_need = self.blue_range

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
        - 拖码个数 >= 单式数-胆数
        - 红胆+红拖 <= MAX_RED_SELECTION(16)，蓝胆+蓝拖 <= MAX_BLUE_SELECTION(6)

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
        # 蓝胆：尊重用户设置（0 = 不要蓝胆），仅做上限截断
        # SSQ 蓝球单式只选 1 个 → 蓝胆上限 0；DLT 蓝球选 2 个 → 蓝胆上限 1
        blue_dan_max = self.blue_count - 1 if self.blue_count >= 2 else 0
        blue_dan_count = max(0, min(blue_dan_count, blue_dan_max))
        red_tuo_min = self.red_count - red_dan_count
        blue_tuo_min = self.blue_count - blue_dan_count
        # 胆+拖总数受用户规则限制：红 <=16，蓝 <=6
        red_tuo_count = max(red_tuo_min,
                            min(red_tuo_count, MAX_RED_SELECTION - red_dan_count))
        blue_tuo_count = max(blue_tuo_min,
                             min(blue_tuo_count, MAX_BLUE_SELECTION - blue_dan_count))

        if self.blue_count == 1 and blue_dan_count == 0:
            # SSQ 蓝球单式只选1个，无胆拖空间，蓝球按复式出号
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

    def predict_fushi(self, recent_data, red_count, blue_count):
        """
        复式预测的线程调用适配方法。
        将 (red_count, blue_count) 转换为 predict_compound(extra_red, extra_blue)。

        Args:
            recent_data: 历史数据 DataFrame
            red_count: 复式红球总数（如8）
            blue_count: 复式蓝球总数（如2）

        Returns:
            (red_numbers, blue_numbers) 号码列表 或 (None, None)
        """
        extra_red = red_count - self.red_count
        extra_blue = blue_count - self.blue_count
        return self.predict_compound(recent_data, extra_red=extra_red, extra_blue=extra_blue)

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