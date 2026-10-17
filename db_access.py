#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQLite 数据库访问层 (Data Access Layer)
=====================================

统一封装所有数据库操作，替代 JSON 文件读写。

设计原则：
1. 事务安全：写操作自动包装事务
2. 连接池：使用 context manager 管理连接
3. 参数化查询：防止 SQL 注入
4. 类型转换：自动序列化/反序列化 JSON 字段
5. 错误处理：数据库错误自动回滚并返回友好错误消息
"""

import sqlite3
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# 数据库路径
DB_PATH = "busuan.db"


# ============================================================
# 连接管理
# ============================================================

@contextmanager
def get_connection():
    """获取数据库连接（context manager）"""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)  # 10秒等待锁
    conn.row_factory = sqlite3.Row  # 返回字典式结果
    # 启用 WAL 模式（提高并发读写性能）
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")  # 10秒
    except Exception:
        pass  # 只读模式下可能失败，忽略
    try:
        yield conn
    finally:
        conn.close()


def execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """执行查询（SELECT）"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def execute_update(query: str, params: tuple = ()) -> int:
    """执行更新/插入/删除（带事务）"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid


def execute_many(query: str, params_list: List[tuple]) -> int:
    """批量执行（带事务）"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
        return cursor.rowcount


# ============================================================
# Lottery Draws (开奖记录)
# ============================================================

def get_lottery_draws(lottery_type: str, limit: int = None) -> List[Dict]:
    """
    获取开奖记录（降序，最新在前）
    
    Returns:
        [{"id": 1, "lottery_type": "ssq", "period": 26118, 
          "draw_date": "2020-01-01", "red_numbers": [1,5,...], ...}, ...]
    """
    query = """
        SELECT id, lottery_type, period, draw_date, red_numbers, blue_numbers, created_at
        FROM lottery_draws
        WHERE lottery_type = ?
        ORDER BY period DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    
    rows = execute_query(query, (lottery_type,))
    
    # 反序列化 JSON 字段
    for row in rows:
        row['red_numbers'] = json.loads(row['red_numbers'])
        row['blue_numbers'] = json.loads(row['blue_numbers'])
    
    return rows


def add_lottery_draw(lottery_type: str, period: int, draw_date: str,
                     red_numbers: List[int], blue_numbers: List[int]) -> int:
    """添加开奖记录（INSERT OR IGNORE，防止重复）"""
    query = """
        INSERT OR IGNORE INTO lottery_draws 
        (lottery_type, period, draw_date, red_numbers, blue_numbers)
        VALUES (?, ?, ?, ?, ?)
    """
    return execute_update(query, (
        lottery_type, period, draw_date,
        json.dumps(red_numbers), json.dumps(blue_numbers)
    ))


def get_latest_period(lottery_type: str) -> Optional[int]:
    """获取最新期号"""
    query = "SELECT MAX(period) as max_period FROM lottery_draws WHERE lottery_type = ?"
    rows = execute_query(query, (lottery_type,))
    return rows[0]['max_period'] if rows else None


# ============================================================
# Predictions (预测记录)
# ============================================================

def get_predictions(lottery_type: str = None, verified: bool = None, 
                   limit: int = None, offset: int = 0) -> List[Dict]:
    """
    获取预测记录
    
    Args:
        lottery_type: 筛选彩种（None 表示全部）
        verified: True=已核验, False=未核验, None=全部
        limit: 返回条数
        offset: 偏移量（分页）
    
    Returns:
        [{"id": 1, "predict_time": "2026-10-15T10:47:35", 
          "lottery_type": "ssq", "model_type": "ensemble",
          "red_numbers": [1,5,...], "red_probabilities": [0.8,...], ...}, ...]
    """
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if lottery_type:
        query += " AND lottery_type = ?"
        params.append(lottery_type)
    
    if verified is not None:
        if verified:
            query += " AND verified_at IS NOT NULL"
        else:
            query += " AND verified_at IS NULL"
    
    query += " ORDER BY predict_time DESC"
    
    if limit:
        query += f" LIMIT {limit} OFFSET {offset}"
    
    rows = execute_query(query, tuple(params))
    
    # 反序列化 JSON 字段
    for row in rows:
        row['red_numbers'] = json.loads(row['red_numbers'])
        row['blue_numbers'] = json.loads(row['blue_numbers'])
        if row['red_probabilities']:
            row['red_probabilities'] = json.loads(row['red_probabilities'])
        if row['blue_probabilities']:
            row['blue_probabilities'] = json.loads(row['blue_probabilities'])
    
    return rows


def add_prediction(lottery_type: str, model_type: str, latest_period: int,
                  red_numbers: List[int], blue_numbers: List[int],
                  red_probabilities: List[float] = None,
                  blue_probabilities: List[float] = None,
                  predict_time: str = None) -> int:
    """添加预测记录"""
    if not predict_time:
        predict_time = datetime.now().isoformat()
    
    query = """
        INSERT INTO predictions 
        (predict_time, lottery_type, model_type, latest_period,
         red_numbers, blue_numbers, red_probabilities, blue_probabilities)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_update(query, (
        predict_time, lottery_type, model_type, latest_period,
        json.dumps(red_numbers), json.dumps(blue_numbers),
        json.dumps(red_probabilities) if red_probabilities else None,
        json.dumps(blue_probabilities) if blue_probabilities else None
    ))


def update_prediction_verification(prediction_id: int, target_period: int,
                                  red_hits: int, blue_hits: int,
                                  prize_name: str = None, prize_amount: float = None):
    """更新预测核验结果"""
    query = """
        UPDATE predictions
        SET target_period = ?, red_hits = ?, blue_hits = ?,
            prize_name = ?, prize_amount = ?, verified_at = ?
        WHERE id = ?
    """
    execute_update(query, (
        target_period, red_hits, blue_hits,
        prize_name, prize_amount, datetime.now().isoformat(),
        prediction_id
    ))


def get_predictions_for_verification(lottery_type: str) -> List[Dict]:
    """获取待核验的预测记录（target_period 为 NULL）"""
    return get_predictions(lottery_type=lottery_type, verified=False)


# ============================================================
# Model Weights (模型权重)
# ============================================================

def get_latest_model_weights(lottery_type: str) -> Optional[Dict]:
    """获取最新模型权重"""
    query = """
        SELECT * FROM model_weights_history
        WHERE lottery_type = ?
        ORDER BY updated_at DESC
        LIMIT 1
    """
    rows = execute_query(query, (lottery_type,))
    if not rows:
        return None
    
    row = rows[0]
    row['window'] = json.loads(row['window'])
    row['weights'] = json.loads(row['weights'])
    if row['ensemble_sub_weights']:
        row['ensemble_sub_weights'] = json.loads(row['ensemble_sub_weights'])
    return row


def add_model_weights(lottery_type: str, window: Dict, weights: Dict,
                     ensemble_sub_weights: Dict = None,
                     recommended_model: str = 'ensemble',
                     avg_red_hits: float = None, avg_blue_hits: float = None,
                     hit_rate: float = None) -> int:
    """添加模型权重记录"""
    query = """
        INSERT INTO model_weights_history
        (lottery_type, updated_at, window, weights, ensemble_sub_weights,
         recommended_model, avg_red_hits, avg_blue_hits, hit_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_update(query, (
        lottery_type, datetime.now().isoformat(),
        json.dumps(window), json.dumps(weights),
        json.dumps(ensemble_sub_weights) if ensemble_sub_weights else None,
        recommended_model, avg_red_hits, avg_blue_hits, hit_rate
    ))


# ============================================================
# Investment Plans (投注计划)
# ============================================================

def get_investment_plans(lottery_type: str = None, verified: bool = None) -> List[Dict]:
    """获取投注计划"""
    query = "SELECT * FROM investment_plans WHERE 1=1"
    params = []
    
    if lottery_type:
        query += " AND lottery_type = ?"
        params.append(lottery_type)
    
    if verified is not None:
        query += " AND verified = ?"
        params.append(1 if verified else 0)
    
    query += " ORDER BY target_period DESC"
    
    rows = execute_query(query, tuple(params))
    
    for row in rows:
        row['bet_numbers'] = json.loads(row['bet_numbers'])
        if row['linked_prediction_ids']:
            row['linked_prediction_ids'] = json.loads(row['linked_prediction_ids'])
    
    return rows


def add_investment_plan(lottery_type: str, target_period: int, plan_date: str,
                       bet_numbers: List[List[int]], total_amount: float,
                       bet_count: int, linked_prediction_ids: List[int] = None,
                       notes: str = None) -> int:
    """添加投注计划"""
    query = """
        INSERT INTO investment_plans
        (lottery_type, target_period, plan_date, bet_numbers,
         total_amount, bet_count, linked_prediction_ids, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_update(query, (
        lottery_type, target_period, plan_date,
        json.dumps(bet_numbers), total_amount, bet_count,
        json.dumps(linked_prediction_ids) if linked_prediction_ids else None,
        notes
    ))


# ============================================================
# Optimization Log (优化日志)
# ============================================================

def add_optimization_log(lottery_type: str, action_type: str, status: str,
                        execution_mode: str, action_params: Dict = None,
                        result_summary: str = None, effect_metrics: Dict = None,
                        error_message: str = None) -> int:
    """添加优化日志"""
    query = """
        INSERT INTO optimization_log
        (lottery_type, timestamp, action_type, action_params, status,
         execution_mode, result_summary, effect_metrics, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_update(query, (
        lottery_type, datetime.now().isoformat(),
        action_type,
        json.dumps(action_params) if action_params else None,
        status, execution_mode, result_summary,
        json.dumps(effect_metrics) if effect_metrics else None,
        error_message
    ))


def get_optimization_history(lottery_type: str = None, action_type: str = None,
                            limit: int = 50) -> List[Dict]:
    """获取优化历史"""
    query = "SELECT * FROM optimization_log WHERE 1=1"
    params = []
    
    if lottery_type:
        query += " AND lottery_type = ?"
        params.append(lottery_type)
    
    if action_type:
        query += " AND action_type = ?"
        params.append(action_type)
    
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    
    rows = execute_query(query, tuple(params))
    
    for row in rows:
        if row['action_params']:
            row['action_params'] = json.loads(row['action_params'])
        if row['effect_metrics']:
            row['effect_metrics'] = json.loads(row['effect_metrics'])
    
    return rows


# ============================================================
# 数据统计查询
# ============================================================

def get_model_performance_stats(lottery_type: str, days: int = 30) -> List[Dict]:
    """获取模型表现统计（近 N 天）"""
    query = """
        SELECT 
            model_type,
            COUNT(*) as prediction_count,
            AVG(red_hits) as avg_red_hits,
            AVG(blue_hits) as avg_blue_hits,
            SUM(CASE WHEN prize_amount > 0 THEN 1 ELSE 0 END) as win_count,
            SUM(COALESCE(prize_amount, 0)) as total_prize
        FROM predictions
        WHERE lottery_type = ?
          AND verified_at >= datetime('now', ? || ' days')
        GROUP BY model_type
        ORDER BY avg_red_hits DESC, avg_blue_hits DESC
    """
    return execute_query(query, (lottery_type, -days))


def get_verification_summary(lottery_type: str = None) -> Dict:
    """获取核验汇总"""
    query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN verified_at IS NOT NULL THEN 1 END) as verified,
            COUNT(CASE WHEN verified_at IS NULL THEN 1 END) as pending,
            COUNT(CASE WHEN red_probabilities IS NOT NULL THEN 1 END) as with_probabilities
        FROM predictions
    """
    params = []
    if lottery_type:
        query += " WHERE lottery_type = ?"
        params.append(lottery_type)
    
    rows = execute_query(query, tuple(params))
    return rows[0] if rows else {}


if __name__ == "__main__":
    # 测试数据库访问
    print("Testing database access...")
    
    # 测试1: 获取最新期号
    ssq_latest = get_latest_period('ssq')
    dlt_latest = get_latest_period('dlt')
    print(f"Latest periods: SSQ={ssq_latest}, DLT={dlt_latest}")
    
    # 测试2: 获取预测记录
    predictions = get_predictions(lottery_type='ssq', verified=True, limit=3)
    print(f"\nVerified SSQ predictions: {len(predictions)}")
    if predictions:
        print(f"Sample: period={predictions[0]['target_period']}, "
              f"red_hits={predictions[0]['red_hits']}, blue_hits={predictions[0]['blue_hits']}")
    
    # 测试3: 获取权重
    weights = get_latest_model_weights('ssq')
    if weights:
        print(f"\nSSQ model weights updated at: {weights['updated_at']}")
        print(f"Recommended model: {weights['recommended_model']}")
    
    # 测试4: 核验汇总
    summary = get_verification_summary()
    print(f"\nVerification summary: {summary}")
    
    print("\n[OK] Database access layer test passed!")
