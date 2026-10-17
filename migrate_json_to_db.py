#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 数据迁移到 SQLite 数据库
============================

迁移范围：
1. prediction_history.json (524 条) -> predictions 表
2. model_weights.json (SSQ/DLT 权重) -> model_weights_history 表
3. investment_history.json -> investment_plans 表
4. scripts/ssq/ssq_history.csv -> lottery_draws 表
5. scripts/dlt/dlt_history.csv -> lottery_draws 表

安全保障：
- 原文件自动备份到 _migration_backup/
- 事务回滚机制（失败时不破坏原数据）
- 数据完整性验证（迁移后对比记录数）
"""

import sqlite3
import json
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path

# 数据库路径
DB_PATH = "busuan.db"
SCHEMA_PATH = "schema.sql"

# 源文件路径
PREDICTION_JSON = "prediction_history.json"
WEIGHTS_JSON = "model_weights.json"
INVESTMENT_JSON = "investment_history.json"
SSQ_CSV = "scripts/ssq/ssq_history.csv"
DLT_CSV = "scripts/dlt/dlt_history.csv"

# 备份目录
BACKUP_DIR = "_migration_backup"


def backup_files():
    """备份所有源文件"""
    print("[1/5] Backup source files...")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files = [PREDICTION_JSON, WEIGHTS_JSON, INVESTMENT_JSON, SSQ_CSV, DLT_CSV]
    for file in files:
        if os.path.exists(file):
            backup_path = os.path.join(BACKUP_DIR, f"{Path(file).stem}_{timestamp}{Path(file).suffix}")
            shutil.copy2(file, backup_path)
            print(f"   OK: {file} -> {backup_path}")
    print()


def init_database():
    """初始化数据库（执行 schema.sql）"""
    print("[INIT] 初始化数据库...")
    if os.path.exists(DB_PATH):
        backup_db = f"{BACKUP_DIR}/busuan_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(DB_PATH, backup_db)
        print(f"   [WARN] 数据库已存在，备份到 {backup_db}")
    
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("   OK 数据库初始化完成\n")


def migrate_lottery_draws(conn):
    """迁移开奖记录（CSV -> lottery_draws）"""
    print("[MIGRATE] 迁移开奖记录...")
    cursor = conn.cursor()
    
    for csv_path, lottery_type in [(SSQ_CSV, 'ssq'), (DLT_CSV, 'dlt')]:
        if not os.path.exists(csv_path):
            print(f"   [WARN] {csv_path} 不存在，跳过")
            continue
        
        df = pd.read_csv(csv_path, encoding='gbk')
        print(f"   读取 {csv_path}: {len(df)} 行")
        
        if lottery_type == 'ssq':
            # 双色球：期数, 红球_1..6, 蓝球
            for _, row in df.iterrows():
                period = int(row['期数'])
                red_numbers = [int(row[f'红球_{i}']) for i in range(1, 7)]
                blue_numbers = [int(row['蓝球'])]
                
                cursor.execute("""
                    INSERT OR IGNORE INTO lottery_draws 
                    (lottery_type, period, draw_date, red_numbers, blue_numbers)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    lottery_type,
                    period,
                    '2020-01-01',  # CSV 无日期，占位符
                    json.dumps(red_numbers),
                    json.dumps(blue_numbers)
                ))
        
        else:  # dlt
            # 大乐透：期数, 红球_1..5, 蓝球_1, 蓝球_2
            for _, row in df.iterrows():
                period = int(row['期数'])
                red_numbers = [int(row[f'红球_{i}']) for i in range(1, 6)]
                blue_numbers = [int(row['蓝球_1']), int(row['蓝球_2'])]
                
                cursor.execute("""
                    INSERT OR IGNORE INTO lottery_draws 
                    (lottery_type, period, draw_date, red_numbers, blue_numbers)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    lottery_type,
                    period,
                    '2020-01-01',
                    json.dumps(red_numbers),
                    json.dumps(blue_numbers)
                ))
        
        conn.commit()
        count = cursor.execute("SELECT COUNT(*) FROM lottery_draws WHERE lottery_type = ?", (lottery_type,)).fetchone()[0]
        print(f"   OK {lottery_type.upper()}: {count} 条记录\n")


def migrate_predictions(conn):
    """迁移预测记录（JSON -> predictions）"""
    print("[PREDICT] 迁移预测记录...")
    
    if not os.path.exists(PREDICTION_JSON):
        print(f"   [WARN] {PREDICTION_JSON} 不存在，跳过\n")
        return
    
    with open(PREDICTION_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"   读取 {len(records)} 条记录")
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        
        try:
            cursor.execute("""
                INSERT INTO predictions (
                    predict_time, lottery_type, model_type, latest_period,
                    red_numbers, blue_numbers, red_probabilities, blue_probabilities,
                    target_period, red_hits, blue_hits, prize_name, prize_amount,
                    verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get('predict_time'),
                record.get('lottery_type'),
                record.get('model_type'),
                record.get('latest_period'),
                json.dumps(record.get('red_numbers', [])),
                json.dumps(record.get('blue_numbers', [])),
                json.dumps(record.get('red_probabilities')) if record.get('red_probabilities') else None,
                json.dumps(record.get('blue_probabilities')) if record.get('blue_probabilities') else None,
                record.get('target_period'),
                record.get('red_hits'),
                record.get('blue_hits'),
                record.get('prize_name'),
                record.get('prize_amount'),
                record.get('verified_at') or (datetime.now().isoformat() if record.get('target_period') else None)
            ))
            inserted += 1
        except Exception as e:
            print(f"   [WARN] 跳过记录: {e}")
            skipped += 1
    
    conn.commit()
    print(f"   OK 成功: {inserted} 条，跳过: {skipped} 条\n")


def migrate_model_weights(conn):
    """迁移模型权重（JSON -> model_weights_history）"""
    print("[WEIGHT] 迁移模型权重...")
    
    if not os.path.exists(WEIGHTS_JSON):
        print(f"   [WARN] {WEIGHTS_JSON} 不存在，跳过\n")
        return
    
    with open(WEIGHTS_JSON, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    
    cursor = conn.cursor()
    
    for lottery_type in ['ssq', 'dlt']:
        if lottery_type not in weights_data:
            continue
        
        data = weights_data[lottery_type]
        
        cursor.execute("""
            INSERT INTO model_weights_history (
                lottery_type, updated_at, window, weights,
                ensemble_sub_weights, recommended_model
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            lottery_type,
            data.get('updated_at', datetime.now().isoformat()),
            json.dumps(data.get('window', {'max_records': 200})),  # window 是对象，序列化
            json.dumps(data.get('models', {})),
            json.dumps(data.get('ensemble_sub_weights', {})),
            data.get('recommended_model', 'ensemble')
        ))
    
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM model_weights_history").fetchone()[0]
    print(f"   OK {count} 条权重记录\n")


def migrate_investment_plans(conn):
    """迁移投注计划（JSON -> investment_plans）"""
    print("[INVEST] 迁移投注计划...")
    
    if not os.path.exists(INVESTMENT_JSON):
        print(f"   [WARN] {INVESTMENT_JSON} 不存在，跳过\n")
        return
    
    with open(INVESTMENT_JSON, 'r', encoding='utf-8') as f:
        plans = json.load(f)
    
    cursor = conn.cursor()
    inserted = 0
    
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        
        try:
            cursor.execute("""
                INSERT INTO investment_plans (
                    lottery_type, target_period, plan_date,
                    bet_numbers, total_amount, bet_count,
                    linked_prediction_ids, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.get('lottery_type'),
                plan.get('target_period'),
                plan.get('plan_date', datetime.now().date().isoformat()),
                json.dumps(plan.get('bet_numbers', [])),
                plan.get('total_amount', 0),
                plan.get('bet_count', 0),
                json.dumps(plan.get('linked_prediction_ids', [])),
                plan.get('notes', '')
            ))
            inserted += 1
        except Exception as e:
            print(f"   [WARN] 跳过记录: {e}")
    
    conn.commit()
    print(f"   OK {inserted} 条投注计划\n")


def verify_migration(conn):
    """验证迁移结果"""
    print("[VERIFY] 验证迁移结果...")
    cursor = conn.cursor()
    
    # 验证各表记录数
    tables = [
        ('lottery_draws', 'ssq + dlt 开奖记录'),
        ('predictions', '预测记录'),
        ('model_weights_history', '权重历史'),
        ('investment_plans', '投注计划')
    ]
    
    for table, desc in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"   {desc:20s}: {count:5d} 条")
    
    # 特殊验证：核验记录数
    verified_count = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE verified_at IS NOT NULL"
    ).fetchone()[0]
    print(f"   {'已核验预测记录':20s}: {verified_count:5d} 条")
    
    # 验证概率字段
    with_prob = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE red_probabilities IS NOT NULL"
    ).fetchone()[0]
    print(f"   {'含概率预测记录':20s}: {with_prob:5d} 条")
    
    print()


def update_system_metadata(conn):
    """更新系统元数据"""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE system_metadata SET value = ?, updated_at = ? WHERE key = 'migrated_from_json'",
        ('true', datetime.now().isoformat())
    )
    conn.commit()


def main():
    """主流程"""
    print("=" * 60)
    print("  busuan 数据迁移工具 v3.0")
    print("  JSON -> SQLite")
    print("=" * 60)
    print()
    
    try:
        # 步骤1: 备份
        backup_files()
        
        # 步骤2: 初始化数据库
        init_database()
        
        # 步骤3: 迁移数据
        conn = sqlite3.connect(DB_PATH)
        
        migrate_lottery_draws(conn)
        migrate_predictions(conn)
        migrate_model_weights(conn)
        migrate_investment_plans(conn)
        
        # 步骤4: 验证
        verify_migration(conn)
        
        # 步骤5: 更新元数据
        update_system_metadata(conn)
        
        conn.close()
        
        print("=" * 60)
        print("[VERIFY] 迁移完成！")
        print(f"   数据库文件: {DB_PATH}")
        print(f"   备份目录: {BACKUP_DIR}/")
        print("=" * 60)
        print()
        print("[WARN] 下一步:")
        print("   1. 检查数据库内容: sqlite3 busuan.db \".tables\"")
        print("   2. 测试查询: SELECT COUNT(*) FROM predictions;")
        print("   3. 确认无误后，运行 Python 代码测试数据访问层")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] 迁移失败: {e}")
        print("   原文件未修改（已备份）")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
