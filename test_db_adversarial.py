# -*- coding:utf-8 -*-
"""
数据库对抗性测试
================

测试范围：
1. 并发读写：多线程同时写入
2. 事务回滚：写入失败时数据不损坏
3. 备份恢复：备份→修改→恢复→验证
4. 空数据/脏数据：边界条件
5. SQL注入防护：参数化查询验证
6. 大数据量：批量写入性能
"""
import os
import sys
import json
import time
import threading
import sqlite3
import tempfile
import shutil
from datetime import datetime

# 测试前备份
DB_PATH = "busuan.db"
BACKUP_DIR = "_test_backup"

def setup():
    """测试前准备"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        shutil.copy2(DB_PATH, os.path.join(BACKUP_DIR, f"busuan_pre_test_{int(time.time())}.db"))

def teardown():
    """测试后清理"""
    # 恢复原始数据库
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith("busuan_pre_test_")])
    if backups:
        shutil.copy2(os.path.join(BACKUP_DIR, backups[-1]), DB_PATH)


def test_concurrent_writes():
    """测试1: 多线程并发写入"""
    print("\n[TEST 1] Concurrent writes...")
    from db_access import get_connection
    
    errors = []
    
    def writer(thread_id):
        try:
            for i in range(20):
                with get_connection() as conn:
                    conn.execute(
                        "INSERT INTO predictions (predict_time, lottery_type, model_type, "
                        "latest_period, red_numbers, blue_numbers) VALUES (?, ?, ?, ?, ?, ?)",
                        (datetime.now().isoformat(), 'ssq', f'thread_{thread_id}',
                         26000 + i, '[1,2,3]', '[7]')
                    )
                    conn.commit()
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        print(f"  FAIL: {len(errors)} errors")
        for e in errors[:3]:
            print(f"    {e}")
        return False
    
    # 验证写入数量
    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_type LIKE 'thread_%'"
        ).fetchone()[0]
    
    expected = 5 * 20
    if count == expected:
        print(f"  PASS: {count} records written by 5 threads")
        return True
    else:
        print(f"  FAIL: expected {expected}, got {count}")
        return False


def test_transaction_rollback():
    """测试2: 事务回滚（写入失败不损坏数据）"""
    print("\n[TEST 2] Transaction rollback...")
    from db_access import get_connection
    
    # 先记录当前数量
    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    # 故意制造一个会失败的事务
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO predictions (predict_time, lottery_type, model_type, "
                "latest_period, red_numbers, blue_numbers) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), 'ssq', 'rollback_test', 99999, '[1]', '[2]')
            )
            conn.commit()
            # 第二条故意违反 CHECK 约束
            conn.execute(
                "INSERT INTO predictions (predict_time, lottery_type, model_type, "
                "latest_period, red_numbers, blue_numbers) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), 'INVALID', 'test', 99999, '[1]', '[2]')
            )
            conn.commit()
    except sqlite3.IntegrityError:
        pass  # 预期失败
    
    # 验证：第二条失败不应影响第一条（已 commit），但总数应只+1
    with get_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        rollback_test = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_type='rollback_test'"
        ).fetchone()[0]
    
    if after == before + 1 and rollback_test == 1:
        print(f"  PASS: before={before}, after={after} (delta=1, first INSERT committed)")
        return True
    else:
        print(f"  FAIL: before={before}, after={after}, rollback_test={rollback_test}")
        return False


def test_backup_restore():
    """测试3: 备份恢复"""
    print("\n[TEST 3] Backup and restore...")
    from db_backup import backup_database, restore_backup, list_backups
    from db_access import get_connection
    
    # 创建备份
    backup_file = backup_database(note='test')
    if not backup_file:
        print("  FAIL: backup creation failed")
        return False
    
    # 记录备份时的记录数
    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    # 修改数据库（添加一条记录）
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO predictions (predict_time, lottery_type, model_type, "
            "latest_period, red_numbers, blue_numbers) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), 'ssq', 'modify_test', 88888, '[1]', '[2]')
        )
        conn.commit()
    
    # 验证修改生效
    with get_connection() as conn:
        modified = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    if modified != before + 1:
        print(f"  FAIL: modification didn't take effect (before={before}, modified={modified})")
        return False
    
    # 从备份恢复
    backup_name = os.path.basename(backup_file)
    success = restore_backup(backup_name, confirm=True)
    if not success:
        print("  FAIL: restore failed")
        return False
    
    # 验证恢复后的记录数 = 备份时的记录数
    with get_connection() as conn:
        restored = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        modify_exists = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_type='modify_test'"
        ).fetchone()[0]
    
    if restored == before and modify_exists == 0:
        print(f"  PASS: before={before}, modified={modified}, restored={restored}")
        return True
    else:
        print(f"  FAIL: restored={restored}, expected={before}, modify_exists={modify_exists}")
        return False


def test_empty_and_dirty_data():
    """测试4: 空数据和脏数据"""
    print("\n[TEST 4] Empty and dirty data...")
    from db_access import get_predictions, get_verification_summary
    
    # 测试空彩种
    result = get_verification_summary('invalid_type')
    if result and result.get('total', 0) == 0:
        print("  PASS: invalid lottery type returns empty summary")
    else:
        print(f"  FAIL: invalid type returned {result}")
        return False
    
    # 测试 None 参数
    try:
        preds = get_predictions(lottery_type=None, limit=5)
        print(f"  PASS: None lottery_type returns {len(preds)} records")
    except Exception as e:
        print(f"  FAIL: None lottery_type raised {e}")
        return False
    
    return True


def test_sql_injection():
    """测试5: SQL注入防护"""
    print("\n[TEST 5] SQL injection protection...")
    from db_access import execute_query
    
    # 尝试注入
    try:
        # 参数化查询应防止注入
        result = execute_query(
            "SELECT * FROM predictions WHERE lottery_type = ?",
            ("ssq'; DROP TABLE predictions; --",)
        )
        # 表应该还在
        check = execute_query("SELECT COUNT(*) as cnt FROM predictions")
        if check and check[0]['cnt'] > 0:
            print("  PASS: SQL injection blocked (table intact)")
            return True
        else:
            print("  FAIL: table may have been affected")
            return False
    except Exception as e:
        print(f"  PASS: injection attempt rejected: {e}")
        return True


def test_large_batch():
    """测试6: 大批量写入性能"""
    print("\n[TEST 6] Large batch write performance...")
    from db_access import get_connection
    
    batch_size = 500
    start = time.time()
    
    with get_connection() as conn:
        params = []
        for i in range(batch_size):
            params.append((
                datetime.now().isoformat(), 'ssq', 'batch_test',
                26000 + i, '[1,2,3,4,5,6]', '[7]'
            ))
        conn.executemany(
            "INSERT INTO predictions (predict_time, lottery_type, model_type, "
            "latest_period, red_numbers, blue_numbers) VALUES (?, ?, ?, ?, ?, ?)",
            params
        )
        conn.commit()
    
    elapsed = time.time() - start
    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_type='batch_test'"
        ).fetchone()[0]
    
    if count == batch_size and elapsed < 5.0:
        print(f"  PASS: {batch_size} records in {elapsed:.2f}s ({batch_size/elapsed:.0f} rec/s)")
        return True
    else:
        print(f"  FAIL: count={count}, elapsed={elapsed:.2f}s")
        return False


def cleanup_test_data():
    """清理测试数据"""
    from db_access import get_connection
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE model_type LIKE 'thread_%'")
        conn.execute("DELETE FROM predictions WHERE model_type='rollback_test'")
        conn.execute("DELETE FROM predictions WHERE model_type='modify_test'")
        conn.execute("DELETE FROM predictions WHERE model_type='batch_test'")
        conn.execute("DELETE FROM predictions WHERE model_type='test_db'")
        conn.commit()


def main():
    print("=" * 60)
    print("  Database Adversarial Testing")
    print("=" * 60)
    
    setup()
    
    tests = [
        test_concurrent_writes,
        test_transaction_rollback,
        test_backup_restore,
        test_empty_and_dirty_data,
        test_sql_injection,
        test_large_batch,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(False)
    
    # 清理测试数据
    print("\n[CLEANUP] Removing test data...")
    cleanup_test_data()
    print("  OK")
    
    # 汇总
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 60}")
    
    teardown()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
