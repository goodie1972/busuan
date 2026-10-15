"""
对抗性测试: JSON安全读写工具
测试线程安全、原子写入、损坏恢复、备份机制
"""
import json
import os
import shutil
import tempfile
import threading
from json_utils import safe_json_write, safe_json_load


def test_atomic_write():
    """测试原子写入（写入一半时"崩溃"也不留损坏文件）"""
    print("\n[TEST 1] 原子写入")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.json')
        
        # 1. 写入初始数据
        data1 = {'version': 1, 'records': [1, 2, 3]}
        assert safe_json_write(target, data1, backup=False)
        assert os.path.exists(target)
        
        # 2. 模拟写入"崩溃"（临时文件未被替换）
        # 正常情况下 os.replace 是原子的，这里测试即使tmp残留也不影响主文件
        tmp_file = target + '.tmp'
        with open(tmp_file, 'w') as f:
            f.write('{"incomplete')  # 损坏的临时文件
        
        # 主文件应仍然完好
        loaded = safe_json_load(target)
        assert loaded == data1, "主文件应不受损坏临时文件影响"
        
        # 3. 再次写入应成功
        data2 = {'version': 2, 'records': [4, 5]}
        assert safe_json_write(target, data2, backup=False)
        loaded = safe_json_load(target)
        assert loaded == data2
        
        print("  [OK] 原子写入机制正常")


def test_backup_and_recovery():
    """测试自动备份与损坏恢复"""
    print("\n[TEST 2] 备份与恢复")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.json')
        
        # 1. 写入第一版
        data1 = {'ver': 1}
        assert safe_json_write(target, data1, backup=True)
        
        # 2. 写入第二版（会自动备份第一版）
        data2 = {'ver': 2}
        assert safe_json_write(target, data2, backup=True)
        
        # 检查备份文件存在
        backups = [f for f in os.listdir(tmpdir) if f.startswith('test.json.bak.')]
        assert len(backups) >= 1, f"应有至少1个备份文件，实际: {backups}"
        
        # 3. 人为损坏主文件
        with open(target, 'w') as f:
            f.write('{"incomplete')
        
        # 4. 加载时应自动从备份恢复
        loaded = safe_json_load(target, default=None, try_backup=True)
        assert loaded is not None, "应从备份恢复"
        assert loaded['ver'] in [1, 2], f"恢复的数据应是v1或v2，实际: {loaded}"
        
        # 5. 恢复后主文件应被修复
        loaded_again = safe_json_load(target, try_backup=False)
        assert loaded_again == loaded, "主文件应已被修复"
        
        print("  [OK] 备份与恢复机制正常")


def test_thread_safety():
    """测试多线程并发写入（文件级锁）"""
    print("\n[TEST 3] 线程安全")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'concurrent.json')
        results = []
        
        def writer(thread_id):
            for i in range(10):
                data = {'thread': thread_id, 'count': i}
                success = safe_json_write(target, data, backup=False)
                results.append(success)
        
        # 启动10个线程同时写同一个文件
        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 检查：所有写入应成功，且最终文件不损坏
        assert all(results), f"所有写入应成功，失败数: {results.count(False)}"
        final_data = safe_json_load(target)
        assert final_data is not None, "并发写入后文件应完整"
        assert 'thread' in final_data and 'count' in final_data
        
        print("  [OK] 线程安全机制正常")


def test_default_fallback():
    """测试文件不存在时返回默认值"""
    print("\n[TEST 4] 默认值回退")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexist = os.path.join(tmpdir, 'nonexist.json')
        
        default = {'status': 'default'}
        loaded = safe_json_load(nonexist, default=default, try_backup=False)
        assert loaded == default, "不存在的文件应返回默认值"
        
        print("  [OK] 默认值回退正常")


def test_old_backup_cleanup():
    """测试旧备份自动清理（保留最近3个）"""
    print("\n[TEST 5] 旧备份清理")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.json')
        
        # 写入10次（每次都会备份）
        for i in range(10):
            data = {'version': i}
            assert safe_json_write(target, data, backup=True)
        
        # 检查备份数量
        backups = sorted([f for f in os.listdir(tmpdir) if f.startswith('test.json.bak.')])
        assert len(backups) <= 4, f"应保留≤4个备份（写入10次后），实际: {len(backups)}"
        
        print(f"  [OK] 旧备份清理正常（保留 {len(backups)} 个）")


def test_large_data():
    """测试大数据量（模拟475条预测记录）"""
    print("\n[TEST 6] 大数据量")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'big.json')
        
        # 生成475条模拟记录
        records = [
            {
                'lottery_type': 'ssq',
                'model_type': 'ensemble',
                'target_period': f'2026{i:04d}',
                'red_numbers': list(range(1, 7)),
                'blue_numbers': [1],
                'has_drawn': False
            }
            for i in range(475)
        ]
        
        # 写入
        assert safe_json_write(target, records, backup=True)
        
        # 加载
        loaded = safe_json_load(target)
        assert len(loaded) == 475, f"应加载475条记录，实际: {len(loaded)}"
        
        # 检查文件大小（应>100KB）
        size = os.path.getsize(target)
        assert size > 100_000, f"475条记录应>100KB，实际: {size}"
        
        print(f"  [OK] 大数据量正常（{size:,} bytes）")


if __name__ == '__main__':
    print("="*60)
    print(" JSON安全读写工具 - 对抗性测试")
    print("="*60)
    
    test_atomic_write()
    test_backup_and_recovery()
    test_thread_safety()
    test_default_fallback()
    test_old_backup_cleanup()
    test_large_data()
    
    print("\n" + "="*60)
    print(" [OK] 全部6项测试通过")
    print("="*60)
