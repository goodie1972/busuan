# -*- coding:utf-8 -*-
"""
模型文件完整性校验 - 对抗性测试
覆盖：空文件/截断/HTML伪装/超大/类型错位/正常加载/备份
"""
import os
import pickle
import tempfile
import struct
from model_integrity import check_pkl_file, safe_load_pkl, backup_model_before_overwrite


def _write_bytes(path, data):
    with open(path, 'wb') as f:
        f.write(data)


def test_missing_file():
    print("\n[TEST 1] 文件不存在")
    ok, msg = check_pkl_file('/nonexistent/model.pkl')
    assert not ok and '不存在' in msg, msg
    print("  [OK] 正确拒绝不存在的文件")


def test_empty_file():
    print("\n[TEST 2] 空文件（0字节）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'empty.pkl')
        _write_bytes(p, b'')
        ok, msg = check_pkl_file(p)
        assert not ok and '空文件' in msg, msg
        obj, msg2 = safe_load_pkl(p)
        assert obj is None and '空文件' in msg2, msg2
    print("  [OK] 正确拒绝空文件")


def test_truncated_file():
    print("\n[TEST 3] 截断文件（<100B）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'trunc.pkl')
        _write_bytes(p, b'\x80\x05partial')  # pickle 头但截断
        ok, msg = check_pkl_file(p)
        assert not ok and '过小' in msg, msg
    print("  [OK] 正确拒绝截断文件")


def test_html_disguised():
    print("\n[TEST 4] HTML 伪装为 pkl")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fake.pkl')
        _write_bytes(p, b'<!DOCTYPE html><html><body>404</body></html>' + b' ' * 500)
        ok, msg = check_pkl_file(p)
        assert not ok and 'HTML' in msg, msg
        obj, msg2 = safe_load_pkl(p)
        assert obj is None, "HTML 伪装文件不应被加载"
    print("  [OK] 正确拒绝 HTML 伪装文件")


def test_oversized_reject():
    print("\n[TEST 5] 超大文件拒绝（防内存炸弹）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'huge.pkl')
        # 写一个 >2GB 的稀疏文件（不占真实磁盘）
        with open(p, 'wb') as f:
            f.truncate(3 * 1024**3)
        ok, msg = check_pkl_file(p)
        assert not ok and '过大' in msg, msg
        os.remove(p)  # 清理稀疏文件
    print("  [OK] 正确拒绝超大文件")


def test_wrong_type_hint():
    print("\n[TEST 6] 类型错位检测（期望 scaler 实际 dict）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'mismatch.pkl')
        import joblib
        # 用一个足够大的 dict（>100B）触发类型错位检测
        big = {'unexpected': 'dict', 'data': list(range(50))}
        joblib.dump(big, p)
        obj, msg = safe_load_pkl(p, expected_hint='Scaler')
        # dict 在白名单豁免内，但 expected_hint 不匹配
        assert obj is None and '类型不符' in msg, msg
    print("  [OK] 正确检测类型错位")


def test_normal_load():
    print("\n[TEST 7] 正常文件加载（MinMaxScaler）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'scaler.pkl')
        import joblib
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(np.array([[1], [2], [3]]))
        joblib.dump(scaler, p)
        obj, msg = safe_load_pkl(p, expected_hint='MinMaxScaler')
        assert obj is not None, msg
        assert hasattr(obj, 'transform'), "应能调用 transform"
        assert 'OK' in msg, msg
    print("  [OK] 正常文件加载成功")


def test_backup_before_overwrite():
    print("\n[TEST 8] 覆盖前自动备份")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'model.pkl')
        _write_bytes(p, b'old model content here padded')
        bak = backup_model_before_overwrite(p)
        assert bak is not None and os.path.exists(bak), "应生成备份"
        # 再次覆盖
        _write_bytes(p, b'new model content here padded')
        bak2 = backup_model_before_overwrite(p)
        assert bak2 is not None
        # 清理保留逻辑：写3次后应≤2个备份
        for i in range(3):
            _write_bytes(p, f'v{i}'.encode() + b' ' * 100)
            backup_model_before_overwrite(p)
        dir_path = os.path.dirname(p)
        baks = [f for f in os.listdir(dir_path) if f.startswith('model.pkl.bak.')]
        assert len(baks) <= 2, f"应保留≤2个备份，实际{len(baks)}"
    print("  [OK] 备份机制正常")


def test_corrupted_pickle():
    print("\n[TEST 9] 损坏的 pickle（魔数正常但内容损坏）")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'corrupt.pkl')
        # pickle 协议头 \x80\x04 但后续是垃圾
        _write_bytes(p, b'\x80\x04' + b'garbage' * 100)
        obj, msg = safe_load_pkl(p)
        assert obj is None and '反序列化失败' in msg, msg
    print("  [OK] 正确处理损坏的 pickle")


if __name__ == '__main__':
    print("="*60)
    print(" 模型文件完整性校验 - 对抗性测试")
    print("="*60)
    test_missing_file()
    test_empty_file()
    test_truncated_file()
    test_html_disguised()
    test_oversized_reject()
    test_wrong_type_hint()
    test_normal_load()
    test_backup_before_overwrite()
    test_corrupted_pickle()
    print("\n" + "="*60)
    print(" [OK] 全部9项测试通过")
    print("="*60)
