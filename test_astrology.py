# -*- coding: utf-8 -*-
"""
离线冒烟测试：紫微斗数 + 梅花易数（不导入 torch/main app）
测试范围：
  1. 引擎独立运行（3组号码、不重复、有推演）
  2. UI 组件创建函数（create_ziwei_tab / create_meihua_tab）
  3. 槽函数逻辑（直接调用引擎模拟 UI 行为）
"""
import sys, os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, '.')

import traceback

def test_engines():
    """测试紫微和梅花引擎"""
    from astrology.ziwei.engine import predict as ziwei_predict
    from astrology.meihua.engine import predict as meihua_predict

    birth = {'year': 1985, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30, 'city': '北京', 'gender': '男'}
    info = {'mode': 'time', 'year': 2026, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30}

    # ===== 紫微 SSQ =====
    ziwei_reds = []
    for v in range(3):
        r, b, interp = ziwei_predict('ssq', 33, 16, 6, 1, birth, target_period=26110, variation=v)
        assert len(r) == 6, f"SSQ red count != 6: {r}"
        assert len(b) == 1, f"SSQ blue count != 1: {b}"
        assert all(1 <= n <= 33 for n in r), f"Red out of range: {r}"
        assert 1 <= b[0] <= 16, f"Blue out of range: {b}"
        assert len(interp) > 10, "Interpretation too short"
        ziwei_reds.append(tuple(r))
    assert len(set(ziwei_reds)) == 3, f"3 groups not unique: {ziwei_reds}"
    print("  紫微 SSQ: 3组唯一 ✓")

    # ===== 紫微 DLT =====
    r, b, _ = ziwei_predict('dlt', 35, 12, 5, 2, birth, target_period=26110, variation=0)
    assert len(r) == 5 and len(b) == 2
    assert all(1 <= n <= 35 for n in r) and all(1 <= n <= 12 for n in b)
    print("  紫微 DLT: 5红2蓝 ✓")

    # ===== 梅花 SSQ =====
    meihua_reds = []
    for v in range(3):
        r, b, interp = meihua_predict('ssq', 33, 16, 6, 1, info, target_period=26110, variation=v)
        assert len(r) == 6 and len(b) == 1
        assert all(1 <= n <= 33 for n in r)
        assert 1 <= b[0] <= 16
        assert len(interp) > 10
        meihua_reds.append(tuple(r))
    assert len(set(meihua_reds)) == 3, f"3 groups not unique: {meihua_reds}"
    print("  梅花 SSQ: 3组唯一 ✓")

    # ===== 梅花 DLT =====
    r, b, _ = meihua_predict('dlt', 35, 12, 5, 2, info, target_period=26110, variation=0)
    assert len(r) == 5 and len(b) == 2
    assert all(1 <= n <= 35 for n in r) and all(1 <= n <= 12 for n in b)
    print("  梅花 DLT: 5红2蓝 ✓")

    # ===== 梅花数字起卦 =====
    num_info = {'mode': 'number', 'num1': 3, 'num2': 8, 'num3': 15,
                'year': 2026, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30}
    r, b, interp = meihua_predict('ssq', 33, 16, 6, 1, num_info, target_period=26110, variation=2)
    assert len(r) == 6 and len(b) == 1
    assert '外应' in interp or '数字' in interp
    print("  梅花数字起卦 ✓")

    return True


def test_ui_components():
    """测试 UI 组件创建函数"""
    from PyQt5.QtWidgets import QApplication, QWidget
    app = QApplication.instance() or QApplication([])

    from ui_components import create_ziwei_tab, create_meihua_tab

    # ===== 紫微 tab =====
    zw_widget = QWidget()
    (zw_predict, zw_reset, zw_name, zw_gender, zw_year, zw_month,
     zw_day, zw_hour, zw_minute, zw_city, zw_target, zw_cards) = create_ziwei_tab(zw_widget)

    assert zw_predict.text() == "紫微排盘·取数"
    assert zw_year.value() == 1990
    assert zw_city.currentText() == "北京"
    assert len(zw_cards) == 3
    for i, (rl, bl, dl) in enumerate(zw_cards):
        assert "待排盘" in rl.text(), f"Card {i} should show 待排盘"
    print("  紫微 UI 组件: 11输入 + 3卡片 ✓")

    # ===== 梅花 tab =====
    mh_widget = QWidget()
    (mh_predict, mh_reset, mh_mode, mh_year, mh_month, mh_day,
     mh_hour, mh_minute, mh_num1, mh_num2, mh_num3, mh_target, mh_cards) = create_meihua_tab(mh_widget)

    assert mh_predict.text() == "梅花起卦·取数"
    assert mh_mode.currentText() == "时间起卦"
    assert mh_year.value() == 2026
    assert len(mh_cards) == 3
    for i, (rl, bl, dl) in enumerate(mh_cards):
        assert "待起卦" in rl.text()
    print("  梅花 UI 组件: 12输入 + 3卡片 ✓")

    return True


def test_slot_logic():
    """模拟槽函数逻辑（直接调用引擎，验证 UI 更新流程）"""
    from PyQt5.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])

    from astrology.ziwei.engine import predict as ziwei_predict
    from astrology.meihua.engine import predict as meihua_predict

    # 模拟 UI 卡片
    cards = [(QLabel("红球: 待排盘"), QLabel("蓝球: 待排盘"), QLabel("推演: 待排盘"))
             for _ in range(3)]

    birth = {'year': 1985, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30, 'city': '北京', 'gender': '男'}
    for v in range(3):
        r, b, interp = ziwei_predict('ssq', 33, 16, 6, 1, birth, target_period=26110, variation=v)
        rl, bl, dl = cards[v]
        rl.setText(f"红球: {' '.join(f'{n:02d}' for n in r)}")
        bl.setText(f"蓝球: {b[0]:02d}")
        dl.setText(interp)

    for i, (rl, bl, dl) in enumerate(cards):
        assert "待排盘" not in rl.text(), f"Card {i} not updated"
        assert "红球:" in rl.text()
        assert len(dl.text()) > 10

    # 验证3组号码不同
    reds = [cards[i][0].text() for i in range(3)]
    assert len(set(reds)) == 3, f"3 groups not unique: {reds}"
    print("  槽函数逻辑: 紫微3组更新+唯一 ✓")

    return True


def test_ml_models_integration():
    """测试 ml_models.py 调用新引擎"""
    from ml_models import LotteryMLModels
    import pandas as pd

    df = pd.DataFrame({'期数': [26110]}, columns=['期数'])

    m = LotteryMLModels('ssq', 'ziwei')
    m.load_models()
    results = []
    for i in range(3):
        r, b = m.predict(df, variation=i)
        assert len(r) == 6 and len(b) == 1
        assert all(1 <= n <= 33 for n in r)
        results.append(tuple(r))
    assert len(set(results)) == 3, f"Not unique: {results}"
    assert hasattr(m, '_astrology_interp'), "Missing _astrology_interp"
    assert len(m._astrology_interp) >= 3
    print("  ml_models 紫微: 3组唯一+推演 ✓")

    m2 = LotteryMLModels('ssq', 'meihua')
    m2.load_models()
    results2 = []
    for i in range(3):
        r, b = m2.predict(df, variation=i)
        assert len(r) == 6 and len(b) == 1
        results2.append(tuple(r))
    assert len(set(results2)) == 3, f"Not unique: {results2}"
    print("  ml_models 梅花: 3组唯一+推演 ✓")

    return True


if __name__ == '__main__':
    tests = [
        ("引擎独立测试", test_engines),
        ("UI 组件创建", test_ui_components),
        ("槽函数逻辑", test_slot_logic),
        ("ml_models 集成", test_ml_models_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            passed += 1
            print(f"  → PASS")
        except Exception as e:
            failed += 1
            print(f"  → FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*40}")
    print(f"总计: {passed} PASS / {failed} FAIL")
    if failed == 0:
        print("===== ALL TESTS PASSED =====")
    sys.exit(0 if failed == 0 else 1)
