"""
胆拖预测 GUI 交互测试（QTest 模拟真实点击）

用法: QT_QPA_PLATFORM=offscreen python test_dantuo_gui.py
验证三种模式的完整交互流程：切卡片 → 设参数 → 点击生成预测 → 检查结果标签
（卡片式布局：单式/复式/胆拖 为 QGroupBox 可勾选互斥卡片）
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import torch  # noqa: F401  必须先于 PyQt5（DLL 冲突规避）

import sys
from PyQt5.QtWidgets import QApplication, QGroupBox
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from lottery_predictor_app_new import LotteryPredictorApp


def _select_card(win, title):
    """选中指定标题的模式卡片（互斥）"""
    cards = [g for g in win.findChildren(QGroupBox) if g.isCheckable()]
    for c in cards:
        if c.title() == title:
            c.setChecked(True)
            return True
    return False


def run_test():
    app = QApplication.instance() or QApplication(sys.argv)
    win = LotteryPredictorApp()
    win.show()
    QTest.qWait(200)  # 让窗口完成布局

    results = []

    # ---------- 测试 1：胆拖模式（SSQ + gbdt） ----------
    print("[1/3] 胆拖模式: SSQ + 梯度提升树")
    win.lottery_combo.setCurrentText("双色球")
    # 找到"梯度提升树"
    for i in range(win.model_combo.count()):
        if win.model_combo.itemText(i) == "梯度提升树":
            win.model_combo.setCurrentIndex(i)
            break
    _select_card(win, "胆拖")
    QTest.qWait(50)
    # 设胆拖参数：红胆2 / 红拖6 / 蓝拖2（SSQ 蓝胆应为 0）
    win.dt_red_dan.setValue(2)
    win.dt_red_tuo.setValue(6)
    win.dt_blue_dan.setValue(0)
    win.dt_blue_tuo.setValue(2)
    QTest.qWait(50)

    old_text = win.result_label.text()
    QTest.mouseClick(win.predict_button, Qt.LeftButton)
    # generate_prediction 同步执行，点击后直接读结果
    QTest.qWait(500)
    new_text = win.result_label.text()
    ok = ("胆拖预测" in new_text) and ("注" in new_text) and (new_text != old_text)
    print(f"    结果标签: {new_text[:80]}...")
    print(f"    验证: 包含'胆拖预测'={'胆拖预测' in new_text}, 包含'注'={'注' in new_text}, 变化={new_text != old_text}")
    results.append(("胆拖 SSQ", ok))

    # ---------- 测试 2：复式模式（SSQ + gbdt） ----------
    print("[2/3] 复式模式: SSQ + 梯度提升树")
    _select_card(win, "复式")
    QTest.qWait(50)
    win.compound_red_spin.setValue(8)
    win.compound_blue_spin.setValue(2)
    QTest.qWait(50)

    QTest.mouseClick(win.predict_button, Qt.LeftButton)
    QTest.qWait(500)
    new_text = win.result_label.text()
    ok = ("复式预测" in new_text) and ("注" in new_text)
    print(f"    结果标签: {new_text[:80]}...")
    print(f"    验证: 包含'复式预测'={'复式预测' in new_text}, 包含'注'={'注' in new_text}")
    results.append(("复式 SSQ", ok))

    # ---------- 测试 3：单式模式（SSQ + gbdt） ----------
    print("[3/3] 单式模式: SSQ + 梯度提升树")
    _select_card(win, "单式")
    QTest.qWait(50)
    QTest.mouseClick(win.predict_button, Qt.LeftButton)
    QTest.qWait(500)
    new_text = win.result_label.text()
    ok = ("预测" in new_text) and ("组" in new_text)
    print(f"    结果标签: {new_text[:80]}...")
    print(f"    验证: 包含'预测'={'预测' in new_text}, 包含'组'={'组' in new_text}")
    results.append(("单式 SSQ", ok))

    # ---------- 汇总 ----------
    print("\n========== 汇总 ==========")
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print(f"\n总计: {'全部 PASS' if all_ok else '有 FAIL'}")
    return all_ok


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
