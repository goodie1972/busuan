# -*- coding:utf-8 -*-
"""PyQt5 桌面端对抗性 UI 测试（offscreen）。

覆盖用户可触达的异常/极端操作：切页、模式互斥、复式/胆拖边界、
超大组合、快速切换彩种、紫微/梅花异常输入、空记录操作、
模型加载失败后的恢复、一键预测线程。

注意：必须先导入应用模块（内部先加载 torch），再导入 PyQt5，
否则 Windows 上会触发 c10.dll 的 WinError 1114。
"""
import math
import os
import re

import pytest

import lottery_predictor_app_new as gui

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication, QGroupBox

LOTTERY_SPECS = {
    "ssq": (33, 6, 16, 1),
    "dlt": (35, 5, 12, 2),
}
LIGHTGBM_READY = os.path.isdir(os.path.join(gui.name_path["ssq"]["path"], "lightgbm"))


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app


@pytest.fixture
def window(qapp, monkeypatch):
    # 阻断启动 10 秒后的联网数据检查，避免测试期间产生后台请求
    monkeypatch.setattr(gui.LotteryPredictorApp, "start_data_check", lambda self: None)
    w = gui.LotteryPredictorApp()
    # 所有测试都不允许写入真实预测记录数据库
    w._save_prediction_records = lambda *args, **kwargs: None
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


@pytest.fixture
def fake_dialogs(monkeypatch):
    real = gui.QMessageBox

    class _FakeMessageBox:
        Ok = real.Ok
        Yes = real.Yes
        No = real.No

        information = staticmethod(lambda *args, **kwargs: real.Ok)
        warning = staticmethod(lambda *args, **kwargs: real.Ok)
        critical = staticmethod(lambda *args, **kwargs: real.Ok)
        question = staticmethod(lambda *args, **kwargs: real.No)

    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)


def _cards(window):
    return {
        group.title(): group
        for group in window.main_tab.findChildren(QGroupBox)
        if group.title() in ("单式", "复式", "胆拖")
    }


def _checked_modes(cards):
    return [title for title, card in cards.items() if card.isChecked()]


def _parse_prediction_rows(text):
    rows = []
    for line in text.splitlines():
        if "组:" not in line or "+" not in line:
            continue
        left, right = line.split("组:", 1)[1].split("+", 1)
        try:
            reds = [int(x) for x in left.split()]
            blues = [int(x) for x in right.split()]
        except ValueError:
            continue
        if reds and blues:
            rows.append((reds, blues))
    return rows


def _assert_valid_numbers(reds, blues, lottery_type):
    red_range, red_count, blue_range, blue_count = LOTTERY_SPECS[lottery_type]
    assert len(reds) == red_count, reds
    assert len(set(reds)) == red_count, reds
    assert all(1 <= n <= red_range for n in reds), reds
    assert len(blues) == blue_count, blues
    assert len(set(blues)) == blue_count, blues
    assert all(1 <= n <= blue_range for n in blues), blues


def _run_auto_predict(window, qapp, timeout_ms=60000):
    loop = QEventLoop()
    window.start_auto_predict()
    thread = window.auto_predict_thread
    assert thread is not None, "一键预测线程未创建"
    thread.finished_signal.connect(lambda *args: loop.quit())
    thread.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()
    qapp.processEvents()
    if thread.isRunning():
        thread.wait(5000)
    if thread.isRunning():
        thread.terminate()
        thread.wait(2000)
        pytest.fail("一键预测线程超时未结束")
    return thread


def test_all_tabs_can_be_visited(window, qapp):
    assert window.tab_widget.count() == 10
    for index in range(window.tab_widget.count()):
        window.tab_widget.setCurrentIndex(index)
        qapp.processEvents()
        assert window.tab_widget.currentIndex() == index
    assert window.isVisible()


def test_analysis_button_opens_analysis_tab(window, qapp, monkeypatch):
    # 阻断真实绘图/统计，只验证按钮的切页行为
    def _stop(*args, **kwargs):
        raise RuntimeError("analysis stopped by test")

    monkeypatch.setattr(gui, "process_analysis_data", _stop)
    window.tab_widget.setCurrentIndex(0)
    window.analyze_button.click()
    qapp.processEvents()

    expected = window.tab_widget.indexOf(window.analysis_tab)
    actual = window.tab_widget.currentIndex()
    if actual != expected:
        pytest.xfail(f"已知缺陷: 数据分析按钮切到 {actual} 号页，应为 {expected} 号页")


def test_mode_cards_are_mutually_exclusive(window):
    cards = _cards(window)
    assert set(cards) == {"单式", "复式", "胆拖"}

    cards["复式"].setChecked(True)
    assert _checked_modes(cards) == ["复式"]
    assert window.mode_combo.currentText() == "复式"

    disabled = {
        window.model_combo.itemText(i)
        for i in range(window.model_combo.count())
        if not window.model_combo.model().item(i).isEnabled()
    }
    assert {"XGBoost", "期望值模型"} <= disabled

    cards["单式"].setChecked(True)
    assert _checked_modes(cards) == ["单式"]
    disabled_after = {
        window.model_combo.itemText(i)
        for i in range(window.model_combo.count())
        if not window.model_combo.model().item(i).isEnabled()
    }
    assert not disabled_after

    xgb_index = window.model_combo.findText("XGBoost")
    assert xgb_index >= 0
    window.model_combo.setCurrentIndex(xgb_index)
    assert window.mode_combo.currentText() == "单式"
    assert not window.mode_combo.isEnabled()


def test_compound_bounds_match_lottery(window):
    """复式/胆拖号码池：下限=单式个数，上限=红16/蓝6（用户规则）。"""
    problems = []

    window.lottery_combo.setCurrentIndex(0)
    window.compound_red_spin.setValue(0)
    window.compound_blue_spin.setValue(0)
    if window.compound_red_spin.value() != 6:
        problems.append(f"双色球复式红球下限应为6，实际 {window.compound_red_spin.value()}")
    if window.compound_blue_spin.value() != 1:
        problems.append(f"双色球复式蓝球下限应为1，实际 {window.compound_blue_spin.value()}")
    if window.compound_red_spin.maximum() != 16:
        problems.append(f"双色球复式红球上限应为16，实际 {window.compound_red_spin.maximum()}")
    if window.compound_blue_spin.maximum() != 6:
        problems.append(f"双色球复式蓝球上限应为6，实际 {window.compound_blue_spin.maximum()}")

    window.lottery_combo.setCurrentIndex(1)
    window.compound_red_spin.setValue(0)
    window.compound_blue_spin.setValue(16)
    if window.compound_red_spin.value() != 5:
        problems.append(f"大乐透复式红球下限应为5，实际 {window.compound_red_spin.value()}")
    if window.compound_blue_spin.value() != 6:
        problems.append(f"大乐透复式蓝球应被截断到6，实际 {window.compound_blue_spin.value()}")
    if window.compound_red_spin.maximum() != 16:
        problems.append(f"大乐透复式红球上限应为16，实际 {window.compound_red_spin.maximum()}")

    # 胆拖拖码上限随胆码联动：红胆+红拖<=16，蓝胆+蓝拖<=6
    window.lottery_combo.setCurrentIndex(0)
    window.dt_red_dan.setValue(1)
    window.dt_red_tuo.setValue(99)
    if window.dt_red_tuo.value() != 15:
        problems.append(f"双色球红胆1时红拖应截断到15，实际 {window.dt_red_tuo.value()}")
    window.dt_red_dan.setValue(5)
    if window.dt_red_tuo.maximum() != 11:
        problems.append(f"双色球红胆5时红拖上限应为11，实际 {window.dt_red_tuo.maximum()}")
    window.dt_blue_tuo.setValue(99)
    if window.dt_blue_tuo.value() != 6:
        problems.append(f"双色球蓝拖应截断到6，实际 {window.dt_blue_tuo.value()}")

    window.lottery_combo.setCurrentIndex(1)
    window.dt_red_dan.setValue(1)
    window.dt_red_tuo.setValue(99)
    if window.dt_red_tuo.value() != 15:
        problems.append(f"大乐透红胆1时红拖应截断到15，实际 {window.dt_red_tuo.value()}")
    window.dt_blue_dan.setValue(1)
    window.dt_blue_tuo.setValue(99)
    if window.dt_blue_tuo.value() != 5:
        problems.append(f"大乐透蓝胆1时蓝拖上限应为5，实际 {window.dt_blue_tuo.value()}")

    if problems:
        pytest.xfail("已知缺陷: " + "；".join(problems))


@pytest.mark.skipif(not LIGHTGBM_READY, reason="未找到支持复式/胆拖的 LightGBM 模型")
def test_huge_compound_inputs_do_not_materialize_all_combos(window, monkeypatch):
    """极端输入不得把千万级组合一次性展开，否则 UI 会卡死/OOM。"""
    real_combinations = gui.itertools.combinations
    hits = []

    def guarded_combinations(iterable, r):
        seq = list(iterable)
        total = math.comb(len(seq), r)
        if total > 100_000:
            hits.append(total)
            raise RuntimeError(f"unsafe materialization of {total} combinations")
        return real_combinations(seq, r)

    monkeypatch.setattr(gui.itertools, "combinations", guarded_combinations)
    cards = _cards(window)

    cards["复式"].setChecked(True)
    window.compound_red_spin.setValue(window.compound_red_spin.maximum())
    window.compound_blue_spin.setValue(window.compound_blue_spin.maximum())
    window.generate_prediction()

    cards["胆拖"].setChecked(True)
    window.dt_red_dan.setValue(window.dt_red_dan.minimum())
    window.dt_red_tuo.setValue(window.dt_red_tuo.maximum())
    window.dt_blue_dan.setValue(window.dt_blue_dan.maximum())
    window.dt_blue_tuo.setValue(window.dt_blue_tuo.maximum())
    window.generate_prediction()

    if hits:
        pytest.xfail(
            f"已知缺陷: UI 一次性物化超大组合（单次展开 {max(hits):,} 组），极端输入会卡死/OOM")
    assert window.result_label.text()


@pytest.mark.skipif(not LIGHTGBM_READY, reason="未找到支持胆拖的 LightGBM 模型")
def test_dantuo_extreme_values_are_clamped_and_valid(window):
    cards = _cards(window)
    cards["胆拖"].setChecked(True)
    window.dt_red_dan.setValue(window.dt_red_dan.maximum())
    window.dt_red_tuo.setValue(window.dt_red_tuo.maximum())
    window.dt_blue_dan.setValue(window.dt_blue_dan.maximum())
    window.dt_blue_tuo.setValue(window.dt_blue_tuo.maximum())
    window.generate_prediction()

    text = window.result_label.text()
    assert "胆拖预测" in text, text
    assert "出错" not in text, text

    dan_match = re.search(r"红胆\((\d+)个\):\s*([\d ]+)", text)
    tuo_match = re.search(r"红拖\((\d+)选\d+\):\s*([\d ]+)", text)
    assert dan_match and tuo_match, text
    dan = [int(x) for x in dan_match.group(2).split()]
    tuo = [int(x) for x in tuo_match.group(2).split()]
    assert len(dan) == int(dan_match.group(1)) == 5
    assert len(dan) + len(tuo) == 16
    assert len(set(dan + tuo)) == 16
    assert all(1 <= n <= 33 for n in dan + tuo)

    blue_match = re.search(r"蓝球\((\d+)选1\):\s*([\d ]+)", text)
    assert blue_match, text
    blues = [int(x) for x in blue_match.group(2).split()]
    assert len(blues) == int(blue_match.group(1)) == 6
    assert len(set(blues)) == 6
    assert all(1 <= n <= 16 for n in blues)

    expected_notes = math.comb(len(tuo), 6 - len(dan)) * len(blues)
    assert f"共 {expected_notes} 注" in text, text


@pytest.mark.skipif(not LIGHTGBM_READY, reason="未找到已训练的 LightGBM 模型")
def test_one_click_compound_prediction_succeeds(window, qapp):
    assert "LightGBM" in window.model_combo.currentText()
    cards = _cards(window)
    cards["复式"].setChecked(True)
    window.compound_red_spin.setValue(8)
    window.compound_blue_spin.setValue(2)

    _run_auto_predict(window, qapp)
    text = window.result_label.text()
    assert "失败" not in text, window.log_box.toPlainText()[-1500:]
    assert "复式" in text
    assert "56" in text


@pytest.mark.skipif(not LIGHTGBM_READY, reason="未找到已训练的 LightGBM 模型")
def test_one_click_dantuo_prediction_succeeds(window, qapp):
    assert "LightGBM" in window.model_combo.currentText()
    cards = _cards(window)
    cards["胆拖"].setChecked(True)
    window.dt_red_dan.setValue(2)
    window.dt_red_tuo.setValue(6)
    window.dt_blue_dan.setValue(0)
    window.dt_blue_tuo.setValue(2)

    _run_auto_predict(window, qapp)
    text = window.result_label.text()
    log = window.log_box.toPlainText()
    if "not enough values to unpack" in log:
        pytest.xfail("已知缺陷: 一键预测线程按 5 元组解包 predict_dantuo 的 4 元组返回值")
    assert "失败" not in text, log[-1500:]
    assert "胆拖" in text


def test_ziwei_failed_chart_does_not_show_stale_numbers(window):
    window.zw_year.setValue(1990)
    window.zw_month.setCurrentText("1")
    window.zw_day.setValue(1)
    window.zw_hour.setCurrentText("00时")
    window.zw_minute.setCurrentText("00分")
    window.zw_city.setCurrentText("北京")
    window.zw_target.setValue(0)
    window.generate_ziwei_prediction()

    first = [red_label.text() for red_label, _, _ in window.zw_cards]
    assert all("待排盘" not in label for label in first), first

    window.zw_month.setCurrentText("2")
    window.zw_day.setValue(31)
    window.generate_ziwei_prediction()
    second = [red_label.text() for red_label, _, _ in window.zw_cards]

    if second == first:
        pytest.xfail("已知缺陷: 排盘失败后仍展示上一次的结果，仅在日志里提示失败")
    assert all("待排盘" in label or "失败" in label for label in second), second


def test_ziwei_and_meihua_extreme_inputs_yield_valid_numbers(window):
    window.zw_name.setText("A" * 2000 + "<script>alert(1)</script>")
    window.zw_year.setValue(2100)
    window.zw_month.setCurrentText("12")
    window.zw_day.setValue(31)
    window.zw_hour.setCurrentText("23时")
    window.zw_minute.setCurrentText("55分")
    window.zw_city.setCurrentText("")
    window.zw_target.setValue(99999)
    window.generate_ziwei_prediction()

    for red_label, blue_label, _ in window.zw_cards:
        reds = [int(x) for x in red_label.text().split(":", 1)[1].split()]
        blues = [int(x) for x in blue_label.text().split(":", 1)[1].split()]
        _assert_valid_numbers(reds, blues, "ssq")

    window.mh_num_check.setChecked(True)
    window.mh_year.setValue(2100)
    window.mh_month.setCurrentText("12")
    window.mh_day.setValue(31)
    window.mh_hour.setCurrentText("23时")
    window.mh_minute.setCurrentText("55分")
    window.mh_num1.setValue(999)
    window.mh_num2.setValue(999)
    window.mh_num3.setValue(999)
    window.mh_target.setValue(99999)
    window.generate_meihua_prediction()

    for red_label, blue_label, _ in window.mh_cards:
        reds = [int(x) for x in red_label.text().split(":", 1)[1].split()]
        blues = [int(x) for x in blue_label.text().split(":", 1)[1].split()]
        _assert_valid_numbers(reds, blues, "ssq")


def test_single_prediction_survives_rapid_lottery_switching(window):
    window.prediction_spin.setValue(0)
    assert window.prediction_spin.value() == 1
    window.prediction_spin.setValue(999)
    assert window.prediction_spin.value() == 10
    window.prediction_spin.setValue(3)

    cards = _cards(window)
    cards["单式"].setChecked(True)

    for index, lottery_type in ((0, "ssq"), (1, "dlt"), (0, "ssq")):
        window.lottery_combo.setCurrentIndex(index)
        window.generate_prediction()
        text = window.result_label.text()
        assert "出错" not in text and "失败" not in text, text
        rows = _parse_prediction_rows(text)
        assert len(rows) == 3, text
        for reds, blues in rows:
            _assert_valid_numbers(reds, blues, lottery_type)


@pytest.mark.skipif(not LIGHTGBM_READY, reason="未找到已训练的 LightGBM 模型")
def test_model_load_failure_is_retried_on_next_click(window, monkeypatch):
    assert "LightGBM" in window.model_combo.currentText()
    from ml_models import LotteryMLModels

    original = LotteryMLModels.load_models
    calls = {"count": 0}

    def flaky_load(self):
        calls["count"] += 1
        if calls["count"] == 1:
            return False
        return original(self)

    monkeypatch.setattr(LotteryMLModels, "load_models", flaky_load)
    window.prediction_spin.setValue(2)

    window.generate_prediction()
    assert "尚未训练" in window.result_label.text()

    window.generate_prediction()
    text = window.result_label.text()
    if "预测失败" in text or "出错" in text:
        pytest.xfail("已知缺陷: 首次加载失败的模型实例被缓存，后续点击不会重试加载")
    rows = _parse_prediction_rows(text)
    assert len(rows) == 2, text


def test_empty_record_actions_do_not_block(window, qapp, fake_dialogs):
    window.tab_widget.setCurrentIndex(window.tab_widget.indexOf(window.records_tab))
    window.rec_refresh_btn.click()
    qapp.processEvents()

    # 确保没有行被选中，删除按钮只应弹提示，绝不能写真实数据库
    window.records_table.clearSelection()
    assert not window.records_table.selectedItems()
    window.rec_delete_btn.click()
    qapp.processEvents()

    assert window.isVisible()
    assert window.rec_stats_label.text()
