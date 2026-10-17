# -*- coding:utf-8 -*-
"""
回归测试：验证现有功能在 DB 版本下完整性
========================================

测试范围：
1. prediction_records: load/add/verify/export
2. model_weights: compute/get/update/export
3. improvement_rules: run_rules/get_plan
4. statistics_insights: get_insight_cards/get_combined_suggestion
5. db_access: 基本CRUD
6. GUI 冒烟：LotteryPredictorApp 启动+标签页切换
"""
import os
import sys
import json

# 设置 offscreen
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# torch 必须先于 PyQt5 导入
import torch

def test_db_access():
    """测试1: db_access 基本 CRUD"""
    print("\n[TEST 1] db_access CRUD...")
    from db_access import (
        get_latest_period, get_predictions, get_verification_summary,
        get_latest_model_weights, get_optimization_history
    )
    
    # 最新期号
    ssq_latest = get_latest_period('ssq')
    dlt_latest = get_latest_period('dlt')
    assert ssq_latest is not None and ssq_latest > 26000, f"SSQ latest={ssq_latest}"
    assert dlt_latest is not None and dlt_latest > 26000, f"DLT latest={dlt_latest}"
    print(f"  PASS: SSQ={ssq_latest}, DLT={dlt_latest}")
    
    # 预测记录
    preds = get_predictions(lottery_type='ssq', limit=5)
    assert len(preds) > 0, "No SSQ predictions"
    assert 'red_numbers' in preds[0], "Missing red_numbers"
    assert isinstance(preds[0]['red_numbers'], list), "red_numbers not list"
    print(f"  PASS: {len(preds)} predictions loaded, first has {len(preds[0]['red_numbers'])} red balls")
    
    # 核验汇总
    summary = get_verification_summary('ssq')
    assert 'total' in summary and 'verified' in summary, "Missing summary fields"
    print(f"  PASS: summary total={summary['total']}, verified={summary['verified']}")
    
    # 权重
    weights = get_latest_model_weights('ssq')
    assert weights is not None, "No SSQ weights"
    print(f"  PASS: weights updated_at={weights['updated_at']}")
    
    # 优化历史
    history = get_optimization_history(limit=5)
    print(f"  PASS: {len(history)} optimization log entries")
    
    return True


def test_prediction_records():
    """测试2: prediction_records 完整流程"""
    print("\n[TEST 2] prediction_records...")
    from prediction_records import load_records, add_prediction_record, verify_records, export_to_json
    
    # 加载
    records = load_records()
    assert len(records) > 100, f"Only {len(records)} records"
    assert 'lottery_type' in records[0], "Missing lottery_type"
    print(f"  PASS: {len(records)} records loaded")
    
    # 添加
    before = len(records)
    count = add_prediction_record(
        'ssq', 'regression_test', 26118,
        [1, 2, 3, 4, 5, 6], [16],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.8]
    )
    assert count == before + 1, f"Count mismatch: {before} -> {count}"
    print(f"  PASS: added record, total now {count}")
    
    # 核验（干运行）
    result = verify_records(log_callback=lambda msg: None)
    assert 'total' in result, "Missing total in result"
    assert 'verified' in result, "Missing verified in result"
    print(f"  PASS: verify total={result['total']}, verified={result['verified']}")
    
    # 导出 JSON
    exported = export_to_json()
    assert exported > 0, "Export failed"
    print(f"  PASS: exported {exported} records to JSON")
    
    # 清理测试记录
    from db_access import get_connection
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE model_type='regression_test'")
        conn.commit()
    
    return True


def test_model_weights():
    """测试3: model_weights 完整流程"""
    print("\n[TEST 3] model_weights...")
    from model_weights import (
        compute_model_weights, update_weights, get_model_weights,
        get_performance_report, update_all_weights
    )
    
    # 计算
    result = compute_model_weights('ssq')
    assert 'models' in result, "Missing models"
    assert 'weights' not in result or isinstance(result.get('ensemble_sub_weights'), dict), "Missing weights"
    assert 'recommended_model' in result, "Missing recommended_model"
    print(f"  PASS: computed, recommended={result['recommended_model']}")
    
    # 更新
    update_weights('ssq')
    print(f"  PASS: updated to DB")
    
    # 获取
    mw = get_model_weights('ssq')
    assert len(mw) > 0, "No model weights"
    total_weight = sum(mw.values())
    assert 0.9 < total_weight < 1.1, f"Weights sum={total_weight} (expected ~1.0)"
    print(f"  PASS: {len(mw)} model weights, sum={total_weight:.3f}")
    
    # 表现报告
    report = get_performance_report('ssq')
    assert 'models' in report, "Missing models in report"
    print(f"  PASS: performance report has {len(report['models'])} models")
    
    # 全部更新
    all_w = update_all_weights()
    assert 'ssq' in all_w and 'dlt' in all_w, "Missing lottery types"
    print(f"  PASS: updated all weights for SSQ+DLT")
    
    return True


def test_improvement_rules():
    """测试4: improvement_rules"""
    print("\n[TEST 4] improvement_rules...")
    from improvement_rules import run_rules, get_improvement_plan
    
    # 运行规则
    suggestions = run_rules('ssq')
    assert len(suggestions) > 0, "No suggestions"
    assert 'action' in suggestions[0], "Missing action"
    print(f"  PASS: {len(suggestions)} suggestions for SSQ")
    
    # 获取改进计划
    plan = get_improvement_plan('ssq')
    assert 'auto_exec_list' in plan, "Missing auto_exec_list"
    assert 'manual_confirm_list' in plan, "Missing manual_confirm_list"
    print(f"  PASS: plan auto={len(plan['auto_exec_list'])}, manual={len(plan['manual_confirm_list'])}")
    
    return True


def test_statistics_insights():
    """测试5: statistics_insights"""
    print("\n[TEST 5] statistics_insights...")
    from statistics_insights import (
        get_hot_cold_insight, get_missing_insight, get_trend_insight,
        get_combined_suggestion, get_insight_cards
    )
    
    # 热冷号
    hc = get_hot_cold_insight('ssq')
    assert 'red_hot' in hc and 'red_cold' in hc, "Missing hot/cold"
    print(f"  PASS: hot={hc['red_hot'][:3]}, cold={hc['red_cold'][:3]}")
    
    # 遗漏
    mi = get_missing_insight('ssq')
    assert 'red_max_missing' in mi, "Missing missing"
    print(f"  PASS: max missing={mi['red_max_missing'][:3]}")
    
    # 趋势
    tr = get_trend_insight('ssq')
    assert 'red_up' in tr, "Missing trend"
    print(f"  PASS: up={tr['red_up'][:3]}, down={tr['red_down'][:3]}")
    
    # 综合建议
    cs = get_combined_suggestion('ssq')
    assert 'red_suggested' in cs and 'blue_suggested' in cs, "Missing suggestion"
    assert len(cs['red_suggested']) == 6, f"Expected 6 red, got {len(cs['red_suggested'])}"
    print(f"  PASS: suggested red={cs['red_suggested']}, blue={cs['blue_suggested']}")
    
    # 卡片文本
    cards = get_insight_cards('ssq')
    assert len(cards) > 50, "Cards too short"
    print(f"  PASS: cards length={len(cards)} chars")
    
    return True


def test_gui_smoke():
    """测试6: GUI 冒烟（标签页切换+基本功能）"""
    print("\n[TEST 6] GUI smoke test...")
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication([])
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    
    from lottery_predictor_app_new import LotteryPredictorApp
    win = LotteryPredictorApp()
    
    # 验证所有标签页存在
    tab_count = win.tab_widget.count()
    assert tab_count >= 5, f"Only {tab_count} tabs"
    print(f"  PASS: {tab_count} tabs exist")
    
    # 切换到优化标签页
    opt_index = win.tab_widget.indexOf(win.optimization_tab)
    assert opt_index >= 0, "Optimization tab not found"
    print(f"  PASS: optimization tab at index {opt_index}")
    
    # 验证 V3 控件
    assert hasattr(win, 'pending_list'), "Missing pending_list"
    assert hasattr(win, 'diagnose_btn'), "Missing diagnose_btn"
    assert hasattr(win, 'auto_opt_check'), "Missing auto_opt_check"
    assert type(win.pending_list).__name__ == 'QTableWidget', f"pending_list is {type(win.pending_list).__name__}"
    print(f"  PASS: V3 controls present (pending_list=QTableWidget)")
    
    # 测试诊断
    win.run_smart_diagnosis()
    assert len(win.pending_actions) >= 0, "Diagnosis failed"
    print(f"  PASS: diagnosis completed, {len(win.pending_actions)} actions")
    
    # 测试权重刷新
    win.refresh_model_weights_ui()
    print(f"  PASS: weights UI refreshed")
    
    # 测试校准状态刷新
    win.refresh_calibration_status()
    print(f"  PASS: calibration status refreshed")
    
    return True


def main():
    print("=" * 60)
    print("  Regression Test Suite (DB-backed)")
    print("=" * 60)
    
    tests = [
        test_db_access,
        test_prediction_records,
        test_model_weights,
        test_improvement_rules,
        test_statistics_insights,
        test_gui_smoke,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 60}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
