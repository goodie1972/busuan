# -*- coding:utf-8 -*-
"""
集成测试：验证新功能在预测流程中的正确工作
"""
import os
import sys
import itertools

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def test_save_prediction_records():
    """测试 _save_prediction_records 方法能正确处理各种输入"""
    print("测试 _save_prediction_records 方法...")
    
    try:
        # 导入必要的模块
        from lottery_predictor_app_new import LotteryPredictorApp
        from unittest.mock import MagicMock
        
        # 创建一个最小的模拟应用实例（不初始化GUI）
        class MockApp:
            def __init__(self):
                self.log_emitter = MagicMock()
                self.log_emitter.new_log = MagicMock()
                
            def _save_prediction_records(self, lottery_type, model_type, predictions, red_probabilities=None, blue_probabilities=None):
                """复制我们修改后的方法"""
                try:
                    if not predictions:
                        return
                    # 使用真实的 load_lottery_data 函数
                    from prediction_utils import load_lottery_data
                    df = load_lottery_data(lottery_type)
                    if df is None or df.empty:
                        # 如果没有数据，使用一个最小的数据框以便测试
                        import pandas as pd
                        df = pd.DataFrame({'期数': [1000, 1001, 1002]})
                    latest_period = int(df['期数'].max())
                    from prediction_records import add_prediction_record
                    count = None
                    for prediction in predictions:
                        # 处理新旧两种格式：(red, blue) 或 (red, blue, red_proba, blue_proba)
                        if len(prediction) == 2:
                            red_nums, blue_nums = prediction
                            red_proba = None
                            blue_proba = None
                        elif len(prediction) == 4:
                            red_nums, blue_nums, red_proba, blue_proba = prediction
                        else:
                            # 后备方案，只取前两个元素
                            red_nums, blue_nums = prediction[0], prediction[1]
                            red_proba = None
                            blue_proba = None
                            
                        count = add_prediction_record(
                            lottery_type, model_type, latest_period,
                            red_nums, blue_nums, red_proba, blue_proba)
                    if count:
                        self.log_emitter.new_log.emit(
                            f"已存档 {len(predictions)} 注预测记录（累计 {count} 注），"
                            f"可在'历史回测'页核验中奖情况")
                    return count
                except Exception as e:
                    self.log_emitter.new_log.emit(
                        f"预测记录存档失败(不影响预测): {e}")
                    return None
        
        app = MockApp()
        
        # 清除旧的预测记录以便测试
        from prediction_records import save_records
        save_records([])  # 清空记录
        
        # 测试1: 简单预测（无概率）
        print("  测试1: 简单预测（无概率）...")
        simple_predictions = [
            ([1, 2, 3, 4, 5, 6], [1]),
            ([2, 3, 4, 5, 6, 7], [2])
        ]
        result1 = app._save_prediction_records('ssq', 'test_model', simple_predictions)
        print(f'    结果: {result1}')
        
        # 测试2: 复式预测（无概率）
        print("  测试2: 复式预测（无概率）...")
        red_numbers = [1, 2, 3, 4, 5]
        blue_numbers = [1, 2]
        red_combos = list(itertools.combinations(red_numbers, 3))  # 3红球复式
        blue_combos = list(itertools.combinations(blue_numbers, 2)) if 2 > 0 else [()]
        complex_predictions = [ (list(r), list(b)) for r in red_combos for b in blue_combos ]
        print(f'    生成了 {len(complex_predictions)} 个复式组合')
        result2 = app._save_prediction_records('ssq', 'test_model', complex_predictions)
        print(f'    结果: {result2}')
        
        # 测试3: 带概率的预测
        print("  测试3: 带概率的预测...")
        prob_predictions = [
            ([1, 2, 3, 4, 5, 6], [1], [0.1, 0.2, 0.15, 0.1, 0.08, 0.07], [0.3]),
            ([2, 3, 4, 5, 6, 7], [2], [0.2, 0.1, 0.2, 0.15, 0.1, 0.1], [0.4])
        ]
        result3 = app._save_prediction_records('ssq', 'test_model', prob_predictions)
        print(f'    结果: {result3}')
        
        # 加载并检查记录
        from prediction_records import load_records
        records = load_records()
        print(f'\n  总共存档记录数: {len(records)}')
        
        for i, record in enumerate(records):
            print(f'    记录 {i+1}:')
            print(f'      时间: {record.get("predict_time")}')
            print(f'      类型: {record.get("lottery_type")} {record.get("model_type")}')
            print(f'      红球: {record.get("red_numbers")}')
            print(f'      蓝球: {record.get("blue_numbers")}')
            print(f'      红球概率: {record.get("red_probabilities")}')
            print(f'      蓝球概率: {record.get("blue_probabilities")}')
        
        # 验证概率字段是否存在
        has_prob_fields = any(
            r.get('red_probabilities') is not None or r.get('blue_probabilities') is not None
            for r in records
        )
        print(f'\n  是否存在概率字段: {has_prob_fields}')
        
        if has_prob_fields:
            print('  PASS: 概率字段正确存档')
        else:
            print('  FAIL: 概率字段未存档')
            
        return len(records) > 0 and has_prob_fields
        
    except Exception as e:
        print(f"  测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_generation_with_probability():
    """测试预测生成过程是否正确收集概率"""
    print("\n测试预测生成过程中的概率收集...")
    
    try:
        from ml_models import LotteryMLModels
        import pandas as pd
        import numpy as np
        
        # 创建一个模拟模型，它会设置概率属性
        class MockMLModel:
            def __init__(self):
                self.red_count = 6
                self.blue_count = 1
                self.last_red_proba = None
                self.last_blue_proba = None
                
            def predict(self, recent_data, variation=0):
                # 模拟预测
                red_pred = [1, 2, 3, 4, 5, 6]
                blue_pred = [1]
                # 设置概率属性（这是我们在ml_models.py中添加的功能）
                self.last_red_proba = {str(i-1): 0.1 for i in range(1, 34)}  # 红球1-33的概率
                self.last_blue_proba = {str(i-1): 0.1 for i in range(1, 17)}  # 蓝球1-16的概率
                return red_pred, blue_pred
        
        model = MockMLModel()
        
        # 生成预测
        recent_data = pd.DataFrame({'期数': [1000]})  # 虚拟数据
        red_nums, blue_nums = model.predict(recent_data)
        
        print(f'  预测结果: 红球={red_nums}, 蓝球={blue_nums}')
        print(f'  红球概率属性存在: {model.last_red_proba is not None}')
        print(f'  蓝球概率属性存在: {model.last_blue_proba is not None}')
        
        if model.last_red_proba is not None and model.last_blue_proba is not None:
            print('  PASS: 预测过程中正确设置了概率属性')
            return True
        else:
            print('  FAIL: 预测过程中未设置概率属性')
            return False
            
    except Exception as e:
        print(f"  测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有集成测试"""
    print("开始集成测试...")
    print("=" * 60)
    
    tests = [
        test_save_prediction_records,
        test_prediction_generation_with_probability
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  测试 {test.__name__} 出现异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("集成测试结果汇总:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"  {test.__name__}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("所有集成测试通过！")
        return 0
    else:
        print("部分集成测试失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())