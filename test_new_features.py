# -*- coding:utf-8 -*-
"""
测试新功能的基本可用性
"""
import os
import sys
import json

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def test_prediction_records():
    """测试预测记录模块的新功能"""
    print("测试预测记录模块...")
    
    try:
        from prediction_records import add_prediction_record, load_records
        
        # 清空旧记录进行测试
        if os.path.exists('prediction_history.json'):
            os.remove('prediction_history.json')
        
        # 测试添加记录（含概率）
        result = add_prediction_record(
            lottery_type='ssq',
            model_type='ensemble',
            latest_period=1000,
            red_numbers=[1, 2, 3, 4, 5, 6],
            blue_numbers=[1],
            red_probabilities=[0.1, 0.2, 0.15, 0.1, 0.08, 0.07],
            blue_probabilities=[0.3]
        )
        
        print(f"  添加记录结果: {result}")
        
        # 浬试加载记录
        records = load_records()
        print(f"  记录数量: {len(records)}")
        
        if records:
            record = records[0]
            print(f"  记录字段: {list(record.keys())}")
            print(f"  是否包含概率: {'red_probabilities' in record and 'blue_probabilities' in record}")
            
            if 'red_probabilities' in record:
                print(f"  红球概率: {record['red_probabilities']}")
            if 'blue_probabilities' in record:
                print(f"  蓝球概率: {record['blue_probabilities']}")
        
        return True
        
    except Exception as e:
        print(f"  预测记录测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_calibration():
    """测试模型校准模块"""
    print("\n测试模型校准模块...")
    
    try:
        from model_calibration import calibrate_lottery, get_calibration_info, load_calibration_model
        
        # 测试获取校准信息（应该返回默认值）
        info = get_calibration_info('ssq')
        print(f"  SSQ 校准信息: {info}")
        
        info = get_calibration_info('dlt')
        print(f"  DLT 校准信息: {info}")
        
        # 测试加载不存在的校准模型
        model = load_calibration_model('ssq', 'red')
        print(f"  SSQ 红球校准模型（不存在时）: {model}")
        
        return True
        
    except Exception as e:
        print(f"  模型校准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_analysis():
    """测试特征分析模块"""
    print("\n测试特征分析模块...")
    
    try:
        from feature_analysis import get_feature_names, extract_features_from_data
        import pandas as pd
        
        # 测试获取特征名称
        ssq_features = get_feature_names('ssq')
        dlt_features = get_feature_names('dlt')
        print(f"  SSQ 特征数量: {len(ssq_features)}")
        print(f"  DLT 特征数量: {len(dlt_features)}")
        print(f"  前5个SSQ特征: {ssq_features[:5]}")
        print(f"  前5个DLT特征: {dlt_features[:5]}")
        
        # 创建简单的测试数据
        test_data = {
            '期数': [1000, 1001, 1002],
            '红球_1': [1, 2, 3],
            '红球_2': [4, 5, 6],
            '红球_3': [7, 8, 9],
            '红球_4': [10, 11, 12],
            '红球_5': [13, 14, 15],
            '红球_6': [16, 17, 18],
            '蓝球': [1, 2, 3]
        }
        df = pd.DataFrame(test_data)
        
        # 测试特征提取
        ssq_feature_df = extract_features_from_data(df, 'ssq')
        print(f"  SSQ 特征提取后列数: {len(ssq_feature_df.columns)}")
        print(f"  SSQ 特征提取后形状: {ssq_feature_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  特征分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrain_module():
    """测试重训练模块的导入"""
    print("\n测试重训练模块导入...")
    
    try:
        from retrain_models_with_feedback import retrain_lottery_models
        print("  重训练模块导入成功")
        return True
    except Exception as e:
        print(f"  重训练模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("开始测试新功能...")
    print("=" * 50)
    
    tests = [
        test_prediction_records,
        test_model_calibration,
        test_feature_analysis,
        test_retrain_module
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  测试 {test.__name__} 出现异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "通过" if result else "失败"
        print(f"  {test.__name__}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("所有测试通过！")
        return 0
    else:
        print("部分测试失败，请检查上面的错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())