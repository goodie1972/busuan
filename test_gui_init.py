# -*- coding:utf-8 -*-
"""
测试GUI初始化和我们新增的方法是否存在
"""
import os
import sys

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def test_gui_class_exists():
    """测试LotteryPredictorApp类是否存在且我们修改的方法存在"""
    print("测试GUI类和方法...")
    
    try:
        # 导入主应用类
        from lottery_predictor_app_new import LotteryPredictorApp
        
        # 检查类是否存在
        print(f"  类存在: {LotteryPredictorApp}")
        
        # 检查我们修改的方法是否存在
        method = getattr(LotteryPredictorApp, '_save_prediction_records', None)
        if method is not None:
            print(f"  方法 _save_prediction_records 存在: {method}")
            # 检查方法签名
            import inspect
            sig = inspect.signature(method)
            print(f"  方法签名: {sig}")
            # 检查是否有我们期望的参数
            params = list(sig.parameters.keys())
            expected_params = ['self', 'lottery_type', 'model_type', 'predictions', 'red_probabilities', 'blue_probabilities']
            if all(p in params for p in expected_params):
                print(f"  方法签名正确: {params}")
                return True
            else:
                print(f"  方法签名不正确. 期望: {expected_params}, 实际: {params}")
                return False
        else:
            print("  方法 _save_prediction_records 不存在")
            return False
            
    except Exception as e:
        print(f"  测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports_work():
    """测试我们新建或修改的模块可以正确导入"""
    print("\n测试模块导入...")
    
    modules_to_test = [
        'model_calibration',
        'feature_analysis', 
        'retrain_models_with_feedback',
        'prediction_records',
        'ml_models'
    ]
    
    results = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  {module_name}: PASS")
            results.append(True)
        except Exception as e:
            print(f"  {module_name}: FAIL - {e}")
            results.append(False)
    
    return all(results)

def main():
    print("开始GUI初始化测试...")
    print("=" * 50)
    
    tests = [
        test_gui_class_exists,
        test_imports_work
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
        status = "PASS" if result else "FAIL"
        print(f"  {test.__name__}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("所有GUI初始化测试通过！")
        return 0
    else:
        print("部分GUI初始化测试失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())