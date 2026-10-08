# -*- coding:utf-8 -*-
"""
最终验证测试：在不显示GUI的情况下测试预测和存档流程
"""
import os
import sys

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def test_full_flow():
    """测试完整的预测->存档流程"""
    print("开始完整流程测试...")
    
    try:
        # 导入必要模块
        from lottery_predictor_app_new import LotteryPredictorApp
        from prediction_records import load_records, save_records
        from unittest.mock import MagicMock
        import pandas as pd
        
        # 清空旧记录
        save_records([])
        print("  已清空预测记录")
        
        # 创建一个最小化的应用实例（我们需要避免实际创建GUI窗口）
        # 方法：重写__init__方法，只保留我们需要的部分
        original_init = LotteryPredictorApp.__init__
        
        def mock_init(self):
            # 只初始化我们需要的属性，避免创建实际的Qt窗口
            self.log_emitter = MagicMock()
            self.log_emitter.new_log = MagicMock()
            self.result_label = MagicMock()
            self.result_label.setText = MagicMock()
            
            # 初始化UI组件的引用（我们不会用到它们，但防止属性错误）
            self.ai_number_label = MagicMock()
            self.ai_number_label.setText = MagicMock()
            self.log_box = MagicMock()
            self.log_box.append = MagicMock()
            self.log_box.customContextMenuRequested = MagicMock()
            self.log_box.customContextMenuRequested.connect = MagicMock()
            
            self.ev_log_box = MagicMock()
            self.ev_log_box.setText = MagicMock()
            self.ev_log_box.customContextMenuRequested = MagicMock()
            self.ev_log_box.customContextMenuRequested.connect = MagicMock()
            
            # 投注计划
            self.investment_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               'investment_history.json')
            
            # 其他需要的属性
            self.stats_window = None
            self.autoretrain_enabled = False
            self.backtest_history = []
            self.last_backtest_result = None
            
            # 初始化机器学习模型字典
            self.ml_models = {}
            
            # 初始化定时器等（我们不会启动它们）
            self.data_check_timer = MagicMock()
            self.update_check_timer = MagicMock()
            
        LotteryPredictorApp.__init__ = mock_init
        
        # 现在创建实例
        print("  创建应用实例...")
        app = LotteryPredictorApp()
        print("  应用实例创建成功")
        
        # 恢复原始init方法（虽然不再需要，但为了整洁）
        LotteryPredictorApp.__init__ = original_init
        
        # 现在我们需要模拟一下数据，这样预测方法不会因为没有数据而失败
        from prediction_utils import load_lottery_data
        original_load_lottery_data = load_lottery_data
        
        def mock_load_lottery_data(lottery_type):
            # 返回一些最小的测试数据
            import pandas as pd
            if lottery_type == 'ssq':
                data = {
                    '期数': list(range(1000, 1010)),
                    '红球_1': [1,2,3,4,5,6,7,8,9,10],
                    '红球_2': [11,12,13,14,15,16,17,18,19,20],
                    '红球_3': [21,22,23,24,25,26,27,28,29,30],
                    '红球_4': [31,32,33,1,2,3,4,5,6,7],
                    '红球_5': [8,9,10,11,12,13,14,15,16,17],
                    '红球_6': [18,19,20,21,22,23,24,25,26,27],
                    '蓝球': [1,2,3,4,5,6,7,8,9,10]
                }
            else:  # dlt
                data = {
                    '期数': list(range(1000, 1010)),
                    '红球_1': [1,2,3,4,5,6,7,8,9,10],
                    '红球_2': [11,12,13,14,15,16,17,18,19,20],
                    '红球_3': [21,22,23,24,25,26,27,28,29,30],
                    '红球_4': [31,32,33,1,2,3,4,5,6,7],
                    '红球_5': [8,9,10,11,12,13,14,15,16,17],
                    '蓝球_1': [1,2,3,4,5,6,7,8,9,10],
                    '蓝球_2': [11,12,13,14,15,16,17,18,19,20]
                }
            return pd.DataFrame(data)
        
        # 补丁加载函数
        import prediction_utils
        prediction_utils.load_lottery_data = mock_load_lottery_data
        
        # 我们还需要确保ml_models.py中的模型不会因为没有训练而崩溃
        # 但我们可以接受预测返回None，因为我们主要测试的是存档机制
        # 或者我们可以mock ml_models.LotteryMLModels的predict方法
        
        from ml_models import LotteryMLModels
        original_predict = LotteryMLModels.predict
        
        def mock_predict(self, recent_data, variation=0):
            # 返回一些固定的预测值，并设置概率属性
            if self.lottery_type == 'ssq':
                red_pred = [1, 2, 3, 4, 5, 6]
                blue_pred = [1]
            else:
                red_pred = [1, 2, 3, 4, 5]
                blue_pred = [1, 2]
            
            # 设置概率属性（这是我们关心的部分）
            if self.lottery_type == 'ssq':
                self.last_red_proba = {str(i): 0.1 for i in range(33)}  # 0-32 对应 1-33
                self.last_blue_proba = {str(i): 0.1 for i in range(16)} # 0-15 对应 1-16
            else:
                self.last_red_proba = {str(i): 0.1 for i in range(35)} # 0-34 对应 1-35
                self.last_blue_proba = {str(i): 0.1 for i in range(12)} # 0-11 对应 1-12
                
            return red_pred, blue_pred
        
        LotteryMLModels.predict = mock_predict
        
        print("  模拟环境设置完成")
        
        # 现在执行一次简单预测
        print("  执行简单预测...")
        # 直接调用内部方法而不经过UI
        # 我们需要设置一些UI状态，这样generate_prediction方法不会因为找不到UI元素而出错
        
        # 设置必要的UI状态（这些是在generate_prediction开始时读取的）
        app.predict_type = "ssq"  # 彩票类型
        app.ai_model_type = "random_forest"  # 模型类型
        app.predict_note = 1  # 注数
        app.predict_method = "简单"  # 预测模式
        
        # 调用预测生成方法
        try:
            app.generate_prediction()
            print("  预测生成方法执行完成")
        except Exception as e:
            print(f"  预测生成过程中出现异常（可能是由于UI元素未完全初始化）: {e}")
            # 即使有异常，我们也继续检查是否已经存档了记录
            pass
        
        # 恢复补丁
        prediction_utils.load_lottery_data = original_load_lottery_data
        LotteryMLModels.predict = original_predict
        
        # 检查记录
        records = load_records()
        print(f"  存档的记录数量: {len(records)}")
        
        if len(records) > 0:
            record = records[0]
            print(f"  第一条记录:")
            print(f"    时间: {record.get('predict_time')}")
            print(f"    类型: {record.get('lottery_type')} {record.get('model_type')}")
            print(f"    红球: {record.get('red_numbers')}")
            print(f"    蓝球: {record.get('blue_numbers')}")
            print(f"    红球概率: {record.get('red_probabilities')}")
            print(f"    蓝球概率: {record.get('blue_probabilities')}")
            
            has_red_proba = record.get('red_probabilities') is not None
            has_blue_proba = record.get('blue_probabilities') is not None
            
            if has_red_proba and has_blue_proba:
                print("  ✅ 成功：预测记录中包含概率信息")
                return True
            else:
                print("  ❌ 失败：预测记录中缺少概率信息")
                return False
        else:
            print("  ❌ 失败：没有存档任何预测记录")
            return False
            
    except Exception as e:
        print(f" 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("最终验证测试")
    print("=" * 50)
    
    success = test_full_flow()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 最终验证测试通过！")
        print("新功能正确工作：预测过程中收集的概率信息被正确存档。")
    else:
        print("❌ 最终验证测试失败。")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())