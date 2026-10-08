# -*- coding:utf-8 -*-
"""
带反馈的模型重训练模块
结合历史预测和开奖数据，特征选择和模型校准来重新训练预测模型
"""
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 确保可以导入项目模块
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from feature_analysis import select_top_features, run_feature_analysis
from model_calibration import calibrate_lottery, get_calibration_info

def retrain_lottery_models(lottery_type: str, use_gpu: bool = False, 
                          model_types: Optional[List[str]] = None) -> bool:
    """
    重新训练指定彩票类型的所有模型
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        use_gpu: 是否使用GPU训练
        model_types: 要训练的模型类型列表，None表示所有支持的模型
        
    Returns:
        重训练是否成功
    """
    try:
        print(f"开始对 {lottery_type} 进行带反馈的模型重训练...")
        
        # 1. 特征分析和选择
        print("  步骤1: 进行特征分析...")
        feature_result = run_feature_analysis(lottery_type)
        if feature_result.get('status') != 'completed':
            print(f"  特征分析失败: {feature_result.get('error', '未知错误')}")
            # 继续使用默认特征
        
        selected_features = feature_result.get('selected_features', [])
        print(f"  已选择 {len(selected_features)} 个特征用于重训练")
        
        # 2. 模型校准
        print("  步骤2: 进行模型校准...")
        calibration_success = calibrate_lottery(lottery_type)
        if calibration_success:
            print("  模型校准完成")
        else:
            print("  模型校准失败或数据不足，将继续使用未校准的模型")
        
        # 3. 重新训练机器学习模型
        print("  步骤3: 重新训练机器学习模型...")
        
        # 如果未指定模型类型，使用默认的所有模型
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble']
        
        # 过滤掉不可用的模型
        available_models = []
        for model_type in model_types:
            if model_type == 'lightgbm':
                try:
                    from lightgbm import LGBMRegressor
                    available_models.append(model_type)
                except ImportError:
                    print(f"  警告: LightGBM 不可用，跳过 {model_type}")
            elif model_type == 'catboost':
                try:
                    from catboost import CatBoostRegressor
                    available_models.append(model_type)
                except ImportError:
                    print(f"  警告: CatBoost 不可用，跳过 {model_type}")
            else:
                available_models.append(model_type)
        
        if not available_models:
            print("  错误: 没有可用的机器学习模型")
            return False
        
        print(f"  将训练以下模型: {available_models}")
        
        # 训练每个模型
        success_count = 0
        for model_type in available_models:
            print(f"    正在训练 {model_type} 模型...")
            try:
                # 使用thread_utils中的TrainModelThread的逻辑，但直接调用训练脚本
                if lottery_type == 'ssq':
                    script_path = os.path.join(project_dir, "scripts", "ssq", "train_ssq_model.py")
                else:
                    script_path = os.path.join(project_dir, "scripts", "dlt", "train_dlt_model.py")
                
                if not os.path.exists(script_path):
                    print(f"    错误: 训练脚本不存在 {script_path}")
                    continue
                
                # 构建训练命令
                cmd = [sys.executable, script_path, "--model", model_type]
                if use_gpu:
                    cmd.append("--gpu")
                
                print(f"    执行命令: {' '.join(cmd)}")
                
                # 运行训练脚本
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1小时超时
                )
                
                if result.returncode == 0:
                    print(f"    {model_type} 模型训练成功")
                    success_count += 1
                else:
                    print(f"    {model_type} 模型训练失败:")
                    print(f"      stdout: {result.stdout[-500:] if result.stdout else 'None'}")
                    print(f"      stderr: {result.stderr[-500:] if result.stderr else 'None'}")
                    
            except subprocess.TimeoutExpired:
                print(f"    {model_type} 模型训练超时（超过1小时）")
            except Exception as e:
                print(f"    {model_type} 模型训练过程中出错: {e}")
        
        print(f"  机器学习模型训练完成: {success_count}/{len(available_models)} 成功")
        
        # 4. 重新训练LSTM-CRF模型（如果需要）
        print("  步骤4: 重新训练LSTM-CRF模型...")
        try:
            if lottery_type == 'ssq':
                lstm_script = os.path.join(project_dir, "scripts", "ssq", "train_ssq_model.py")
            else:
                lstm_script = os.path.join(project_dir, "scripts", "dlt", "train_dlt_model.py")
            
            if os.path.exists(lstm_script):
                # LSTM-CRF模型通过特殊参数训练
                lstm_cmd = [sys.executable, lstm_script, "--model", "lstm-crf"]
                if use_gpu:
                    lstm_cmd.append("--gpu")
                
                print(f"    执行LSTM-CRF训练命令: {' '.join(lstm_cmd)}")
                
                lstm_result = subprocess.run(
                    lstm_cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200  # LSTM训练可能需要更长时间，2小时超时
                )
                
                if lstm_result.returncode == 0:
                    print("    LSTM-CRF模型训练成功")
                    lstm_success = True
                else:
                    print(f"    LSTM-CRF模型训练失败:")
                    print(f"      stdout: {lstm_result.stdout[-500:] if lstm_result.stdout else 'None'}")
                    print(f"      stderr: {lstm_result.stderr[-500:] if lstm_result.stderr else 'None'}")
                    lstm_success = False
            else:
                print(f"    警告: LSTM-CRF训练脚本不存在 {lstm_script}")
                lstm_success = False
                
        except subprocess.TimeoutExpired:
            print("    LSTM-CRF模型训练超时（超过2小时）")
            lstm_success = False
        except Exception as e:
            print(f"    LSTM-CRF模型训练过程中出错: {e}")
            lstm_success = False
        
        # 5. 生成重训练报告
        print("  步骤5: 生成重训练报告...")
        retrain_report = {
            'lottery_type': lottery_type,
            'timestamp': datetime.now().isoformat(),
            'feature_analysis': feature_result,
            'model_training': {
                'attempted_models': available_models,
                'successful_count': success_count,
                'lstm_crf_success': lstm_success if 'lstm_success' in locals() else False,
                'use_gpu': use_gpu
            },
            'calibration': get_calibration_info(lottery_type),
            'status': 'completed' if success_count > 0 or (locals().get('lstm_success', False)) else 'failed'
        }
        
        # 保存重训练报告
        retrain_dir = os.path.join(project_dir, 'retrain_reports')
        if not os.path.exists(retrain_dir):
            os.makedirs(retrain_dir)
        
        report_file = os.path.join(
            retrain_dir, 
            f'{lottery_type}_retrain_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(retrain_report, f, ensure_ascii=False, indent=2)
        
        print(f"  重训练报告已保存: {report_file}")
        
        overall_success = success_count > 0 or (locals().get('lstm_success', False))
        if overall_success:
            print(f"{lottery_type} 带反馈的模型重训练完成！")
        else:
            print(f"{lottery_type} 带反馈的模型重训练失败：没有成功训练的模型")
        
        return overall_success
        
    except Exception as e:
        print(f"重训练过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def retrain_all_lotteries(use_gpu: bool = False) -> Dict[str, bool]:
    """
    重新训练所有彩票类型的模型
    
    Args:
        use_gpu: 是否使用GPU训练
        
    Returns:
        每种彩票类型的重训练结果字典
    """
    results = {}
    for lottery_type in ['ssq', 'dlt']:
        print(f"\n{'='*50}")
        print(f"开始重新训练 {lottery_type.upper()} 模型")
        print(f"{'='*50}")
        results[lottery_type] = retrain_lottery_models(lottery_type, use_gpu=use_gpu)
    
    print(f"\n{'='*50}")
    print("所有彩票类型重训练完成")
    print(f"{'='*50}")
    for lt, success in results.items():
        status = "成功" if success else "失败"
        print(f"  {lt.upper()}: {status}")
    
    return results

if __name__ == "__main__":
    # 命令行接口
    import argparse
    
    parser = argparse.ArgumentParser(description='带反馈的彩票预测模型重训练')
    parser.add_argument('--type', choices=['ssq', 'dlt', 'all'], default='all',
                       help='要重训练的彩票类型')
    parser.add_argument('--gpu', action='store_true',
                       help='是否使用GPU训练')
    parser.add_argument('--model', type=str, nargs='+',
                       choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                       help='要训练的特定模型类型（可多选）')
    
    args = parser.parse_args()
    
    if args.type == 'all':
        results = retrain_all_lotteries(use_gpu=args.gpu)
        # 设置退出码：如果所有都成功则退出码0，否则退出码1
        sys.exit(0 if all(results.values()) else 1)
    else:
        success = retrain_lottery_models(args.type, use_gpu=args.gpu, model_types=args.model)
        sys.exit(0 if success else 1)