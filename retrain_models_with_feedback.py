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
                          model_types: Optional[List[str]] = None,
                          recent_periods: Optional[int] = None,
                          feature_mode: str = 'original',
                          include_lstm: bool = False) -> bool:
    """
    重新训练指定彩票类型的所有模型
    
    Args:
        lottery_type: 'ssq' 或 'dlt'
        use_gpu: 是否使用GPU训练
        model_types: 要训练的模型类型列表，None表示所有支持的模型
        recent_periods: 近期窗口重训(抗漂移)——仅用最近N期数据重训ML模型；
                        None=全量数据。增强模式下暖机50期仍会自动保留
        feature_mode: 'original'=70维位图 / 'enhanced'=+21时序衍生特征
        
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
        if selected_features:
            _feat_names = list(selected_features[0].keys()) if (isinstance(selected_features, list)
                          and selected_features and isinstance(selected_features[0], dict)) else list(selected_features)
            print(f"  特征分析完成，入选特征 {len(selected_features)} 个(参考，当前模型为全特征训练): {_feat_names[:8]}")
        else:
            print("  特征分析未给出结果，继续全特征重训练")
        
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
        
        # 训练每个模型（直接走 LotteryMLModels 真实路径——旧版subprocess调LSTM脚本+--model参数从未生效）
        success_count = 0
        trained_details = {}
        try:
            from ml_models import LotteryMLModels
            from backtest import _load_csv_data
            ml_import_ok = True
        except Exception as e:
            print(f"  错误: ML模型模块导入失败: {e}")
            ml_import_ok = False
        
        df_full = None
        if ml_import_ok:
            try:
                df_full = _load_csv_data(lottery_type)
            except Exception as e:
                print(f"  错误: 历史数据加载失败: {e}")
        
        if ml_import_ok and df_full is not None and len(df_full) > 0:
            # 近期窗口重训(抗漂移): 只取最近N期，至少留足够样本
            df_train = df_full
            if recent_periods and recent_periods > 0:
                min_needed = 120 if feature_mode == 'original' else 170  # 暖机+最少样本
                take = max(int(recent_periods), min_needed)
                df_train = (df_full.sort_values('期数', ascending=False)
                            .head(take).sort_values('期数').reset_index(drop=True))
                print(f"  近期窗口重训: 使用最近 {len(df_train)} 期(全量 {len(df_full)} 期)")
            
            for model_type in available_models:
                print(f"    正在训练 {model_type} 模型...", flush=True)
                try:
                    ml_model = LotteryMLModels(
                        lottery_type=lottery_type,
                        model_type=model_type,
                        log_callback=lambda s: print(f"      {s}") if False else None,
                        use_gpu=use_gpu
                    )
                    if feature_mode == 'enhanced':
                        ml_model.feature_mode = 'enhanced'
                    ok = ml_model.train(df_train)
                    if ok is not False:
                        # train() 无显式返回值时以 models 非空判成功
                        ok = bool(getattr(ml_model, 'models', None))
                    if ok:
                        print(f"    {model_type} 模型训练成功({feature_mode})")
                        success_count += 1
                        trained_details[model_type] = 'success'
                    else:
                        print(f"    {model_type} 模型训练失败")
                        trained_details[model_type] = 'failed'
                except Exception as e:
                    print(f"    {model_type} 模型训练出错: {e}")
                    trained_details[model_type] = f'error: {e}'
            
            # 重训后刷新动态权重（旧模型被替换，旧权重不再代表当前模型）
            try:
                from model_weights import update_weights
                update_weights(lottery_type)
                print("  已按最新核验表现刷新动态权重")
            except Exception as e:
                print(f"  权重刷新失败(不影响重训结果): {e}")
        
        print(f"  机器学习模型训练完成: {success_count}/{len(available_models)} 成功")
        
        # 4. 重新训练LSTM-CRF模型（如果需要）
        print("  步骤4: 重新训练LSTM-CRF模型...")
        if not include_lstm:
            print("    (默认跳过 LSTM-CRF 重训，耗时约2小时；如需请加 --with-lstm)")
            lstm_success = False
        else:
            try:
                if lottery_type == 'ssq':
                    lstm_script = os.path.join(project_dir, "scripts", "ssq", "train_ssq_model.py")
                else:
                    lstm_script = os.path.join(project_dir, "scripts", "dlt", "train_dlt_model.py")

                if os.path.exists(lstm_script):
                    # LSTM 训练脚本本身即 LSTM-CRF，不接受 --model 参数(旧版传该参数导致 argparse 报错，从未成功)
                    lstm_cmd = [sys.executable, lstm_script]
                    if use_gpu:
                        lstm_cmd.append("--gpu")

                    print(f"    执行LSTM-CRF训练命令: {' '.join(lstm_cmd)}")
                    lstm_result = subprocess.run(lstm_cmd, capture_output=True, text=True, timeout=7200)

                    if lstm_result.returncode == 0:
                        print("    LSTM-CRF模型训练成功")
                        lstm_success = True
                    else:
                        print("    LSTM-CRF模型训练失败:")
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
                'trained_details': trained_details,
                'feature_mode': feature_mode,
                'recent_periods': recent_periods,
                'lstm_crf_success': locals().get('lstm_success', False),
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
        
        def _json_default(o):
            """numpy 标量/布尔等非原生类型 → 原生，避免报告序列化失败"""
            import numpy as _np
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, _np.integer):
                return int(o)
            if isinstance(o, _np.floating):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return str(o)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(retrain_report, f, ensure_ascii=False, indent=2, default=_json_default)
        
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

def retrain_all_lotteries(use_gpu: bool = False, recent_periods: Optional[int] = None,
                          feature_mode: str = 'original', include_lstm: bool = False) -> Dict[str, bool]:
    """
    重新训练所有彩票类型的模型
    """
    results = {}
    for lottery_type in ['ssq', 'dlt']:
        print(f"\n{'='*50}")
        print(f"开始重新训练 {lottery_type.upper()} 模型")
        print(f"{'='*50}")
        results[lottery_type] = retrain_lottery_models(
            lottery_type, use_gpu=use_gpu, recent_periods=recent_periods,
            feature_mode=feature_mode, include_lstm=include_lstm)
    
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
    
    parser.add_argument('--recent', type=int, default=None,
                       help='近期窗口重训(抗漂移): 仅用最近N期重训ML模型，默认全量')
    parser.add_argument('--feature-mode', choices=['original', 'enhanced'], default='original',
                       help='特征模式: original=70维位图(默认) / enhanced=+21时序衍生特征')
    parser.add_argument('--with-lstm', action='store_true',
                       help='同时重训 LSTM-CRF(耗时约2小时，默认跳过)')

    args = parser.parse_args()

    if args.type == 'all':
        results = retrain_all_lotteries(use_gpu=args.gpu, recent_periods=args.recent,
                                        feature_mode=args.feature_mode, include_lstm=args.with_lstm)
        # 设置退出码：如果所有都成功则退出码0，否则退出码1
        sys.exit(0 if all(results.values()) else 1)
    else:
        success = retrain_lottery_models(args.type, use_gpu=args.gpu, model_types=args.model,
                                         recent_periods=args.recent, feature_mode=args.feature_mode,
                                         include_lstm=args.with_lstm)
        sys.exit(0 if success else 1)