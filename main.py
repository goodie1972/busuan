#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Entry Point for Lottery Prophet Application
Author: Yang Zhao
"""

import os
import sys
import logging
import argparse
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置运行环境，包括路径和 DLL 加载"""
    # 修复 PyInstaller 打包后的 torch DLL 加载问题
    fix_torch_dlls()

    # 在打包环境中，将工作目录切换到 exe 所在目录，
    # 使 ./scripts/ ./model/ ./data/ 等相对路径指向 exe 同级的目录
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        try:
            os.chdir(exe_dir)
        except OSError:
            pass

    # 优先使用 exe 所在目录下的 scripts（打包后 exe 同级已复制 scripts/），
    # 否则退回到模块所在位置的 scripts
    base_dir = os.getcwd()
    scripts_dir = os.path.join(base_dir, 'scripts')
    if not os.path.isdir(scripts_dir):
        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    
    # 确保必要的目录存在
    for lottery_type in ['dlt', 'ssq']:
        model_dir = os.path.join('model', lottery_type)
        os.makedirs(model_dir, exist_ok=True)
        
        data_dir = os.path.join('data', lottery_type)
        os.makedirs(data_dir, exist_ok=True)


def fix_torch_dlls():
    """
    修复 PyInstaller 打包后 torch DLL 加载失败的问题。
    在 torch 被导入之前，先设置环境变量并预加载 DLL 文件。
    """
    # 无条件先设置环境变量，解决 OpenMP 运行时冲突（libiomp5 与 numpy/scipy 冲突
    # 会导致 WinError 1114）。此设置必须在 torch 被 import 前完成，且不依赖打包路径。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 获取基础路径（兼容 PyInstaller 打包环境）
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    torch_lib = os.path.join(base_path, 'torch', 'lib')
    if not os.path.isdir(torch_lib):
        # 也尝试 PyInstaller 的 _internal 路径
        internal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_internal')
        torch_lib = os.path.join(internal_dir, 'torch', 'lib')
        if not os.path.isdir(torch_lib):
            return  # 不是打包环境，或者没有 torch lib，跳过

    # 设置环境变量，解决 OpenMP 冲突
    # （已在 fix_torch_dlls 中设置；此处为直接运行 backtest.py 时兜底，
    #   必须在 torch 被 import 前完成，否则与 numpy 的 OpenMP 运行时冲突 WinError 1114）
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 将 torch/lib 目录加入 PATH，确保 DLL 依赖能正确解析
    torch_lib = os.path.abspath(torch_lib)
    if torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

    # 预加载 torch DLL（按依赖顺序）
    import ctypes
    dll_order = [
        "libiomp5md.dll",
        "c10.dll",
        "torch_cpu.dll",
        "torch_python.dll",
        "torch.dll",
        "torch_global_deps.dll",
    ]
    for dll_name in dll_order:
        dll_path = os.path.join(torch_lib, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
            except Exception:
                pass  # 让 torch 自己处理

def fetch_data(lottery_type):
    """获取彩票数据"""
    logger.info(f"开始获取{lottery_type}彩票数据...")
    
    if lottery_type == 'dlt':
        from scripts.fetch_dlt_data import main as fetch_dlt
        fetch_dlt()
    elif lottery_type == 'ssq':
        from scripts.fetch_ssq_data import main as fetch_ssq
        fetch_ssq()
    else:
        logger.error(f"不支持的彩票类型: {lottery_type}")
        return False
    
    logger.info(f"{lottery_type}彩票数据获取完成")
    return True

def train_model(lottery_type, model_type):
    """训练彩票预测模型"""
    logger.info(f"开始训练{lottery_type}彩票的{model_type}模型...")
    
   
    from lottery_predictor_app_new import train_model as app_train_model
    
   
    def log_to_console(message):
        logger.info(message)
    

    success = app_train_model(lottery_type, model_type, log_callback=log_to_console)
    
    if success:
        logger.info(f"{lottery_type}彩票的{model_type}模型训练完成")
    else:
        logger.error(f"{lottery_type}彩票的{model_type}模型训练失败")
    
    return success

def predict(lottery_type, model_type, compound_red=0, compound_blue=0,
            dantuo=False, dan_red=2, tuo_red=6, dan_blue=0, tuo_blue=2):
    """使用训练好的模型进行预测"""
    # 胆拖预测：胆码固定 + 拖码组合
    if dantuo:
        logger.info(f"使用{model_type}模型胆拖预测{lottery_type}彩票...")
        try:
            from ml_models import LotteryMLModels
            from scripts.data_analysis import load_lottery_data
            ml_model = LotteryMLModels(lottery_type=lottery_type, model_type=model_type)
            if not ml_model.load_models():
                logger.error(f"模型{model_type}尚未训练，请先训练模型。")
                return None
            df = load_lottery_data(lottery_type)
            recent_data = df.sort_values('期数', ascending=False).head(ml_model.feature_window)
            r_dan, r_tuo, b_dan, b_tuo = ml_model.predict_dantuo(
                recent_data, red_dan_count=dan_red, red_tuo_count=tuo_red,
                blue_dan_count=dan_blue, blue_tuo_count=tuo_blue)
            if r_dan is None:
                logger.error("胆拖预测失败（期望值/xgboost 模型可能不支持）")
                return None
            from math import comb
            red_pick = ml_model.red_count - len(r_dan)
            blue_pick = ml_model.blue_count - len(b_dan)
            n_notes = comb(len(r_tuo), red_pick) * \
                (comb(len(b_tuo), blue_pick) if blue_pick > 0 else comb(len(b_tuo), ml_model.blue_count))
            result = {
                'dan_red': r_dan, 'tuo_red': r_tuo,
                'dan_blue': b_dan, 'tuo_blue': b_tuo,
                'notes': n_notes, 'amount': n_notes * 2,
            }
            logger.info(f"胆拖红胆({len(r_dan)}个): {r_dan}  红拖({len(r_tuo)}选{red_pick}): {r_tuo}")
            if len(b_dan) > 0:
                logger.info(f"胆拖蓝胆({len(b_dan)}个): {b_dan}  蓝拖({len(b_tuo)}选{blue_pick}): {b_tuo}")
            else:
                logger.info(f"胆拖蓝球({len(b_tuo)}选{ml_model.blue_count}): {b_tuo}")
            logger.info(f"共 {n_notes} 注, 投注金额 {n_notes * 2} 元")
            return result
        except Exception as e:
            logger.error(f"胆拖预测失败: {e}")
            return None

    # 复式预测：直接调 predict_compound，输出号码池+注数（不走多组单式循环）
    if compound_red > 0 or compound_blue > 0:
        logger.info(f"使用{model_type}模型复式预测{lottery_type}彩票...")
        try:
            from ml_models import LotteryMLModels
            from scripts.data_analysis import load_lottery_data
            ml_model = LotteryMLModels(lottery_type=lottery_type, model_type=model_type)
            if not ml_model.load_models():
                logger.error(f"模型{model_type}尚未训练，请先训练模型。")
                return None
            df = load_lottery_data(lottery_type)
            recent_data = df.sort_values('期数', ascending=False).head(ml_model.feature_window)
            # 输入为总数（如8 = 8选6），换算为额外个数
            red_extra = max(compound_red - ml_model.red_count, 0)
            blue_extra = max(compound_blue - ml_model.blue_count, 0)
            if red_extra <= 0 and blue_extra <= 0:
                logger.error(f"复式总数未超过单式(红{compound_red}/{ml_model.red_count} 蓝{compound_blue}/{ml_model.blue_count})")
                return None
            red_numbers, blue_numbers = ml_model.predict_compound(
                recent_data, extra_red=red_extra, extra_blue=blue_extra)
            if red_numbers is None or blue_numbers is None:
                logger.error(f"复式预测失败")
                return None
            from math import comb
            n_notes = comb(len(red_numbers), ml_model.red_count) * comb(len(blue_numbers), ml_model.blue_count)
            result = {
                'compound_red': red_numbers,
                'compound_blue': blue_numbers,
                'notes': n_notes,
                'amount': n_notes * 2,
            }
            logger.info(f"复式红球({len(red_numbers)}选{ml_model.red_count}): {red_numbers}")
            logger.info(f"复式蓝球({len(blue_numbers)}选{ml_model.blue_count}): {blue_numbers}")
            logger.info(f"共 {n_notes} 注, 投注金额 {n_notes * 2} 元")
            return result
        except Exception as e:
            logger.error(f"复式预测失败: {e}")
            return None

    logger.info(f"使用{model_type}模型预测{lottery_type}彩票...")
    
    from lottery_predictor_app_new import predict_next_draw as app_predict
    results = app_predict(lottery_type, model_type)
    
    if results:
        logger.info(f"预测结果: {results}")
    else:
        logger.error(f"预测失败")
    
    return results

def run_backtest(lottery_type, model_type, periods=None, output=None):
    """历史数据回测：逐期滚动预测并与实际开奖对比"""
    logger.info(f"开始回测{lottery_type}彩票的{model_type}模型...")

    from backtest import run_backtest as bt_run
    result = bt_run(lottery_type, model_type, periods=periods, output_path=output,
                    log_callback=lambda msg: logger.info(msg))
    if result:
        logger.info("回测完成")
    else:
        logger.error("回测失败")
    return result is not None

def run_app():
    """运行完整的GUI应用程序"""
    logger.info("启动彩票预测应用程序...")
    
    from lottery_predictor_app_new import main as app_main
    app_main()

def verify_records_cli():
    """命令行核验预测记录：实际生成过的预测 vs 对应期开奖"""
    logger.info("开始核验预测记录...")
    from prediction_records import verify_records, format_verify_summary
    report = verify_records(log_callback=lambda msg: logger.info(msg))
    if report is None:
        logger.error("核验失败：数据加载异常")
        return False
    print("\n" + format_verify_summary(report))
    return True

def main():
    """主函数，解析命令行参数并运行对应功能"""
    parser = argparse.ArgumentParser(description='彩票预测系统')
    

    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
   
    fetch_parser = subparsers.add_parser('fetch', help='获取彩票数据')
    fetch_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    
 
    train_parser = subparsers.add_parser('train', help='训练预测模型')
    train_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    train_parser.add_argument('--model', default='lightgbm', 
                             choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                             help='模型类型')
    
  
    predict_parser = subparsers.add_parser('predict', help='预测下一期彩票号码')
    predict_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    predict_parser.add_argument('--model', default='lightgbm',
                               choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                               help='模型类型')
    predict_parser.add_argument('--compound-red', type=int, default=0, metavar='N',
                               help='复式红球总数(如ssq传8 = 8个红球里选6, 0=不启用)')
    predict_parser.add_argument('--compound-blue', type=int, default=0, metavar='N',
                               help='复式蓝球总数(如ssq传2 = 2个蓝球里选1, 0=不启用)')
    predict_parser.add_argument('--dantuo', action='store_true',
                               help='胆拖模式：胆码固定，拖码组合（与复式互斥）')
    predict_parser.add_argument('--dan-red', type=int, default=2, metavar='N',
                               help='胆拖红胆个数(SSQ 1-5, DLT 1-4)')
    predict_parser.add_argument('--tuo-red', type=int, default=6, metavar='N',
                               help='胆拖红拖个数')
    predict_parser.add_argument('--dan-blue', type=int, default=0, metavar='N',
                               help='胆拖蓝胆个数(SSQ只能0, DLT 0-1)')
    predict_parser.add_argument('--tuo-blue', type=int, default=2, metavar='N',
                               help='胆拖蓝拖个数')
    
    
    backtest_parser = subparsers.add_parser('backtest', help='历史数据回测')
    backtest_parser.add_argument('lottery_type', choices=['dlt', 'ssq'], help='彩票类型')
    backtest_parser.add_argument('--model', default='ensemble',
                                choices=['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble'],
                                help='模型类型')
    backtest_parser.add_argument('--periods', type=int, default=None,
                                help='仅回测最近N期(默认全部)')
    backtest_parser.add_argument('--output', default=None, help='报告JSON保存路径')

    records_parser = subparsers.add_parser('records', help='核验预测记录(实际生成过的预测 vs 开奖)')
    records_parser.add_argument('--verify', action='store_true', default=True,
                                help='核验并显示摘要(默认)')

    app_parser = subparsers.add_parser('app', help='运行GUI应用程序')
    

    args = parser.parse_args()
    
    
    setup_environment()
    
 
    if args.command == 'fetch':
        return fetch_data(args.lottery_type)
    elif args.command == 'train':
        return train_model(args.lottery_type, args.model)
    elif args.command == 'predict':
        return predict(args.lottery_type, args.model,
                       compound_red=args.compound_red, compound_blue=args.compound_blue,
                       dantuo=args.dantuo, dan_red=args.dan_red, tuo_red=args.tuo_red,
                       dan_blue=args.dan_blue, tuo_blue=args.tuo_blue)
    elif args.command == 'backtest':
        return run_backtest(args.lottery_type, args.model,
                            periods=args.periods, output=args.output)
    elif args.command == 'records':
        return verify_records_cli()
    elif args.command == 'app':
        return run_app()
    else:
      
        logger.info("未指定子命令，默认启动GUI应用程序")
        return run_app()

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"程序开始运行: {start_time}")
        result = main()
        end_time = datetime.now()
        logger.info(f"程序结束运行: {end_time}")
        logger.info(f"总运行时间: {end_time - start_time}")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"程序运行出错: {e}")
        sys.exit(1) 