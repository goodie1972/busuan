#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彩票数据获取与模型训练一体化脚本
"""
import os
import sys
import argparse
import subprocess
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('fetch_and_train')

def fetch_data(lottery_type):
    """
    获取彩票历史数据
    
    Args:
        lottery_type: 彩票类型，'dlt'表示大乐透，'ssq'表示双色球，'all'表示两者都获取
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if lottery_type in ['dlt', 'all']:
        logger.info("开始获取大乐透历史数据...")
        fetch_script = os.path.join(script_dir, 'scripts', 'dlt', 'fetch_dlt_data.py')
        
        try:
            subprocess.run([sys.executable, fetch_script], check=True)
            logger.info("大乐透历史数据获取完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"大乐透历史数据获取失败: {str(e)}")
            return False
    
    if lottery_type in ['ssq', 'all']:
        logger.info("开始获取双色球历史数据...")
        fetch_script = os.path.join(script_dir, 'scripts', 'ssq', 'fetch_ssq_data.py')
        
        try:
            subprocess.run([sys.executable, fetch_script], check=True)
            logger.info("双色球历史数据获取完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"双色球历史数据获取失败: {str(e)}")
            return False
    
    return True

def train_model(lottery_type, use_gpu, epochs):
    """训练预测模型"""
    logger.info(f"开始训练{'双色球' if lottery_type == 'ssq' else '大乐透'}模型...")
    
    # 检查GPU实际可用性
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_device = torch.cuda.get_device_name(0)
                logger.info(f"检测到可用GPU: {gpu_device}")
            else:
                logger.warning("未检测到CUDA可用的GPU，将使用CPU训练")
                use_gpu = False
        except Exception as e:
            logger.warning(f"检查GPU可用性时出错: {str(e)}")
            logger.warning("将使用CPU训练")
            use_gpu = False
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'scripts/{lottery_type}/train_{lottery_type}_model.py')
    
    cmd = [sys.executable, script_path, '--epochs', str(epochs)]
    if use_gpu:
        cmd.append('--gpu')
    
    try:
        subprocess.run(cmd, check=True)
        model_file = os.path.join('model', lottery_type, f'{lottery_type}_model.pth')
        if os.path.exists(model_file):
            logger.info(f"{'双色球' if lottery_type == 'ssq' else '大乐透'}模型训练完成")
            logger.info(f"{'双色球' if lottery_type == 'ssq' else '大乐透'}模型文件已生成: {model_file}")
            return True
        else:
            logger.error(f"{'双色球' if lottery_type == 'ssq' else '大乐透'}模型文件未生成")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{'双色球' if lottery_type == 'ssq' else '大乐透'}模型训练失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='彩票数据获取与模型训练一体化工具')
    parser.add_argument('--type', type=str, choices=['dlt', 'ssq', 'all'], default='all',
                        help='彩票类型: dlt(大乐透), ssq(双色球), all(两者都处理)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU训练')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过数据获取步骤')
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"开始处理 {args.type} 数据和模型，GPU: {args.gpu}，训练轮数: {args.epochs}")
    
    # 获取数据
    if not args.skip_fetch:
        logger.info("步骤1: 获取历史数据")
        if not fetch_data(args.type):
            logger.error("数据获取失败，流程中断")
            return
    else:
        logger.info("跳过数据获取步骤")
    
    # 训练模型
    logger.info("步骤2: 训练预测模型")
    if not train_model(args.type, args.gpu, args.epochs):
        logger.error("模型训练失败，流程中断")
        return
    
    # 检查模型文件
    if args.type in ['dlt', 'all']:
        model_path = os.path.join('model', 'dlt', 'dlt_model.pth')
        if os.path.exists(model_path):
            logger.info(f"大乐透模型文件已生成: {model_path}")
        else:
            logger.warning(f"未找到大乐透模型文件: {model_path}")
    
    if args.type in ['ssq', 'all']:
        model_path = os.path.join('model', 'ssq', 'ssq_model.pth')
        if os.path.exists(model_path):
            logger.info(f"双色球模型文件已生成: {model_path}")
        else:
            logger.warning(f"未找到双色球模型文件: {model_path}")
    
    end_time = time.time()
    logger.info(f"所有处理完成，耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main() 