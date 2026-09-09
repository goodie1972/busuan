#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彩票预测模型训练脚本
"""
import os
import sys
import argparse
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('train_models')

def train_model(model_type, use_gpu=False, epochs=100):
    """
    训练指定类型的彩票预测模型
    
    Args:
        model_type: 模型类型，'dlt'表示大乐透，'ssq'表示双色球，'all'表示两者都训练
        use_gpu: 是否使用GPU训练
        epochs: 训练轮数
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type in ['dlt', 'all']:
        logger.info("开始训练大乐透模型...")
        dlt_script = os.path.join(script_dir, 'scripts', 'dlt', 'train_dlt_model.py')
        
        cmd = [sys.executable, dlt_script]
        if use_gpu:
            cmd.append('--gpu')
        cmd.extend(['--epochs', str(epochs)])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("大乐透模型训练完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"大乐透模型训练失败: {str(e)}")
    
    if model_type in ['ssq', 'all']:
        logger.info("开始训练双色球模型...")
        ssq_script = os.path.join(script_dir, 'scripts', 'ssq', 'train_ssq_model.py')
        
        cmd = [sys.executable, ssq_script]
        if use_gpu:
            cmd.append('--gpu')
        cmd.extend(['--epochs', str(epochs)])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("双色球模型训练完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"双色球模型训练失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练彩票预测模型')
    parser.add_argument('--type', type=str, choices=['dlt', 'ssq', 'all'], default='all',
                        help='要训练的模型类型: dlt(大乐透), ssq(双色球), all(两者都训练)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU训练')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    args = parser.parse_args()
    
    logger.info(f"开始训练 {args.type} 模型，轮数: {args.epochs}，GPU: {args.gpu}")
    train_model(args.type, args.gpu, args.epochs)
    logger.info("所有模型训练完成")

if __name__ == "__main__":
    main() 