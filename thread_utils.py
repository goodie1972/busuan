# -*- coding:utf-8 -*-
"""
Thread Utilities for Model Training and Data Updates
Author: Yang Zhao
"""
import os
import sys
import time
import logging
import subprocess
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from ml_models import LotteryMLModels, MODEL_TYPES

from model_utils import name_path

class LogEmitter(QObject):
    """日志发射器，用于在线程中发送日志信息"""
    new_log = pyqtSignal(str)

class TrainModelThread(QThread):
    """模型训练线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    pause_signal = pyqtSignal(bool)
    
    def __init__(self, lottery_type, use_gpu=False, model_type='lstm-crf'):
        super().__init__()
        self.lottery_type = lottery_type
        self.use_gpu = use_gpu
        self.model_type = model_type
        self.is_paused = False
        self.should_terminate = False
        
    def run(self):
        try:
            # 检查模型类型
            if self.model_type == 'lstm-crf':
                # 训练LSTM-CRF模型
                self._train_lstm_crf()
            else:
                # 训练机器学习模型
                self._train_ml_model()
        except Exception as e:
            self.log_signal.emit(f"训练过程中出错: {str(e)}")
        finally:
            self.finished_signal.emit()
            
    def _train_lstm_crf(self):
        """训练LSTM-CRF模型"""
        # 检查应用是否正在以打包后的状态运行
        if getattr(sys, 'frozen', False):
            # 在打包环境中运行
            if self.lottery_type == 'dlt':
                script_path = os.path.join(os.path.dirname(sys.executable), "scripts", "dlt", "train_dlt_model.py")
            else:
                script_path = os.path.join(os.path.dirname(sys.executable), "scripts", "ssq", "train_ssq_model.py")
        else:
            # 在开发环境中运行
            if self.lottery_type == 'dlt':
                script_path = "./scripts/dlt/train_dlt_model.py"
            else:
                script_path = "./scripts/ssq/train_ssq_model.py"
        
        # GPU可用性检查
        gpu_available = torch.cuda.is_available()
        if self.use_gpu and not gpu_available:
            self.log_signal.emit("警告: 已选择使用GPU但CUDA不可用，将使用CPU训练。")
            self.use_gpu = False
        
        # 构建命令，根据是否使用GPU添加--gpu参数
        command = [sys.executable, script_path]
        if self.use_gpu:
            command.append("--gpu")
            self.log_signal.emit(f"GPU训练已启用，使用设备: {torch.cuda.get_device_name(0)}")
        else:
            self.log_signal.emit("使用CPU训练")
        
        self.log_signal.emit(f"启动训练脚本: {' '.join(command)}")
        
        try:
            # 使用Popen启动进程，捕获输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出
            for line in iter(process.stdout.readline, ''):
                if self.should_terminate:
                    process.terminate()
                    self.log_signal.emit("训练已终止。")
                    break
                
                if line:
                    self.log_signal.emit(line.strip())
                
                # 处理暂停
                while self.is_paused and not self.should_terminate:
                    time.sleep(0.1)
            
            # 等待进程结束
            process.wait()
            
            if process.returncode != 0 and not self.should_terminate:
                self.log_signal.emit(f"训练脚本以非零退出码结束: {process.returncode}")
            elif not self.should_terminate:
                self.log_signal.emit("训练完成。")
                
        except Exception as e:
            self.log_signal.emit(f"运行训练脚本时出错: {str(e)}")
    
    def _train_ml_model(self):
        """训练机器学习模型"""
        try:
            from scripts.data_analysis import load_lottery_data
            
            # GPU可用性检查
            gpu_available = torch.cuda.is_available()
            if self.use_gpu and not gpu_available:
                self.log_signal.emit("警告: 已选择使用GPU但CUDA不可用，将使用CPU训练。")
                self.use_gpu = False
            
            if self.use_gpu:
                self.log_signal.emit(f"GPU训练已启用，使用设备: {torch.cuda.get_device_name(0)}")
            else:
                self.log_signal.emit("使用CPU训练")
            
            self.log_signal.emit(f"开始训练{self.lottery_type}预测模型({MODEL_TYPES[self.model_type]})...")
            
            # 加载数据
            df = load_lottery_data(self.lottery_type)
            if df is None or df.empty:
                self.log_signal.emit("加载数据失败，请检查数据文件。")
                return
                
            self.log_signal.emit(f"成功加载{len(df)}条历史数据。")
            
            
            def enhanced_log(message):
                
                if message:
                    self.log_signal.emit(message)
                
               
                if self.is_paused and not self.should_terminate:
                    self.log_signal.emit("训练已暂停，等待恢复...")
                    while self.is_paused and not self.should_terminate:
                        time.sleep(0.1)
                    if not self.should_terminate:
                        self.log_signal.emit("训练已恢复...")
                
               
                if self.should_terminate:
                    raise Exception("训练被用户终止")
            
           
            ml_model = LotteryMLModels(
                lottery_type=self.lottery_type, 
                model_type=self.model_type,
                log_callback=enhanced_log,  
                use_gpu=self.use_gpu 
            )
            
     
            self.log_signal.emit("准备训练数据...")
            
         
            if self.should_terminate:
                self.log_signal.emit("训练已终止。")
                return
            
            # 开始训练
            try:
                ml_model.train(df)
                
                if not self.should_terminate:
                    self.log_signal.emit(f"{MODEL_TYPES[self.model_type]}模型训练完成。")
            except Exception as e:
                if str(e) == "训练被用户终止":
                    self.log_signal.emit("训练已被用户终止。")
                else:
                    raise  # 重新抛出其他异常
            
        except Exception as e:
            if not self.should_terminate:  # 只在非用户终止的情况下显示错误
                self.log_signal.emit(f"训练{MODEL_TYPES[self.model_type]}模型时出错: {str(e)}")
                import traceback
                self.log_signal.emit(traceback.format_exc())
    
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        self.pause_signal.emit(self.is_paused)
        
    def is_paused(self):
        """获取当前暂停状态"""
        return self.is_paused
        
    def terminate(self):
        """终止线程"""
        self.should_terminate = True
        super().terminate()

class UpdateDataThread(QThread):
    """数据更新线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, lottery_type):
        super().__init__()
        self.lottery_type = lottery_type
        
    def run(self):
        """运行线程"""
        self.log_signal.emit(f"开始更新{name_path[self.lottery_type]['name']}历史数据...")
        
        # 检查应用是否正在以打包后的状态运行
        if getattr(sys, 'frozen', False):
            # 在打包环境中运行
            if self.lottery_type == 'dlt':
                script_path = os.path.join(os.path.dirname(sys.executable), "scripts", "dlt", "fetch_dlt_data.py")
            else:
                script_path = os.path.join(os.path.dirname(sys.executable), "scripts", "ssq", "fetch_ssq_data.py")
        else:
            # 在开发环境中运行
            if self.lottery_type == 'dlt':
                script_path = "./scripts/dlt/fetch_dlt_data.py"
            else:
                script_path = "./scripts/ssq/fetch_ssq_data.py"
        
        self.log_signal.emit(f"启动数据更新脚本: {script_path}")
        
        try:
            # 执行数据更新脚本
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_signal.emit(line.strip())
            
            # 等待进程结束
            process.wait()
            
            if process.returncode != 0:
                self.log_signal.emit(f"数据更新脚本以非零退出码结束: {process.returncode}")
            else:
                self.log_signal.emit("数据更新完成。")
                
        except Exception as e:
            self.log_signal.emit(f"更新数据时出错: {str(e)}")
        finally:
            self.finished_signal.emit()


class BacktestThread(QThread):
    """历史回测线程（逐期滚动预测并与真实开奖对比）"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)  # True=回测成功，False=失败

    def __init__(self, lottery_type, model_type, periods=None, seed=42):
        super().__init__()
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.periods = periods
        self.seed = seed
        self.report = None  # 回测报告 dict（run 内填充，finished 后主线程读取）

    def run(self):
        """运行回测"""
        try:
            # 延迟 import：backtest 模块内部避免加载 scripts/data_analysis
            # （其 PyQt5 依赖与 torch 的 DLL 初始化冲突，见 pyqt5-torch-dll-conflict）
            import io
            import contextlib
            from backtest import run_backtest
            # 重定向 stdout/stderr：ml_models 内部打印的模型信息与
            # sklearn joblib 的 [Parallel] 进度会刷爆终端，回测只保留
            # 经 log_callback 转发的进度日志
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                report = run_backtest(
                    self.lottery_type,
                    self.model_type,
                    periods=self.periods,
                    seed=self.seed,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )
            self.report = report
            self.finished_signal.emit(report is not None)
        except Exception as e:
            self.log_signal.emit(f"回测过程中出错: {str(e)}")
            try:
                import traceback
                self.log_signal.emit(traceback.format_exc())
            except Exception:
                pass
            self.finished_signal.emit(False) 