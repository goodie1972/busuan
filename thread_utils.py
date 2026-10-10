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


class AutoPredictThread(QThread):
    """一键智能预测线程：自动 fetch → train → predict → 存档
    全流程在后台完成，通过信号通知主线程更新 UI。
    """
    log_signal = pyqtSignal(str)
    step_signal = pyqtSignal(str)       # 当前步骤名
    finished_signal = pyqtSignal(bool, str)  # (是否成功, 结果文本)
    predictions_signal = pyqtSignal(object)  # 结构化预测列表 [(red, blue), ...]，供主线程存档

    def __init__(self, lottery_type, model_type='gbdt', num_predictions=3,
                 use_gpu=False, skip_fetch=False, skip_train=False):
        super().__init__()
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.num_predictions = num_predictions
        self.use_gpu = use_gpu
        self.skip_fetch = skip_fetch
        self.skip_train = skip_train
        self.should_terminate = False

    def run(self):
        try:
            import subprocess, sys as _sys

            # ===== 第1步：更新数据 =====
            if not self.skip_fetch:
                self.step_signal.emit("正在更新数据...")
                self.log_signal.emit(f"═══ 第1步/3: 更新{name_path[self.lottery_type]['name']}历史数据 ═══")

                if getattr(_sys, 'frozen', False):
                    base = os.path.dirname(_sys.executable)
                else:
                    base = "."

                script = os.path.join(base, "scripts", self.lottery_type,
                                      f"fetch_{self.lottery_type}_data.py")
                self.log_signal.emit(f"执行: {script}")

                proc = subprocess.Popen(
                    [_sys.executable, script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, bufsize=1,
                )
                for line in iter(proc.stdout.readline, ''):
                    if self.should_terminate:
                        proc.terminate()
                        self.log_signal.emit("已终止。")
                        self.finished_signal.emit(False, "")
                        return
                    if line:
                        self.log_signal.emit(line.strip())
                proc.wait()
                if proc.returncode != 0:
                    self.log_signal.emit(f"数据更新失败 (exit {proc.returncode})")
                    self.finished_signal.emit(False, "")
                    return
                self.log_signal.emit("数据更新完成。")
            else:
                self.log_signal.emit("跳过数据更新。")

            if self.should_terminate:
                self.finished_signal.emit(False, "")
                return

            # ===== 第2步：训练模型 =====
            if not self.skip_train:
                self.step_signal.emit("正在训练模型...")
                self.log_signal.emit(f"═══ 第2步/3: 训练{MODEL_TYPES.get(self.model_type, self.model_type)}模型 ═══")

                from scripts.data_analysis import load_lottery_data
                df = load_lottery_data(self.lottery_type)
                if df is None or df.empty:
                    self.log_signal.emit("加载数据失败。")
                    self.finished_signal.emit(False, "")
                    return

                self.log_signal.emit(f"已加载 {len(df)} 条历史数据。")

                def _log(msg):
                    if msg:
                        self.log_signal.emit(msg)
                    if self.should_terminate:
                        raise Exception("用户终止")

                ml_model = LotteryMLModels(
                    lottery_type=self.lottery_type,
                    model_type=self.model_type,
                    log_callback=_log,
                    use_gpu=self.use_gpu,
                )
                ml_model.train(df)
                self.log_signal.emit("模型训练完成。")
            else:
                self.log_signal.emit("跳过模型训练。")

            if self.should_terminate:
                self.finished_signal.emit(False, "")
                return

            # ===== 第3步：生成预测 =====
            self.step_signal.emit("正在生成预测...")
            self.log_signal.emit(f"═══ 第3步/3: 生成{self.num_predictions}组预测 ═══")

            from scripts.data_analysis import load_lottery_data
            df = load_lottery_data(self.lottery_type)

            ml_model = LotteryMLModels(
                lottery_type=self.lottery_type,
                model_type=self.model_type,
                log_callback=lambda m: self.log_signal.emit(m) if m else None,
                use_gpu=self.use_gpu,
            )
            if not ml_model.load_models():
                self.log_signal.emit(f"模型加载失败，请先训练 {self.model_type} 模型。")
                self.finished_signal.emit(False, "")
                return

            import pandas as pd
            # 取最近 feature_window 期做输入
            feature_window = getattr(ml_model, 'feature_window', 10)
            recent = df.sort_values('期数', ascending=False).head(feature_window)
            recent = recent.sort_values('期数', ascending=True)

            lottery_name = name_path[self.lottery_type]['name']
            result_lines = [f"【一键智能预测】{lottery_name} · {MODEL_TYPES.get(self.model_type, self.model_type)}"]
            result_lines.append(f"基于最新 {len(df)} 条历史数据，生成 {self.num_predictions} 组预测：\n")

            all_predictions = []
            for i in range(self.num_predictions):
                red, blue = ml_model.predict(recent, variation=i)
                if self.lottery_type == 'dlt':
                    line = f"  第{i+1}组: {' '.join(f'{n:02d}' for n in red)} + {' '.join(f'{n:02d}' for n in blue)}"
                else:
                    line = f"  第{i+1}组: {' '.join(f'{n:02d}' for n in red)} + {blue[0]:02d}"
                result_lines.append(line)
                all_predictions.append((red, blue))

            result_text = "\n".join(result_lines)
            self.log_signal.emit("预测生成完成。")
            self.predictions_signal.emit(all_predictions)
            self.finished_signal.emit(True, result_text)

        except Exception as e:
            if "用户终止" in str(e):
                self.log_signal.emit("一键预测已被终止。")
            else:
                self.log_signal.emit(f"一键预测出错: {str(e)}")
                import traceback
                self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, "")

    def terminate(self):
        self.should_terminate = True
        super().terminate()


class DataCheckThread(QThread):
    """轻量检查官方是否有新开奖数据（只取1条，不下载全量）"""
    new_data_signal = pyqtSignal(bool, str)  # (是否有新数据, 提醒文本)

    def __init__(self, lottery_type='dlt'):
        super().__init__()
        self.lottery_type = lottery_type
        self.should_terminate = False

    def run(self):
        try:
            import requests
            import pandas as pd

            # 1. 读取本地最新期数
            local_file = os.path.join('scripts', self.lottery_type,
                                      f'{self.lottery_type}_history.csv')
            if not os.path.exists(local_file):
                self.new_data_signal.emit(False, "")
                return

            local_latest = 0
            for enc in ['utf-8', 'gbk', 'utf-8-sig']:
                try:
                    df = pd.read_csv(local_file, encoding=enc)
                    local_latest = int(df['期数'].max())
                    break
                except Exception:
                    continue

            # 2. 轻量获取官方最新期数（只取1条）
            if self.lottery_type == 'ssq':
                remote_latest = self._check_ssq()
            else:
                remote_latest = self._check_dlt()

            if remote_latest is None:
                self.new_data_signal.emit(False, "")
                return

            # 3. 比较
            if remote_latest > local_latest:
                lottery_name = name_path[self.lottery_type]['name']
                msg = f"{lottery_name}有新开奖（第{remote_latest}期），建议更新数据"
                self.new_data_signal.emit(True, msg)
            else:
                self.new_data_signal.emit(False, "")

        except Exception:
            self.new_data_signal.emit(False, "")

    def _check_ssq(self):
        """轻量获取双色球最新期数（只取1条）"""
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.cwl.gov.cn/",
        }
        s = requests.Session()
        s.headers.update(headers)
        s.get("https://www.cwl.gov.cn/", timeout=10)
        url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        params = {"name": "ssq", "pageNo": "1", "pageSize": "1", "systemType": "PC"}
        resp = s.get(url, params=params, timeout=15)
        data = resp.json()
        items = data.get("result", [])
        if items:
            code = items[0].get("code", "")
            if code:
                sc = str(code).strip()
                if len(sc) >= 7 and sc.isdigit():
                    return int(sc[1:])
                return int(sc)
        return None

    def _check_dlt(self):
        """轻量获取大乐透最新期数（只取1条）"""
        import requests
        url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
        params = {"gameNo": "85", "provinceId": "0", "pageSize": "1",
                   "isVerify": "1", "pageNo": "1"}
        headers = {"User-Agent": "Mozilla/5.0",
                    "Referer": "https://static.sporttery.cn/"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        data = resp.json()
        items = data.get("value", {}).get("list", [])
        if items:
            draw_num = items[0].get("lotteryDrawNum", "")
            if draw_num:
                return int(draw_num)
        return None


# ============ 模型优化相关线程 ============

class CalibrationThread(QThread):
    """模型校准后台线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)  # 是否成功

    def __init__(self, lottery_type='ssq'):
        super().__init__()
        self.lottery_type = lottery_type
        self.should_terminate = False

    def run(self):
        try:
            from model_calibration import calibrate_lottery, get_calibration_info, collect_calibration_data

            self.log_signal.emit(f"开始校准 {self.lottery_type} ...")

            # 先收集数据看有多少
            red_data, blue_data = collect_calibration_data(self.lottery_type)
            red_count = sum(len(v) for v in red_data.values()) if red_data else 0
            blue_count = sum(len(v) for v in blue_data.values()) if blue_data else 0
            self.log_signal.emit(f"红球样本: {red_count}, 蓝球样本: {blue_count}")

            if red_count == 0 and blue_count == 0:
                # 给出精确诊断，避免"请先核验"这类误导性提示
                from prediction_records import load_records
                all_records = load_records()
                lt_records = [r for r in all_records
                              if r.get('lottery_type') in (None, self.lottery_type)]
                verified = [r for r in lt_records if r.get('target_period') is not None]
                with_proba = [r for r in verified if r.get('red_probabilities')
                              or r.get('blue_probabilities')]
                if not lt_records:
                    self.log_signal.emit("没有可用的校准数据：该彩种暂无任何预测记录")
                elif not verified:
                    self.log_signal.emit("没有可用的校准数据：已有预测记录但都未核验，"
                                         "请先在'历史回测'页核验")
                elif not with_proba:
                    self.log_signal.emit("没有可用的校准数据：已核验的记录都不含模型概率"
                                         "（概率存档功能启用前的旧记录无法补充概率）")
                    self.log_signal.emit("解决：从下次起正常生成预测+核验即可自动积累，"
                                         "建议积累30注以上再训练校准模型")
                else:
                    self.log_signal.emit("没有可用的校准数据：已核验记录的概率数据不完整")
                self.finished_signal.emit(False)
                return

            success = calibrate_lottery(self.lottery_type)
            if success:
                info = get_calibration_info(self.lottery_type)
                self.log_signal.emit(f"校准完成! 红球: {'已校准' if info.get('red_calibrated') else '未校准'}, "
                                     f"蓝球: {'已校准' if info.get('blue_calibrated') else '未校准'}")
            else:
                self.log_signal.emit("校准失败或数据不足")

            self.finished_signal.emit(success)

        except Exception as e:
            self.log_signal.emit(f"校准出错: {e}")
            self.finished_signal.emit(False)


class FeatureAnalysisThread(QThread):
    """特征分析后台线程"""
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)  # 特征重要性列表
    drift_signal = pyqtSignal(dict)   # 漂移检测结果
    finished_signal = pyqtSignal(bool)

    def __init__(self, lottery_type='ssq', top_k=20, drift_window=30, check_drift=False):
        super().__init__()
        self.lottery_type = lottery_type
        self.top_k = top_k
        self.drift_window = drift_window
        self.check_drift = check_drift
        self.should_terminate = False

    def run(self):
        try:
            from feature_analysis import run_feature_analysis, select_top_features, detect_feature_drift

            # 仅漂移检测模式：直接检测，不跑完整分析
            if self.check_drift:
                self.log_signal.emit(f"检测特征漂移 {self.lottery_type} (窗口={self.drift_window})...")
                try:
                    drift_result = detect_feature_drift(self.lottery_type, self.drift_window)
                    self.drift_signal.emit(drift_result)
                except Exception as e:
                    self.drift_signal.emit({'error': str(e)})
                self.finished_signal.emit(True)
                return

            self.log_signal.emit(f"开始特征分析 {self.lottery_type}...")

            # 运行特征分析（内部已包含漂移检测）
            result = run_feature_analysis(self.lottery_type)
            if result.get('status') == 'completed':
                selected = result.get('selected_features', [])
                self.log_signal.emit(f"特征分析完成，选择了 {len(selected)} 个特征")
            else:
                self.log_signal.emit(f"特征分析: {result.get('error', '未知')}")

            # 获取 Top 特征
            try:
                top_features = select_top_features(self.lottery_type, top_k=self.top_k)
                self.log_signal.emit(f"已获取 Top {self.top_k} 特征")
            except Exception as e:
                self.log_signal.emit(f"获取 Top 特征失败: {e}")
                top_features = []

            self.result_signal.emit({
                'status': result.get('status'),
                'top_features': top_features,
                'all_features': result.get('feature_importance', [])
            })

            # 把完整分析中已算出的漂移结果回传给UI
            if result.get('feature_drift'):
                self.drift_signal.emit(result['feature_drift'])

            self.finished_signal.emit(True)

        except Exception as e:
            self.log_signal.emit(f"特征分析出错: {e}")
            self.finished_signal.emit(False)


class RetrainThread(QThread):
    """反馈重训练后台线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, lottery_type=None, use_gpu=False, model_types=None, retrain_all=False):
        super().__init__()
        self.lottery_type = lottery_type
        self.use_gpu = use_gpu
        self.model_types = model_types
        self.retrain_all = retrain_all
        self.should_terminate = False

    def run(self):
        try:
            from retrain_models_with_feedback import retrain_lottery_models, retrain_all_lotteries

            if self.retrain_all:
                self.log_signal.emit("开始重训练全部彩票模型...")
                # retrain_all_lotteries 只接受 use_gpu，返回 {lottery: bool}
                results = retrain_all_lotteries(use_gpu=self.use_gpu)
                all_ok = all(results.values()) if results else False
                for lt, ok in results.items():
                    self.log_signal.emit(f"  {lt.upper()}: {'成功' if ok else '失败'}")
                self.log_signal.emit(f"全部彩票重训练{'完成' if all_ok else '部分失败'}")
                self.finished_signal.emit(all_ok)
            else:
                self.log_signal.emit(f"开始重训练 {self.lottery_type} 模型...")
                success = retrain_lottery_models(
                    lottery_type=self.lottery_type,
                    use_gpu=self.use_gpu,
                    model_types=self.model_types
                )
                self.log_signal.emit(f"{self.lottery_type} 重训练{'完成' if success else '失败'}")
                self.finished_signal.emit(success)

        except Exception as e:
            self.log_signal.emit(f"重训练出错: {e}")
            self.finished_signal.emit(False)