# -*- coding:utf-8 -*-
"""
Lottery Predictor Application
Author: Yang Zhao
"""
import sys
import os
import io
import pandas as pd
import numpy as np
import torch
import time
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QComboBox, QWidget, QTextEdit, QSpinBox, QHBoxLayout,
    QTabWidget, QScrollArea, QGridLayout, QCheckBox, QGroupBox, QFormLayout,
    QMenu, QAction
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer
from PyQt5.QtGui import QPixmap


from model_utils import (
    name_path, load_resources_pytorch, sample_crf_sequences
)
from ml_models import (
    LotteryMLModels, MODEL_TYPES
)
from thread_utils import (
    TrainModelThread, UpdateDataThread, LogEmitter, BacktestThread
)
from prediction_utils import (
    process_predictions, randomize_numbers
)
from theme_manager import ThemeManager, CustomThemeDialog
from ui_components import (
    create_main_tab, create_analysis_tab, create_advanced_statistics_tab,
    create_expected_value_tab, create_backtest_tab,
    create_ziwei_tab, create_meihua_tab
)
from data_processing import (
    process_analysis_data, get_trend_features, prepare_recent_trend_data,
    format_quality_report, format_frequency_stats, format_hot_cold_stats,
    format_gap_stats, format_pattern_stats, format_trend_stats
)
from scripts.data_analysis import (
    plot_frequency_distribution, plot_hot_cold_numbers, 
    plot_gap_statistics, plot_patterns, plot_trend_analysis,
    load_lottery_data
)
from scripts.advanced_statistics import (
    calculate_advanced_statistics, plot_advanced_statistics,
    plot_distribution_analysis
)

# ---------------- 主窗口类 ----------------
class LotteryPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.log_emitter = LogEmitter()
        self.log_emitter.new_log.connect(self.update_log)
        self.train_thread = None
        self.update_thread = None
        self.pause_state = False
        
        # 初始化主题管理器
        self.theme_manager = ThemeManager()
        
        # 获取GPU信息
        self.has_cuda = False
        self.cuda_info = "不可用"
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self.cuda_info = f"可用 ({torch.cuda.get_device_name(0)})"
        except:
            pass
        
        self.initUI()
        
        # 应用当前主题
        self.apply_theme()
        
        app = QApplication.instance()
        app.aboutToQuit.connect(self.cleanup_resources)
        
        self.current_stats = None
        self.current_df = None
        self.enhanced_df = None
        self.current_lottery_type = None
        
        # 初始化机器学习模型实例字典
        self.ml_models = {}
        
        # 保存统计数据窗口的引用
        self.stats_window = None
        
        # 连接日志框的自定义右键菜单信号
        self.log_box.customContextMenuRequested.connect(self.show_log_context_menu)
        
        # 连接期望值模型标签页的自定义右键菜单信号
        if hasattr(self, 'ev_log_box'):
            self.ev_log_box.customContextMenuRequested.connect(self.show_ev_log_context_menu)

    def initUI(self):
        self.setWindowTitle(f"卜算 - 彩票娱乐软件 - GPU: {self.cuda_info}")
        self.setGeometry(50, 50, 2112, 1360)

        self.tab_widget = QTabWidget()
        
        # 创建期望值模型标签页 - 放在第二个位置
        self.expectedvalue_tab = QWidget()
        self.ev_predict_button, self.ev_train_button, self.ev_update_data_button, \
        self.ev_lottery_combo, self.ev_prediction_spin, self.ev_gpu_checkbox, \
        self.ev_result_label, self.ev_log_box = create_expected_value_tab(self.expectedvalue_tab)
        
        # 连接期望值模型标签页的信号和槽
        self.ev_predict_button.clicked.connect(self.generate_ev_prediction)
        self.ev_train_button.clicked.connect(self.train_ev_model)
        self.ev_update_data_button.clicked.connect(self.update_lottery_data)
        
        # 创建主标签页 - 放在第一个位置
        self.main_tab = QWidget()
        self.predict_button, self.train_button, self.pause_button, self.analyze_button, self.update_data_button, \
        self.lottery_combo, self.prediction_spin, self.gpu_checkbox, self.result_label, self.log_box, \
        self.theme_combo, self.customize_theme_button, self.model_combo, \
        self.mode_combo, self.compound_red_spin, self.compound_blue_spin, \
        self.dt_red_dan, self.dt_red_tuo, self.dt_blue_dan, self.dt_blue_tuo = create_main_tab(self.main_tab)
        
        # 连接信号和槽
        self.predict_button.clicked.connect(self.generate_prediction)
        self.train_button.clicked.connect(self.train_model)
        self.pause_button.clicked.connect(self.pause_resume_training)
        self.analyze_button.clicked.connect(self.analyze_data)
        self.update_data_button.clicked.connect(self.update_lottery_data)
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        self.customize_theme_button.clicked.connect(self.customize_theme)
        # 模型切换时更新预测模式可用性
        self.model_combo.currentTextChanged.connect(self.update_prediction_modes)
        
        # 创建数据分析标签页
        self.analysis_tab = QWidget()
        self.analysis_combo, self.trend_feature_combo, self.chart_label, self.stats_text, \
        self.advanced_stats_button, self.distribution_analysis_button, \
        self.load_analysis_button, self.analysis_suggestion_button = create_analysis_tab(self.analysis_tab)
        
        # 连接信号和槽
        self.analysis_combo.currentIndexChanged.connect(self.update_analysis_view)
        self.trend_feature_combo.currentIndexChanged.connect(lambda: self.update_analysis_view(4))
        self.advanced_stats_button.triggered.connect(self.show_advanced_statistics)
        self.distribution_analysis_button.triggered.connect(self.show_distribution_analysis)
        self.load_analysis_button.clicked.connect(self.analyze_data)
        self.analysis_suggestion_button.clicked.connect(self.generate_analysis_suggestions)
        
        # 创建高级统计分析标签页
        self.advanced_stats_tab = QWidget()
        self.advanced_stats_lottery_combo, self.run_advanced_stats_button, \
        self.run_distribution_analysis_button, self.show_stats_data_button, \
        self.stats_result_label, self.adv_stats_suggestion_button = create_advanced_statistics_tab(self.advanced_stats_tab)
        
        # 连接信号和槽
        self.run_advanced_stats_button.triggered.connect(self.run_advanced_statistics)
        self.run_distribution_analysis_button.triggered.connect(self.run_distribution_analysis)
        self.show_stats_data_button.clicked.connect(self.show_statistics_data)
        self.adv_stats_suggestion_button.clicked.connect(self.generate_analysis_suggestions)
        
        # 创建历史回测标签页
        self.backtest_tab = QWidget()
        self.bt_start_button, self.bt_lottery_combo, self.bt_model_combo, \
        self.bt_periods_spin, self.bt_result_text, self.bt_status_label, \
        self.bt_verify_button = create_backtest_tab(self.backtest_tab)
        
        # 连接信号和槽
        self.bt_start_button.clicked.connect(self.start_backtest)
        self.bt_verify_button.clicked.connect(self.verify_prediction_records)
        self.backtest_thread = None
        
        # 添加标签页 - 注意顺序调整：预测放在第一个，期望值放在第二个
        self.tab_widget.addTab(self.main_tab, "预测")
        self.tab_widget.addTab(self.expectedvalue_tab, "期望值模型")
        self.tab_widget.addTab(self.analysis_tab, "数据分析")
        self.tab_widget.addTab(self.advanced_stats_tab, "高级统计")
        self.tab_widget.addTab(self.backtest_tab, "历史回测")
        
        # ===== 紫微斗数标签页 =====
        self.ziwei_tab = QWidget()
        (self.zw_predict_btn, self.zw_reset_btn,
         self.zw_name, self.zw_gender, self.zw_year, self.zw_month,
         self.zw_day, self.zw_hour, self.zw_minute, self.zw_city, self.zw_target,
         self.zw_cards) = create_ziwei_tab(self.ziwei_tab)
        self.zw_predict_btn.clicked.connect(self.generate_ziwei_prediction)
        self.zw_reset_btn.clicked.connect(self.reset_ziwei)
        self.tab_widget.addTab(self.ziwei_tab, "紫微斗数")
        
        # ===== 梅花易数标签页 =====
        self.meihua_tab = QWidget()
        (self.mh_predict_btn, self.mh_reset_btn,
         self.mh_mode, self.mh_year, self.mh_month, self.mh_day,
         self.mh_hour, self.mh_minute,
         self.mh_num1, self.mh_num2, self.mh_num3, self.mh_target,
         self.mh_cards) = create_meihua_tab(self.meihua_tab)
        self.mh_predict_btn.clicked.connect(self.generate_meihua_prediction)
        self.mh_reset_btn.clicked.connect(self.reset_meihua)
        self.tab_widget.addTab(self.meihua_tab, "梅花易数")
        
        # 统一 tab 标签字体到 11pt，并加 padding/高度防截断
        from PyQt5.QtGui import QFont
        tab_font = QFont()
        tab_font.setPointSize(11)
        tab_font.setBold(True)
        self.tab_widget.setFont(tab_font)
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 18px;
                min-height: 22px;
            }
        """)
        
        self.setCentralWidget(self.tab_widget)
        
        self.training_thread = None
        self.is_training_paused = False

    def apply_theme(self):
        """应用当前选择的主题"""
        stylesheet = self.theme_manager.generate_stylesheet()
        # 追加全局字体样式（不被主题覆盖）
        stylesheet += """
            QGroupBox { font-size: 11pt; font-weight: bold; }
            QLabel { font-size: 11pt; }
            QComboBox { font-size: 11pt; }
            QSpinBox { font-size: 11pt; }
            QPushButton { font-size: 11pt; }
            QCheckBox { font-size: 11pt; }
            QTextEdit { font-size: 11pt; }
            QPlainTextEdit { font-size: 11pt; }
            QLineEdit { font-size: 11pt; }
        """
        self.setStyleSheet(stylesheet)
    
    def change_theme(self, theme_name):
        """切换主题"""
        if self.theme_manager.set_theme(theme_name):
            self.apply_theme()
            self.log_emitter.new_log.emit(f"已切换到{theme_name}主题")
    
    def customize_theme(self):
        """打开自定义主题对话框"""
        dialog = CustomThemeDialog(self.theme_manager, self)
        if dialog.exec_():
            # 如果用户点击了保存
            if self.theme_combo.currentText() == "自定义":
                # 如果当前已经是自定义主题，则刷新
                self.apply_theme()
            else:
                # 否则切换到自定义主题
                self.theme_combo.setCurrentText("自定义")
    
    def update_prediction_modes(self):
        """根据模型类型启用/禁用预测模式（单式/复式/胆拖）"""
        model_text = self.model_combo.currentText()
        # 不支持复式/胆拖的模型
        unsupported_models = ["期望值模型", "XGBoost", "紫微斗数", "梅花易数"]
        
        is_unsupported = model_text in unsupported_models
        
        # 复式/胆拖相关控件
        self.compound_red_spin.setEnabled(not is_unsupported)
        self.compound_blue_spin.setEnabled(not is_unsupported)
        self.dt_red_dan.setEnabled(not is_unsupported)
        self.dt_red_tuo.setEnabled(not is_unsupported)
        self.dt_blue_dan.setEnabled(not is_unsupported)
        self.dt_blue_tuo.setEnabled(not is_unsupported)
        
        # 模式下拉框：如果是不支持的模型，强制选"单式"并禁用
        if is_unsupported:
            self.mode_combo.setCurrentText("单式")
        self.mode_combo.setEnabled(not is_unsupported)
        
        # 训练按钮：紫微/梅花无需训练
        no_train_models = ["紫微斗数", "梅花易数"]
        self.train_button.setEnabled(model_text not in no_train_models)
        
        if is_unsupported:
            self.log_emitter.new_log.emit(f"模型 {model_text} 仅支持单式预测，复式/胆拖已禁用")
        elif model_text in no_train_models:
            self.log_emitter.new_log.emit(f"模型 {model_text} 无需训练，直接预测")

    def train_model(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        
        # 获取选择的模型类型
        model_text = self.model_combo.currentText()
        
        # 从文本映射回模型键
        model_type = None
        for key, value in {'lstm-crf': 'LSTM-CRF (默认)'}.items():
            if value == model_text:
                model_type = key
                break
        if model_type is None:
            for key, value in MODEL_TYPES.items():
                if value == model_text:
                    model_type = key
                    break

        # 检查GPU状态
        use_gpu = self.gpu_checkbox.isChecked()
        if use_gpu and not torch.cuda.is_available():
            self.log_box.append("<font color='orange'>警告: GPU被选中但CUDA不可用，将使用CPU进行训练</font>")
            use_gpu = False
        elif use_gpu:
            gpu_info = torch.cuda.get_device_name(0)
            self.log_box.append(f"<font color='green'>使用GPU训练: {gpu_info}</font>")
        else:
            self.log_box.append("<font color='blue'>使用CPU训练</font>")

        self.train_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.lottery_combo.setEnabled(False)
        self.gpu_checkbox.setEnabled(False)
        self.model_combo.setEnabled(False)

        self.log_box.clear()
        
        if model_type == 'lstm-crf':
            # 训练LSTM-CRF模型
            self.log_emitter.new_log.emit(f"开始训练{lottery_name}预测模型(LSTM-CRF)...")
            self.training_thread = TrainModelThread(lottery_type, use_gpu)
            self.training_thread.log_signal.connect(self.update_log)
            self.training_thread.finished_signal.connect(self.on_train_finished)
            self.training_thread.pause_signal.connect(self.on_pause_state_changed)
            self.training_thread.start()
        else:
            # 训练机器学习模型
            self.log_emitter.new_log.emit(f"开始训练{lottery_name}预测模型({MODEL_TYPES[model_type]})...")
            
            # 创建训练线程并传递模型类型
            self.training_thread = TrainModelThread(
                lottery_type=lottery_type, 
                use_gpu=use_gpu,
                model_type=model_type
            )
            self.training_thread.log_signal.connect(self.update_log)
            self.training_thread.finished_signal.connect(self.on_train_finished)
            self.training_thread.pause_signal.connect(self.on_pause_state_changed)
            self.training_thread.start()

    def on_train_finished(self):
        self.train_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.lottery_combo.setEnabled(True)
        self.gpu_checkbox.setEnabled(torch.cuda.is_available())
        self.model_combo.setEnabled(True)
        self.log_emitter.new_log.emit("训练已完成。")
        
    def on_pause_state_changed(self, is_paused):
        if is_paused:
            self.pause_button.setText("继续训练")
        else:
            self.pause_button.setText("暂停训练")

    def generate_prediction(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        num_predictions = self.prediction_spin.value()
        
        # 获取选择的模型类型
        model_text = self.model_combo.currentText()
        
        # 从文本映射回模型键
        model_type = None
        for key, value in {'lstm-crf': 'LSTM-CRF (默认)'}.items():
            if value == model_text:
                model_type = key
                break
        if model_type is None:
            for key, value in MODEL_TYPES.items():
                if value == model_text:
                    model_type = key
                    break
        
        result_text = f"预测的{num_predictions}个{lottery_name}号码：\n"
        # 本次预测的所有号码（红, 蓝），用于自动存档核验
        collected_predictions = []
        
        try:
            if model_type == 'lstm-crf':
                # 使用LSTM-CRF模型预测
                red_model, blue_model, scaler_X = load_resources_pytorch(lottery_type)
                self.log_emitter.new_log.emit(f"已加载 PyTorch 模型和缩放器 for {lottery_name}")
                
                # 获取输入维度
                input_dim = scaler_X.n_features_in_
                self.log_emitter.new_log.emit(f"模型输入维度: {input_dim}")

                for i in range(num_predictions):
                    with torch.no_grad():
                        # 生成随机输入并进行缩放
                        random_input = np.random.normal(0, 1, (1, 10, input_dim))  # 使用10作为序列长度
                        random_input_reshaped = random_input.reshape(-1, input_dim)
                        scaled_input = scaler_X.transform(random_input_reshaped)
                        scaled_input = scaled_input.reshape(1, 10, input_dim)
                        scaled_input = torch.tensor(scaled_input, dtype=torch.float32)

                        # 红球预测
                        red_lstm_out = red_model.lstm(scaled_input)
                        red_fc_out = red_model.fc(red_lstm_out[0])
                        red_emissions = red_fc_out.view(-1, red_model.output_seq_length, red_model.output_dim)
                        red_mask = torch.ones(red_emissions.size()[:2], dtype=torch.uint8)
                        red_sampled_sequences = sample_crf_sequences(red_model.crf, red_emissions, red_mask, num_samples=1, temperature=1.0)

                        if not red_sampled_sequences:
                            raise ValueError("未能生成红球预测序列。")

                        red_predicted = red_sampled_sequences[0]

                        # 蓝球预测
                        blue_lstm_out = blue_model.lstm(scaled_input)
                        blue_fc_out = blue_model.fc(blue_lstm_out[0])
                        blue_emissions = blue_fc_out.view(-1, blue_model.output_seq_length, blue_model.output_dim)
                        blue_mask = torch.ones(blue_emissions.size()[:2], dtype=torch.uint8)
                        blue_sampled_sequences = sample_crf_sequences(blue_model.crf, blue_emissions, blue_mask, num_samples=1, temperature=1.0)

                        if not blue_sampled_sequences:
                            raise ValueError("未能生成蓝球预测序列。")

                        blue_predicted = blue_sampled_sequences[0]

                        # 处理预测结果
                        numbers = process_predictions(red_predicted, blue_predicted, lottery_type)
                        
                        # 增加随机性
                        extra_randomness = randomize_numbers(numbers, lottery_type)
                        
                        # 格式化显示结果
                        if lottery_type == "dlt":
                            result_text += f"  第 {i+1} 组: {' '.join(map(str, extra_randomness[:5]))} + {' '.join(map(str, extra_randomness[5:]))}\n"
                            collected_predictions.append(
                                (list(extra_randomness[:5]), list(extra_randomness[5:])))
                        else:
                            result_text += f"  第 {i+1} 组: {' '.join(map(str, extra_randomness[:6]))} + {str(extra_randomness[6])}\n"
                            collected_predictions.append(
                                (list(extra_randomness[:6]), [int(extra_randomness[6])]))
            else:
                # 使用机器学习模型预测
                model_key = f"{lottery_type}_{model_type}"
                
                # 检查模型是否已经训练
                if model_key not in self.ml_models:
                    self.log_emitter.new_log.emit(f"初始化 {MODEL_TYPES[model_type]} 模型...")
                    
                    # 如果模型不存在，创建一个新实例
                    use_gpu = self.gpu_checkbox.isChecked()
                    self.ml_models[model_key] = LotteryMLModels(
                        lottery_type=lottery_type, 
                        model_type=model_type,
                        log_callback=self.log_emitter.new_log.emit,  # 添加日志回调
                        use_gpu=use_gpu
                    )
                    
                    # 特殊处理期望值模型
                    if model_type == 'expected_value':
                        self.log_emitter.new_log.emit(f"检查期望值模型文件...")
                        
                        # 检查期望值模型目录
                        from expected_value_model import ExpectedValueLotteryModel
                        
                        # 直接创建期望值模型实例
                        ev_model = ExpectedValueLotteryModel(
                            lottery_type=lottery_type,
                            log_callback=self.log_emitter.new_log.emit,
                            use_gpu=use_gpu
                        )
                        
                        # 尝试加载模型
                        load_success = ev_model.load()
                        if load_success:
                            self.log_emitter.new_log.emit(f"期望值模型加载成功")
                            # 将加载的模型设置到ml_models中
                            self.ml_models[model_key].models = {'red': ev_model, 'blue': ev_model}
                            self.ml_models[model_key].raw_models = {'expected_value_model': ev_model}
                        else:
                            raise ValueError(f"期望值模型加载失败，请先训练模型。")
                    else:
                        if not self.ml_models[model_key].load_models():
                            raise ValueError(f"模型{MODEL_TYPES[model_type]}尚未训练，请先训练模型。")
                
                ml_model = self.ml_models[model_key]
                self.log_emitter.new_log.emit(f"已加载 {MODEL_TYPES[model_type]} 模型 for {lottery_name}")
                
                df = load_lottery_data(lottery_type)
                recent_data = df.sort_values('期数', ascending=False).head(ml_model.feature_window)
                
                predict_mode = self.mode_combo.currentText()
                if predict_mode == "复式":
                    # 输入的是"总数"（如双色球红8 = 8选6），换算为额外个数
                    red_total = self.compound_red_spin.value()
                    blue_total = self.compound_blue_spin.value()
                    red_extra = max(red_total - ml_model.red_count, 0)
                    blue_extra = max(blue_total - ml_model.blue_count, 0)
                    red_numbers, blue_numbers = ml_model.predict_compound(
                        recent_data, extra_red=red_extra, extra_blue=blue_extra)
                    if red_numbers is None or blue_numbers is None:
                        raise ValueError("复式预测失败（期望值/xgboost 模型可能不支持复式）")
                    from math import comb
                    n_notes = comb(len(red_numbers), ml_model.red_count) * \
                        comb(len(blue_numbers), ml_model.blue_count)
                    result_text = (
                        f"【复式预测】{MODEL_TYPES[model_type]} 模型\n"
                        f"最新期: {int(df['期数'].max())}\n"
                        f"红球({len(red_numbers)}选{ml_model.red_count}): "
                        f"{' '.join(map(str, red_numbers))}\n"
                        f"蓝球({len(blue_numbers)}选{ml_model.blue_count}): "
                        f"{' '.join(map(str, blue_numbers))}\n"
                        f"共 {n_notes} 注，投注金额 {n_notes * 2} 元\n"
                        f"（按模型概率从高到低取号；复式结果不写入核验存档）"
                    )
                    self.result_label.setText(result_text)
                    self.log_emitter.new_log.emit(
                        f"复式预测完成: 红{len(red_numbers)}选{ml_model.red_count} "
                        f"蓝{len(blue_numbers)}选{ml_model.blue_count}，{n_notes}注/{n_notes * 2}元")
                    return

                if predict_mode == "胆拖":
                    red_dan = self.dt_red_dan.value()
                    red_tuo = self.dt_red_tuo.value()
                    blue_dan = self.dt_blue_dan.value()
                    blue_tuo = self.dt_blue_tuo.value()
                    r_dan, r_tuo, b_dan, b_tuo = ml_model.predict_dantuo(
                        recent_data, red_dan_count=red_dan, red_tuo_count=red_tuo,
                        blue_dan_count=blue_dan, blue_tuo_count=blue_tuo)
                    if r_dan is None:
                        raise ValueError("胆拖预测失败（期望值/xgboost 模型可能不支持胆拖）")
                    from math import comb
                    red_pick = ml_model.red_count - len(r_dan)
                    blue_pick = ml_model.blue_count - len(b_dan)
                    n_notes = comb(len(r_tuo), red_pick) * \
                        (comb(len(b_tuo), blue_pick) if blue_pick > 0 else comb(len(b_tuo), ml_model.blue_count))
                    dan_label = (
                        f"红胆({len(r_dan)}个): {' '.join(map(str, r_dan))}  "
                        f"红拖({len(r_tuo)}选{red_pick}): {' '.join(map(str, r_tuo))}\n")
                    if len(b_dan) > 0:
                        dan_label += (
                            f"蓝胆({len(b_dan)}个): {' '.join(map(str, b_dan))}  "
                            f"蓝拖({len(b_tuo)}选{blue_pick}): {' '.join(map(str, b_tuo))}\n")
                    else:
                        dan_label += f"蓝球({len(b_tuo)}选{ml_model.blue_count}): {' '.join(map(str, b_tuo))}\n"
                    result_text = (
                        f"【胆拖预测】{MODEL_TYPES[model_type]} 模型\n"
                        f"最新期: {int(df['期数'].max())}\n"
                        f"{dan_label}"
                        f"共 {n_notes} 注，投注金额 {n_notes * 2} 元\n"
                        f"（胆码固定，拖码按模型概率排序；胆拖结果不写入核验存档）"
                    )
                    self.result_label.setText(result_text)
                    self.log_emitter.new_log.emit(
                        f"胆拖预测完成: 红胆{len(r_dan)}+红拖{len(r_tuo)} "
                        f"蓝胆{len(b_dan)}+蓝拖{len(b_tuo)}，{n_notes}注/{n_notes * 2}元")
                    return

                for i in range(num_predictions):
                    # 传递 variation=i 给紫微/梅花等传统算法，生成不同结果
                    red_predictions, blue_predictions = ml_model.predict(recent_data, variation=i)
                    
                    if red_predictions is None or blue_predictions is None:
                        raise ValueError(f"预测失败，请检查数据或重新训练模型。")
                    
                    if lottery_type == "dlt":
                        result_text += f"  第 {i+1} 组: {' '.join(map(str, red_predictions))} + {' '.join(map(str, blue_predictions))}\n"
                        collected_predictions.append(
                            (list(red_predictions), list(blue_predictions)))
                    else:
                        result_text += f"  第 {i+1} 组: {' '.join(map(str, red_predictions))} + {str(blue_predictions[0])}\n"
                        collected_predictions.append(
                            (list(red_predictions), [int(blue_predictions[0])]))

            self.result_label.setText(result_text)
            # 自动存档预测记录（供'历史回测'页核验，失败不影响结果）
            self._save_prediction_records(lottery_type, model_type,
                                          collected_predictions)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_emitter.new_log.emit(f"生成预测时出错: {e}")
            self.log_emitter.new_log.emit(f"错误详情:\n{error_details}")
            self.result_label.setText(f"生成预测时出错: {e}")

    def update_log(self, text):
        self.log_box.append(text)

    def _save_prediction_records(self, lottery_type, model_type, predictions):
        """
        将本次生成的预测号码自动存档，供'历史回测'页核验。
        存档失败不影响预测结果展示。
        """
        try:
            if not predictions:
                return
            # 预测时已知的最新开奖期（核验时找其后第一期）
            df = load_lottery_data(lottery_type)
            latest_period = int(df['期数'].max())
            from prediction_records import add_prediction_record
            count = None
            for red_nums, blue_nums in predictions:
                count = add_prediction_record(
                    lottery_type, model_type, latest_period,
                    red_nums, blue_nums)
            if count:
                self.log_emitter.new_log.emit(
                    f"已存档 {len(predictions)} 注预测记录（累计 {count} 注），"
                    f"可在'历史回测'页核验中奖情况")
        except Exception as e:
            self.log_emitter.new_log.emit(
                f"预测记录存档失败(不影响预测): {e}")

    def generate_ziwei_prediction(self):
        """紫微斗数排盘预测：3组号码"""
        try:
            from astrology.ziwei.engine import predict as ziwei_predict
            import pandas as pd

            # 获取彩种
            lottery_text = self.lottery_combo.currentText()
            lottery_type = 'ssq' if '双色' in lottery_text or 'ssq' in lottery_text.lower() else 'dlt'

            # 构造出生信息
            birth_info = {
                'year': self.zw_year.value(),
                'month': int(self.zw_month.currentText()),
                'day': self.zw_day.value(),
                'hour': int(self.zw_hour.currentText().replace('时', '')),
                'minute': int(self.zw_minute.currentText().replace('分', '')),
                'city': self.zw_city.currentText(),
                'gender': self.zw_gender.currentText(),
            }

            # 获取最新期号
            target_period = self.zw_target.value()
            if target_period == 0:
                try:
                    from scripts.data_analysis import load_lottery_data
                    df = load_lottery_data(lottery_type)
                    target_period = int(df['期数'].max())
                except Exception:
                    target_period = 0

            # 彩种配置
            if lottery_type == 'ssq':
                red_range, blue_range, red_count, blue_count = 33, 16, 6, 1
            else:
                red_range, blue_range, red_count, blue_count = 35, 12, 5, 2

            # 生成3组
            for v in range(3):
                red, blue, interp = ziwei_predict(
                    lottery_type=lottery_type,
                    red_range=red_range, blue_range=blue_range,
                    red_count=red_count, blue_count=blue_count,
                    birth_info=birth_info,
                    target_period=target_period,
                    variation=v,
                )
                red_label, blue_label, detail_label = self.zw_cards[v]
                red_label.setText(f"红球: {' '.join(f'{n:02d}' for n in red)}")
                if lottery_type == 'ssq':
                    blue_label.setText(f"蓝球: {blue[0]:02d}")
                else:
                    blue_label.setText(f"蓝球: {' '.join(f'{n:02d}' for n in blue)}")
                detail_label.setText(interp)

            self.log_emitter.new_log.emit(
                f"紫微排盘完成: {birth_info['year']}年{birth_info['month']}月{birth_info['day']}日 "
                f"{birth_info['hour']}时 {birth_info['city']} {birth_info['gender']} → 3组号码已生成")
        except Exception as e:
            self.log_emitter.new_log.emit(f"紫微排盘失败: {e}")

    def reset_ziwei(self):
        """重置紫微结果"""
        for i in range(3):
            red_label, blue_label, detail_label = self.zw_cards[i]
            red_label.setText("红球: 待排盘")
            blue_label.setText("蓝球: 待排盘")
            detail_label.setText("推演: 待排盘")

    def generate_meihua_prediction(self):
        """梅花易数起卦预测：3组号码"""
        try:
            from astrology.meihua.engine import predict as meihua_predict

            # 获取彩种
            lottery_text = self.lottery_combo.currentText()
            lottery_type = 'ssq' if '双色' in lottery_text or 'ssq' in lottery_text.lower() else 'dlt'

            # 构造输入信息
            mode_text = self.mh_mode.currentText()
            input_info = {
                'mode': 'number' if '数字' in mode_text else 'time',
                'year': self.mh_year.value(),
                'month': int(self.mh_month.currentText()),
                'day': self.mh_day.value(),
                'hour': int(self.mh_hour.currentText().replace('时', '')),
                'minute': int(self.mh_minute.currentText().replace('分', '')),
                'num1': self.mh_num1.value(),
                'num2': self.mh_num2.value(),
                'num3': self.mh_num3.value(),
            }

            # 获取最新期号
            target_period = self.mh_target.value()
            if target_period == 0:
                try:
                    from scripts.data_analysis import load_lottery_data
                    df = load_lottery_data(lottery_type)
                    target_period = int(df['期数'].max())
                except Exception:
                    target_period = 0

            # 彩种配置
            if lottery_type == 'ssq':
                red_range, blue_range, red_count, blue_count = 33, 16, 6, 1
            else:
                red_range, blue_range, red_count, blue_count = 35, 12, 5, 2

            # 生成3组
            for v in range(3):
                red, blue, interp = meihua_predict(
                    lottery_type=lottery_type,
                    red_range=red_range, blue_range=blue_range,
                    red_count=red_count, blue_count=blue_count,
                    input_info=input_info,
                    target_period=target_period,
                    variation=v,
                )
                red_label, blue_label, detail_label = self.mh_cards[v]
                red_label.setText(f"红球: {' '.join(f'{n:02d}' for n in red)}")
                if lottery_type == 'ssq':
                    blue_label.setText(f"蓝球: {blue[0]:02d}")
                else:
                    blue_label.setText(f"蓝球: {' '.join(f'{n:02d}' for n in blue)}")
                detail_label.setText(interp)

            self.log_emitter.new_log.emit(
                f"梅花起卦完成: {mode_text} → 3组号码已生成")
        except Exception as e:
            self.log_emitter.new_log.emit(f"梅花起卦失败: {e}")

    def reset_meihua(self):
        """重置梅花结果"""
        for i in range(3):
            red_label, blue_label, detail_label = self.mh_cards[i]
            red_label.setText("红球: 待起卦")
            blue_label.setText("蓝球: 待起卦")
            detail_label.setText("卦象: 待起卦")

    def verify_prediction_records(self):
        """核验预测记录：统计实际生成过的预测的中奖情况"""
        try:
            from prediction_records import verify_records, format_verify_summary
            self.bt_status_label.setText("核验中...")
            report = verify_records(
                log_callback=lambda msg: self.log_emitter.new_log.emit(msg))
            self.bt_result_text.append("\n" + format_verify_summary(report))
            self.bt_status_label.setText("核验完成")
        except Exception as e:
            self.bt_result_text.append(f"\n[核验失败: {e}]")
            self.bt_status_label.setText("核验失败")
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def analyze_data(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        self.current_lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        
        try:
            self.tab_widget.setCurrentIndex(1)
            
            self.chart_label.setText(f"正在加载{lottery_name}数据...")
            self.stats_text.clear()
            QApplication.processEvents()
            
            # 使用数据处理模块加载和处理数据
            self.current_df, self.current_stats, self.enhanced_df, report = process_analysis_data(self.current_lottery_type)
            
            # 更新趋势特征下拉列表
            trend_features = get_trend_features(self.enhanced_df)
            self.trend_feature_combo.clear()
            self.trend_feature_combo.addItems(trend_features)
            
            # 更新视图
            self.update_analysis_view(0)
            
            # 显示数据质量报告
            quality_report = format_quality_report(report)
            self.stats_text.setText(quality_report)
            
        except Exception as e:
            self.chart_label.setText(f"数据分析错误: {str(e)}")
            self.log_emitter.new_log.emit(f"数据分析错误: {str(e)}")
    
    def update_analysis_view(self, index):
        if self.current_stats is None or self.current_df is None:
            return
        
        try:
            if index == 0:  # 频率分布
                self.trend_feature_combo.setVisible(False)
                pixmap = plot_frequency_distribution(self.current_stats, self.current_lottery_type)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.width(), self.chart_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 使用数据处理模块格式化统计信息
                stats_text = format_frequency_stats(self.current_stats)
                self.stats_text.setText(stats_text)
                
            elif index == 1:  # 热冷号码
                self.trend_feature_combo.setVisible(False)
                pixmap = plot_hot_cold_numbers(self.current_stats)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.width(), self.chart_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 使用数据处理模块格式化统计信息
                stats_text = format_hot_cold_stats(self.current_stats)
                self.stats_text.setText(stats_text)
                
            elif index == 2:  # 遗漏分析
                self.trend_feature_combo.setVisible(False)
                pixmap = plot_gap_statistics(self.current_stats, self.current_lottery_type)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.width(), self.chart_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 使用数据处理模块格式化统计信息
                stats_text = format_gap_stats(self.current_stats)
                self.stats_text.setText(stats_text)
                
            elif index == 3:  # 号码模式
                self.trend_feature_combo.setVisible(False)
                pixmap = plot_patterns(self.current_stats)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.width(), self.chart_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 使用数据处理模块格式化统计信息
                stats_text = format_pattern_stats(self.current_stats)
                self.stats_text.setText(stats_text)
                
            elif index == 4:  # 趋势分析
                self.trend_feature_combo.setVisible(True)
                
                # 准备最近10期的趋势数据
                recent_df = prepare_recent_trend_data(self.enhanced_df)
                
                # 更新统计数据
                if recent_df is not None:
                    trend_data = {
                        "期数": recent_df['期数'].tolist(),
                        "红球和值": recent_df['红球和'].tolist(),
                        "红球最大差值": recent_df['红球最大差值'].tolist()
                    }
                    self.current_stats["趋势数据"] = trend_data
                
                # 绘制趋势分析图
                pixmap = plot_trend_analysis(self.current_stats)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.width(), self.chart_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 使用数据处理模块格式化统计信息
                stats_text = format_trend_stats(recent_df)
                self.stats_text.setText(stats_text)
        
        except Exception as e:
            self.chart_label.setText(f"图表生成错误: {str(e)}")
            logging.error(f"图表生成错误: {str(e)}", exc_info=True)
            self.log_emitter.new_log.emit(f"图表生成错误: {str(e)}")

    def update_lottery_data(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']

        self.update_data_button.setEnabled(False)
        self.lottery_combo.setEnabled(False)

        self.log_box.clear()
        self.log_emitter.new_log.emit(f"开始更新{lottery_name}历史数据...")

        self.update_thread = UpdateDataThread(lottery_type)
        self.update_thread.log_signal.connect(self.update_log)
        self.update_thread.finished_signal.connect(self.on_update_finished)
        self.update_thread.start()

    def on_update_finished(self):
        self.update_data_button.setEnabled(True)
        self.lottery_combo.setEnabled(True)
        self.update_log("数据更新线程已结束。")

    def pause_resume_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.toggle_pause()

    def cleanup_resources(self):
        """在应用程序关闭前清理资源"""
        logging.info("正在清理资源...")
        
        if self.training_thread and self.training_thread.isRunning():
            logging.info("停止训练线程")
            self.training_thread.terminate()
            self.training_thread.wait()
        
        if self.update_thread and self.update_thread.isRunning():
            logging.info("停止数据更新线程")
            self.update_thread.terminate()
            self.update_thread.wait()
        
        if self.backtest_thread and self.backtest_thread.isRunning():
            logging.info("停止回测线程")
            self.backtest_thread.terminate()
            self.backtest_thread.wait()
        
        # 关闭统计窗口
        if self.stats_window is not None and self.stats_window.isVisible():
            logging.info("关闭统计窗口")
            self.stats_window.close()
        
        try:
            logging.info("应用程序正常关闭")
            for handler in logging.root.handlers:
                handler.flush()
                handler.close()
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def start_backtest(self):
        """开始历史回测"""
        if self.backtest_thread and self.backtest_thread.isRunning():
            self.bt_status_label.setText("回测正在进行中，请等待...")
            return
        
        # 从控件读取参数
        lottery_index = self.bt_lottery_combo.currentIndex()
        lottery_keys = list(name_path.keys())
        lottery_type = lottery_keys[lottery_index]
        model_type = self.bt_model_combo.currentData()
        periods = self.bt_periods_spin.value() or None
        
        self.bt_status_label.setText("回测运行中...")
        self.bt_start_button.setEnabled(False)
        self.bt_result_text.setPlainText("正在回测，请稍候...（完整历史回测可能需要数分钟）")
        
        # 创建并启动回测线程
        self.backtest_thread = BacktestThread(lottery_type, model_type, periods=periods)
        self.backtest_thread.log_signal.connect(self.on_backtest_log)
        self.backtest_thread.finished_signal.connect(self.on_backtest_finished)
        self.backtest_thread.start()
    
    def on_backtest_log(self, message):
        """回测过程日志：实时显示在回测tab结果区，并转发到主日志框"""
        # 实时显示在回测tab（用户当前所在位置）
        self.bt_result_text.append(message)
        # 同步进度到状态栏
        if "进度" in message and "期" in message:
            self.bt_status_label.setText(message)
        # 限制结果区行数，防止日志过多拖慢界面
        doc = self.bt_result_text.document()
        if doc.blockCount() > 3000:
            cursor = self.bt_result_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.movePosition(QtGui.QTextCursor.Down,
                                QtGui.QTextCursor.KeepAnchor, 1500)
            cursor.removeSelectedText()
        # 转发主日志框
        self.log_emitter.new_log.emit(message)
    
    def on_backtest_finished(self, success):
        """回测完成，展示摘要"""
        self.bt_start_button.setEnabled(True)
        if success and self.backtest_thread and self.backtest_thread.report:
            try:
                from backtest import format_summary_text
                # 追加摘要，保留前面的进度日志
                self.bt_result_text.append("\n" +
                    format_summary_text(self.backtest_thread.report))
            except Exception as e:
                self.bt_result_text.append(f"\n[生成摘要失败: {e}]")
            self.bt_status_label.setText("回测完成")
        else:
            self.bt_result_text.append("\n[回测失败，详见日志]")
            self.bt_status_label.setText("回测失败")

    def show_advanced_statistics(self):
        """显示高级统计分析结果"""
        try:
            # 切换到高级统计标签页 (index 3)
            self.tab_widget.setCurrentIndex(3)
            
            # 运行高级统计分析
            self.run_advanced_statistics()
            
        except Exception as e:
            self.log_box.append(f"高级统计分析出错：{str(e)}")
    
    def show_distribution_analysis(self):
        """显示分布分析结果"""
        try:
            # 切换到高级统计标签页 (index 3)
            self.tab_widget.setCurrentIndex(3)
            
            # 运行分布分析
            self.run_distribution_analysis()
            
        except Exception as e:
            self.log_box.append(f"分布分析出错：{str(e)}")

    def run_advanced_statistics(self):
        """运行高级统计分析"""
        try:
            # 显示加载消息
            self.stats_result_label.setText("正在计算高级统计指标，请稍候...")
            QApplication.processEvents()
            
            # 获取彩票类型
            lottery_type = self.advanced_stats_lottery_combo.currentText()
            if lottery_type == "双色球":
                lottery_type = "ssq"
            else:
                lottery_type = "dlt"
            
            # 加载数据
            df = load_lottery_data(lottery_type)
            
            # 生成统计图表
            pixmap = plot_advanced_statistics(df, lottery_type)
            
            # 显示结果
            self.stats_result_label.setPixmap(pixmap)
            
        except Exception as e:
            self.log_box.append(f"高级统计分析出错：{str(e)}")
            self.stats_result_label.setText(f"分析出错：{str(e)}")
    
    def run_distribution_analysis(self):
        """运行分布分析"""
        try:
            # 显示加载消息
            self.stats_result_label.setText("正在进行分布分析，请稍候...")
            QApplication.processEvents()
            
            # 获取彩票类型
            lottery_type = self.advanced_stats_lottery_combo.currentText()
            if lottery_type == "双色球":
                lottery_type = "ssq"
            else:
                lottery_type = "dlt"
            
            # 加载数据
            df = load_lottery_data(lottery_type)
            
            # 生成分布分析图表
            pixmap = plot_distribution_analysis(df, lottery_type)
            
            # 显示结果
            self.stats_result_label.setPixmap(pixmap)
            
        except Exception as e:
            self.log_box.append(f"分布分析出错：{str(e)}")
            self.stats_result_label.setText(f"分析出错：{str(e)}")

    def show_statistics_data(self):
        """显示详细统计数据"""
        try:
            # 获取彩票类型
            lottery_type = self.advanced_stats_lottery_combo.currentText()
            if lottery_type == "双色球":
                lottery_type = "ssq"
            else:
                lottery_type = "dlt"
            
            self.log_box.append(f"正在计算{lottery_type.upper()}的高级统计数据...")
            
            # 确保导入正确的模块
            try:
                from scripts.data_analysis import load_lottery_data
                from scripts.advanced_statistics import calculate_advanced_statistics
            except ImportError as e:
                self.log_box.append(f"导入模块失败: {str(e)}")
                return
            
            # 加载数据
            df = load_lottery_data(lottery_type)
            if df is None or df.empty:
                self.log_box.append("加载数据失败，请先获取最新数据。")
                return
            
            self.log_box.append(f"成功加载{len(df)}条历史数据，正在计算统计指标...")
            
            # 打印列名以便调试
            self.log_box.append(f"数据列名: {list(df.columns)}")
            
            # 计算统计指标
            stats_dict = calculate_advanced_statistics(df, lottery_type)
            
            # 检查是否有错误
            if "error" in stats_dict:
                self.log_box.append(f"计算统计指标出错: {stats_dict['error']}")
                return
            
            # 如果已有统计窗口，先关闭它
            if self.stats_window is not None:
                self.stats_window.close()
            
            # 创建统计数据窗口
            self.stats_window = QWidget()
            self.stats_window.setWindowTitle(f"{lottery_type.upper()} 高级统计数据")
            stats_layout = QVBoxLayout()
            
            # 创建文本编辑器用于显示统计数据
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            
            # 格式化统计信息
            stats_text_content = f"<h2>{lottery_type.upper()} 高级统计分析结果</h2>"
            
            # 确定红球和蓝球的列名
            if lottery_type == 'dlt':
                red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
                blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
            else:  # ssq
                red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
                # 确保正确获取蓝球列
                blue_cols = []
                for col in df.columns:
                    if col.startswith('蓝球_') or col.startswith('蓝球'):
                        blue_cols.append(col)
                        if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                            break
            
            # 添加红球统计数据
            stats_text_content += "<h3>红球统计指标</h3>"
            stats_text_content += "<table border='1' cellspacing='0' cellpadding='5'>"
            
            # 添加表头
            stats_text_content += "<tr><th>指标</th>"
            for col in red_cols:
                stats_text_content += f"<th>{col}</th>"
            stats_text_content += "</tr>"
            
            # 添加统计指标
            metrics = [
                ('均值', 'mean'), 
                ('中位数', 'median'), 
                ('标准差', 'std'),
                ('偏度', 'skewness'),
                ('峰度', 'kurtosis'),
                ('众数', 'mode'),
                ('第一四分位数', 'q1'),
                ('第三四分位数', 'q3'),
                ('四分位距', 'iqr'),
                ('范围', 'range'),
                ('方差', 'variance'),
                ('变异系数', 'coefficient_of_variation'),
                ('中位数绝对偏差', 'mad'),
                ('标准误', 'sem'),
                ('熵值', 'entropy')
            ]
            
            for metric_name, metric_key in metrics:
                stats_text_content += f"<tr><td>{metric_name}</td>"
                for col in red_cols:
                    if f'{col}_stats' in stats_dict and metric_key in stats_dict[f'{col}_stats']:
                        value = stats_dict[f'{col}_stats'][metric_key]
                        if isinstance(value, (int, float)):
                            stats_text_content += f"<td>{value:.4f}</td>"
                        else:
                            stats_text_content += f"<td>{value}</td>"
                    else:
                        stats_text_content += "<td>N/A</td>"
                stats_text_content += "</tr>"
            
            stats_text_content += "</table>"
            
            # 添加蓝球统计数据
            if blue_cols:
                stats_text_content += "<h3>蓝球统计指标</h3>"
                stats_text_content += "<table border='1' cellspacing='0' cellpadding='5'>"
                
                # 添加表头
                stats_text_content += "<tr><th>指标</th>"
                for col in blue_cols:
                    stats_text_content += f"<th>{col}</th>"
                stats_text_content += "</tr>"
                
                # 添加统计指标
                for metric_name, metric_key in metrics:
                    stats_text_content += f"<tr><td>{metric_name}</td>"
                    for col in blue_cols:
                        if f'{col}_stats' in stats_dict and metric_key in stats_dict[f'{col}_stats']:
                            value = stats_dict[f'{col}_stats'][metric_key]
                            if isinstance(value, (int, float)):
                                stats_text_content += f"<td>{value:.4f}</td>"
                            else:
                                stats_text_content += f"<td>{value}</td>"
                        else:
                            stats_text_content += "<td>N/A</td>"
                    stats_text_content += "</tr>"
                
                stats_text_content += "</table>"
            else:
                stats_text_content += "<p>未找到蓝球数据列</p>"
            
            # 添加正态性检验结果
            norm_test_cols = [col for col in red_cols + blue_cols 
                             if f'{col}_stats' in stats_dict and 'normality_test' in stats_dict[f'{col}_stats']]
            if norm_test_cols:
                stats_text_content += "<h3>正态性检验结果</h3>"
                stats_text_content += "<p>D'Agostino和Pearson的正态性检验:</p>"
                stats_text_content += "<table border='1' cellspacing='0' cellpadding='5'>"
                stats_text_content += "<tr><th>号码</th><th>统计量</th><th>p值</th><th>结论</th></tr>"
                
                for col in norm_test_cols:
                    test_stat, p_value = stats_dict[f'{col}_stats']['normality_test']
                    conclusion = "正态分布" if p_value > 0.05 else "非正态分布"
                    stats_text_content += f"<tr><td>{col}</td><td>{test_stat:.4f}</td><td>{p_value:.4f}</td><td>{conclusion}</td></tr>"
                
                stats_text_content += "</table>"
            
            # 添加游程检验结果
            runs_test_cols = [col for col in red_cols + blue_cols 
                             if f'{col}_stats' in stats_dict and 'runs_test' in stats_dict[f'{col}_stats']]
            if runs_test_cols:
                stats_text_content += "<h3>游程检验结果</h3>"
                stats_text_content += "<p>检验序列是否随机:</p>"
                stats_text_content += "<table border='1' cellspacing='0' cellpadding='5'>"
                stats_text_content += "<tr><th>号码</th><th>统计量</th><th>p值</th><th>结论</th></tr>"
                
                for col in runs_test_cols:
                    runs_stat, p_value = stats_dict[f'{col}_stats']['runs_test']
                    conclusion = "随机序列" if p_value > 0.05 else "非随机序列"
                    stats_text_content += f"<tr><td>{col}</td><td>{runs_stat:.4f}</td><td>{p_value:.4f}</td><td>{conclusion}</td></tr>"
                
                stats_text_content += "</table>"
            
            # 设置HTML内容
            stats_text.setHtml(stats_text_content)
            stats_layout.addWidget(stats_text)
            
            # 设置窗口
            self.stats_window.setLayout(stats_layout)
            self.stats_window.resize(800, 600)
            self.stats_window.show()
            
            self.log_box.append("统计数据计算和显示完成。")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_box.append(f"显示统计数据出错：{str(e)}")
            self.log_box.append(f"错误详情：{error_details}")

    def generate_analysis_suggestions(self):
        """生成选号参考建议 - 将统计指标翻译成人话"""
        try:
            # 获取当前彩票类型 - 根据调用源确定使用哪个下拉框
            sender = self.sender()
            if sender == self.analysis_suggestion_button:
                # 数据分析页面：用 analysis_combo
                selected_index = self.analysis_combo.currentIndex()
                selected_key = list(name_path.keys())[selected_index]
                lottery_type = selected_key
            elif sender == self.adv_stats_suggestion_button:
                # 高级统计页面：用 advanced_stats_lottery_combo (显示名"双色球"/"大乐透")
                display_name = self.advanced_stats_lottery_combo.currentText()
                lottery_type = 'ssq' if display_name == '双色球' else 'dlt'
            elif hasattr(self, 'current_lottery_type') and self.current_lottery_type:
                lottery_type = self.current_lottery_type
            else:
                # 兜底：主界面
                selected_index = self.lottery_combo.currentIndex()
                selected_key = list(name_path.keys())[selected_index]
                lottery_type = selected_key
            
            lottery_name = name_path[lottery_type]['name']
            self.log_box.append(f"正在为{lottery_name}生成选号参考建议...")
            QApplication.processEvents()
            
            # 加载数据
            from scripts.data_analysis import load_lottery_data
            df = load_lottery_data(lottery_type)
            if df is None or df.empty:
                self.log_box.append("数据为空，请先点击'加载分析数据'")
                return
            
            # 确定红蓝球列
            if lottery_type == 'dlt':
                red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
                blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
                red_count, blue_count = 5, 2
                red_range, blue_range = 35, 12
            else:
                red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
                blue_cols = []
                for col in df.columns:
                    if col.startswith('蓝球_') or col == '蓝球':
                        blue_cols.append(col)
                        break
                red_count, blue_count = 6, 1
                red_range, blue_range = 33, 16
            
            # 计算各种指标
            suggestions = []
            total_periods = len(df)
            
            # 1. 频率分析 - 热冷号
            red_freq = {}
            for col in red_cols:
                red_freq[col] = df[col].value_counts().to_dict()
            
            # 找出热号（出现频率 > 平均频率 * 1.2）和冷号（< 平均 * 0.8）
            avg_freq = total_periods / red_range
            hot_reds = []
            cold_reds = []
            for num in range(1, red_range + 1):
                freq = sum(red_freq.get(col, {}).get(num, 0) for col in red_cols)
                if freq > avg_freq * 1.2:
                    hot_reds.append((num, freq))
                elif freq < avg_freq * 0.8:
                    cold_reds.append((num, freq))
            
            hot_reds.sort(key=lambda x: x[1], reverse=True)
            cold_reds.sort(key=lambda x: x[1])
            
            if hot_reds:
                top_hot = [str(n) for n, f in hot_reds[:5]]
                suggestions.append(f"🔥 <b>热号关注</b>：红球 {', '.join(top_hot)} 近期出现频率显著高于平均（>{avg_freq:.1f}次），可考虑防守")
            if cold_reds:
                top_cold = [str(n) for n, f in cold_reds[:5]]
                suggestions.append(f"❄️ <b>冷号补防</b>：红球 {', '.join(top_cold)} 长期出现偏少（<{avg_freq:.1f}次），理论回补概率较大")
            
            # 2. 遗漏分析
            recent_n = min(20, total_periods)
            recent_df = df.sort_values('期数', ascending=False).head(recent_n)
            
            missing_reds = []
            for num in range(1, red_range + 1):
                # 计算遗漏期数
                missed = 0
                for _, row in recent_df.iterrows():
                    if num not in [row[c] for c in red_cols]:
                        missed += 1
                    else:
                        break
                if missed > recent_n * 0.5:  # 遗漏超过一半的近期
                    missing_reds.append((num, missed))
            
            missing_reds.sort(key=lambda x: x[1], reverse=True)
            if missing_reds:
                top_missing = [f"{n}(遗漏{m}期)" for n, m in missing_reds[:4]]
                suggestions.append(f"📊 <b>遗漏预警</b>：红球 {', '.join(top_missing)} 近{recent_n}期未开出，关注回补机会")
            
            # 3. 奇偶比分析
            recent_20 = df.sort_values('期数', ascending=False).head(20)
            odd_counts = []
            for _, row in recent_20.iterrows():
                reds = [row[c] for c in red_cols]
                odd = sum(1 for r in reds if r % 2 == 1)
                odd_counts.append(odd)
            
            avg_odd = sum(odd_counts) / len(odd_counts)
            if avg_odd > red_count * 0.6:
                suggestions.append(f"🔢 <b>奇偶趋势</b>：近20期红球奇数占比 {avg_odd/red_count:.0%}（理论50%），本期防偶数回补")
            elif avg_odd < red_count * 0.4:
                suggestions.append(f"🔢 <b>奇偶趋势</b>：近20期红球偶数占比 {(1-avg_odd/red_count):.0%}，本期防奇数回补")
            
            # 4. 大小比分析
            mid = red_range // 2
            large_counts = []
            for _, row in recent_20.iterrows():
                reds = [row[c] for c in red_cols]
                large = sum(1 for r in reds if r > mid)
                large_counts.append(large)
            
            avg_large = sum(large_counts) / len(large_counts)
            if avg_large > red_count * 0.6:
                suggestions.append(f"📈 <b>大小趋势</b>：近20期大号(>{mid})占比 {avg_large/red_count:.0%}，本期关注小号回补")
            elif avg_large < red_count * 0.4:
                suggestions.append(f"📈 <b>大小趋势</b>：近20期小号(≤{mid})占比 {(1-avg_large/red_count):.0%}，本期关注大号回补")
            
            # 5. 和值区间
            sum_vals = []
            for _, row in recent_20.iterrows():
                reds = [row[c] for c in red_cols]
                sum_vals.append(sum(reds))
            
            avg_sum = sum(sum_vals) / len(sum_vals)
            min_sum = sum(range(1, red_count + 1))
            max_sum = sum(range(red_range - red_count + 1, red_range + 1))
            theoretical_avg = (min_sum + max_sum) / 2
            
            if avg_sum > theoretical_avg * 1.05:
                suggestions.append(f"🎯 <b>和值参考</b>：近20期和值均值 {avg_sum:.0f} 偏高（理论{theoretical_avg:.0f}），建议关注 {int(theoretical_avg-10)}-{int(theoretical_avg+5)} 区间")
            elif avg_sum < theoretical_avg * 0.95:
                suggestions.append(f"🎯 <b>和值参考</b>：近20期和值均值 {avg_sum:.0f} 偏低（理论{theoretical_avg:.0f}），建议关注 {int(theoretical_avg-5)}-{int(theoretical_avg+10)} 区间")
            else:
                suggestions.append(f"🎯 <b>和值参考</b>：近20期和值均值 {avg_sum:.0f} 接近理论值 {theoretical_avg:.0f}，建议关注 {int(theoretical_avg-10)}-{int(theoretical_avg+10)} 区间")
            
            # 6. AC值（算术复杂度）
            ac_vals = []
            for _, row in recent_20.iterrows():
                reds = sorted([row[c] for c in red_cols])
                diffs = set()
                for i in range(len(reds)):
                    for j in range(i+1, len(reds)):
                        diffs.add(reds[j] - reds[i])
                ac = len(diffs) - (red_count - 1)
                ac_vals.append(ac)
            
            avg_ac = sum(ac_vals) / len(ac_vals)
            max_ac = red_count * (red_count - 1) // 2 - (red_count - 1)
            suggestions.append(f"🔬 <b>AC值参考</b>：近20期平均AC值 {avg_ac:.1f}（范围0-{max_ac}），建议选号AC值在 {max(0, int(avg_ac-2))}-{int(avg_ac+2)} 之间")
            
            # 7. 蓝球建议
            if blue_cols:
                blue_col = blue_cols[0]
                blue_freq = df[blue_col].value_counts().to_dict()
                avg_b_freq = total_periods / blue_range
                hot_blues = [(n, f) for n, f in blue_freq.items() if f > avg_b_freq * 1.3]
                cold_blues = [(n, f) for n, f in blue_freq.items() if f < avg_b_freq * 0.7]
                hot_blues.sort(key=lambda x: x[1], reverse=True)
                cold_blues.sort(key=lambda x: x[1])
                
                if hot_blues:
                    suggestions.append(f"🔵 <b>蓝球热号</b>：{', '.join(str(n) for n, _ in hot_blues[:3])} 频率偏高")
                if cold_blues:
                    suggestions.append(f"🔵 <b>蓝球冷号</b>：{', '.join(str(n) for n, _ in cold_blues[:3])} 长期未出，可做防守")
            
            # 显示建议
            html = f"""
            <h2 style='color:#C9702D;'>{lottery_name} 选号参考建议 <small>（基于近{total_periods}期历史数据）</small></h2>
            <p style='color:#888;font-size:10pt;'>⚠️ 仅供娱乐参考，不构成投注建议，彩票本质是随机事件</p>
            <hr style='border-color:#8B6F47;'>
            """
            
            for i, s in enumerate(suggestions, 1):
                html += f"<p style='font-size:11pt; margin:8px 0;'><b>{i}.</b> {s}</p>"
            
            html += f"""
            <hr style='border-color:#8B6F47;'>
            <p style='color:#888;font-size:10pt;'>
            数据来源：{total_periods}期历史开奖 | 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 
            理论中奖概率：双色球约1/1772万，大乐透约1/2142万
            </p>
            """
            
            # 显示在对应页面的结果区域
            if hasattr(self, 'stats_result_label'):
                self.stats_result_label.setText(html)
            if hasattr(self, 'chart_label'):
                self.chart_label.setText(html)
            
            self.log_box.append(f"✅ 生成选号参考完成，共 {len(suggestions)} 条建议")
            
        except Exception as e:
            import traceback
            self.log_box.append(f"生成选号参考出错：{str(e)}")
            self.log_box.append(traceback.format_exc())

    def show_log_context_menu(self, position):
        """
        显示日志文本框的右键菜单
        
        Args:
            position: 鼠标右键点击的位置
        """
        context_menu = QMenu(self)
        clear_action = QAction("清除日志", self)
        clear_action.triggered.connect(self.clear_log)
        context_menu.addAction(clear_action)
        
        # 在鼠标位置显示菜单
        context_menu.exec_(self.log_box.mapToGlobal(position))

    def clear_log(self):
        """清除日志文本框的内容"""
        self.log_box.clear()

    def generate_ev_prediction(self):
        """使用期望值模型生成预测"""
        selected_index = self.ev_lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        num_predictions = self.ev_prediction_spin.value()
        
        result_text = f"期望值模型预测的{num_predictions}个{lottery_name}号码：\n"
        
        try:
            # 从expected_value_model.py导入模型
            from expected_value_model import ExpectedValueLotteryModel
            
            # 清空日志
            self.ev_log_box.clear()
            self.ev_log_box.append(f"<b>===== {lottery_name}期望值预测计算过程 =====</b>")
            
            # 创建期望值模型实例
            model = ExpectedValueLotteryModel(
                lottery_type=lottery_type,
                log_callback=lambda msg: self.ev_log_box.append(msg),
                use_gpu=self.ev_gpu_checkbox.isChecked(),
                verbose=True  # 启用详细日志
            )
            
            self.ev_log_box.append(f"加载{lottery_name}期望值模型...")
            
            # 加载模型
            if not model.load():
                self.ev_log_box.append(f"<font color='red'>错误: 期望值模型未训练或加载失败，请先训练模型。</font>")
                self.ev_result_label.setText(f"错误: 模型未训练或加载失败，请先训练模型。")
                return
            
            # 加载最近数据用于预测
            df = load_lottery_data(lottery_type)
            if df is None or df.empty:
                self.ev_log_box.append("<font color='red'>错误: 无法加载历史数据。</font>")
                return
                
            recent_data = df.sort_values('期数', ascending=False).head(10)
            self.ev_log_box.append(f"使用最近{len(recent_data)}期数据进行预测...")
            
            # 显示最近的开奖数据
            self.ev_log_box.append("<b>最近的开奖数据：</b>")
            for _, row in recent_data.head(5).iterrows():
                if lottery_type == "dlt":
                    ball_info = f"期数: {row['期数']} 红球: {row['红球_1']} {row['红球_2']} {row['红球_3']} {row['红球_4']} {row['红球_5']} 蓝球: {row['蓝球_1']} {row['蓝球_2']}"
                else:
                    ball_info = f"期数: {row['期数']} 红球: {row['红球_1']} {row['红球_2']} {row['红球_3']} {row['红球_4']} {row['红球_5']} {row['红球_6']} 蓝球: {row['蓝球']}"
                self.ev_log_box.append(ball_info)
            
            # 显示概率分布
            self.ev_log_box.append("<b>号码概率分布:</b>")
            
            # 红球概率分布
            self.ev_log_box.append("<font color='red'><b>红球概率分布 (前10名):</b></font>")
            sorted_red_probs = sorted(model.red_probs.items(), key=lambda x: x[1], reverse=True)
            for num, prob in sorted_red_probs[:10]:
                actual_num = num + 1  # 转换为1-based索引
                self.ev_log_box.append(f"  {actual_num}号: {prob:.6f}")
            
            # 蓝球概率分布
            if lottery_type == "dlt":
                self.ev_log_box.append("<font color='blue'><b>蓝球概率分布:</b></font>")
                sorted_blue_probs = sorted(model.blue_probs.items(), key=lambda x: x[1], reverse=True)
                for num, prob in sorted_blue_probs:
                    actual_num = num + 1  # 转换为1-based索引
                    self.ev_log_box.append(f"  {actual_num}号: {prob:.6f}")
            else:
                self.ev_log_box.append("<font color='blue'><b>蓝球概率分布 (前5名):</b></font>")
                sorted_blue_probs = sorted(model.blue_probs.items(), key=lambda x: x[1], reverse=True)
                for num, prob in sorted_blue_probs[:5]:
                    actual_num = num + 1  # 转换为1-based索引
                    self.ev_log_box.append(f"  {actual_num}号: {prob:.6f}")
            
            # 生成预测
            self.ev_log_box.append("<b>开始生成预测...</b>")
            red_predictions, blue_predictions = model.predict(recent_data, num_predictions=num_predictions)
            
            # 显示预测过程
            self.ev_log_box.append("<b>期望值计算过程:</b>")
            self.ev_log_box.append("综合考虑历史数据频率、最近走势以及号码组合优势，计算每组号码的期望值")
            self.ev_log_box.append("期望值 = 号码概率 * 潜在回报 - (1 - 号码概率) * 投入成本")
            
            # 格式化结果
            self.ev_log_box.append("<b>生成的预测组合:</b>")
            for i in range(num_predictions):
                if lottery_type == "dlt":
                    # 大乐透结果格式化：5个红球 + 2个蓝球
                    red_balls = [int(num+1) for num in red_predictions[i]]  # 转换为1-based索引
                    blue_balls = [int(num+1) for num in blue_predictions[i]]  # 转换为1-based索引
                    
                    # 排序
                    red_balls.sort()
                    blue_balls.sort()
                    
                    # 显示概率值
                    red_probs_text = ", ".join([f"{rb}({model.red_probs[rb-1]:.4f})" for rb in red_balls])
                    blue_probs_text = ", ".join([f"{bb}({model.blue_probs[bb-1]:.4f})" for bb in blue_balls])
                    
                    self.ev_log_box.append(f"<b>第 {i+1} 组:</b>")
                    self.ev_log_box.append(f"  红球: {red_probs_text}")
                    self.ev_log_box.append(f"  蓝球: {blue_probs_text}")
                    
                    result_text += f"  第 {i+1} 组: {' '.join(map(str, red_balls))} + {' '.join(map(str, blue_balls))}\n"
                else:
                    # 双色球结果格式化：6个红球 + 1个蓝球
                    red_balls = [int(num+1) for num in red_predictions[i]]  # 转换为1-based索引
                    blue_ball = int(blue_predictions[i][0])+1  # 转换为1-based索引
                    
                    # 排序红球
                    red_balls.sort()
                    
                    # 显示概率值
                    red_probs_text = ", ".join([f"{rb}({model.red_probs[rb-1]:.4f})" for rb in red_balls])
                    blue_prob_text = f"{blue_ball}({model.blue_probs[blue_ball-1]:.4f})"
                    
                    self.ev_log_box.append(f"<b>第 {i+1} 组:</b>")
                    self.ev_log_box.append(f"  红球: {red_probs_text}")
                    self.ev_log_box.append(f"  蓝球: {blue_prob_text}")
                    
                    result_text += f"  第 {i+1} 组: {' '.join(map(str, red_balls))} + {blue_ball}\n"
            
            # 添加期望值分析总结
            self.ev_log_box.append("<b>期望值分析总结:</b>")
            self.ev_log_box.append("1. 各组号码都基于历史概率和期望值原理选取")
            self.ev_log_box.append("2. 红球选择优先考虑历史频率较高且近期表现稳定的号码")
            self.ev_log_box.append("3. 蓝球选择偏向于高概率且与红球组合优势较大的号码")
            self.ev_log_box.append("4. 各组合之间保持一定差异性，增加中奖概率")
            
            self.ev_result_label.setText(result_text)
            self.ev_log_box.append(f"<font color='green'>期望值模型预测完成，生成了{num_predictions}组预测结果。</font>")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.ev_log_box.append(f"<font color='red'>生成预测时出错: {e}</font>")
            self.ev_log_box.append(f"<font color='red'>错误详情:</font>\n{error_details}")
            self.ev_result_label.setText(f"生成预测时出错: {e}")
    
    def train_ev_model(self):
        """训练期望值模型"""
        selected_index = self.ev_lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        
        # 检查GPU状态
        use_gpu = self.ev_gpu_checkbox.isChecked()
        if use_gpu and not torch.cuda.is_available():
            self.ev_log_box.append("<font color='orange'>警告: GPU被选中但CUDA不可用，将使用CPU进行训练</font>")
            use_gpu = False
        elif use_gpu:
            gpu_info = torch.cuda.get_device_name(0)
            self.ev_log_box.append(f"<font color='green'>使用GPU训练: {gpu_info}</font>")
        else:
            self.ev_log_box.append("<font color='blue'>使用CPU训练</font>")
        
        self.ev_train_button.setEnabled(False)
        self.ev_lottery_combo.setEnabled(False)
        self.ev_gpu_checkbox.setEnabled(False)
        
        self.ev_log_box.clear()
        self.ev_log_box.append(f"开始训练{lottery_name}期望值模型...")
        
        try:
            # 从expected_value_model.py导入模型
            from expected_value_model import ExpectedValueLotteryModel
            
            # 加载数据
            df = load_lottery_data(lottery_type)
            if df is None or df.empty:
                self.ev_log_box.append("<font color='red'>错误: 无法加载历史数据。</font>")
                self.ev_train_button.setEnabled(True)
                self.ev_lottery_combo.setEnabled(True)
                self.ev_gpu_checkbox.setEnabled(torch.cuda.is_available())
                return
            
            self.ev_log_box.append(f"成功加载{len(df)}条历史数据")
            
            # 创建并训练模型
            model = ExpectedValueLotteryModel(
                lottery_type=lottery_type,
                log_callback=lambda msg: self.ev_log_box.append(msg),
                use_gpu=use_gpu
            )
            
            # 训练模型
            model.train(df)
            
            # 更新模型实例
            model_key = f"{lottery_type}_expected_value"
            self.ml_models[model_key] = LotteryMLModels(
                lottery_type=lottery_type, 
                model_type='expected_value',
                log_callback=lambda msg: self.ev_log_box.append(msg),
                use_gpu=use_gpu
            )
            
            # 将训练好的期望值模型设置到ml_models中
            self.ml_models[model_key].models = {'red': model, 'blue': model}
            self.ml_models[model_key].raw_models = {'expected_value_model': model}
            
            self.ev_log_box.append(f"<font color='green'>{lottery_name}期望值模型训练完成！</font>")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.ev_log_box.append(f"<font color='red'>训练期望值模型时出错: {e}</font>")
            self.ev_log_box.append(f"<font color='red'>错误详情:</font>\n{error_details}")
        finally:
            self.ev_train_button.setEnabled(True)
            self.ev_lottery_combo.setEnabled(True)
            self.ev_gpu_checkbox.setEnabled(torch.cuda.is_available())
    
    def show_ev_log_context_menu(self, position):
        """
        显示期望值模型日志文本框的右键菜单
        
        Args:
            position: 鼠标右键点击的位置
        """
        context_menu = QMenu(self)
        clear_action = QAction("清除日志", self)
        clear_action.triggered.connect(self.clear_ev_log)
        context_menu.addAction(clear_action)
        
        # 在鼠标位置显示菜单
        context_menu.exec_(self.ev_log_box.mapToGlobal(position))

    def clear_ev_log(self):
        """清除期望值模型日志文本框的内容"""
        self.ev_log_box.clear()

def main():
    app = QApplication(sys.argv)
    main_window = LotteryPredictorApp()
    main_window.show()
    sys.exit(app.exec_())

# 函数用于支持从命令行调用
def train_model(lottery_type, model_type, log_callback=None):
    """
    训练指定类型的彩票预测模型
    
    Args:
        lottery_type: 彩票类型，'dlt'表示大乐透，'ssq'表示双色球
        model_type: 模型类型，可选值为MODEL_TYPES中的键
        log_callback: 日志回调函数，用于将训练过程的日志输出到外部
        
    Returns:
        bool: 模型训练是否成功
    """
    try:
        # 设置日志输出函数，如果没有提供则使用print函数
        if log_callback is None:
            log_callback = print
            
        log_callback(f"开始训练{lottery_type}预测模型({MODEL_TYPES.get(model_type, model_type)})...")
        
        # 确定GPU可用性
        use_gpu = torch.cuda.is_available()
        device_info = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        log_callback(f"GPU训练已{'' if use_gpu else '不'}启用，使用设备: {device_info}")
        
        # 加载数据
        from scripts.data_analysis import load_lottery_data
        df = load_lottery_data(lottery_type)
        
        if df is None or df.empty:
            log_callback("加载数据失败，请检查数据文件。")
            return False
            
        log_callback(f"成功加载{len(df)}条历史数据。")
        
        # 创建模型实例
        ml_model = LotteryMLModels(
            lottery_type=lottery_type, 
            model_type=model_type,
            log_callback=log_callback,
            use_gpu=use_gpu
        )
        
        # 训练模型
        log_callback("准备训练数据...")
        ml_model.train(df)
        
        log_callback(f"{MODEL_TYPES.get(model_type, model_type)}模型训练完成。")
        return True
        
    except Exception as e:
        import traceback
        log_callback(f"训练{MODEL_TYPES.get(model_type, model_type)}模型时出错: {str(e)}")
        log_callback(traceback.format_exc())
        return False

def predict_next_draw(lottery_type, model_type, num_predictions=5):
    """
    使用训练好的模型预测下一期彩票号码
    
    Args:
        lottery_type: 彩票类型，'dlt'表示大乐透，'ssq'表示双色球
        model_type: 模型类型，可选值为MODEL_TYPES中的键
        num_predictions: 要生成的预测组数
        
    Returns:
        list: 预测结果列表，每个元素是一组预测号码
    """
    try:
        # 使用机器学习模型预测
        from scripts.data_analysis import load_lottery_data
        
        # 创建模型实例并加载
        ml_model = LotteryMLModels(
            lottery_type=lottery_type, 
            model_type=model_type
        )
        
        if not ml_model.load_models():
            print(f"模型{MODEL_TYPES.get(model_type, model_type)}尚未训练，请先训练模型。")
            return None
        
        # 加载近期数据用于预测
        df = load_lottery_data(lottery_type)
        recent_data = df.sort_values('期数', ascending=False).head(ml_model.feature_window)
        
        # 结果列表
        results = []
        
        for i in range(num_predictions):
            # 生成预测（传递 variation 给紫微/梅花）
            red_predictions, blue_predictions = ml_model.predict(recent_data, variation=i)
            
            if red_predictions is None or blue_predictions is None:
                print(f"预测失败，请检查数据或重新训练模型。")
                return None
            
            # 根据彩票类型组织结果
            if lottery_type == "dlt":
                # 大乐透: 5个红球 + 2个蓝球
                result = {
                    'red': red_predictions,
                    'blue': blue_predictions
                }
            else:
                # 双色球: 6个红球 + 1个蓝球
                result = {
                    'red': red_predictions,
                    'blue': blue_predictions[:1] if len(blue_predictions) > 0 else []
                }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        import traceback
        print(f"预测时出错: {str(e)}")
        print(traceback.format_exc())
        return None
        
if __name__ == "__main__":
    main() 