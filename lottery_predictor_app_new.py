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
import itertools
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QComboBox, QWidget, QTextEdit, QSpinBox, QHBoxLayout,
    QTabWidget, QScrollArea, QGridLayout, QCheckBox, QGroupBox, QFormLayout,
    QMenu, QAction, QMessageBox, QInputDialog, QLineEdit, QFileDialog,
    QTableWidgetItem, QTableWidget, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer
from PyQt5.QtGui import QPixmap, QTextDocument
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog


from model_utils import (
    name_path, load_resources_pytorch, sample_crf_sequences
)
from ml_models import (
    LotteryMLModels, MODEL_TYPES
)
from thread_utils import (
    TrainModelThread, UpdateDataThread, LogEmitter, BacktestThread,
    AutoPredictThread, DataCheckThread,
    CalibrationThread, FeatureAnalysisThread, RetrainThread
)
from prediction_utils import (
    process_predictions, randomize_numbers
)
from theme_manager import ThemeManager, CustomThemeDialog
from ui_components import (
    create_main_tab, create_analysis_tab, create_advanced_statistics_tab,
    create_expected_value_tab, create_backtest_tab,
    create_ziwei_tab, create_meihua_tab,
    create_prediction_records_tab,
    create_investment_plan_tab_new,
    create_optimization_tab
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
        self.auto_predict_thread = None
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
        
        # 投注计划管理相关
        self.investment_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           'investment_history.json')
        
        # 自动重训策略相关
        self.autoretrain_enabled = False
        self.backtest_history = []  # 保存历史回测结果
        self.last_backtest_result = None
        
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
        self.dt_red_dan, self.dt_red_tuo, self.dt_blue_dan, self.dt_blue_tuo, \
        self.auto_predict_button, self.compare_button, self.autoretrain_checkbox = create_main_tab(self.main_tab)
        
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
        
        # 一键智能预测按钮
        self.auto_predict_button.clicked.connect(self.start_auto_predict)
        # 多模型对比按钮
        self.compare_button.clicked.connect(self.compare_models)
        # 自动重训策略复选框
        self.autoretrain_checkbox.stateChanged.connect(self.toggle_autoretrain)
        
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
         self.mh_num_check, self.mh_year, self.mh_month, self.mh_day,
         self.mh_hour, self.mh_minute,
         self.mh_num1, self.mh_num2, self.mh_num3, self.mh_target,
         self.mh_cards) = create_meihua_tab(self.meihua_tab)
        self.mh_predict_btn.clicked.connect(self.generate_meihua_prediction)
        self.mh_reset_btn.clicked.connect(self.reset_meihua)
        # 勾选「直觉数字」时才启用三个数字输入框
        self.mh_num_check.toggled.connect(
            lambda checked: [self.mh_num1.setEnabled(checked),
                             self.mh_num2.setEnabled(checked),
                             self.mh_num3.setEnabled(checked)]
        )
        self.tab_widget.addTab(self.meihua_tab, "梅花易数")

        # ===== 预测记录标签页 =====
        self.records_tab = QWidget()
        (self.rec_refresh_btn, self.rec_verify_btn, self.rec_verify_all_btn,
         self.rec_export_btn, self.rec_delete_btn,
         self.records_table, self.rec_filter_lottery, self.rec_filter_model,
         self.rec_filter_status, self.rec_stats_label
        ) = create_prediction_records_tab(self.records_tab)
        self.rec_refresh_btn.clicked.connect(self.refresh_prediction_records)
        self.rec_verify_btn.clicked.connect(self.verify_selected_prediction_records)
        self.rec_verify_all_btn.clicked.connect(self.verify_all_prediction_records)
        self.rec_export_btn.clicked.connect(self.export_prediction_records)
        self.rec_delete_btn.clicked.connect(self.delete_selected_prediction_records)
        self.rec_filter_lottery.currentIndexChanged.connect(self.refresh_prediction_records)
        self.rec_filter_model.currentIndexChanged.connect(self.refresh_prediction_records)
        self.rec_filter_status.currentIndexChanged.connect(self.refresh_prediction_records)
        self.tab_widget.addTab(self.records_tab, "预测记录")

        # ===== 投注计划管理标签页（重做） =====
        self.investment_tab = QWidget()
        (self.invest_load_btn, self.invest_confirm_btn, self.invest_verify_btn,
         self.invest_export_btn, self.invest_clear_btn,
         self.prediction_select_table, self.invest_multiplier_spin,
         self.investment_table, self.total_invested_label,
         self.total_won_label, self.net_profit_label,
         self.roi_label, self.win_rate_label
        ) = create_investment_plan_tab_new(self.investment_tab)
        self.invest_load_btn.clicked.connect(self.load_predictions_for_investment)
        self.invest_confirm_btn.clicked.connect(self.confirm_investment_from_predictions)
        self.invest_verify_btn.clicked.connect(self.verify_investment_records)
        self.invest_export_btn.clicked.connect(self.export_investment_records)
        self.invest_clear_btn.clicked.connect(self.clear_investment_records)
        self.tab_widget.addTab(self.investment_tab, "投注计划")

        # ===== 模型优化标签页 =====
        self.optimization_tab = QWidget()
        (self.opt_lottery_combo,
         self.opt_refresh_calib_btn, self.opt_train_calib_btn, self.opt_show_calib_btn,
         self.opt_calib_status_label, self.opt_calib_samples_label, self.opt_calib_update_label,
         self.opt_calib_table,
         self.opt_top_k_spin, self.opt_drift_window_spin,
         self.opt_run_feat_btn, self.opt_refresh_feat_btn, self.opt_drift_check_btn,
         self.opt_feat_table,
         self.opt_use_gpu_check, self.opt_model_multi_edit,
         self.opt_retrain_one_btn, self.opt_retrain_all_btn, self.opt_retrain_status_label,
         self.opt_log_box) = create_optimization_tab(self.optimization_tab)

        # 连接校准信号槽
        self.opt_refresh_calib_btn.clicked.connect(self.refresh_calibration_status)
        self.opt_train_calib_btn.clicked.connect(self.start_calibration)
        self.opt_show_calib_btn.clicked.connect(self.show_calibration_curve)
        # 连接特征分析信号槽
        self.opt_run_feat_btn.clicked.connect(self.run_feature_analysis_ui)
        self.opt_refresh_feat_btn.clicked.connect(self.refresh_feature_list)
        self.opt_drift_check_btn.clicked.connect(self.check_feature_drift_ui)
        # 连接重训练信号槽
        self.opt_retrain_one_btn.clicked.connect(self.retrain_current_lottery)
        self.opt_retrain_all_btn.clicked.connect(self.retrain_all_lotteries_ui)
        # 彩票类型切换时刷新
        self.opt_lottery_combo.currentIndexChanged.connect(self.refresh_calibration_status)
        self.opt_lottery_combo.currentIndexChanged.connect(self.refresh_feature_list)

        self.tab_widget.addTab(self.optimization_tab, "模型优化")

        # 后台线程引用
        self.calibration_thread = None
        self.feature_analysis_thread = None
        self.retrain_thread = None

        # 号码过滤标签页已隐藏
        
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
        
        # ===== 为所有标签页添加「复制结果」+「打印结果」按钮 =====
        self._add_copy_print_buttons()

        self.setCentralWidget(self.tab_widget)
        
        # 加载投注记录
        self.load_investment_records()
        
        # 数据更新定时提醒相关
        self.data_check_timer = QTimer()
        self.data_check_thread = None
        self.last_data_check_time = 0
        
        self.training_thread = None
        self.is_training_paused = False

        # 启动数据更新定时提醒（每30分钟检查一次，首次启动后10秒开始）
        self.data_check_timer.timeout.connect(self.start_data_check)
        self.data_check_timer.start(60 * 60 * 1000)  # 1小时
        QTimer.singleShot(10000, self.start_data_check)  # 首次延迟10秒

    def _add_copy_print_buttons(self):
        """为每个有结果的标签页添加「复制结果」+「打印结果」按钮
        预测页/期望值页：按钮放在结果框内部；其他页：放在底部
        """
        # (tab_widget, result_extractor_name, tab_label, result_label_attr)
        tabs = [
            (self.main_tab, 'main', '预测', 'result_label'),
            (self.expectedvalue_tab, 'ev', '期望值模型', 'ev_result_label'),
            (self.analysis_tab, 'analysis', '数据分析', None),
            (self.advanced_stats_tab, 'advanced', '高级统计', None),
            (self.backtest_tab, 'backtest', '历史回测', None),
            (self.ziwei_tab, 'ziwei', '紫微斗数', None),
            (self.meihua_tab, 'meihua', '梅花易数', None),
            (self.investment_tab, 'investment', '投注计划', None),
            (self.records_tab, 'records', '预测记录', None),
        ]
        for widget, key, label, rl_attr in tabs:
            layout = widget.layout()
            if layout is None:
                continue
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            copy_btn = QPushButton(f"📋 复制{label}结果")
            copy_btn.setStyleSheet("font-size: 11pt; padding: 6px 16px;")
            print_btn = QPushButton(f"🖨 打印{label}结果")
            print_btn.setStyleSheet("font-size: 11pt; padding: 6px 16px;")
            copy_btn.clicked.connect(lambda _, k=key: self._copy_result(k))
            print_btn.clicked.connect(lambda _, k=key: self._print_result(k))
            btn_layout.addWidget(copy_btn)
            btn_layout.addWidget(print_btn)

            # 预测页/期望值页：按钮放到结果框内部
            if rl_attr and hasattr(self, rl_attr):
                rl = getattr(self, rl_attr)
                parent_layout = rl.parent().layout()  # result_group 的 layout
                if parent_layout:
                    parent_layout.addLayout(btn_layout)
                else:
                    layout.addLayout(btn_layout)
            else:
                # 其他页：放到底部
                layout.addLayout(btn_layout)

    def _get_tab_result_text(self, tab_key):
        """提取指定标签页的结果文本"""
        text = ""
        if tab_key == 'main':
            text = self.result_label.text()
        elif tab_key == 'ev':
            text = self.ev_result_label.text()
        elif tab_key == 'analysis':
            text = self.stats_text.toPlainText()
        elif tab_key == 'advanced':
            text = self.stats_result_label.text()
        elif tab_key == 'backtest':
            text = self.bt_result_text.toPlainText()
        elif tab_key == 'ziwei':
            lines = ["═══ 紫微斗数排盘结果 ═══"]
            for i, (rl, bl, dl) in enumerate(self.zw_cards):
                lines.append(f"\n【第{i+1}组】")
                lines.append(f"  {rl.text()}")
                lines.append(f"  {bl.text()}")
                lines.append(f"  {dl.text()}")
            text = "\n".join(lines)
        elif tab_key == 'meihua':
            lines = ["═══ 梅花易数起卦结果 ═══"]
            for i, (rl, bl, dl) in enumerate(self.mh_cards):
                lines.append(f"\n【第{i+1}组】")
                lines.append(f"  {rl.text()}")
                lines.append(f"  {bl.text()}")
                lines.append(f"  {dl.text()}")
            text = "\n".join(lines)
        elif tab_key == 'investment':
            lines = ["═══ 投注计划记录 ═══"]
            for row in range(self.investment_table.rowCount()):
                vals = []
                for col in range(self.investment_table.columnCount()):
                    item = self.investment_table.item(row, col)
                    vals.append(item.text() if item else "")
                lines.append(f"  {' | '.join(vals)}")
            # 统计信息
            lines.append(f"\n总投入: {self.total_invested_label.text()}")
            lines.append(f"总中奖: {self.total_won_label.text()}")
            lines.append(f"净收益: {self.net_profit_label.text()}")
            lines.append(f"ROI: {self.roi_label.text()}")
            lines.append(f"中奖率: {self.win_rate_label.text()}")
            text = "\n".join(lines)
        elif tab_key == 'records':
            lines = ["═══ 预测记录 ═══"]
            for row in range(self.records_table.rowCount()):
                vals = []
                for col in range(self.records_table.columnCount()):
                    item = self.records_table.item(row, col)
                    vals.append(item.text() if item else "")
                lines.append(f"  {' | '.join(vals)}")
            # 统计信息
            lines.append(f"\n{self.rec_stats_label.text()}")
            text = "\n".join(lines)
        return text.strip() if text else "(暂无结果)"

    def _copy_result(self, tab_key):
        """复制结果到系统剪贴板"""
        text = self._get_tab_result_text(tab_key)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.statusBar().showMessage(f"已复制结果到剪贴板 ({len(text)} 字符)", 3000)

    def _print_result(self, tab_key):
        """打印结果"""
        text = self._get_tab_result_text(tab_key)
        if not text or text == "(暂无结果)":
            self.statusBar().showMessage("暂无结果可打印", 3000)
            return
        # 先弹出打印对话框
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle(f"打印{tab_key}结果")
        if dialog.exec_() != QPrintDialog.Accepted:
            return
        # 用 QTextDocument 渲染到 printer
        doc = QTextDocument()
        # 添加标题
        title = f"卜算 - {tab_key} 结果\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n\n"
        doc.setPlainText(title + text)
        doc.print_(printer)
        self.statusBar().showMessage("打印已发送", 3000)

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
        # 移除" (默认)"后缀以匹配 MODEL_TYPES
        model_text_clean = model_text.replace(" (默认)", "").replace(" (默认)", "")
        
        # 从文本映射回模型键
        model_type = None
        for key, value in MODEL_TYPES.items():
            if value == model_text or value == model_text_clean:
                model_type = key
                break
        # LSTM-CRF 特殊处理（不在 MODEL_TYPES 中）
        if model_type is None and 'LSTM' in model_text:
            model_type = 'lstm-crf'

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
        # 移除" (默认)"后缀以匹配 MODEL_TYPES
        model_text_clean = model_text.replace(" (默认)", "").replace(" (默认)", "")
        
        # 从文本映射回模型键
        model_type = None
        for key, value in MODEL_TYPES.items():
            if value == model_text or value == model_text_clean:
                model_type = key
                break
        # LSTM-CRF 特殊处理（不在 MODEL_TYPES 中）
        if model_type is None and 'LSTM' in model_text:
            model_type = 'lstm-crf'
        
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
                        f"（按模型概率从高到低取号；已写入所有复式组合至核验存档）"
                    )
                    self.result_label.setText(result_text)
                    self.log_emitter.new_log.emit(
                        f"复式预测完成: 红{len(red_numbers)}选{ml_model.red_count} "
                        f"蓝{len(blue_numbers)}选{ml_model.blue_count}，{n_notes}注/{n_notes * 2}元")
                    # 生成所有复式组合并存档
                    red_combos = list(itertools.combinations(red_numbers, ml_model.red_count))
                    blue_combos = list(itertools.combinations(blue_numbers, ml_model.blue_count)) if ml_model.blue_count > 0 else [()]
                    predictions_to_save = [ (list(r), list(b)) for r in red_combos for b in blue_combos ]
                    # 获取概率信息用于存档
                    red_probabilities = getattr(ml_model, 'last_red_proba', None)
                    blue_probabilities = getattr(ml_model, 'last_blue_proba', None)
                    self._save_prediction_records(lottery_type, model_type, predictions_to_save, red_probabilities, blue_probabilities)
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
                        f"（胆码固定，拖码按模型概率排序；已写入所有胆拖组合至核验存档）"
                    )
                    self.result_label.setText(result_text)
                    self.log_emitter.new_log.emit(
                        f"胆拖预测完成: 红胆{len(r_dan)}+红拖{len(r_tuo)} "
                        f"蓝胆{len(b_dan)}+蓝拖{len(b_tuo)}，{n_notes}注/{n_notes * 2}元")
                    # 生成所有胆拖组合并存档
                    red_combos = list(itertools.combinations(r_tuo, red_pick)) if red_pick > 0 else [tuple()]
                    blue_combos = list(itertools.combinations(b_tuo, blue_pick)) if blue_pick > 0 else [tuple()]
                    predictions_to_save = []
                    # 收集概率信息用于存档
                    red_probabilities = getattr(ml_model, 'last_red_proba', None)
                    blue_probabilities = getattr(ml_model, 'last_blue_proba', None)
                    for r_combo in red_combos:
                        red_nums = sorted(list(r_dan) + list(r_combo))
                        for b_combo in blue_combos:
                            blue_nums = sorted(list(b_dan) + list(b_combo))
                            predictions_to_save.append((red_nums, blue_nums, red_probabilities, blue_probabilities))
                    self._save_prediction_records(lottery_type, model_type, predictions_to_save, red_probabilities, blue_probabilities)
                    return

                for i in range(num_predictions):
                    # 传递 variation=i 给紫微/梅花等传统算法，生成不同结果
                    red_predictions, blue_predictions = ml_model.predict(recent_data, variation=i)
                    
                    if red_predictions is None or blue_predictions is None:
                        raise ValueError(f"预测失败，请检查数据或重新训练模型。")
                    
                    # 收集预测概率信息（用于后续校准）
                    red_probabilities = getattr(ml_model, 'last_red_proba', None)
                    blue_probabilities = getattr(ml_model, 'last_blue_proba', None)
                    
                    if lottery_type == "dlt":
                        result_text += f"  第 {i+1} 组: {' '.join(map(str, red_predictions))} + {' '.join(map(str, blue_predictions))}\n"
                        collected_predictions.append(
                            (list(red_predictions), list(blue_predictions), red_probabilities, blue_probabilities))
                    else:
                        result_text += f"  第 {i+1} 组: {' '.join(map(str, red_predictions))} + {str(blue_predictions[0])}\n"
                        collected_predictions.append(
                            (list(red_predictions), [int(blue_predictions[0])], red_probabilities, blue_probabilities))

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

    def _save_prediction_records(self, lottery_type, model_type, predictions, red_probabilities=None, blue_probabilities=None):
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
            for prediction in predictions:
                # 处理新旧两种格式：(red, blue) 或 (red, blue, red_proba, blue_proba)
                if len(prediction) == 2:
                    red_nums, blue_nums = prediction
                    red_proba = None
                    blue_proba = None
                elif len(prediction) == 4:
                    red_nums, blue_nums, red_proba, blue_proba = prediction
                else:
                    # 后备方案，只取前两个元素
                    red_nums, blue_nums = prediction[0], prediction[1]
                    red_proba = None
                    blue_proba = None
                    
                count = add_prediction_record(
                    lottery_type, model_type, latest_period,
                    red_nums, blue_nums, red_proba, blue_proba)
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

            # 构造输入信息：根据勾选框决定模式
            use_number = self.mh_num_check.isChecked()
            input_info = {
                'mode': 'number' if use_number else 'time',
                'year': self.mh_year.value(),
                'month': int(self.mh_month.currentText()),
                'day': self.mh_day.value(),
                'hour': int(self.mh_hour.currentText().replace('时', '')),
                'minute': int(self.mh_minute.currentText().replace('分', '')),
                'num1': self.mh_num1.value() if use_number else 0,
                'num2': self.mh_num2.value() if use_number else 0,
                'num3': self.mh_num3.value() if use_number else 0,
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
                f"梅花起卦完成: {'数字起卦' if use_number else '时间起卦'} → 3组号码已生成")
        except Exception as e:
            self.log_emitter.new_log.emit(f"梅花起卦失败: {e}")

    def reset_meihua(self):
        """重置梅花结果"""
        for i in range(3):
            red_label, blue_label, detail_label = self.mh_cards[i]
            red_label.setText("红球: 待起卦")
            blue_label.setText("蓝球: 待起卦")
            detail_label.setText("卦象: 待起卦")
    
    def toggle_autoretrain(self, state):
        """切换自动重训策略"""
        self.autoretrain_enabled = (state == 2)  # Qt.Checked = 2
        status = "启用" if self.autoretrain_enabled else "禁用"
        self.log_emitter.new_log.emit(f"自动重训策略已{status}")
        if self.autoretrain_enabled:
            self.statusBar().showMessage("自动重训策略已启动，将监控回测表现", 3000)
        else:
            self.statusBar().showMessage("自动重训策略已停止", 3000)
    
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
    
    def toggle_autoretrain(self, state):
        """切换自动重训策略"""
        self.autoretrain_enabled = (state == 2)  # Qt.Checked = 2
        status = "启用" if self.autoretrain_enabled else "禁用"
        self.log_emitter.new_log.emit(f"自动重训策略已{status}")
        if self.autoretrain_enabled:
            self.statusBar().showMessage("自动重训策略已启动，将监控回测表现", 3000)
        else:
            self.statusBar().showMessage("自动重训策略已停止", 3000)
    
    # ===== 预测记录管理功能 =====
    def refresh_prediction_records(self):
        """刷新预测记录表格"""
        try:
            from prediction_records import load_records
            records = load_records()

            # 筛选
            lottery_filter = self.rec_filter_lottery.currentText()
            model_filter = self.rec_filter_model.currentText()
            status_filter = self.rec_filter_status.currentText()

            lottery_map = {"双色球": "ssq", "大乐透": "dlt"}
            model_map = {
                "LSTM-CRF": "lstm-crf", "随机森林": "random_forest",
                "XGBoost": "xgboost", "梯度提升树": "gbdt",
                "集成模型": "ensemble", "LightGBM": "lightgbm",
                "CatBoost": "catboost", "紫微斗数": "ziwei",
                "梅花易数": "meihua"
            }

            filtered = []
            for rec in records:
                lt = rec.get('lottery_type', '')
                if lottery_filter != "全部" and lottery_map.get(lottery_filter) != lt:
                    continue
                mt = rec.get('model_type', '')
                mt_display = {v: k for k, v in model_map.items()}.get(mt, mt)
                if model_filter != "全部" and mt_display != model_filter:
                    continue
                prize = rec.get('prize_name')
                if status_filter == "待核验" and prize is not None:
                    continue
                if status_filter == "已核验" and prize is None:
                    continue
                if status_filter == "已中奖" and (prize is None or "未中" in str(prize)):
                    continue
                filtered.append(rec)

            self.records_table.setRowCount(0)
            for i, rec in enumerate(reversed(filtered)):
                row = self.records_table.rowCount()
                self.records_table.insertRow(row)
                self.records_table.setItem(row, 0, QTableWidgetItem(str(len(filtered) - i)))
                self.records_table.setItem(row, 1, QTableWidgetItem(rec.get('predict_time', '?')[:19]))
                lt_name = "双色球" if rec.get('lottery_type') == 'ssq' else "大乐透"
                self.records_table.setItem(row, 2, QTableWidgetItem(lt_name))
                self.records_table.setItem(row, 3, QTableWidgetItem(rec.get('model_type', '?')))
                self.records_table.setItem(row, 4, QTableWidgetItem(str(rec.get('latest_period', '?'))))
                red = rec.get('red_numbers', [])
                blue = rec.get('blue_numbers', [])
                self.records_table.setItem(row, 5, QTableWidgetItem(" ".join(f"{n:02d}" for n in red)))
                self.records_table.setItem(row, 6, QTableWidgetItem(" ".join(f"{n:02d}" for n in blue)))
                prize = rec.get('prize_name')
                if prize is None:
                    status = "待核验"
                    prize_text = "-"
                    amount_text = "-"
                elif "未中" in str(prize):
                    status = "已核验"
                    prize_text = prize
                    amount_text = "0"
                else:
                    status = "已中奖"
                    prize_text = prize
                    amount_text = str(rec.get('prize_amount', 0))
                self.records_table.setItem(row, 7, QTableWidgetItem(status))
                self.records_table.setItem(row, 8, QTableWidgetItem(prize_text))
                self.records_table.setItem(row, 9, QTableWidgetItem(amount_text))

            verified = sum(1 for r in filtered if r.get('prize_name') is not None)
            won = sum(1 for r in filtered if r.get('prize_name') and "未中" not in str(r.get('prize_name')))
            self.rec_stats_label.setText(
                f"共 {len(filtered)} 条记录 | {verified} 条已核验 | {won} 条中奖")
        except Exception as e:
            self.log_emitter.new_log.emit(f"刷新预测记录失败: {e}")

    def verify_selected_prediction_records(self):
        """核验选中的预测记录"""
        try:
            from prediction_records import verify_records
            verify_records(log_callback=lambda msg: self.log_emitter.new_log.emit(msg))
            self.refresh_prediction_records()
            self.log_emitter.new_log.emit("预测记录核验完成")
        except Exception as e:
            self.log_emitter.new_log.emit(f"核验失败: {e}")

    def verify_all_prediction_records(self):
        """全部核验"""
        self.verify_selected_prediction_records()

    def export_prediction_records(self):
        """导出预测记录为CSV"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self, "导出预测记录", "prediction_records.csv", "CSV Files (*.csv)")
            if not path:
                return
            import csv
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["预测时间", "彩票", "模型", "最新期号",
                                 "红球", "蓝球", "核验状态", "奖级", "奖金"])
                for r in range(self.records_table.rowCount()):
                    row_data = []
                    for c in range(self.records_table.columnCount()):
                        item = self.records_table.item(r, c)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            self.log_emitter.new_log.emit(f"已导出 {self.records_table.rowCount()} 条预测记录到 {path}")
        except Exception as e:
            self.log_emitter.new_log.emit(f"导出失败: {e}")

    def delete_selected_prediction_records(self):
        """删除选中的预测记录（通过匹配内容定位原始记录）"""
        try:
            selected_rows = sorted(set(item.row() for item in self.records_table.selectedItems()), reverse=True)
            if not selected_rows:
                QMessageBox.information(self, "提示", "请先选择要删除的记录")
                return
            from prediction_records import load_records, save_records
            records = load_records()
            # 通过匹配表格内容（预测时间+彩票+模型+期号+红球+蓝球）定位原始记录
            to_delete = set()
            for row in selected_rows:
                if row >= self.records_table.rowCount():
                    continue
                predict_time = self.records_table.item(row, 1).text() if self.records_table.item(row, 1) else ""
                lottery_name = self.records_table.item(row, 2).text() if self.records_table.item(row, 2) else ""
                model = self.records_table.item(row, 3).text() if self.records_table.item(row, 3) else ""
                period = self.records_table.item(row, 4).text() if self.records_table.item(row, 4) else ""
                red_str = self.records_table.item(row, 5).text() if self.records_table.item(row, 5) else ""
                blue_str = self.records_table.item(row, 6).text() if self.records_table.item(row, 6) else ""
                # 在原始记录中查找匹配
                for idx, rec in enumerate(records):
                    if idx in to_delete:
                        continue
                    rec_lt = "双色球" if rec.get('lottery_type') == 'ssq' else "大乐透"
                    rec_red = " ".join(f"{n:02d}" for n in rec.get('red_numbers', []))
                    rec_blue = " ".join(f"{n:02d}" for n in rec.get('blue_numbers', []))
                    if (rec.get('predict_time', '')[:19] == predict_time and
                        rec_lt == lottery_name and
                        rec.get('model_type', '') == model and
                        str(rec.get('latest_period', '')) == period and
                        rec_red == red_str and rec_blue == blue_str):
                        to_delete.add(idx)
                        break
            records = [r for i, r in enumerate(records) if i not in to_delete]
            save_records(records)
            self.refresh_prediction_records()
            self.log_emitter.new_log.emit(f"已删除 {len(to_delete)} 条预测记录")
        except Exception as e:
            self.log_emitter.new_log.emit(f"删除失败: {e}")

    # ===== 投注计划管理功能（重做） =====
    def load_predictions_for_investment(self):
        """加载最近预测记录到投注选择表"""
        try:
            from prediction_records import load_records
            records = load_records()
            # 取最近 20 条
            recent = list(reversed(records[-20:]))
            self.prediction_select_table.setRowCount(0)
            for rec in recent:
                row = self.prediction_select_table.rowCount()
                self.prediction_select_table.insertRow(row)
                cb = QCheckBox()
                self.prediction_select_table.setCellWidget(row, 0, cb)
                self.prediction_select_table.setItem(row, 1, QTableWidgetItem(rec.get('predict_time', '?')[:19]))
                lt_name = "双色球" if rec.get('lottery_type') == 'ssq' else "大乐透"
                self.prediction_select_table.setItem(row, 2, QTableWidgetItem(lt_name))
                self.prediction_select_table.setItem(row, 3, QTableWidgetItem(rec.get('model_type', '?')))
                red = rec.get('red_numbers', [])
                blue = rec.get('blue_numbers', [])
                self.prediction_select_table.setItem(row, 4, QTableWidgetItem(" ".join(f"{n:02d}" for n in red)))
                self.prediction_select_table.setItem(row, 5, QTableWidgetItem(" ".join(f"{n:02d}" for n in blue)))
                self.prediction_select_table.setItem(row, 6, QTableWidgetItem(str(rec.get('latest_period', '?'))))
            self.log_emitter.new_log.emit(f"已加载 {len(recent)} 条最近预测记录")
        except Exception as e:
            self.log_emitter.new_log.emit(f"加载预测记录失败: {e}")

    def confirm_investment_from_predictions(self):
        """从预测记录确认投注"""
        try:
            multiplier = self.invest_multiplier_spin.value()
            cost_per_ticket = 2  # 每注2元
            added = 0
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M")

            for r in range(self.prediction_select_table.rowCount()):
                cb = self.prediction_select_table.cellWidget(r, 0)
                if cb and cb.isChecked():
                    lottery_item = self.prediction_select_table.item(r, 2)
                    model_item = self.prediction_select_table.item(r, 3)
                    red_item = self.prediction_select_table.item(r, 4)
                    blue_item = self.prediction_select_table.item(r, 5)
                    period_item = self.prediction_select_table.item(r, 6)
                    if not all([lottery_item, model_item, red_item, blue_item]):
                        continue
                    lottery_name = lottery_item.text()
                    period = period_item.text() if period_item else "?"
                    numbers = f"{red_item.text()} + {blue_item.text()}"
                    amount = cost_per_ticket * multiplier
                    model = model_item.text()

                    row = self.investment_table.rowCount()
                    self.investment_table.insertRow(row)
                    self.investment_table.setItem(row, 0, QTableWidgetItem(now))
                    self.investment_table.setItem(row, 1, QTableWidgetItem(lottery_name))
                    self.investment_table.setItem(row, 2, QTableWidgetItem(period))
                    self.investment_table.setItem(row, 3, QTableWidgetItem(model))
                    self.investment_table.setItem(row, 4, QTableWidgetItem(numbers))
                    self.investment_table.setItem(row, 5, QTableWidgetItem(str(multiplier)))
                    self.investment_table.setItem(row, 6, QTableWidgetItem(f"{amount:.2f}"))
                    self.investment_table.setItem(row, 7, QTableWidgetItem("待开奖"))
                    self.investment_table.setItem(row, 8, QTableWidgetItem("0.00"))
                    added += 1

            if added == 0:
                QMessageBox.information(self, "提示", "请先勾选要投注的预测记录")
                return
            self.update_investment_stats()
            self.save_investment_records()
            self.log_emitter.new_log.emit(f"已添加 {added} 条投注记录，倍数 {multiplier}，投入 {added * cost_per_ticket * multiplier:.2f} 元")
        except Exception as e:
            self.log_emitter.new_log.emit(f"确认投注失败: {e}")

    def verify_investment_records(self):
        """核验选中的投注记录"""
        try:
            selected_rows = sorted(set(item.row() for item in self.investment_table.selectedItems()))
            if not selected_rows:
                QMessageBox.information(self, "提示", "请先选择要核验的投注记录")
                return
            from backtest import SSQ_PRIZE_TABLE, DLT_PRIZE_TABLE, _load_csv_data
            verified = 0
            skipped = 0
            for row in selected_rows:
                period_item = self.investment_table.item(row, 2)
                numbers_item = self.investment_table.item(row, 4)
                lottery_item = self.investment_table.item(row, 1)
                multiplier_item = self.investment_table.item(row, 5)
                if not all([period_item, numbers_item, lottery_item]):
                    continue
                lottery_name = lottery_item.text()
                lottery_type = 'ssq' if '双色球' in lottery_name else 'dlt'
                period = int(period_item.text())
                # 解析号码
                parts = numbers_item.text().split("+")
                red_str = parts[0].strip().split()
                blue_str = parts[1].strip().split() if len(parts) > 1 else []
                red = [int(x) for x in red_str]
                blue = [int(x) for x in blue_str]
                # 查找开奖数据
                df = _load_csv_data(lottery_type)
                if df is None:
                    self.log_emitter.new_log.emit(f"无法加载开奖数据: {lottery_type}")
                    break
                draw = df[df['期数'] == period]
                if draw.empty:
                    skipped += 1
                    continue
                draw_row = draw.iloc[0]
                actual_red = [int(draw_row[col]) for col in draw_row.index if '红球' in str(col)]
                actual_blue = [int(draw_row[col]) for col in draw_row.index if '蓝球' in str(col)]
                red_hits = len(set(red) & set(actual_red))
                blue_hits = len(set(blue) & set(actual_blue))
                # 判断奖级
                prize_table = SSQ_PRIZE_TABLE if lottery_type == 'ssq' else DLT_PRIZE_TABLE
                key = (red_hits, blue_hits)
                multiplier = int(multiplier_item.text()) if multiplier_item else 1
                prize_entry = prize_table.get(key)
                if prize_entry is None:
                    prize = "未中奖"
                    prize_amount = 0
                else:
                    prize = prize_entry[0]
                    # 奖金：None=浮动奖(按固定估算)，否则用表中金额
                    if prize_entry[1] is not None:
                        prize_amount = prize_entry[1] * multiplier
                    else:
                        # 一等奖/二等奖浮动，按估算值
                        prize_amount = (5000000 if "一" in prize else 150000) * multiplier
                self.investment_table.setItem(row, 7, QTableWidgetItem(prize))
                # 计算盈亏
                cost = 2 * multiplier
                net_profit = prize_amount - cost
                self.investment_table.setItem(row, 8, QTableWidgetItem(f"{net_profit:.2f}"))
                verified += 1
            self.update_investment_stats()
            self.save_investment_records()
            msg = f"已核验 {verified} 条投注记录"
            if skipped > 0:
                msg += f"（{skipped} 条期号未找到，跳过）"
            self.log_emitter.new_log.emit(msg)
        except Exception as e:
            self.log_emitter.new_log.emit(f"核验投注失败: {e}")

    def export_investment_records(self):
        """导出投注记录为CSV"""
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "导出投注记录", "investment_records.csv", "CSV Files (*.csv)")
            if not path:
                return
            import csv
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["投注时间", "彩票", "期号", "模型", "号码",
                                 "倍数", "投入(元)", "奖级", "盈亏(元)"])
                for r in range(self.investment_table.rowCount()):
                    row_data = []
                    for c in range(self.investment_table.columnCount()):
                        item = self.investment_table.item(r, c)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            self.log_emitter.new_log.emit(f"已导出投注记录到 {path}")
        except Exception as e:
            self.log_emitter.new_log.emit(f"导出失败: {e}")

    def clear_investment_records(self):
        """清空全部投注记录"""
        reply = QMessageBox.question(self, "确认", "确定清空全部投注记录？",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.investment_table.setRowCount(0)
            self.update_investment_stats()
            self.save_investment_records()
            self.log_emitter.new_log.emit("已清空全部投注记录")

    def update_investment_stats(self):
        """更新投注统计"""
        try:
            total_invested = 0.0
            total_won = 0.0
            won_count = 0
            total_count = self.investment_table.rowCount()
            for r in range(total_count):
                amount_item = self.investment_table.item(r, 6)
                profit_item = self.investment_table.item(r, 8)
                prize_item = self.investment_table.item(r, 7)
                if amount_item:
                    total_invested += float(amount_item.text())
                if profit_item:
                    profit = float(profit_item.text())
                    # 盈亏 = 奖金 - 成本，总中奖 = 盈亏 + 成本（即奖金）
                    if prize_item and prize_item.text() not in ("待开奖", "未中奖", ""):
                        cost = float(amount_item.text()) if amount_item else 0
                        total_won += profit + cost
                        won_count += 1
            net = total_won - total_invested
            roi = (net / total_invested * 100) if total_invested > 0 else 0
            win_rate = (won_count / total_count * 100) if total_count > 0 else 0
            self.total_invested_label.setText(f"总投入: {total_invested:.2f} 元")
            self.total_won_label.setText(f"总中奖: {total_won:.2f} 元")
            self.net_profit_label.setText(f"净盈亏: {net:.2f} 元")
            self.roi_label.setText(f"ROI: {roi:.2f}%")
            self.win_rate_label.setText(f"中奖率: {win_rate:.2f}%")
        except Exception as e:
            self.log_emitter.new_log.emit(f"更新统计失败: {e}")

    def save_investment_records(self):
        """保存投注记录到JSON文件"""
        try:
            records = []
            for r in range(self.investment_table.rowCount()):
                record = {}
                for c in range(self.investment_table.columnCount()):
                    item = self.investment_table.item(r, c)
                    record[c] = item.text() if item else ""
                records.append(record)
            with open(self.investment_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_emitter.new_log.emit(f"保存投注记录失败: {e}")

    def load_investment_records(self):
        """从JSON文件加载投注记录"""
        try:
            if not hasattr(self, 'investment_file'):
                self.investment_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                   'investment_history.json')
            if not os.path.exists(self.investment_file):
                return
            import json
            with open(self.investment_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            self.investment_table.setRowCount(0)
            for record in records:
                row = self.investment_table.rowCount()
                self.investment_table.insertRow(row)
                for c in range(self.investment_table.columnCount()):
                    self.investment_table.setItem(row, c, QTableWidgetItem(str(record.get(str(c), ""))))
            self.update_investment_stats()
        except Exception as e:
            self.log_emitter.new_log.emit(f"加载投注记录失败: {e}")

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

    def start_data_check(self):
        """启动数据更新检查线程（防重复启动）"""
        if self.data_check_thread and self.data_check_thread.isRunning():
            return
        # 每次检查前更新last时间（简单防抖）
        import time
        now = time.time()
        if now - self.last_data_check_time < 30:  # 30秒内不重复检查
            return
        self.last_data_check_time = now

        current_lottery = list(name_path.keys())[self.lottery_combo.currentIndex()]
        self.data_check_thread = DataCheckThread(lottery_type=current_lottery)
        self.data_check_thread.new_data_signal.connect(self.on_data_check_result)
        self.data_check_thread.start()

    def on_data_check_result(self, has_new: bool, message: str):
        """处理数据检查结果"""
        if has_new and message:
            # 状态栏提醒（3秒）
            self.statusBar().showMessage(message, 60000)  # 1分钟
            # 标题栏追加提醒（不覆盖GPU信息）
            base_title = f"卜算 - 彩票娱乐软件 - GPU: {self.cuda_info}"
            self.setWindowTitle(base_title + " ｜ " + message)
        else:
            # 无新数据时恢复原标题
            base_title = f"卜算 - 彩票娱乐软件 - GPU: {self.cuda_info}"
            self.setWindowTitle(base_title)

    def start_auto_predict(self):
        """一键智能预测按钮点击处理"""
        if self.auto_predict_thread and self.auto_predict_thread.isRunning():
            self.update_log("一键智能预测正在进行中，请稍候...")
            return

        # 获取当前选择
        lottery_index = self.lottery_combo.currentIndex()
        lottery_keys = list(name_path.keys())
        lottery_type = lottery_keys[lottery_index]

        model_text = self.model_combo.currentText()
        model_text_clean = model_text.replace(" (默认)", "").replace(" (默认)", "")
        model_type = None
        for key, value in MODEL_TYPES.items():
            if value == model_text or value == model_text_clean:
                model_type = key
                break
        if model_type is None and 'LSTM' in model_text:
            model_type = 'lstm-crf'
        if model_type is None:
            self.update_log("错误：无法识别模型类型")
            return

        use_gpu = self.gpu_checkbox.isChecked()
        num_pred = self.prediction_spin.value()

        # 禁用相关控件防止重复点击
        self.auto_predict_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.update_data_button.setEnabled(False)
        self.lottery_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.gpu_checkbox.setEnabled(False)
        self.prediction_spin.setEnabled(False)
        self.statusBar().showMessage("一键智能预测已启动...")

        # 清空结果区
        self.result_label.setText("一键智能预测进行中，请稍候...")

        # 创建并启动线程
        self.auto_predict_thread = AutoPredictThread(
            lottery_type=lottery_type,
            model_type=model_type,
            num_predictions=num_pred,
            use_gpu=use_gpu,
            skip_fetch=False,
            skip_train=False
        )
        self.auto_predict_thread.log_signal.connect(self.update_log)
        self.auto_predict_thread.step_signal.connect(lambda s: self.statusBar().showMessage(s))
        self.auto_predict_thread.predictions_signal.connect(self._auto_predict_collect_predictions)
        self.auto_predict_thread.finished_signal.connect(self.on_auto_predict_finished)
        self.auto_predict_thread.start()

    def _auto_predict_collect_predictions(self, predictions):
        """收集一键预测产出的结构化预测结果，待完成回调中统一存档"""
        self._auto_predict_pending = predictions

    def on_auto_predict_finished(self, success, result_text):
        """一键智能预测完成回调"""
        # 恢复控件状态
        self.auto_predict_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.update_data_button.setEnabled(True)
        self.lottery_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.gpu_checkbox.setEnabled(torch.cuda.is_available())
        self.prediction_spin.setEnabled(True)
        self.statusBar().clearMessage()

        if success:
            self.result_label.setText(result_text)
            self.update_log("一键智能预测完成。")
            # 自动存档：复用普通预测的存档机制（读取最新期号、逐注写入）
            pending = getattr(self, '_auto_predict_pending', None)
            if pending:
                # AutoPredictThread 内保存了正确的 key（'ssq'/'dlt'）与 model_type
                t_lottery = self.auto_predict_thread.lottery_type
                t_model = self.auto_predict_thread.model_type
                self._save_prediction_records(t_lottery, t_model, pending)
                self._auto_predict_pending = None
            else:
                self.update_log("（未获取到结构化预测结果，本次未存档）")
        else:
            self.result_label.setText("一键智能预测失败，请查看日志了解详情。")
            self.update_log("一键智能预测失败。")

    def compare_models(self):
        """多模型对比预测：同期数、相同参数下，比较多个模型的预测结果"""
        if self.compare_button and self.compare_button.isEnabled() == False:
            return
            
        # 获取当前选择
        lottery_index = self.lottery_combo.currentIndex()
        lottery_keys = list(name_path.keys())
        lottery_type = lottery_keys[lottery_index]
        lottery_name = name_path[lottery_type]['name']
        num_predictions = self.prediction_spin.value()
        
        # 禁用相关控件防止重复点击
        self.compare_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.update_data_button.setEnabled(False)
        self.auto_predict_button.setEnabled(False)
        self.lottery_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.gpu_checkbox.setEnabled(False)
        self.prediction_spin.setEnabled(False)
        self.statusBar().showMessage("多模型对比进行中，请稍候...")
        
        # 清空结果区
        self.result_label.setText("多模型对比进行中，请稍候...")
        self.log_box.clear()
        
        # 记录开始时间
        import time
        start_time = time.time()
        
        # 要对比的模型列表（排除不支持复式/胆拖的模型和传统算法）
        compare_models = ['random_forest', 'xgboost', 'gbdt', 'lightgbm', 'catboost', 'ensemble']
        model_names = {key: MODEL_TYPES[key] for key in compare_models if key in MODEL_TYPES}
        
        # 结果存储
        all_results = {}
        success_count = 0
        
        try:
            df = load_lottery_data(lottery_type)
            recent_data = df.sort_values('期数', ascending=False).head(10)  # 使用较小窗口以加快速度
            
            for model_key, model_name in model_names.items():
                self.log_emitter.new_log.emit(f"正在测试 {model_name}...")
                
                try:
                    # 检查模型是否已经训练
                    model_full_key = f"{lottery_type}_{model_key}"
                    if model_full_key not in self.ml_models:
                        self.log_emitter.new_log.emit(f"初始化 {model_name} 模型...")
                        
                        use_gpu = self.gpu_checkbox.isChecked()
                        self.ml_models[model_full_key] = LotteryMLModels(
                            lottery_type=lottery_type, 
                            model_type=model_key,
                            log_callback=self.log_emitter.new_log.emit,
                            use_gpu=use_gpu
                        )
                    
                    ml_model = self.ml_models[model_full_key]
                    
                    # 特殊处理期望值模型（不参与对比，因为通常不提供概率）
                    if model_key == 'expected_value':
                        self.log_emitter.new_log.emit(f"跳过 {model_name}（通常不提供具体号码预测）")
                        continue
                    
                    # 检查模型是否可用
                    if not ml_model.load_models():
                        self.log_emitter.new_log.emit(f"跳过 {model_name}（模型尚未训练）")
                        continue
                    
                    # 单式预测
                    red_numbers, blue_numbers = ml_model.predict(recent_data, variation=0)
                    
                    if red_numbers is None or blue_numbers is None:
                        raise ValueError("预测返回空结果")
                    
                    # 格式化结果
                    if lottery_type == "dlt":
                        result_str = f"{' '.join(map(str, red_numbers))} + {' '.join(map(str, blue_numbers))}"
                    else:
                        result_str = f"{' '.join(map(str, red_numbers))} + {str(blue_numbers[0])}"
                    
                    all_results[model_name] = {
                        'red': list(red_numbers),
                        'blue': list(blue_numbers) if lottery_type == "dlt" else [int(blue_numbers[0])],
                        'display': result_str,
                        'success': True
                    }
                    success_count += 1
                    self.log_emitter.new_log.emit(f"{model_name} 预测成功: {result_str}")
                    
                except Exception as model_e:
                    self.log_emitter.new_log.emit(f"{model_name} 预测失败: {str(model_e)}")
                    all_results[model_name] = {
                        'success': False,
                        'error': str(model_e)
                    }
            
            # 生成对比结果文本
            result_text = f"【多模型对比预测】{lottery_name}\n"
            result_text += f"最新期: {int(df['期数'].max())}\n"
            result_text += f"对比模型: {len(model_names)} 个\n"
            result_text += f"成功预测: {success_count} 个\n\n"
            
            # 显示每个模型的预测结果
            for model_name, result in all_results.items():
                if result['success']:
                    result_text += f"{model_name}: {result['display']}\n"
                else:
                    result_text += f"{model_name}: 预测失败 ({result.get('error', '未知错误')})\n"
            
            # 添加简单共识分析（如果有足够成功的预测）
            if success_count >= 2:
                result_text += "\n--- 号码频率分析（成功模型） ---\n"
                
                if lottery_type == "dlt":
                    # 红球频率
                    red_counts = {}
                    blue_counts = {}
                    for model_name, result in all_results.items():
                        if result['success']:
                            for num in result['red']:
                                red_counts[num] = red_counts.get(num, 0) + 1
                            for num in result['blue']:
                                blue_counts[num] = blue_counts.get(num, 0) + 1
                    
                    if red_counts:
                        max_red_count = max(red_counts.values())
                        hot_red = [str(num) for num, cnt in red_counts.items() if cnt == max_red_count]
                        result_text += f"热门红球 (出现{max_red_count}次): {' '.join(hot_red)}\n"
                    
                    if blue_counts:
                        max_blue_count = max(blue_counts.values())
                        hot_blue = [str(num) for num, cnt in blue_counts.items() if cnt == max_blue_count]
                        result_text += f"热门蓝球 (出现{max_blue_count}次): {' '.join(hot_blue)}\n"
                else:
                    # 双色球
                    red_counts = {}
                    blue_counts = {}
                    for model_name, result in all_results.items():
                        if result['success']:
                            for num in result['red']:
                                red_counts[num] = red_counts.get(num, 0) + 1
                            blue_num = result['blue'][0]
                            blue_counts[blue_num] = blue_counts.get(blue_num, 0) + 1
                    
                    if red_counts:
                        max_red_count = max(red_counts.values())
                        hot_red = [str(num) for num, cnt in red_counts.items() if cnt == max_red_count]
                        result_text += f"热门红球 (出现{max_red_count}次): {' '.join(hot_red)}\n"
                    
                    if blue_counts:
                        max_blue_count = max(blue_counts.values())
                        hot_blue = [str(num) for num, cnt in blue_counts.items() if cnt == max_blue_count]
                        result_text += f"热门蓝球 (出现{max_blue_count}次): {' '.join(hot_blue)}\n"
            
            used_time = time.time() - start_time
            result_text += f"\n总用时: {used_time:.1f} 秒"
            
            self.result_label.setText(result_text)
            
            # 恢复控件状态
            self.compare_button.setEnabled(True)
            self.predict_button.setEnabled(True)
            self.train_button.setEnabled(True)
            self.update_data_button.setEnabled(True)
            self.auto_predict_button.setEnabled(True)
            self.lottery_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.gpu_checkbox.setEnabled(True)
            self.prediction_spin.setEnabled(True)
            
            self.statusBar().showMessage(f"多模型对比完成 - 成功 {success_count}/{len(model_names)} 模型", 5000)
            self.log_emitter.new_log.emit(f"多模型对比完成: 成功 {success_count}/{len(model_names)} 模型, 用时 {used_time:.1f}s")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_emitter.new_log.emit(f"多模型对比时出错: {e}")
            self.log_emitter.new_log.emit(f"错误详情:\n{error_details}")
            self.result_label.setText(f"多模型对比时出错: {e}")
            
            # 即使出错也要恢复控件状态
            self.compare_button.setEnabled(True)
            self.predict_button.setEnabled(True)
            self.train_button.setEnabled(True)
            self.update_data_button.setEnabled(True)
            self.auto_predict_button.setEnabled(True)
            self.lottery_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.gpu_checkbox.setEnabled(True)
            self.prediction_spin.setEnabled(True)
            
            self.statusBar().showMessage("多模型对比失败", 5000)

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

    # ==================== 模型优化标签页 - 校准 ====================

    def _get_opt_lottery_type(self):
        """获取优化标签页当前选中的彩票类型"""
        return 'ssq' if self.opt_lottery_combo.currentIndex() == 0 else 'dlt'

    def _opt_log(self, msg):
        """写入模型优化日志框（带时间戳）"""
        from datetime import datetime
        self.opt_log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _set_opt_buttons_enabled(self, enabled):
        """批量启用/禁用优化页操作按钮"""
        for btn in [self.opt_refresh_calib_btn, self.opt_train_calib_btn,
                    self.opt_show_calib_btn, self.opt_run_feat_btn,
                    self.opt_refresh_feat_btn, self.opt_drift_check_btn,
                    self.opt_retrain_one_btn, self.opt_retrain_all_btn]:
            btn.setEnabled(enabled)

    def refresh_calibration_status(self, *args):
        """刷新校准状态显示"""
        try:
            from model_calibration import get_calibration_info
            lottery_type = self._get_opt_lottery_type()
            info = get_calibration_info(lottery_type)

            red_ok = info.get('red_calibrated', False)
            blue_ok = info.get('blue_calibrated', False)
            red_samples = info.get('red_samples', 0)
            blue_samples = info.get('blue_samples', 0)
            last_update = info.get('last_update')

            if red_ok and blue_ok:
                self.opt_calib_status_label.setText("校准状态: 已校准")
                self.opt_calib_status_label.setStyleSheet(
                    "font-size: 11pt; font-weight: bold; color: #2E7D32;")
            elif red_ok or blue_ok:
                self.opt_calib_status_label.setText("校准状态: 部分校准")
                self.opt_calib_status_label.setStyleSheet(
                    "font-size: 11pt; font-weight: bold; color: #FF9800;")
            else:
                self.opt_calib_status_label.setText("校准状态: 未校准")
                self.opt_calib_status_label.setStyleSheet(
                    "font-size: 11pt; font-weight: bold; color: #F44336;")

            self.opt_calib_samples_label.setText(
                f"校准样本: 红球 {red_samples} / 蓝球 {blue_samples}")
            self.opt_calib_update_label.setText(
                f"更新时间: {last_update[:19] if last_update else '-'}")

            # 填充校准状态表格
            from PyQt5.QtGui import QColor
            self.opt_calib_table.setRowCount(2)
            for row, (ball, ok, samples) in enumerate([
                ('红球', red_ok, red_samples), ('蓝球', blue_ok, blue_samples)
            ]):
                status_text = "已校准" if ok else "未校准"
                color = "#2E7D32" if ok else "#F44336"
                for col, text in enumerate([ball, status_text, str(samples), '-', 
                                            last_update[:19] if last_update else '-']):
                    item = QTableWidgetItem(text)
                    if col == 1:
                        item.setForeground(QColor(color))
                    self.opt_calib_table.setItem(row, col, item)
        except Exception as e:
            self._opt_log(f"刷新校准状态失败: {e}")

    def start_calibration(self):
        """启动校准训练（后台线程）"""
        lottery_type = self._get_opt_lottery_type()
        reply = QMessageBox.question(
            self, "确认校准",
            f"确定对 {lottery_type.upper()} 训练校准模型？\n"
            f"将使用已核验的预测记录作为训练数据。",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self._set_opt_buttons_enabled(False)
        self._opt_log(f"开始训练 {lottery_type.upper()} 校准模型...")

        self.calibration_thread = CalibrationThread(lottery_type)
        self.calibration_thread.log_signal.connect(self._opt_log)
        self.calibration_thread.finished_signal.connect(self._on_calibration_done)
        self.calibration_thread.start()

    def _on_calibration_done(self, success):
        """校准完成回调"""
        self._set_opt_buttons_enabled(True)
        self._opt_log(f"校准流程结束: {'成功' if success else '失败'}")
        self.refresh_calibration_status()
        if success:
            QMessageBox.information(self, "完成", "校准模型训练完成！")

    def show_calibration_curve(self):
        """显示校准曲线"""
        try:
            import matplotlib
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            from sklearn.isotonic import IsotonicRegression
            from model_calibration import (load_calibration_model,
                                           collect_calibration_data)
            import numpy as np

            lottery_type = self._get_opt_lottery_type()
            red_model = load_calibration_model(lottery_type, 'red')
            blue_model = load_calibration_model(lottery_type, 'blue')

            if red_model is None and blue_model is None:
                QMessageBox.warning(self, "提示", "尚无校准模型，请先训练。")
                return

            fig, ax = plt.subplots(figsize=(10, 7))
            # 理想校准线
            x = np.linspace(0, 1, 100)
            ax.plot(x, x, 'k--', alpha=0.5, label='理想校准线')

            for model, ball_name, color in [
                (red_model, '红球', '#E53935'),
                (blue_model, '蓝球', '#1E88E5')
            ]:
                if model is None:
                    continue
                calibrated = model.predict(x)
                ax.plot(x, calibrated, color=color, linewidth=2,
                        label=f'{ball_name}校准曲线')

            # 散点：实际 (原始概率, 是否命中)
            red_data, blue_data = collect_calibration_data(lottery_type)
            if red_data:
                probs, outcomes = zip(*red_data)
                ax.scatter(probs, outcomes, alpha=0.3, s=20,
                           color='#E53935', edgecolors='none', label='红球数据点')
            if blue_data:
                probs, outcomes = zip(*blue_data)
                ax.scatter(probs, outcomes, alpha=0.3, s=20,
                           color='#1E88E5', edgecolors='none', label='蓝球数据点')

            ax.set_xlabel('预测概率')
            ax.set_ylabel('实际频率')
            ax.set_title(f'{lottery_type.upper()} 模型校准曲线')
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            import traceback
            self._opt_log(f"显示校准曲线失败: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "错误", f"显示校准曲线失败:\n{e}")

    # ==================== 模型优化标签页 - 特征分析 ====================

    def refresh_feature_list(self, *args):
        """刷新特征重要性列表（从已保存的分析结果读取）"""
        try:
            from feature_analysis import load_feature_importance
            lottery_type = self._get_opt_lottery_type()
            data = load_feature_importance()
            importance = data.get(lottery_type, {}) if data else {}

            if not importance:
                self.opt_feat_table.setRowCount(0)
                self._opt_log(f"暂无 {lottery_type.upper()} 的特征分析结果，请先运行分析")
                return

            # 按重要性排序
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_k = self.opt_top_k_spin.value()
            display = sorted_features[:top_k]

            self.opt_feat_table.setRowCount(len(display))
            for row, (feat, score) in enumerate(display):
                self.opt_feat_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                self.opt_feat_table.setItem(row, 1, QTableWidgetItem(feat))
                self.opt_feat_table.setItem(row, 2, QTableWidgetItem(f"{score:.6f}"))
                # 简单分类
                if '遗漏' in feat or 'frequency' in feat.lower():
                    cat = "频率类"
                elif '和值' in feat or 'sum' in feat.lower():
                    cat = "形态类"
                elif '跨度' in feat or 'span' in feat.lower():
                    cat = "形态类"
                elif 'AC' in feat or '奇' in feat or '偶' in feat:
                    cat = "结构类"
                else:
                    cat = "其他"
                self.opt_feat_table.setItem(row, 3, QTableWidgetItem(cat))
        except Exception as e:
            self._opt_log(f"刷新特征列表失败: {e}")

    def run_feature_analysis_ui(self):
        """运行完整特征分析（后台线程）"""
        lottery_type = self._get_opt_lottery_type()
        reply = QMessageBox.question(
            self, "确认运行",
            f"对 {lottery_type.upper()} 运行完整特征分析？\n"
            f"包括特征重要性、漂移检测和特征选择。",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self._set_opt_buttons_enabled(False)
        self._opt_log(f"开始 {lottery_type.upper()} 特征分析...")

        self.feature_analysis_thread = FeatureAnalysisThread(
            lottery_type, top_k=self.opt_top_k_spin.value(),
            drift_window=self.opt_drift_window_spin.value())
        self.feature_analysis_thread.log_signal.connect(self._opt_log)
        self.feature_analysis_thread.result_signal.connect(self._on_feature_result)
        self.feature_analysis_thread.drift_signal.connect(self._on_drift_result)
        self.feature_analysis_thread.finished_signal.connect(self._on_feature_done)
        self.feature_analysis_thread.start()

    def _on_feature_result(self, result):
        """特征分析结果回调：填充表格"""
        top_features = result.get('top_features', [])
        all_features = result.get('all_features', {})

        # 优先用重要性排序
        if isinstance(all_features, dict) and all_features:
            sorted_features = sorted(all_features.items(),
                                     key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                                     reverse=True)
            display = sorted_features[:self.opt_top_k_spin.value()]
        elif top_features:
            display = [(f, 0.0) for f in top_features]
        else:
            display = []

        self.opt_feat_table.setRowCount(len(display))
        for row, (feat, score) in enumerate(display):
            self.opt_feat_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.opt_feat_table.setItem(row, 1, QTableWidgetItem(str(feat)))
            self.opt_feat_table.setItem(row, 2, QTableWidgetItem(f"{score:.6f}" if score else '-'))
            self.opt_feat_table.setItem(row, 3, QTableWidgetItem("其他"))

    def _on_drift_result(self, drift_result):
        """漂移检测结果回调"""
        if not drift_result:
            return
        if 'error' in drift_result:
            self._opt_log(f"漂移检测出错: {drift_result['error']}")
            return
        if 'reason' in drift_result and not drift_result.get('drift_detected'):
            self._opt_log(f"漂移检测: {drift_result['reason']}")
            return

        if drift_result.get('drift_detected'):
            drifted = [k for k, v in drift_result.get('feature_drift', {}).items()
                       if v.get('is_drifting')]
            self._opt_log(f"警告: 检测到 {len(drifted)} 个特征发生漂移: {', '.join(drifted[:10])}")
        else:
            self._opt_log("漂移检测: 未发现显著特征漂移")

    def _on_feature_done(self, success):
        """特征分析完成"""
        self._set_opt_buttons_enabled(True)
        self._opt_log(f"特征分析流程结束: {'成功' if success else '失败'}")
        self.refresh_feature_list()
        if success:
            QMessageBox.information(self, "完成", "特征分析完成！")

    def check_feature_drift_ui(self):
        """单独运行漂移检测（后台线程）"""
        lottery_type = self._get_opt_lottery_type()
        self._set_opt_buttons_enabled(False)
        self._opt_log(f"开始检测 {lottery_type.upper()} 特征漂移...")

        self.feature_analysis_thread = FeatureAnalysisThread(
            lottery_type, top_k=5,
            drift_window=self.opt_drift_window_spin.value(),
            check_drift=True)
        self.feature_analysis_thread.log_signal.connect(self._opt_log)
        self.feature_analysis_thread.result_signal.connect(lambda r: None)
        self.feature_analysis_thread.drift_signal.connect(self._on_drift_result)
        self.feature_analysis_thread.finished_signal.connect(self._on_feature_done)
        self.feature_analysis_thread.start()

    # ==================== 模型优化标签页 - 反馈重训练 ====================

    def retrain_current_lottery(self):
        """重训练当前彩票的ML模型"""
        lottery_type = self._get_opt_lottery_type()
        use_gpu = self.opt_use_gpu_check.isChecked()

        reply = QMessageBox.question(
            self, "确认重训练",
            f"重训练 {lottery_type.upper()} 的 ML 模型？\n\n"
            f"使用GPU: {'是' if use_gpu else '否'}\n"
            f"模型: {self.opt_model_multi_edit.text() or '全部'}\n\n"
            f"此过程可能耗时较长，确定继续？",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self._set_opt_buttons_enabled(False)
        self._opt_log(f"开始重训练 {lottery_type.upper()} 模型 (GPU={'开' if use_gpu else '关'})...")

        self.retrain_thread = RetrainThread(
            lottery_type=lottery_type, use_gpu=use_gpu)
        self.retrain_thread.log_signal.connect(self._opt_log)
        self.retrain_thread.finished_signal.connect(self._on_retrain_done)
        self.retrain_thread.start()

    def retrain_all_lotteries_ui(self):
        """重训练全部彩票模型"""
        use_gpu = self.opt_use_gpu_check.isChecked()

        reply = QMessageBox.question(
            self, "确认全部重训练",
            f"重训练全部彩票 (SSQ + DLT) 的 ML 模型？\n\n"
            f"使用GPU: {'是' if use_gpu else '否'}\n\n"
            f"此过程可能耗时很长，确定继续？",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self._set_opt_buttons_enabled(False)
        self._opt_log("开始重训练全部彩票模型...")

        self.retrain_thread = RetrainThread(
            retrain_all=True, use_gpu=use_gpu)
        self.retrain_thread.log_signal.connect(self._opt_log)
        self.retrain_thread.finished_signal.connect(self._on_retrain_done)
        self.retrain_thread.start()

    def _on_retrain_done(self, success):
        """重训练完成回调"""
        from datetime import datetime
        self._set_opt_buttons_enabled(True)
        status = "成功" if success else "失败"
        self._opt_log(f"重训练结束: {status}")
        self.opt_retrain_status_label.setText(
            f"上次重训练: {datetime.now().strftime('%Y-%m-%d %H:%M')} ({status})")
        if success:
            QMessageBox.information(self, "完成", "模型重训练完成！")
        else:
            QMessageBox.warning(self, "警告", "部分模型重训练失败，请查看日志。")

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