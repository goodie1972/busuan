# -*- coding:utf-8 -*-
"""
UI Components for Lottery Predictor Application
Author: Yang Zhao
"""
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QComboBox, QWidget, 
    QTextEdit, QSpinBox, QHBoxLayout, QTabWidget, QScrollArea, 
    QGridLayout, QCheckBox, QGroupBox, QFormLayout, QMainWindow,
    QMenu, QAction
)
from PyQt5.QtCore import Qt
from theme_manager import ThemeManager, CustomThemeDialog
import torch
from model_utils import name_path

def create_main_tab(main_tab):
    """
    创建主标签页的UI组件
    
    Args:
        main_tab: QWidget，主标签页容器
        
    Returns:
        tuple: 主标签页中的关键UI组件
    """
    main_layout = QVBoxLayout(main_tab)
    main_layout.setSpacing(6)    
    main_layout.setContentsMargins(8, 8, 8, 8)  
    

    cuda_available = False
    cuda_device = "不可用"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except:
        pass
    

    top_widget = QWidget()
    top_layout = QGridLayout()
    

    lottery_label = QLabel("彩票类型:")
    lottery_combo = QComboBox()
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    
    model_label = QLabel("模型类型:")
    model_combo = QComboBox()

    model_combo.addItem("LSTM-CRF (默认)")

    from ml_models import MODEL_TYPES
    for model_key, model_name in MODEL_TYPES.items():
        model_combo.addItem(model_name)
    
    top_layout.addWidget(lottery_label, 0, 0)
    top_layout.addWidget(lottery_combo, 0, 1)
    top_layout.addWidget(model_label, 0, 2)
    top_layout.addWidget(model_combo, 0, 3)
    
    prediction_label = QLabel("预测数量:")
    prediction_spin = QSpinBox()
    prediction_spin.setRange(1, 10)
    prediction_spin.setValue(5)
    
    gpu_checkbox = QCheckBox("使用GPU训练")
    gpu_checkbox.setChecked(cuda_available)
    gpu_checkbox.setEnabled(cuda_available)
    if not cuda_available:
        gpu_checkbox.setToolTip("您的系统未安装GPU版本的PyTorch或没有可用的CUDA设备")
    else:
        gpu_checkbox.setToolTip(f"使用GPU加速训练 ({cuda_device})")
    
    predict_button = QPushButton("生成预测")
    
    top_layout.addWidget(prediction_label, 1, 0)
    top_layout.addWidget(prediction_spin, 1, 1)
    top_layout.addWidget(gpu_checkbox, 1, 2)
    top_layout.addWidget(predict_button, 1, 3)
    
    main_layout.addWidget(top_widget)
    
  
    theme_layout = QHBoxLayout()
    theme_label = QLabel("主题样式:")
    theme_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
    theme_combo = QComboBox()
    theme_combo.addItems(ThemeManager().get_theme_names())
    theme_combo.setCurrentText(ThemeManager().current_theme)
    
    customize_theme_button = QPushButton("自定义主题")
    
    theme_layout.addStretch()
    theme_layout.addWidget(theme_label)
    theme_layout.addWidget(theme_combo)
    theme_layout.addWidget(customize_theme_button)
    
    main_layout.addLayout(theme_layout)
    
  
    control_layout = QHBoxLayout()
    control_layout.setSpacing(10)
    
    settings_group = QGroupBox("预测设置")
    settings_layout = QFormLayout(settings_group)
    settings_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    settings_layout.setSpacing(6)
    
    settings_layout.addRow("彩票类型:", lottery_combo)
    settings_layout.addRow("预测数量:", prediction_spin)
    settings_layout.addRow("预测模型:", model_combo)

    compound_check = QCheckBox("复式模式")
    compound_check.setToolTip("勾选后按模型概率取前N个号码组成号码池，而非单式固定数量")
    compound_red_spin = QSpinBox()
    compound_red_spin.setRange(0, 35)
    compound_red_spin.setValue(0)
    compound_red_spin.setToolTip("复式红球总数(如双色球填8 = 8个红球里选6，0 = 不启用)")
    compound_blue_spin = QSpinBox()
    compound_blue_spin.setRange(0, 16)
    compound_blue_spin.setValue(0)
    compound_blue_spin.setToolTip("复式蓝球总数(如双色球填2 = 2个蓝球里选1，0 = 不启用)")
    settings_layout.addRow("复式模式:", compound_check)
    settings_layout.addRow("复式红球数:", compound_red_spin)
    settings_layout.addRow("复式蓝球数:", compound_blue_spin)

    gpu_group = QGroupBox("GPU设置")
    gpu_layout = QVBoxLayout(gpu_group)
    gpu_layout.setSpacing(6)
    
    gpu_layout.addWidget(gpu_checkbox)
    
    control_layout.addWidget(settings_group, 2)
    control_layout.addWidget(gpu_group, 1)
    
    main_layout.addLayout(control_layout)
    
   
    button_layout = QHBoxLayout()
    button_layout.setSpacing(6)
    
    train_button = QPushButton("训练模型")
    train_button.setMinimumHeight(30)
    
    pause_button = QPushButton("暂停训练")
    pause_button.setMinimumHeight(30)
    pause_button.setEnabled(False)
    
    analyze_button = QPushButton("数据分析")
    analyze_button.setMinimumHeight(30)
    
    update_data_button = QPushButton("更新数据")
    update_data_button.setMinimumHeight(30)
    
    button_layout.addWidget(predict_button)
    button_layout.addWidget(train_button)
    button_layout.addWidget(pause_button)
    button_layout.addWidget(analyze_button)
    button_layout.addWidget(update_data_button)
    
    main_layout.addLayout(button_layout)

    
    content_layout = QHBoxLayout()
    
    
    result_group = QGroupBox("预测结果")
    result_layout = QVBoxLayout(result_group)
    
    result_label = QLabel("点击'生成预测'按钮查看预测结果")
    result_label.setAlignment(Qt.AlignCenter)
    result_label.setWordWrap(True)
    result_label.setStyleSheet("padding: 10px; background-color: white; border: 1px solid #DDDDDD;")
    result_label.setMinimumHeight(200)
    
    result_layout.addWidget(result_label)
    
    
    log_group = QGroupBox("训练和预测日志")
    log_layout = QVBoxLayout(log_group)
    
    log_box = QTextEdit()
    log_box.setReadOnly(True)
    log_box.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    
    # 启用右键菜单
    log_box.setContextMenuPolicy(Qt.CustomContextMenu)
    
    log_layout.addWidget(log_box)
    
    content_layout.addWidget(result_group, 1)
    content_layout.addWidget(log_group, 2)
    
    main_layout.addLayout(content_layout, 1)
    
    return (predict_button, train_button, pause_button, analyze_button, update_data_button,
            lottery_combo, prediction_spin, gpu_checkbox, result_label, log_box,
            theme_combo, customize_theme_button, model_combo,
            compound_check, compound_red_spin, compound_blue_spin)


def create_analysis_tab(analysis_tab):
    """
    创建数据分析标签页的UI组件
    
    Args:
        analysis_tab: QWidget，数据分析标签页容器
        
    Returns:
        tuple: 数据分析标签页中的关键UI组件
    """
    analysis_layout = QVBoxLayout(analysis_tab)
    analysis_layout.setSpacing(6)
    analysis_layout.setContentsMargins(8, 8, 8, 8)
    

    analysis_control_layout = QHBoxLayout()
    
    chart_group = QGroupBox("图表选择")
    chart_layout = QHBoxLayout(chart_group)
    
    chart_layout.addWidget(QLabel("分析图表:"))
    analysis_combo = QComboBox()
    analysis_combo.addItems(["频率分布", "热冷号码", "遗漏分析", "号码模式", "趋势分析"])
    analysis_combo.setMinimumWidth(150)
    chart_layout.addWidget(analysis_combo)
    
    trend_feature_combo = QComboBox()
    trend_feature_combo.setVisible(False)  
    trend_feature_combo.setMinimumWidth(150)
    chart_layout.addWidget(trend_feature_combo)
    chart_layout.addStretch()
    
    analysis_control_layout.addWidget(chart_group)
    analysis_layout.addLayout(analysis_control_layout)
    
   
    analysis_content = QHBoxLayout()
    
    chart_group = QGroupBox("数据可视化")
    chart_layout = QVBoxLayout(chart_group)
    
    chart_label = QLabel("请先点击'数据分析'按钮加载数据")
    chart_label.setAlignment(Qt.AlignCenter)
    chart_label.setMinimumHeight(350)
    chart_label.setStyleSheet("border: 1px dashed #CCCCCC; background-color: white;")
    
    chart_layout.addWidget(chart_label)
    
    stats_group = QGroupBox("统计信息摘要")
    stats_layout = QVBoxLayout(stats_group)
    
    stats_text = QTextEdit()
    stats_text.setReadOnly(True)
    stats_text.setMaximumHeight(150)
    stats_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    
    stats_layout.addWidget(stats_text)
    
    analysis_content.addWidget(chart_group, 7)
    analysis_content.addWidget(stats_group, 3)
    
    analysis_layout.addLayout(analysis_content, 1)
    
    # 添加新的统计分析按钮
    advanced_stats_button = QPushButton("高级统计分析")
    analysis_layout.addWidget(advanced_stats_button)
    
    distribution_analysis_button = QPushButton("分布分析")
    analysis_layout.addWidget(distribution_analysis_button)
    
    return (analysis_combo, trend_feature_combo, chart_label, stats_text,
            advanced_stats_button, distribution_analysis_button)


def create_advanced_statistics_tab(advanced_stats_tab):
    """
    创建高级统计分析标签页的UI组件
    
    Args:
        advanced_stats_tab: QWidget，高级统计分析标签页容器
        
    Returns:
        tuple: 高级统计分析标签页中的关键UI组件
    """
    # 创建整体布局
    advanced_layout = QVBoxLayout(advanced_stats_tab)
    advanced_layout.setSpacing(10)
    advanced_layout.setContentsMargins(8, 8, 8, 8)
    
    # 创建控制区域布局
    control_layout = QHBoxLayout()
    
    # 彩票类型选择
    lottery_label = QLabel("彩票类型:")
    lottery_combo = QComboBox()
    lottery_combo.addItems(["双色球", "大乐透"])
    
    # 添加按钮
    run_stats_button = QPushButton("运行高级统计分析")
    run_distribution_button = QPushButton("运行分布分析")
    show_data_button = QPushButton("显示详细统计数据")
    
    control_layout.addWidget(lottery_label)
    control_layout.addWidget(lottery_combo)
    control_layout.addWidget(run_stats_button)
    control_layout.addWidget(run_distribution_button)
    control_layout.addWidget(show_data_button)
    
    advanced_layout.addLayout(control_layout)
    
    # 创建结果显示区域
    result_label = QLabel("点击'运行分析'按钮查看统计分析结果")
    result_label.setAlignment(Qt.AlignCenter)
    result_label.setMinimumHeight(500)
    result_label.setStyleSheet("background-color: white; border: 1px solid #DDDDDD;")
    
    # 使用QScrollArea包裹结果显示区域，以支持滚动
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(result_label)
    
    advanced_layout.addWidget(scroll_area, 1)
    
    return (lottery_combo, run_stats_button, run_distribution_button, show_data_button, result_label)


def create_expected_value_tab(expected_value_tab):
    """
    创建期望值模型专用标签页的UI组件
    
    Args:
        expected_value_tab: QWidget，期望值模型标签页容器
        
    Returns:
        tuple: 期望值模型标签页中的关键UI组件
    """
    # 创建整体布局
    ev_layout = QVBoxLayout(expected_value_tab)
    ev_layout.setSpacing(10)
    ev_layout.setContentsMargins(8, 8, 8, 8)
    
    # 创建标题
    title_label = QLabel("期望值模型预测")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
    ev_layout.addWidget(title_label)
    
    # 创建设置区域
    settings_group = QGroupBox("模型设置")
    settings_layout = QFormLayout(settings_group)
    
    # 彩票类型选择
    lottery_combo = QComboBox()
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    settings_layout.addRow("彩票类型:", lottery_combo)
    
    # 预测数量
    prediction_spin = QSpinBox()
    prediction_spin.setRange(1, 10)
    prediction_spin.setValue(5)
    settings_layout.addRow("预测数量:", prediction_spin)
    
    # 训练设置
    train_group = QGroupBox("训练设置")
    train_layout = QVBoxLayout(train_group)
    
    gpu_checkbox = QCheckBox("使用GPU训练")
    cuda_available = torch.cuda.is_available()
    gpu_checkbox.setChecked(cuda_available)
    gpu_checkbox.setEnabled(cuda_available)
    train_layout.addWidget(gpu_checkbox)
    
    # 添加控制按钮
    button_layout = QHBoxLayout()
    
    train_button = QPushButton("训练期望值模型")
    train_button.setMinimumHeight(30)
    
    predict_button = QPushButton("生成期望值预测")
    predict_button.setMinimumHeight(30)
    
    update_data_button = QPushButton("更新历史数据")
    update_data_button.setMinimumHeight(30)
    
    button_layout.addWidget(train_button)
    button_layout.addWidget(predict_button)
    button_layout.addWidget(update_data_button)
    
    # 组织布局
    control_layout = QHBoxLayout()
    control_layout.addWidget(settings_group, 1)
    control_layout.addWidget(train_group, 1)
    
    ev_layout.addLayout(control_layout)
    ev_layout.addLayout(button_layout)
    
    # 创建结果显示区域
    results_group = QGroupBox("预测结果")
    results_layout = QVBoxLayout(results_group)
    
    result_label = QLabel("期望值模型预测结果将显示在这里")
    result_label.setAlignment(Qt.AlignCenter)
    result_label.setWordWrap(True)
    result_label.setStyleSheet("padding: 10px; background-color: white; border: 1px solid #DDDDDD;")
    result_label.setMinimumHeight(150)
    
    results_layout.addWidget(result_label)
    
    # 创建日志显示区域
    log_group = QGroupBox("期望值模型训练日志")
    log_layout = QVBoxLayout(log_group)
    
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    log_text.setContextMenuPolicy(Qt.CustomContextMenu)
    
    log_layout.addWidget(log_text)
    
    # 结果和日志区域布局
    content_layout = QHBoxLayout()
    content_layout.addWidget(results_group, 1)
    content_layout.addWidget(log_group, 2)
    
    ev_layout.addLayout(content_layout, 1)
    
    # 添加说明文本
    info_label = QLabel(
        "期望值模型基于历史数据统计和博弈论中的期望值概念计算最优号码组合。"
        "此模型分析历史开奖模式、号码频率和组合特征，为每个可能的号码分配期望值，"
        "并基于这些期望值生成预测结果。"
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet("font-style: italic; color: #666666; margin-top: 5px;")
    
    ev_layout.addWidget(info_label)
    
    return (predict_button, train_button, update_data_button,
            lottery_combo, prediction_spin, gpu_checkbox, 
            result_label, log_text)


# 回测可用模型类型（显示名 -> 模型标识）
BACKTEST_MODEL_ITEMS = [
    ("集成模型", "ensemble"),
    ("GBDT", "gbdt"),
    ("LightGBM", "lightgbm"),
    ("XGBoost", "xgboost"),
    ("随机森林", "random_forest"),
    ("CatBoost", "catboost"),
]


def create_backtest_tab(backtest_tab):
    """
    创建历史回测标签页的UI组件

    Args:
        backtest_tab: QWidget，历史回测标签页容器

    Returns:
        tuple: (start_button, lottery_combo, model_combo, periods_spin,
                result_text, status_label)
    """
    bt_layout = QVBoxLayout(backtest_tab)
    bt_layout.setSpacing(10)
    bt_layout.setContentsMargins(8, 8, 8, 8)

    # 创建标题
    title_label = QLabel("历史回测")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
    bt_layout.addWidget(title_label)

    # 创建设置区域
    settings_group = QGroupBox("回测设置")
    settings_layout = QFormLayout(settings_group)

    # 彩票类型选择
    lottery_combo = QComboBox()
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    settings_layout.addRow("彩票类型:", lottery_combo)

    # 模型类型选择
    model_combo = QComboBox()
    for display, value in BACKTEST_MODEL_ITEMS:
        model_combo.addItem(display, value)
    settings_layout.addRow("模型类型:", model_combo)

    # 回测期数（0 = 全部历史）
    periods_spin = QSpinBox()
    periods_spin.setRange(0, 10000)
    periods_spin.setValue(0)
    periods_spin.setSpecialValueText("全部历史")
    periods_spin.setSuffix(" 期")
    settings_layout.addRow("回测期数:", periods_spin)

    bt_layout.addWidget(settings_group)

    # 控制按钮与状态
    control_layout = QHBoxLayout()
    start_button = QPushButton("开始回测")
    start_button.setMinimumHeight(32)
    start_button.setStyleSheet("font-weight: bold;")
    verify_button = QPushButton("核验预测记录")
    verify_button.setMinimumHeight(32)
    status_label = QLabel("就绪")
    status_label.setStyleSheet("color: #666666;")
    control_layout.addWidget(start_button)
    control_layout.addWidget(verify_button)
    control_layout.addWidget(status_label, 1)
    bt_layout.addLayout(control_layout)

    # 结果显示区域
    results_group = QGroupBox("回测结果")
    results_layout = QVBoxLayout(results_group)
    result_text = QTextEdit()
    result_text.setReadOnly(True)
    result_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
    results_layout.addWidget(result_text)
    bt_layout.addWidget(results_group, 1)

    # 说明文本
    info_label = QLabel(
        "回测使用逐期滚动方式：每期只用截至当期之前的历史数据预测下一期，并与实际开奖对比。"
        "内置 10 万注随机选号蒙特卡洛基线作为参照，可客观判断模型是否优于随机选号。"
        "注意：模型训练时见过全部历史数据，回测结果偏乐观，请作为相对评估而非对未来收益的承诺；"
        "彩票为独立随机事件，长期期望收益为负。"
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet("font-style: italic; color: #666666; margin-top: 5px;")
    bt_layout.addWidget(info_label)

    return (start_button, lottery_combo, model_combo, periods_spin,
            result_text, status_label, verify_button)


def create_main_window():
    """
    创建主窗口实例
    
    Returns:
        LotteryPredictorApp: 主窗口实例
    """
    from lottery_predictor_app_new import LotteryPredictorApp
    return LotteryPredictorApp() 