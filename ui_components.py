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
    

    # ==================== 全局样式 ====================
    label_style = "font-size: 11pt;"
    spin_style = "font-size: 11pt; min-height: 26px;"

    # ==================== 第一行：彩票类型 + 预测模型 ====================
    row1 = QHBoxLayout()
    row1.setSpacing(12)
    lottery_label = QLabel("彩票类型:")
    lottery_label.setStyleSheet(label_style)
    lottery_combo = QComboBox()
    lottery_combo.setStyleSheet(spin_style)
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    model_label = QLabel("预测模型:")
    model_label.setStyleSheet(label_style)
    model_combo = QComboBox()
    model_combo.setStyleSheet(spin_style)
    model_combo.addItem("LSTM-CRF (默认)")
    from ml_models import MODEL_TYPES
    for model_key, model_name in MODEL_TYPES.items():
        model_combo.addItem(model_name)
    row1.addWidget(lottery_label)
    row1.addWidget(lottery_combo, 1)
    row1.addSpacing(20)
    row1.addWidget(model_label)
    row1.addWidget(model_combo, 1)
    row1.addStretch()
    main_layout.addLayout(row1)

    # ==================== 第二行：三张并排模式卡片 ====================
    # 卡片通用样式
    card_style = "QGroupBox { font-size: 11pt; font-weight: bold; border: 2px solid #B0B0B0; border-radius: 8px; margin-top: 10px; padding: 8px 6px 6px 6px;} QGroupBox::title { subpadding-left: 10px; }"

    cards_layout = QHBoxLayout()
    cards_layout.setSpacing(10)

    # ---- 卡片1：单式 ----
    single_card = QGroupBox("单式")
    single_card.setStyleSheet(card_style)
    single_card.setCheckable(True)
    single_card.setChecked(True)  # 默认选中
    single_v = QVBoxLayout(single_card)
    single_v.setSpacing(8)
    prediction_label = QLabel("预测数量:")
    prediction_label.setStyleSheet(label_style)
    prediction_spin = QSpinBox()
    prediction_spin.setRange(1, 10)
    prediction_spin.setValue(5)
    prediction_spin.setStyleSheet(spin_style)
    single_v.addWidget(prediction_label)
    single_v.addWidget(prediction_spin)
    single_v.addStretch()
    cards_layout.addWidget(single_card, 1)

    # ---- 卡片2：复式 ----
    compound_card = QGroupBox("复式")
    compound_card.setStyleSheet(card_style)
    compound_card.setCheckable(True)
    compound_card.setChecked(False)
    compound_v = QVBoxLayout(compound_card)
    compound_v.setSpacing(6)
    compound_red_spin = QSpinBox()
    compound_red_spin.setRange(0, 35)
    compound_red_spin.setValue(8)
    compound_red_spin.setStyleSheet(spin_style)
    compound_red_spin.setToolTip("复式红球总数(如双色球填8 = 8个红球里选6)")
    compound_blue_spin = QSpinBox()
    compound_blue_spin.setRange(0, 16)
    compound_blue_spin.setValue(2)
    compound_blue_spin.setStyleSheet(spin_style)
    compound_blue_spin.setToolTip("复式蓝球总数(如双色球填2 = 2个蓝球里选1)")
    lbl_r = QLabel("红球数:"); lbl_r.setStyleSheet(label_style)
    lbl_b = QLabel("蓝球数:"); lbl_b.setStyleSheet(label_style)
    compound_v.addWidget(lbl_r)
    compound_v.addWidget(compound_red_spin)
    compound_v.addWidget(lbl_b)
    compound_v.addWidget(compound_blue_spin)
    compound_v.addStretch()
    cards_layout.addWidget(compound_card, 1)

    # ---- 卡片3：胆拖 ----
    dantuo_card = QGroupBox("胆拖")
    dantuo_card.setStyleSheet(card_style)
    dantuo_card.setCheckable(True)
    dantuo_card.setChecked(False)
    dantuo_v = QVBoxLayout(dantuo_card)
    dantuo_v.setSpacing(4)
    dt_red_dan = QSpinBox()
    dt_red_dan.setRange(1, 5)
    dt_red_dan.setValue(2)
    dt_red_dan.setStyleSheet(spin_style)
    dt_red_dan.setToolTip("红球胆码个数(SSQ最多5, DLT最多4)")
    dt_red_tuo = QSpinBox()
    dt_red_tuo.setRange(1, 33)
    dt_red_tuo.setValue(6)
    dt_red_tuo.setStyleSheet(spin_style)
    dt_red_tuo.setToolTip("红球拖码个数")
    dt_blue_dan = QSpinBox()
    dt_blue_dan.setRange(0, 2)
    dt_blue_dan.setValue(0)
    dt_blue_dan.setStyleSheet(spin_style)
    dt_blue_dan.setToolTip("蓝球胆码个数(SSQ只能0, DLT最多1)")
    dt_blue_tuo = QSpinBox()
    dt_blue_tuo.setRange(1, 16)
    dt_blue_tuo.setValue(2)
    dt_blue_tuo.setStyleSheet(spin_style)
    dt_blue_tuo.setToolTip("蓝球拖码个数")
    for lbl, w in [("红胆:", dt_red_dan), ("红拖:", dt_red_tuo), ("蓝胆:", dt_blue_dan), ("蓝拖:", dt_blue_tuo)]:
        l = QLabel(lbl); l.setStyleSheet(label_style)
        dantuo_v.addWidget(l)
        dantuo_v.addWidget(w)
    dantuo_v.addStretch()
    cards_layout.addWidget(dantuo_card, 1)

    main_layout.addLayout(cards_layout)

    # ==================== 模式互斥逻辑 ====================
    mode_cards = {'单式': single_card, '复式': compound_card, '胆拖': dantuo_card}
    # 用于外部读取当前模式的 QComboBox（保持返回值兼容）
    mode_combo = QComboBox()
    mode_combo.addItems(["单式", "复式", "胆拖"])
    mode_combo.setVisible(False)  # 隐藏，仅作数据载体

    def _on_card_toggled(checked_card):
        """选中一张卡时取消其他两张（不禁用卡片本身，保持可点击）"""
        def _toggled(state):
            if state:
                for name, card in mode_cards.items():
                    if card is not checked_card:
                        card.blockSignals(True)
                        card.setChecked(False)
                        card.blockSignals(False)
                mode_combo.setCurrentText(checked_card.title())
        return _toggled

    for card in mode_cards.values():
        card.toggled.connect(_on_card_toggled(card))

    # ==================== 第三行：主题 + GPU + 操作按钮 ====================
    row3 = QHBoxLayout()
    row3.setSpacing(8)
    theme_label = QLabel("主题:")
    theme_label.setStyleSheet(label_style)
    theme_combo = QComboBox()
    theme_combo.setStyleSheet(spin_style)
    theme_combo.addItems(ThemeManager().get_theme_names())
    theme_combo.setCurrentText(ThemeManager().current_theme)
    customize_theme_button = QPushButton("自定义主题")
    customize_theme_button.setStyleSheet(label_style)
    gpu_checkbox = QCheckBox("GPU训练")
    gpu_checkbox.setStyleSheet(label_style)
    gpu_checkbox.setChecked(cuda_available)
    gpu_checkbox.setEnabled(cuda_available)
    if not cuda_available:
        gpu_checkbox.setToolTip("未检测到GPU版PyTorch")
    else:
        gpu_checkbox.setToolTip(f"GPU加速 ({cuda_device})")
    row3.addWidget(theme_label)
    row3.addWidget(theme_combo)
    row3.addWidget(customize_theme_button)
    row3.addSpacing(20)
    row3.addWidget(gpu_checkbox)
    row3.addStretch()
    main_layout.addLayout(row3)

    # 操作按钮行
    button_layout = QHBoxLayout()
    button_layout.setSpacing(6)
    predict_button = QPushButton("生成预测")
    predict_button.setMinimumHeight(34)
    predict_button.setStyleSheet("font-size: 11pt; font-weight: bold;")
    train_button = QPushButton("训练模型")
    train_button.setMinimumHeight(34)
    train_button.setStyleSheet(label_style)
    pause_button = QPushButton("暂停训练")
    pause_button.setMinimumHeight(34)
    pause_button.setEnabled(False)
    pause_button.setStyleSheet(label_style)
    analyze_button = QPushButton("数据分析")
    analyze_button.setMinimumHeight(34)
    analyze_button.setStyleSheet(label_style)
    update_data_button = QPushButton("更新数据")
    update_data_button.setMinimumHeight(34)
    update_data_button.setStyleSheet(label_style)
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
    # 预测结果字体：比默认大两个号并加粗
    from PyQt5.QtGui import QFont
    result_font = QFont(result_label.font())
    result_font.setPointSize(result_font.pointSize() + 4)
    result_font.setBold(True)
    result_label.setFont(result_font)
    
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
            mode_combo, compound_red_spin, compound_blue_spin,
            dt_red_dan, dt_red_tuo, dt_blue_dan, dt_blue_tuo)


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
    # 预测结果字体：比默认大两个号并加粗（与主预测页一致）
    from PyQt5.QtGui import QFont
    result_font = QFont(result_label.font())
    result_font.setPointSize(result_font.pointSize() + 4)
    result_font.setBold(True)
    result_label.setFont(result_font)
    
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