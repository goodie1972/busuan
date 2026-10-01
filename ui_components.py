# -*- coding:utf-8 -*-
"""
UI Components for Lottery Predictor Application
Author: Yang Zhao
"""
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QComboBox, QWidget, 
    QTextEdit, QSpinBox, QHBoxLayout, QTabWidget, QScrollArea, 
    QGridLayout, QCheckBox, QGroupBox, QFormLayout, QMainWindow,
    QMenu, QAction, QToolButton
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
    lottery_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
    lottery_combo = QComboBox()
    lottery_combo.setStyleSheet(spin_style)
    lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
    model_label = QLabel("预测模型:")
    model_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
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
    card_style = "QGroupBox { font-size: 11pt; font-weight: bold; border: 2px solid #B0B0B0; border-radius: 8px; margin-top: 10px; padding: 8px 6px 6px 6px;} QGroupBox::title { padding-left: 10px; }"

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
                # 切换到复式/胆拖时禁用不支持该模式的模型选项
                mode_text = checked_card.title()
                unsupported = {"LSTM-CRF", "XGBoost", "期望值"}
                for i in range(model_combo.count()):
                    item_text = model_combo.itemText(i)
                    should_disable = mode_text in ("复式", "胆拖") and \
                        any(k in item_text for k in unsupported)
                    item = model_combo.model().item(i)
                    if item:
                        item.setEnabled(not should_disable)
                # 如果当前选中被禁用，切到第一个可用项
                cur_item = model_combo.model().item(model_combo.currentIndex())
                if cur_item and not cur_item.isEnabled():
                    for i in range(model_combo.count()):
                        if model_combo.model().item(i).isEnabled():
                            model_combo.setCurrentIndex(i)
                            break
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
    result_label.setStyleSheet("padding: 10px; border: 1px solid #8B6F47; border-radius: 3px; font-size: 14pt; font-weight: bold;")
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
    log_box.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt;")
    
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
    
    # 统一按钮样式
    label_style = "font-size: 11pt; padding: 6px 12px;"
    
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
    chart_label.setStyleSheet("""
        QLabel {
            border: 1px dashed #8B6F47; 
            border-radius: 3px;
        }
    """)
    
    chart_layout.addWidget(chart_label)
    
    stats_group = QGroupBox("统计信息摘要")
    stats_layout = QVBoxLayout(stats_group)
    
    stats_text = QTextEdit()
    stats_text.setReadOnly(True)
    stats_text.setMaximumHeight(150)
    stats_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt;")
    
    stats_layout.addWidget(stats_text)
    
    analysis_content.addWidget(chart_group, 7)
    analysis_content.addWidget(stats_group, 3)
    
    analysis_layout.addLayout(analysis_content, 1)
    
    # --- 底部按钮区 ---
    btn_layout = QHBoxLayout()
    
    # 主按钮：选号参考（醒目，主色调）
    suggestion_button = QPushButton("💡 生成选号参考")
    suggestion_button.setMinimumHeight(40)
    suggestion_button.setStyleSheet("""
        QPushButton {
            font-size: 12pt; font-weight: bold; color: white;
            background-color: #C9702D; border: none; border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover { background-color: #D4853E; }
        QPushButton:pressed { background-color: #A65D24; }
    """)
    btn_layout.addWidget(suggestion_button)
    
    # 次级按钮：加载分析数据
    load_analysis_button = QPushButton("加载分析数据")
    load_analysis_button.setMinimumHeight(34)
    load_analysis_button.setStyleSheet(label_style)
    btn_layout.addWidget(load_analysis_button)
    
    # 折叠菜单：专业分析（隐藏复杂统计）
    pro_menu = QMenu()
    pro_menu.setStyleSheet("font-size: 11pt;")
    action_advanced = QAction("高级统计分析", pro_menu)
    action_distribution = QAction("分布分析", pro_menu)
    pro_menu.addAction(action_advanced)
    pro_menu.addAction(action_distribution)
    
    pro_button = QToolButton()
    pro_button.setText("专业分析 ▼")
    pro_button.setMinimumHeight(34)
    pro_button.setStyleSheet(label_style)
    pro_button.setMenu(pro_menu)
    pro_button.setPopupMode(QToolButton.InstantPopup)
    btn_layout.addWidget(pro_button)
    
    analysis_layout.addLayout(btn_layout)
    
    # 为了兼容性，保留原按钮引用（通过 menu actions 触发）
    advanced_stats_button = action_advanced
    distribution_analysis_button = action_distribution
    
    return (analysis_combo, trend_feature_combo, chart_label, stats_text,
            advanced_stats_button, distribution_analysis_button, 
            load_analysis_button, suggestion_button)


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
    
    # 主按钮：生成选号参考
    suggestion_button = QPushButton("💡 生成选号参考")
    suggestion_button.setMinimumHeight(40)
    suggestion_button.setStyleSheet("""
        QPushButton {
            font-size: 12pt; font-weight: bold; color: white;
            background-color: #C9702D; border: none; border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover { background-color: #D4853E; }
        QPushButton:pressed { background-color: #A65D24; }
    """)
    
    # 折叠菜单：专业分析
    pro_menu = QMenu()
    pro_menu.setStyleSheet("font-size: 11pt;")
    action_advanced = QAction("运行高级统计分析", pro_menu)
    action_distribution = QAction("运行分布分析", pro_menu)
    pro_menu.addAction(action_advanced)
    pro_menu.addAction(action_distribution)
    
    pro_button = QToolButton()
    pro_button.setText("专业分析 ▼")
    pro_button.setMinimumHeight(34)
    pro_button.setStyleSheet("font-size: 11pt; padding: 6px 12px;")
    pro_button.setMenu(pro_menu)
    pro_button.setPopupMode(QToolButton.InstantPopup)
    
    # 显示详细统计数据按钮
    show_data_button = QPushButton("显示详细统计数据")
    show_data_button.setMinimumHeight(34)
    show_data_button.setStyleSheet("font-size: 11pt; padding: 6px 12px;")
    
    control_layout.addWidget(lottery_label)
    control_layout.addWidget(lottery_combo)
    control_layout.addWidget(suggestion_button)
    control_layout.addWidget(pro_button)
    control_layout.addWidget(show_data_button)
    control_layout.addStretch()
    
    advanced_layout.addLayout(control_layout)
    
    # 创建结果显示区域
    result_label = QLabel("点击'生成选号参考'获取选号建议，或点击'专业分析'查看详细统计图表")
    result_label.setAlignment(Qt.AlignCenter)
    result_label.setMinimumHeight(500)
    result_label.setStyleSheet("""
        QLabel {
            padding: 10px; 
            border: 1px solid #8B6F47; 
            border-radius: 3px; 
            font-size: 14pt; 
            font-weight: bold;
        }
    """)
    
    # 使用QScrollArea包裹结果显示区域，以支持滚动
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(result_label)
    
    advanced_layout.addWidget(scroll_area, 1)
    
    # 兼容性引用
    run_stats_button = action_advanced
    run_distribution_button = action_distribution
    
    return (lottery_combo, run_stats_button, run_distribution_button, show_data_button, result_label, suggestion_button)


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
    result_label.setStyleSheet("padding: 10px; border: 1px solid #8B6F47; border-radius: 3px; font-size: 14pt; font-weight: bold;")
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
    log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt;")
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
    result_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt;")
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


def create_ziwei_tab(ziwei_tab):
    """
    创建紫微斗数标签页
    
    输入：姓名/性别/出生年月日时/城市/目标期号
    输出：3组号码（正财/偏财/本命三视角）
    """
    from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
                                  QLabel, QComboBox, QSpinBox, QLineEdit, QGroupBox,
                                  QTextEdit, QScrollArea, QFrame)
    from PyQt5.QtGui import QFont
    from astrology.common.cities import get_city_names

    main_layout = QVBoxLayout(ziwei_tab)
    main_layout.setSpacing(8)
    main_layout.setContentsMargins(12, 12, 12, 12)

    label_style = "font-size: 11pt; font-weight: bold;"
    input_style = "font-size: 11pt; min-height: 30px;"

    # ==================== 输入区 ====================
    input_group = QGroupBox("排盘信息")
    form = QFormLayout()
    form.setSpacing(6)

    name_edit = QLineEdit()
    name_edit.setPlaceholderText("选填")
    name_edit.setStyleSheet(input_style)

    gender_combo = QComboBox()
    gender_combo.addItems(["男", "女"])
    gender_combo.setStyleSheet(input_style)

    year_spin = QSpinBox()
    year_spin.setRange(1900, 2100)
    year_spin.setValue(1990)
    year_spin.setStyleSheet(input_style)

    month_combo = QComboBox()
    month_combo.addItems([str(m) for m in range(1, 13)])
    month_combo.setStyleSheet(input_style)

    day_spin = QSpinBox()
    day_spin.setRange(1, 31)
    day_spin.setValue(1)
    day_spin.setStyleSheet(input_style)

    hour_combo = QComboBox()
    hour_combo.addItems([f"{h:02d}时" for h in range(24)])
    hour_combo.setStyleSheet(input_style)

    minute_combo = QComboBox()
    minute_combo.addItems([f"{m:02d}分" for m in range(0, 60, 5)])
    minute_combo.setStyleSheet(input_style)

    city_combo = QComboBox()
    city_combo.addItems(get_city_names())
    city_combo.setCurrentText("北京")
    city_combo.setStyleSheet(input_style)
    city_combo.setEditable(True)

    target_spin = QSpinBox()
    target_spin.setRange(0, 99999)
    target_spin.setValue(0)
    target_spin.setStyleSheet(input_style)

    form.addRow("姓名:", name_edit)
    form.addRow("性别:", gender_combo)
    form.addRow("出生年:", year_spin)
    form.addRow("出生月:", month_combo)
    form.addRow("出生日:", day_spin)
    form.addRow("出生时:", hour_combo)
    form.addRow("出生分:", minute_combo)
    form.addRow("出生城市:", city_combo)
    form.addRow("目标期号(0=自动):", target_spin)

    input_group.setLayout(form)
    main_layout.addWidget(input_group)

    # ==================== 操作按钮 ====================
    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(8)
    predict_btn = QPushButton("紫微排盘·取数")
    predict_btn.setMinimumHeight(38)
    predict_btn.setStyleSheet("font-size: 12pt; font-weight: bold;")
    reset_btn = QPushButton("重置")
    reset_btn.setMinimumHeight(38)
    reset_btn.setStyleSheet(label_style)
    btn_layout.addWidget(predict_btn)
    btn_layout.addWidget(reset_btn)
    btn_layout.addStretch()
    main_layout.addLayout(btn_layout)

    # ==================== 结果展示区 ====================
    result_group = QGroupBox("紫微三组号码")
    result_layout = QVBoxLayout()

    # 3 张组卡片
    cards = []
    variation_names = ["第一组·正财视角", "第二组·偏财视角", "第三组·本命视角"]
    for i, vname in enumerate(variation_names):
        card_frame = QFrame()
        card_frame.setFrameStyle(QFrame.Box)
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(10, 6, 10, 6)

        title_label = QLabel(vname)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        card_layout.addWidget(title_label)

        red_label = QLabel("红球: 待排盘")
        red_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        card_layout.addWidget(red_label)

        blue_label = QLabel("蓝球: 待排盘")
        blue_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        card_layout.addWidget(blue_label)

        detail_label = QLabel("推演: 待排盘")
        detail_label.setStyleSheet("font-size: 10pt;")
        detail_label.setWordWrap(True)
        card_layout.addWidget(detail_label)

        cards.append((red_label, blue_label, detail_label))
        result_layout.addWidget(card_frame)

    result_group.setLayout(result_layout)
    main_layout.addWidget(result_group, 1)

    return (predict_btn, reset_btn,
            name_edit, gender_combo, year_spin, month_combo,
            day_spin, hour_combo, minute_combo, city_combo, target_spin,
            cards)


def create_meihua_tab(meihua_tab):
    """
    创建梅花易数标签页
    
    输入：起卦方式(时间/数字) + 对应参数
    输出：3组号码（本卦主象/互卦过程/外应灵数）
    """
    from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
                                  QLabel, QComboBox, QSpinBox, QLineEdit, QGroupBox,
                                  QFrame)
    from PyQt5.QtGui import QFont

    main_layout = QVBoxLayout(meihua_tab)
    main_layout.setSpacing(8)
    main_layout.setContentsMargins(12, 12, 12, 12)

    label_style = "font-size: 11pt; font-weight: bold;"
    input_style = "font-size: 11pt; min-height: 30px;"

    # ==================== 输入区 ====================
    input_group = QGroupBox("起卦信息")
    form = QFormLayout()
    form.setSpacing(6)

    mode_combo = QComboBox()
    mode_combo.addItems(["时间起卦", "数字起卦"])
    mode_combo.setStyleSheet(input_style)

    year_spin = QSpinBox()
    year_spin.setRange(1900, 2100)
    year_spin.setValue(2026)
    year_spin.setStyleSheet(input_style)

    month_combo = QComboBox()
    month_combo.addItems([str(m) for m in range(1, 13)])
    month_combo.setStyleSheet(input_style)

    day_spin = QSpinBox()
    day_spin.setRange(1, 31)
    day_spin.setValue(1)
    day_spin.setStyleSheet(input_style)

    hour_combo = QComboBox()
    hour_combo.addItems([f"{h:02d}时" for h in range(24)])
    hour_combo.setStyleSheet(input_style)

    minute_combo = QComboBox()
    minute_combo.addItems([f"{m:02d}分" for m in range(0, 60, 5)])
    minute_combo.setStyleSheet(input_style)

    num1_spin = QSpinBox()
    num1_spin.setRange(1, 999)
    num1_spin.setValue(3)
    num1_spin.setStyleSheet(input_style)

    num2_spin = QSpinBox()
    num2_spin.setRange(1, 999)
    num2_spin.setValue(8)
    num2_spin.setStyleSheet(input_style)

    num3_spin = QSpinBox()
    num3_spin.setRange(1, 999)
    num3_spin.setValue(15)
    num3_spin.setStyleSheet(input_style)

    target_spin = QSpinBox()
    target_spin.setRange(0, 99999)
    target_spin.setValue(0)
    target_spin.setStyleSheet(input_style)

    form.addRow("起卦方式:", mode_combo)
    form.addRow("年:", year_spin)
    form.addRow("月:", month_combo)
    form.addRow("日:", day_spin)
    form.addRow("时:", hour_combo)
    form.addRow("分:", minute_combo)
    form.addRow("数字1(数字起卦用):", num1_spin)
    form.addRow("数字2(数字起卦用):", num2_spin)
    form.addRow("数字3(数字起卦用):", num3_spin)
    form.addRow("目标期号(0=自动):", target_spin)

    input_group.setLayout(form)
    main_layout.addWidget(input_group)

    # ==================== 操作按钮 ====================
    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(8)
    predict_btn = QPushButton("梅花起卦·取数")
    predict_btn.setMinimumHeight(38)
    predict_btn.setStyleSheet("font-size: 12pt; font-weight: bold;")
    reset_btn = QPushButton("重置")
    reset_btn.setMinimumHeight(38)
    reset_btn.setStyleSheet(label_style)
    btn_layout.addWidget(predict_btn)
    btn_layout.addWidget(reset_btn)
    btn_layout.addStretch()
    main_layout.addLayout(btn_layout)

    # ==================== 结果展示区 ====================
    result_group = QGroupBox("梅花三组号码")
    result_layout = QVBoxLayout()

    cards = []
    variation_names = ["第一组·本卦主象", "第二组·互卦过程", "第三组·外应灵数"]
    for i, vname in enumerate(variation_names):
        card_frame = QFrame()
        card_frame.setFrameStyle(QFrame.Box)
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(10, 6, 10, 6)

        title_label = QLabel(vname)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        card_layout.addWidget(title_label)

        red_label = QLabel("红球: 待起卦")
        red_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        card_layout.addWidget(red_label)

        blue_label = QLabel("蓝球: 待起卦")
        blue_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        card_layout.addWidget(blue_label)

        detail_label = QLabel("卦象: 待起卦")
        detail_label.setStyleSheet("font-size: 10pt;")
        detail_label.setWordWrap(True)
        card_layout.addWidget(detail_label)

        cards.append((red_label, blue_label, detail_label))
        result_layout.addWidget(card_frame)

    result_group.setLayout(result_layout)
    main_layout.addWidget(result_group, 1)

    return (predict_btn, reset_btn,
            mode_combo, year_spin, month_combo, day_spin,
            hour_combo, minute_combo,
            num1_spin, num2_spin, num3_spin, target_spin,
            cards)