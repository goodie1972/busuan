def create_investment_plan_tab(investment_tab):
    """
    创建投注计划管理标签页
    
    功能：
    1. 记录每次实际投入的金额、注数、选号
    2. 自动关联对应期数的预测存档（如果有）
    3. 开奖后核验中奖情况并计算盈亏
    4. 统计总投入、总奖金、ROI、命中率等指标
    5. 支持导出/导入 CSV 记录
    """
    from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                                  QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                                  QPushButton, QTableWidget, QTableWidgetItem,
                                  QHeaderView, QAbstractItemView, QMessageBox,
                                  QFileDialog, QCheckBox, QLineEdit, QTextEdit)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QColor
    import json
    import os
    from datetime import datetime
    
    # 主布局
    main_layout = QVBoxLayout(investment_tab)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(12, 12, 12, 12)
    
    # 标题
    title_label = QLabel("投注计划管理")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
    main_layout.addWidget(title_label)
    
    # 投注记录表格
    table_group = QGroupBox("投注记录")
    table_layout = QVBoxLayout(table_group)
    
    # 表格工具栏
    toolbar_layout = QHBoxLayout()
    
    self.add_investment_btn = QPushButton("添加投注记录")
    self.add_investment_btn.setStyleSheet("font-size: 11pt; font-weight: bold; background-color: #4CAF50; color: white;")
    self.add_investment_btn.setMinimumHeight(32)
    
    self.verify_investment_btn = QPushButton("核验选中记录")
    self.verify_investment_btn.setStyleSheet("font-size: 11pt; font-weight: bold; background-color: #2196F3; color: white;")
    self.verify_investment_btn.setMinimumHeight(32)
    
    self.export_investment_btn = QPushButton("导出记录")
    self.export_investment_btn.setStyleSheet("font-size: 11pt; background-color: #FF9800; color: white;")
    self.export_investment_btn.setMinimumHeight(32)
    
    self.import_investment_btn = QPushButton("导入记录")
    self.import_investment_btn.setStyleSheet("font-size: 11pt; background-color: #9C27B0; color: white;")
    self.import_investment_btn.setMinimumHeight(32)
    
    self.clear_investment_btn = QPushButton("清空记录")
    self.clear_investment_btn.setStyleSheet("font-size: 11pt; background-color: #F44336; color: white;")
    self.clear_investment_btn.setMinimumHeight(32)
    
    toolbar_layout.addWidget(self.add_investment_btn)
    toolbar_layout.addWidget(self.verify_investment_btn)
    toolbar_layout.addWidget(self.export_investment_btn)
    toolbar_layout.addWidget(self.import_investment_btn)
    toolbar_layout.addWidget(self.clear_investment_btn)
    toolbar_layout.addStretch()
    
    table_layout.addLayout(toolbar_layout)
    
    # 投注记录表格
    self.investment_table = QTableWidget()
    self.investment_table.setColumnCount(10)
    self.investment_table.setHorizontalHeaderLabels([
        "日期", "彩票类型", "期号", "投入金额(元)", "投入注数", 
        "选号", "对应预测", "开奖号码", "中奖情况", "盈亏(元)"
    ])
    
    # 设置表格属性
    self.investment_table.setAlternatingRowColors(True)
    self.investment_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    self.investment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self.investment_table.horizontalHeader().setStretchLastSection(True)
    self.investment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    
    # 设置列宽
    header = self.investment_table.horizontalHeader()
    header.resizeSection(0, 120)   # 日期
    header.resizeSection(1, 80)    # 彩票类型
    header.resizeSection(2, 80)    # 期号
    header.resizeSection(3, 100)   # 投入金额
    header.resizeSection(4, 80)    # 投入注数
    header.resizeSection(5, 200)   # 选号
    header.resizeSection(6, 200)   # 对应预测
    header.resizeSection(7, 200)   # 开奖号码
    header.resizeSection(8, 120)   # 中奖情况
    header.resizeSection(9, 100)   # 盈亏
    
    table_layout.addWidget(self.investment_table)
    main_layout.addWidget(table_group)
    
    # 统计信息面板
    stats_group = QGroupBox("投注统计")
    stats_layout = QFormLayout(stats_group)
    
    self.total_invested_label = QLabel("0.00 元")
    self.total_invested_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2196F3;")
    
    self.total_won_label = QLabel("0.00 元")
    self.total_won_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #4CAF50;")
    
    self.net_profit_label = QLabel("0.00 元")
    self.net_profit_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
    
    self.roi_label = QLabel("0.00%")
    self.roi_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
    
    self.win_rate_label = QLabel("0.00%")
    self.win_rate_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
    
    self.total_records_label = QLabel("0 条")
    self.total_records_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
    
    stats_layout.addRow("总投入:", self.total_invested_label)
    stats_layout.addRow("总中奖:", self.total_won_label)
    stats_layout.addRow("净收益:", self.net_profit_label)
    stats_layout.addRow("投资回报率(ROI):", self.roi_label)
    stats_layout.addRow("中奖率:", self.win_rate_label)
    stats_layout.addRow("记录总数:", self.total_records_label)
    
    main_layout.addWidget(stats_group)
    
    # 投注输入面板
    input_group = QGroupBox("添加新投注记录")
    input_layout = QFormLayout(input_group)
    
    # 投注时间
    self.invest_time_edit = QDateTimeEdit()
    self.invest_time_edit.setDateTime(QDateTime.currentDateTime())
    self.invest_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
    self.invest_time_edit.setCalendarPopup(True)
    input_layout.addRow("投注时间:", self.invest_time_edit)
    
    # 彩票类型
    self.invest_lottery_combo = QComboBox()
    for key in name_path.keys():
        self.invest_lottery_combo.addItem(name_path[key]['name'], key)
    input_layout.addRow("彩票类型:", self.invest_lottery_como)
    
    # 期号
    self.invest_period_spin = QSpinBox()
    self.invest_period_spin.setMinimum(1)
    self.invest_period_spin.setMaximum(999999)
    self.invest_period_spin.setValue(1)
    input_layout.addRow("期号:", self.invest_period_spin)
    
    # 投入金额
    self.invest_amount_spin = QDoubleSpinBox()
    self.invest_amount_spin.setMinimum(0.0)
    self.invest_amount_spin.setMaximum(1000000.0)
    self.invest_amount_spin.setSingleStep(2.0)
    self.invest_amount_spin.setValue(2.0)
    self.invest_amount_spin.setSuffix(" 元")
    input_layout.addRow("投入金额:", self.invest_amount_spin)
    
    # 投入注数
    self.invest_count_spin = QSpinBox()
    self.invest_count_spin.setMinimum(1)
    self.invest_count_spin.setMaximum(10000)
    self.invest_count_spin.setValue(1)
    input_layout.addRow("投入注数:", self.invest_count_spin)
    
    # 是否追加
    self.invest_extra_check = QCheckBox("追加投注")
    input_layout.addRow("", self.invest_extra_check)
    
    # 选号输入
    self.invest_numbers_edit = QLineEdit()
    self.invest_numbers_edit.setPlaceholderText("格式示例: 双色球 01 02 03 04 05 06 + 07；大乐透 01 02 03 04 05 + 06 07")
    self.invest_numbers_edit.setStyleSheet("font-size: 11pt; padding: 5px;")
    input_layout.addRow("选号:", self.invest_numbers_edit)
    
    # 关联预测按钮
    self.link_prediction_btn = QPushButton("查找对应预测")
    self.link_prediction_btn.setStyleSheet("font-size: 10pt;")
    self.link_prediction_btn.setMaximumWidth(120)
    input_layout.addRow("", self.link_prediction_btn)
    
    # 对应预测显示
    self.prediction_info_label = QLabel("暂无关联预测")
    self.prediction_info_label.setStyleSheet("font-size: 10pt; color: #666; font-style: italic;")
    input_layout.addRow("关联预测:", self.prediction_info_label)
    
    # 投注说明
    invest_note_label = QLabel(
        "<p style='color:#888;font-size:9pt;'>"
        "💡 使用提示："
        "<br>• 投入金额建议使用 Kelly 公式计算结果（如可用）"
        "<br>• 选号格式请严格按照示例输入，红蓝球之间用空格+加号+空格分隔"
        "<br>• 系统会自动尝试关联最近的预测存档进行对比分析"
        "<br>• 核验时会根据开奖数据自动计算中奖情况和盈亏"
        "</p>"
    )
    invest_note_label.setTextFormat(Qt.RichText)
    invest_note_label.setWordWrap(True)
    input_layout.addRow("", invest_note_label)
    
    main_layout.addWidget(input_group)
    
    # 底部操作按钮
    bottom_layout = QHBoxLayout()
    
    self.save_investment_btn = QPushButton("保存投注记录")
    self.save_investment_btn.setStyleSheet("font-size: 11pt; font-weight: bold; background-color: #FF9800; color: white;")
    self.save_investment_btn.setMinimumHeight(35)
    
    self.reset_investment_form_btn = QPushButton("重置表单")
    self.reset_investment_form_btn.setStyleSheet("font-size: 11pt; background-color: #607D8B; color: white;")
    self.reset_investment_form_btn.setMinimumHeight(35)
    
    bottom_layout.addWidget(self.save_investment_btn)
    bottom_layout.addWidget(self.reset_investment_form_btn)
    bottom_layout.addStretch()
    
    main_layout.addLayout(bottom_layout)
    
    # 初始化数据
    self.investment_records = []  # 内存中的投注记录
    self.investment_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       'investment_history.json')
    
    # 返回关键控件的引用
    return (
        self.add_investment_btn, self.verify_investment_btn, self.export_investment_btn,
        self.import_investment_btn, self.clear_investment_btn,
        self.investment_table,
        self.total_invested_label, self.total_won_label, self.net_profit_label,
        self.roi_label, self.win_rate_label, self.total_records_label,
        self.invest_time_edit, self.invest_lottery_combo, self.invest_period_spin,
        self.invest_amount_spin, self.invest_count_spin, self.invest_extra_check,
        self.invest_numbers_edit, self.link_prediction_btn, self.prediction_info_label,
        self.save_investment_btn, self.reset_investment_form_btn
    )