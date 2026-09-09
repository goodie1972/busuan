# -*- coding:utf-8 -*-
"""
主题管理器模块 - 用于处理应用的颜色主题
Author: Zhao Yang
"""
import json
import logging
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QPushButton, QLabel, QGroupBox, QComboBox,
    QColorDialog
)

class ThemeManager:
    """主题管理器类，用于管理和应用自定义主题"""
    
    def __init__(self):
        self.settings = QSettings("LottoProphet", "LotteryPredictor")
        
        # 预定义主题
        self.themes = {
            "浅色": {
                "background": "#F5F5F5",
                "primary": "#2B579A",
                "primary_hover": "#3A67AE",
                "primary_pressed": "#1D3C6E",
                "text": "#333333",
                "border": "#CCCCCC",
                "panel": "white",
                "disabled_bg": "#CCCCCC",
                "disabled_text": "#888888",
                "tab_bg": "#F0F0F0",
                "accent": "#007ACC",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336"
            },
            "深色": {
                "background": "#121212",
                "primary": "#1976D2",
                "primary_hover": "#1E88E5",
                "primary_pressed": "#0D47A1",
                "text": "#E0E0E0",
                "border": "#404040",
                "panel": "#1E1E1E",
                "disabled_bg": "#3A3A3A",
                "disabled_text": "#666666",
                "tab_bg": "#252525",
                "accent": "#00B0FF",
                "success": "#66BB6A",
                "warning": "#FFA726",
                "error": "#EF5350"
            },
            "蓝色": {
                "background": "#E3F2FD",
                "primary": "#1565C0",
                "primary_hover": "#1976D2",
                "primary_pressed": "#0D47A1",
                "text": "#01579B",
                "border": "#81D4FA",
                "panel": "#FFFFFF",
                "disabled_bg": "#B3E5FC",
                "disabled_text": "#0288D1",
                "tab_bg": "#BBDEFB",
                "accent": "#00B0FF",
                "success": "#00C853",
                "warning": "#FF9100",
                "error": "#FF1744"
            },
            "绿色": {
                "background": "#E8F5E9",
                "primary": "#2E7D32",
                "primary_hover": "#388E3C",
                "primary_pressed": "#1B5E20",
                "text": "#1B5E20",
                "border": "#A5D6A7",
                "panel": "#FFFFFF",
                "disabled_bg": "#C8E6C9",
                "disabled_text": "#388E3C",
                "tab_bg": "#C8E6C9",
                "accent": "#00C853",
                "success": "#00E676",
                "warning": "#FFAB00",
                "error": "#FF3D00"
            },
            "紫色": {
                "background": "#F3E5F5",
                "primary": "#7B1FA2",
                "primary_hover": "#8E24AA",
                "primary_pressed": "#6A1B9A",
                "text": "#4A148C",
                "border": "#CE93D8",
                "panel": "#FFFFFF",
                "disabled_bg": "#E1BEE7",
                "disabled_text": "#8E24AA",
                "tab_bg": "#D1C4E9",
                "accent": "#D500F9",
                "success": "#00E676",
                "warning": "#FFAB00",
                "error": "#FF3D00"
            },
            "自定义": {}  # 用户自定义主题
        }
        
        # 加载保存的主题配置
        self.load_custom_theme()
        self.current_theme = self.settings.value("current_theme", "浅色")
        
    def get_theme(self, theme_name=None):
        """获取主题配置"""
        if theme_name is None:
            theme_name = self.current_theme
        
        if theme_name in self.themes:
            return self.themes[theme_name]
        return self.themes["浅色"]  # 默认返回浅色主题
    
    def set_theme(self, theme_name):
        """设置当前主题"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            self.settings.setValue("current_theme", theme_name)
            return True
        return False
    
    def get_theme_names(self):
        """获取所有主题名称"""
        return list(self.themes.keys())
    
    def load_custom_theme(self):
        """从设置加载自定义主题"""
        custom_theme = self.settings.value("custom_theme", None)
        if custom_theme:
            try:
                if isinstance(custom_theme, str):
                    self.themes["自定义"] = json.loads(custom_theme)
                else:
                    self.themes["自定义"] = custom_theme
            except Exception as e:
                logging.error(f"加载自定义主题错误: {e}")
                # 出错时使用默认浅色主题作为自定义主题的基础
                self.themes["自定义"] = self.themes["浅色"].copy()
        else:
            # 如果没有自定义主题，使用浅色主题作为基础
            self.themes["自定义"] = self.themes["浅色"].copy()
    
    def save_custom_theme(self, theme_dict):
        """保存自定义主题"""
        self.themes["自定义"] = theme_dict
        self.settings.setValue("custom_theme", json.dumps(theme_dict))
    
    def generate_stylesheet(self, theme_name=None):
        """根据主题生成样式表"""
        theme = self.get_theme(theme_name)
        
        # 确保所有需要的颜色都有默认值
        default_theme = self.themes["浅色"]
        for key in ["background", "primary", "primary_hover", "primary_pressed", 
                   "text", "border", "panel", "disabled_bg", "disabled_text", "tab_bg"]:
            if key not in theme or not theme[key]:
                theme[key] = default_theme[key]
        
        return f"""
            QMainWindow {{
                background-color: {theme['background']};
            }}
            QPushButton {{
                background-color: {theme['primary']};
                color: white;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['primary_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {theme['disabled_bg']};
                color: {theme['disabled_text']};
            }}
            QLabel {{
                color: {theme['text']};
            }}
            QComboBox, QSpinBox, QLineEdit {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                padding: 3px;
                background-color: {theme['panel']};
                min-height: 22px;
                color: {theme['text']};
            }}
            QTextEdit {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background-color: {theme['panel']};
                font-family: "Segoe UI", sans-serif;
                font-size: 11pt;
                color: {theme['text']};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['border']};
                background-color: {theme['panel']};
                border-radius: 3px;
            }}
            QTabBar::tab {{
                background-color: {theme['tab_bg']};
                border: 1px solid {theme['border']};
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 6px 12px;
                margin-right: 2px;
                font-weight: bold;
                color: {theme['text']};
            }}
            QTabBar::tab:selected {{
                background-color: {theme['panel']};
                border-bottom: none;
            }}
            QGroupBox {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                margin-top: 1ex;
                font-weight: bold;
                color: {theme['text']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: {theme['text']};
            }}
            QCheckBox {{
                color: {theme['text']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: {theme['background']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme['primary']};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0px;
            }}
        """


class CustomThemeDialog(QDialog):
    """自定义主题对话框"""
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.custom_theme = theme_manager.get_theme("自定义").copy()
        
        # 确保自定义主题包含所有必要颜色
        required_colors = ["background", "primary", "primary_hover", "primary_pressed", 
                          "text", "border", "panel", "disabled_bg", "disabled_text", "tab_bg"]
        
        # 如果自定义主题缺少必要颜色，从当前主题中补充
        current_theme = theme_manager.get_theme()
        for color in required_colors:
            if color not in self.custom_theme or not self.custom_theme[color]:
                self.custom_theme[color] = current_theme[color]
        
        self.setWindowTitle("自定义主题")
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # 创建颜色选择按钮
        self.color_buttons = {}
        for color_key in ["background", "primary", "text", "border", "panel"]:
            color_name = {
                "background": "背景色",
                "primary": "主色调",
                "text": "文本颜色",
                "border": "边框颜色",
                "panel": "面板颜色"
            }.get(color_key, color_key)
            
            btn = QPushButton()
            btn.setFixedSize(80, 25)
            btn.setStyleSheet(f"background-color: {self.custom_theme.get(color_key, '#FFFFFF')};")
            btn.clicked.connect(lambda checked, k=color_key: self.choose_color(k))
            
            form_layout.addRow(f"{color_name}:", btn)
            self.color_buttons[color_key] = btn
        
        layout.addLayout(form_layout)
        
        # 预览区域
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("这是主题预览文本")
        self.preview_label.setAlignment(Qt.AlignCenter)
        
        preview_button = QPushButton("预览按钮")
        preview_button.setEnabled(True)
        
        disabled_button = QPushButton("禁用按钮")
        disabled_button.setEnabled(False)
        
        preview_combo = QComboBox()
        preview_combo.addItems(["预览下拉框", "选项1", "选项2"])
        
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(preview_button)
        preview_layout.addWidget(disabled_button)
        preview_layout.addWidget(preview_combo)
        
        layout.addWidget(preview_group)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save_theme)
        
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.update_preview()
    
    def choose_color(self, color_key):
        """打开颜色选择对话框"""
        current_color = QColor(self.custom_theme.get(color_key, "#FFFFFF"))
        color = QColorDialog.getColor(current_color, self)
        
        if color.isValid():
            color_hex = color.name()
            self.custom_theme[color_key] = color_hex
            self.color_buttons[color_key].setStyleSheet(f"background-color: {color_hex};")
            
            # 自动更新相关颜色
            if color_key == "primary":
                # 根据主色调自动计算hover和pressed状态颜色
                primary_color = QColor(color_hex)
                
                # 计算hover颜色 (稍微亮一些)
                hover_color = QColor(
                    min(primary_color.red() + 20, 255),
                    min(primary_color.green() + 20, 255),
                    min(primary_color.blue() + 20, 255)
                )
                self.custom_theme["primary_hover"] = hover_color.name()
                
                # 计算pressed颜色 (稍微暗一些)
                pressed_color = QColor(
                    max(primary_color.red() - 20, 0),
                    max(primary_color.green() - 20, 0),
                    max(primary_color.blue() - 20, 0)
                )
                self.custom_theme["primary_pressed"] = pressed_color.name()
            
            self.update_preview()
    
    def update_preview(self):
        """更新预览区域"""
        preview_style = f"""
            QGroupBox {{
                border: 1px solid {self.custom_theme.get('border', '#CCCCCC')};
                border-radius: 3px;
                margin-top: 1ex;
                font-weight: bold;
                color: {self.custom_theme.get('text', '#333333')};
                background-color: {self.custom_theme.get('background', '#F5F5F5')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: {self.custom_theme.get('text', '#333333')};
            }}
            QLabel {{
                color: {self.custom_theme.get('text', '#333333')};
            }}
            QPushButton {{
                background-color: {self.custom_theme.get('primary', '#2B579A')};
                color: white;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.custom_theme.get('primary_hover', '#3A67AE')};
            }}
            QPushButton:pressed {{
                background-color: {self.custom_theme.get('primary_pressed', '#1D3C6E')};
            }}
            QPushButton:disabled {{
                background-color: {self.custom_theme.get('disabled_bg', '#CCCCCC')};
                color: {self.custom_theme.get('disabled_text', '#888888')};
            }}
            QComboBox {{
                border: 1px solid {self.custom_theme.get('border', '#CCCCCC')};
                border-radius: 3px;
                padding: 3px;
                background-color: {self.custom_theme.get('panel', 'white')};
                min-height: 22px;
                color: {self.custom_theme.get('text', '#333333')};
            }}
        """
        
        # 为预览组件应用样式
        self.findChild(QGroupBox).setStyleSheet(preview_style)
    
    def save_theme(self):
        """保存自定义主题"""
        # 确保所有必要的颜色都已定义
        required_colors = ["background", "primary", "primary_hover", "primary_pressed", 
                          "text", "border", "panel", "disabled_bg", "disabled_text"]
        
        default_theme = self.theme_manager.get_theme("浅色")
        for color in required_colors:
            if color not in self.custom_theme:
                self.custom_theme[color] = default_theme[color]
        
        self.theme_manager.save_custom_theme(self.custom_theme)
        self.accept() 