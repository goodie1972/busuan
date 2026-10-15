"""
输入验证工具模块
提供彩票号码、期号、金额等用户输入的边界检查与类型验证
"""
from typing import List, Optional, Tuple


# 彩票规则常量
LOTTERY_RULES = {
    'ssq': {
        'red_range': (1, 33),
        'red_count': 6,
        'blue_range': (1, 16),
        'blue_count': 1,
        'name': '双色球'
    },
    'dlt': {
        'red_range': (1, 35),
        'red_count': 5,
        'blue_range': (1, 12),
        'blue_count': 2,
        'name': '大乐透'
    }
}


def validate_lottery_type(lottery_type: str) -> Tuple[bool, str]:
    """
    验证彩票类型
    
    Args:
        lottery_type: 彩票类型字符串
        
    Returns:
        (是否有效, 错误信息)
    """
    if not isinstance(lottery_type, str):
        return False, f"彩票类型必须是字符串，实际类型: {type(lottery_type).__name__}"
    
    lottery_type = lottery_type.lower().strip()
    if lottery_type not in LOTTERY_RULES:
        return False, f"不支持的彩票类型: {lottery_type}，支持: {list(LOTTERY_RULES.keys())}"
    
    return True, ""


def validate_numbers(lottery_type: str, red_numbers: List[int], 
                    blue_numbers: List[int]) -> Tuple[bool, str]:
    """
    验证号码是否符合规则
    
    Args:
        lottery_type: 彩票类型 ('ssq' / 'dlt')
        red_numbers: 红球号码列表
        blue_numbers: 蓝球号码列表
        
    Returns:
        (是否有效, 错误信息)
    """
    # 1. 验证彩票类型
    valid, msg = validate_lottery_type(lottery_type)
    if not valid:
        return False, msg
    
    rules = LOTTERY_RULES[lottery_type.lower()]
    name = rules['name']
    
    # 2. 验证红球
    if not isinstance(red_numbers, (list, tuple)):
        return False, f"红球号码必须是列表，实际类型: {type(red_numbers).__name__}"
    
    if len(red_numbers) != rules['red_count']:
        return False, f"{name}红球必须{rules['red_count']}个，实际: {len(red_numbers)}个"
    
    for num in red_numbers:
        if not isinstance(num, int):
            return False, f"红球号码必须是整数，发现: {num} (类型: {type(num).__name__})"
        if not (rules['red_range'][0] <= num <= rules['red_range'][1]):
            return False, f"{name}红球范围{rules['red_range']}，发现: {num}"
    
    if len(set(red_numbers)) != len(red_numbers):
        return False, f"红球号码不能重复: {red_numbers}"
    
    # 3. 验证蓝球
    if not isinstance(blue_numbers, (list, tuple)):
        return False, f"蓝球号码必须是列表，实际类型: {type(blue_numbers).__name__}"
    
    if len(blue_numbers) != rules['blue_count']:
        return False, f"{name}蓝球必须{rules['blue_count']}个，实际: {len(blue_numbers)}个"
    
    for num in blue_numbers:
        if not isinstance(num, int):
            return False, f"蓝球号码必须是整数，发现: {num} (类型: {type(num).__name__})"
        if not (rules['blue_range'][0] <= num <= rules['blue_range'][1]):
            return False, f"{name}蓝球范围{rules['blue_range']}，发现: {num}"
    
    if len(set(blue_numbers)) != len(blue_numbers):
        return False, f"蓝球号码不能重复: {blue_numbers}"
    
    return True, ""


def validate_period(period: str) -> Tuple[bool, str]:
    """
    验证期号格式
    
    Args:
        period: 期号字符串（如 '2024001'）
        
    Returns:
        (是否有效, 错误信息)
    """
    if not isinstance(period, str):
        return False, f"期号必须是字符串，实际类型: {type(period).__name__}"
    
    period = period.strip()
    
    if not period:
        return False, "期号不能为空"
    
    if not period.isdigit():
        return False, f"期号必须全为数字，实际: {period}"
    
    if len(period) != 7:
        return False, f"期号长度应为7位（如2024001），实际: {len(period)}位"
    
    # 检查年份合理性（2000-2099）
    year = int(period[:4])
    if not (2000 <= year <= 2099):
        return False, f"期号年份不合理: {year}"
    
    # 检查期数合理性（001-999）
    issue = int(period[4:])
    if not (1 <= issue <= 999):
        return False, f"期号期数不合理: {issue}"
    
    return True, ""


def validate_amount(amount: float, min_val: float = 2.0, 
                   max_val: float = 100000.0) -> Tuple[bool, str]:
    """
    验证金额
    
    Args:
        amount: 金额
        min_val: 最小金额（默认2元，单式一注）
        max_val: 最大金额（默认10万元）
        
    Returns:
        (是否有效, 错误信息)
    """
    if not isinstance(amount, (int, float)):
        return False, f"金额必须是数字，实际类型: {type(amount).__name__}"
    
    if amount < min_val:
        return False, f"金额不能低于{min_val}元，实际: {amount}元"
    
    if amount > max_val:
        return False, f"金额不能超过{max_val}元，实际: {amount}元"
    
    # 检查是否是2的倍数（单注2元）
    if amount % 2 != 0:
        return False, f"金额必须是2的倍数，实际: {amount}元"
    
    return True, ""


def validate_compound_params(lottery_type: str, red_count: int, 
                            blue_count: int) -> Tuple[bool, str]:
    """
    验证复式投注参数
    
    Args:
        lottery_type: 彩票类型
        red_count: 红球个数
        blue_count: 蓝球个数
        
    Returns:
        (是否有效, 错误信息)
    """
    valid, msg = validate_lottery_type(lottery_type)
    if not valid:
        return False, msg
    
    rules = LOTTERY_RULES[lottery_type.lower()]
    name = rules['name']
    
    # 红球数量检查
    min_red = rules['red_count']
    max_red = rules['red_range'][1]
    
    if not isinstance(red_count, int):
        return False, f"红球个数必须是整数，实际类型: {type(red_count).__name__}"
    
    if not (min_red <= red_count <= max_red):
        return False, f"{name}复式红球个数范围{min_red}-{max_red}，实际: {red_count}"
    
    # 蓝球数量检查
    min_blue = rules['blue_count']
    max_blue = rules['blue_range'][1]
    
    if not isinstance(blue_count, int):
        return False, f"蓝球个数必须是整数，实际类型: {type(blue_count).__name__}"
    
    if not (min_blue <= blue_count <= max_blue):
        return False, f"{name}复式蓝球个数范围{min_blue}-{max_blue}，实际: {blue_count}"
    
    return True, ""


def validate_dantuo_params(lottery_type: str, dan_red: int, tuo_red: int,
                          dan_blue: int, tuo_blue: int) -> Tuple[bool, str]:
    """
    验证胆拖投注参数
    
    Args:
        lottery_type: 彩票类型
        dan_red: 红球胆码个数
        tuo_red: 红球拖码个数
        dan_blue: 蓝球胆码个数
        tuo_blue: 蓝球拖码个数
        
    Returns:
        (是否有效, 错误信息)
    """
    valid, msg = validate_lottery_type(lottery_type)
    if not valid:
        return False, msg
    
    rules = LOTTERY_RULES[lottery_type.lower()]
    name = rules['name']
    required_red = rules['red_count']
    required_blue = rules['blue_count']
    
    # 检查胆拖总和
    if dan_red + tuo_red < required_red:
        return False, f"{name}红球胆+拖必须≥{required_red}个，实际: {dan_red}+{tuo_red}={dan_red+tuo_red}"
    
    if dan_blue + tuo_blue < required_blue:
        return False, f"{name}蓝球胆+拖必须≥{required_blue}个，实际: {dan_blue}+{tuo_blue}={dan_blue+tuo_blue}"
    
    # 检查胆码不能超过要求数量
    if dan_red >= required_red:
        return False, f"{name}红球胆码必须<{required_red}个，实际: {dan_red}"
    
    if dan_blue >= required_blue:
        return False, f"{name}蓝球胆码必须<{required_blue}个，实际: {dan_blue}"
    
    # 检查拖码至少1个
    if tuo_red < 1:
        return False, f"{name}红球拖码至少1个，实际: {tuo_red}"
    
    if tuo_blue < 1:
        return False, f"{name}蓝球拖码至少1个，实际: {tuo_blue}"
    
    return True, ""


def sanitize_input(text: str, max_length: int = 100) -> str:
    """
    清洗用户输入文本（防止注入/溢出）
    
    Args:
        text: 输入文本
        max_length: 最大长度
        
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 去除首尾空白
    text = text.strip()
    
    # 截断到最大长度
    if len(text) > max_length:
        text = text[:max_length]
    
    # 移除控制字符（保留空格、换行、制表符）
    text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t ')
    
    return text
