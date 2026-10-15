"""
输入验证测试
验证边界条件、类型检查、规则约束
"""
from input_validator import (
    validate_lottery_type, validate_numbers, validate_period,
    validate_amount, validate_compound_params, validate_dantuo_params,
    sanitize_input
)


def test_lottery_type_validation():
    """测试彩票类型验证"""
    print("\n[TEST 1] 彩票类型验证")
    
    # 正常情况
    assert validate_lottery_type('ssq')[0] == True
    assert validate_lottery_type('SSQ')[0] == True
    assert validate_lottery_type('dlt')[0] == True
    assert validate_lottery_type(' DLT ')[0] == True
    
    # 异常情况
    assert validate_lottery_type('invalid')[0] == False
    assert validate_lottery_type(123)[0] == False
    assert validate_lottery_type('')[0] == False
    
    print("  [OK] 彩票类型验证正常")


def test_numbers_validation():
    """测试号码验证"""
    print("\n[TEST 2] 号码验证")
    
    # SSQ 正常
    assert validate_numbers('ssq', [1,2,3,4,5,6], [7])[0] == True
    
    # SSQ 异常：红球超范围
    assert validate_numbers('ssq', [1,2,3,4,5,34], [7])[0] == False
    
    # SSQ 异常：红球数量错误
    assert validate_numbers('ssq', [1,2,3,4,5], [7])[0] == False
    
    # SSQ 异常：红球重复
    assert validate_numbers('ssq', [1,1,2,3,4,5], [7])[0] == False
    
    # SSQ 异常：蓝球超范围
    assert validate_numbers('ssq', [1,2,3,4,5,6], [17])[0] == False
    
    # SSQ 异常：蓝球数量错误
    assert validate_numbers('ssq', [1,2,3,4,5,6], [7,8])[0] == False
    
    # DLT 正常
    assert validate_numbers('dlt', [1,2,3,4,5], [6,7])[0] == True
    
    # DLT 异常：红球超范围
    assert validate_numbers('dlt', [1,2,3,4,36], [6,7])[0] == False
    
    # DLT 异常：蓝球超范围
    assert validate_numbers('dlt', [1,2,3,4,5], [6,13])[0] == False
    
    # 类型错误
    assert validate_numbers('ssq', '123456', [7])[0] == False
    assert validate_numbers('ssq', [1,2,3,4,5,'6'], [7])[0] == False
    
    print("  [OK] 号码验证正常")


def test_period_validation():
    """测试期号验证"""
    print("\n[TEST 3] 期号验证")
    
    # 正常情况
    assert validate_period('2024001')[0] == True
    assert validate_period('2026153')[0] == True
    
    # 异常：长度错误
    assert validate_period('202401')[0] == False
    assert validate_period('20240001')[0] == False
    
    # 异常：含非数字
    assert validate_period('2024A01')[0] == False
    
    # 异常：年份不合理
    assert validate_period('1999001')[0] == False
    assert validate_period('2100001')[0] == False
    
    # 异常：期数不合理
    assert validate_period('2024000')[0] == False
    
    # 类型错误
    assert validate_period(2024001)[0] == False
    assert validate_period('')[0] == False
    
    print("  [OK] 期号验证正常")


def test_amount_validation():
    """测试金额验证"""
    print("\n[TEST 4] 金额验证")
    
    # 正常情况
    assert validate_amount(2.0)[0] == True
    assert validate_amount(100)[0] == True
    assert validate_amount(50000)[0] == True
    
    # 异常：低于最小值
    assert validate_amount(1.0)[0] == False
    
    # 异常：超过最大值
    assert validate_amount(200000)[0] == False
    
    # 异常：不是2的倍数
    assert validate_amount(3.0)[0] == False
    assert validate_amount(99)[0] == False
    
    # 类型错误
    assert validate_amount('100')[0] == False
    
    print("  [OK] 金额验证正常")


def test_compound_params():
    """测试复式参数验证"""
    print("\n[TEST 5] 复式参数验证")
    
    # SSQ 正常
    assert validate_compound_params('ssq', 7, 2)[0] == True
    assert validate_compound_params('ssq', 6, 1)[0] == True
    assert validate_compound_params('ssq', 33, 16)[0] == True
    
    # SSQ 异常：红球过少
    assert validate_compound_params('ssq', 5, 1)[0] == False
    
    # SSQ 异常：红球过多
    assert validate_compound_params('ssq', 34, 1)[0] == False
    
    # SSQ 异常：蓝球过少
    assert validate_compound_params('ssq', 6, 0)[0] == False
    
    # DLT 正常
    assert validate_compound_params('dlt', 6, 3)[0] == True
    
    # 类型错误
    assert validate_compound_params('ssq', '7', 2)[0] == False
    
    print("  [OK] 复式参数验证正常")


def test_dantuo_params():
    """测试胆拖参数验证"""
    print("\n[TEST 6] 胆拖参数验证")
    
    # SSQ 正常
    assert validate_dantuo_params('ssq', 2, 4, 0, 1)[0] == True
    assert validate_dantuo_params('ssq', 1, 5, 0, 2)[0] == True
    
    # SSQ 异常：胆+拖不足
    assert validate_dantuo_params('ssq', 2, 3, 0, 1)[0] == False
    
    # SSQ 异常：胆码过多
    assert validate_dantuo_params('ssq', 6, 1, 0, 1)[0] == False
    
    # SSQ 异常：拖码为0
    assert validate_dantuo_params('ssq', 5, 0, 0, 1)[0] == False
    
    # DLT 正常
    assert validate_dantuo_params('dlt', 2, 3, 1, 1)[0] == True
    
    # DLT 异常：蓝球胆码过多
    assert validate_dantuo_params('dlt', 2, 3, 2, 1)[0] == False
    
    print("  [OK] 胆拖参数验证正常")


def test_sanitize_input():
    """测试输入清洗"""
    print("\n[TEST 7] 输入清洗")
    
    # 正常情况
    assert sanitize_input('  hello  ') == 'hello'
    assert sanitize_input('abc123') == 'abc123'
    
    # 截断超长输入
    long_text = 'a' * 200
    assert len(sanitize_input(long_text, max_length=100)) == 100
    
    # 移除控制字符
    assert '\x00' not in sanitize_input('hello\x00world')
    
    # 类型转换
    assert sanitize_input(123) == '123'
    
    print("  [OK] 输入清洗正常")


if __name__ == '__main__':
    print("="*60)
    print(" 输入验证工具 - 测试")
    print("="*60)
    
    test_lottery_type_validation()
    test_numbers_validation()
    test_period_validation()
    test_amount_validation()
    test_compound_params()
    test_dantuo_params()
    test_sanitize_input()
    
    print("\n" + "="*60)
    print(" [OK] 全部7项测试通过")
    print("="*60)
