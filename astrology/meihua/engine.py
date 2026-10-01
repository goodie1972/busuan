# -*- coding: utf-8 -*-
"""
梅花易数排盘引擎
基于邵雍《梅花易数》三种起卦法 + 三层取象

三组取数路径：
  variation=0  本卦主象   → 时间起卦，取本卦上下卦数+动爻
  variation=1  互卦过程   → 时间起卦，取互卦上下卦+体用五行数
  variation=2  外应灵数   → 数字起卦，取用户输入数+卦爻变化数
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import hashlib
import random

from astrology.common.ganzhi import TIANGAN, DIZHI, TIANGAN_WUXING, DIZHI_WUXING

# ==================== 常量 ====================

# 先天八卦序数：乾1 兑2 离3 震4 巽5 坎6 艮7 坤8
BAGUA_NAMES = ["坤", "乾", "兑", "离", "震", "巽", "坎", "艮"]  # index 0-7 对应 8/1/2/3/4/5/6/7
BAGUA_NUMBERS = {"乾": 1, "兑": 2, "离": 3, "震": 4, "巽": 5, "坎": 6, "艮": 7, "坤": 8}

# 八卦五行
BAGUA_WUXING = {
    "乾": "金", "兑": "金", "离": "火", "震": "木",
    "巽": "木", "坎": "水", "艮": "土", "坤": "土",
}

# 八卦方位数（后天八卦方位）
BAGUA_FANGWEI = {
    "乾": 6, "坎": 1, "艮": 8, "震": 3, "巽": 4, "离": 9, "坤": 2, "兑": 7,
}

# 先天八卦数（用于取数）
XIANTIAN_NUM = {"乾": 1, "兑": 2, "离": 3, "震": 4, "巽": 5, "坎": 6, "艮": 7, "坤": 8}

# 五行数字（河图洛书）
WUXING_NUM = {"水": 1, "火": 2, "木": 3, "金": 4, "土": 5}

# 卦名（64卦）
GUA64 = [
    "坤为地", "山地剥", "水地比", "风地观", "雷地豫", "火地晋", "泽地萃", "天地否",
    "天风姤", "泽风大过", "火风鼎", "雷风恒", "风天小畜", "水风井", "山风蛊", "地风升",
    "水雷屯", "泽雷随", "火雷噬嗑", "震为雷", "风雷益", "水雷屯", "山雷颐", "地雷复",
    "山泽损", "水山蹇", "风山渐", "雷山小过", "火山旅", "泽山咸", "艮为山", "地山谦",
    "地火明夷", "山火贲", "水火既济", "风火家人", "雷火丰", "离为火", "泽火革", "天火同人",
    "天泽履", "兑为泽", "火泽睽", "雷泽归妹", "风泽中孚", "水泽节", "山泽损", "地泽临",
    "地水师", "山水蒙", "水水坎", "风水涣", "雷水解", "火水未济", "泽水困", "天水讼",
    "天地否", "山天大畜", "水天需", "风天小畜", "雷天大壮", "火天大有", "泽天夬", "乾为天",
]


@dataclass
class GuaXiang:
    """卦象"""
    upper: str    # 上卦名（乾兑离震巽坎艮坤）
    lower: str    # 下卦名
    moving_line: int  # 动爻（1-6）
    
    @property
    def gua_name(self) -> str:
        """获取64卦名"""
        up_idx = list(BAGUA_WUXING.keys()).index(self.upper)
        low_idx = list(BAGUA_WUXING.keys()).index(self.lower)
        # 64卦排列：上卦为外层，下卦为内层
        idx = up_idx * 8 + low_idx
        return GUA64[idx] if 0 <= idx < 64 else "未知"
    
    @property
    def ben_gua(self) -> str:
        """本卦名"""
        return self.gua_name
    
    @property
    def upper_num(self) -> int:
        return XIANTIAN_NUM.get(self.upper, 1)
    
    @property
    def lower_num(self) -> int:
        return XIANTIAN_NUM.get(self.lower, 1)
    
    @property
    def upper_wuxing(self) -> str:
        return BAGUA_WUXING.get(self.upper, "土")
    
    @property
    def lower_wuxing(self) -> str:
        return BAGUA_WUXING.get(self.lower, "土")
    
    @property
    def ti_gua(self) -> str:
        """体卦（不动的那部分）"""
        # 动爻在上卦(4-6)→ 下卦为体；动爻在下卦(1-3)→ 上卦为体
        if self.moving_line <= 3:
            return self.upper  # 上卦不动→体
        return self.lower
    
    @property
    def yong_gua(self) -> str:
        """用卦（动的那部分）"""
        if self.moving_line <= 3:
            return self.lower
        return self.upper
    
    def bian_gua_upper(self) -> str:
        """变卦上卦"""
        if self.moving_line > 3:
            # 上卦动，取反（阴阳变）
            idx = list(BAGUA_WUXING.keys()).index(self.upper)
            return list(BAGUA_WUXING.keys())[7 - idx]  # 简化：取对称卦
        return self.upper
    
    def bian_gua_lower(self) -> str:
        """变卦下卦"""
        if self.moving_line <= 3:
            idx = list(BAGUA_WUXING.keys()).index(self.lower)
            return list(BAGUA_WUXING.keys())[7 - idx]
        return self.lower

    def hu_gua_upper(self) -> str:
        """互卦上卦（取本卦3-5爻）"""
        # 简化：互卦上 = 本卦上卦与下卦的组合
        idx = (list(BAGUA_WUXING.keys()).index(self.upper) + 
               list(BAGUA_WUXING.keys()).index(self.lower)) % 8
        return list(BAGUA_WUXING.keys())[idx]
    
    def hu_gua_lower(self) -> str:
        """互卦下卦（取本卦2-4爻）"""
        idx = (list(BAGUA_WUXING.keys()).index(self.lower) + 2) % 8
        return list(BAGUA_WUXING.keys())[idx]


def _seed_hash(*args) -> int:
    s = "|".join(str(a) for a in args)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:10], 16)


def time_qigua(year: int, month: int, day: int, hour: int, 
               minute: int = 0, variation: int = 0) -> GuaXiang:
    """
    时间起卦法
    
    上卦 = (年数 + 月数 + 日数) % 8
    下卦 = (年数 + 月数 + 日数 + 时数) % 8
    动爻 = (年数 + 月数 + 日数 + 时数 + 分数) % 6 + 1
    
    variation 微调：不同组取不同的"年数"取法
    """
    # 年数取法：variation=0 用年份后两位，variation=1 用农历年序，variation=2 用天干序
    year_num = year % 100
    if variation == 1:
        year_num = (year - 1900) % 60  # 六十甲子序
    elif variation == 2:
        year_num = (year % 10) + (year % 12)  # 天干+地支序和

    month_num = month
    day_num = day
    hour_num = hour
    minute_num = minute

    upper_idx = (year_num + month_num + day_num) % 8
    lower_idx = (year_num + month_num + day_num + hour_num) % 8
    moving = (year_num + month_num + day_num + hour_num + minute_num + variation) % 6 + 1

    bagua_list = ["坤", "乾", "兑", "离", "震", "巽", "坎", "艮"]
    upper = bagua_list[upper_idx]
    lower = bagua_list[lower_idx]

    return GuaXiang(upper=upper, lower=lower, moving_line=moving)


def number_qigua(num1: int, num2: int, num3: int, variation: int = 0) -> GuaXiang:
    """
    数字起卦法
    
    上卦 = num1 % 8
    下卦 = num2 % 8
    动爻 = (num1 + num2 + num3 + variation) % 6 + 1
    """
    upper_idx = num1 % 8
    lower_idx = num2 % 8
    moving = (num1 + num2 + num3 + variation) % 6 + 1

    bagua_list = ["坤", "乾", "兑", "离", "震", "巽", "坎", "艮"]
    return GuaXiang(
        upper=bagua_list[upper_idx],
        lower=bagua_list[lower_idx],
        moving_line=moving,
    )


def predict(
    lottery_type: str,
    red_range: int, blue_range: int,
    red_count: int, blue_count: int,
    input_info: dict,
    target_period: int = 0,
    variation: int = 0,
) -> Tuple[List[int], List[int], str]:
    """
    梅花易数预测

    Args:
        lottery_type: 'ssq' / 'dlt'
        red_range, blue_range: 号码范围
        red_count, blue_count: 号码个数
        input_info: dict with keys
            - mode: 'time' / 'number'
            - year/month/day/hour/minute (time mode)
            - num1/num2/num3 (number mode)
        target_period: 目标期号
        variation: 0=本卦主象, 1=互卦过程, 2=外应灵数

    Returns:
        (red_numbers, blue_numbers, interpretation)
    """
    mode = input_info.get('mode', 'time')

    if mode == 'number' and variation == 2:
        # 外应灵数：用用户输入的数字起卦
        gua = number_qigua(
            input_info.get('num1', 3),
            input_info.get('num2', 8),
            input_info.get('num3', 15),
            variation=variation,
        )
    else:
        # 时间起卦
        gua = time_qigua(
            year=input_info.get('year', 2026),
            month=input_info.get('month', 1),
            day=input_info.get('day', 1),
            hour=input_info.get('hour', 12),
            minute=input_info.get('minute', 0),
            variation=variation,
        )

    seed = _seed_hash(gua.upper, gua.lower, gua.moving_line, target_period, variation)

    rng = random.Random(seed)
    red_numbers = set()
    blue_numbers = set()
    interpretation_parts = []

    # 取数路径根据 variation 不同
    if variation == 0:
        # 本卦主象：本卦上下卦数 + 动爻数
        core_nums = [
            gua.upper_num,      # 上卦先天数 1-8
            gua.lower_num,      # 下卦先天数 1-8
            gua.moving_line,    # 动爻 1-6
            BAGUA_FANGWEI.get(gua.upper, 1),  # 上卦方位数
            BAGUA_FANGWEI.get(gua.lower, 1),  # 下卦方位数
        ]
        interpretation_parts.append(f"本卦: 上{gua.upper}({gua.upper_num}) 下{gua.lower}({gua.lower_num})")
        interpretation_parts.append(f"动爻: 第{gua.moving_line}爻")
        interpretation_parts.append(f"卦名: {gua.gua_name}")

    elif variation == 1:
        # 互卦过程：互卦上下卦 + 体用五行
        hu_upper = gua.hu_gua_upper()
        hu_lower = gua.hu_gua_lower()
        ti = gua.ti_gua
        yong = gua.yong_gua
        ti_wx = BAGUA_WUXING.get(ti, "土")
        yong_wx = BAGUA_WUXING.get(yong, "土")

        core_nums = [
            XIANTIAN_NUM.get(hu_upper, 1),
            XIANTIAN_NUM.get(hu_lower, 1),
            WUXING_NUM.get(ti_wx, 5),
            WUXING_NUM.get(yong_wx, 5),
            (XIANTIAN_NUM.get(hu_upper, 1) + XIANTIAN_NUM.get(hu_lower, 1)) % 8 + 1,
        ]
        interpretation_parts.append(f"互卦: 上{hu_upper} 下{hu_lower}")
        interpretation_parts.append(f"体卦: {ti}({ti_wx}) 用卦: {yong}({yong_wx})")
        # 体用生克
        sheng = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
        if sheng.get(ti_wx) == yong_wx:
            interpretation_parts.append("体生用 → 泄气，谨慎行事")
        elif sheng.get(yong_wx) == ti_wx:
            interpretation_parts.append("用生体 → 得气，有贵人助")
        elif ti_wx == yong_wx:
            interpretation_parts.append("体用比和 → 平稳")
        else:
            interpretation_parts.append("体用相克 → 变动大")

    else:
        # 外应灵数：数字起卦 + 方位数 + 变卦数
        bian_upper = gua.bian_gua_upper()
        bian_lower = gua.bian_gua_lower()
        core_nums = [
            input_info.get('num1', 3),
            input_info.get('num2', 8),
            input_info.get('num3', 15),
            XIANTIAN_NUM.get(bian_upper, 1),
            XIANTIAN_NUM.get(bian_lower, 1),
            (gua.upper_num + gua.lower_num) % 8 + 1,
        ]
        interpretation_parts.append(f"外应数: {input_info.get('num1', 3)}, {input_info.get('num2', 8)}, {input_info.get('num3', 15)}")
        interpretation_parts.append(f"变卦: 上{bian_upper} 下{bian_lower}")
        interpretation_parts.append(f"卦名: {gua.gua_name}")

    # 映射核心数字到号码范围
    for num in core_nums:
        if len(red_numbers) >= red_count:
            break
        # 扩展映射：乘以固定系数 + 偏移
        mapped = (num * 4 + variation * 3) % red_range + 1
        if 1 <= mapped <= red_range:
            red_numbers.add(mapped)

    # 期数参与
    period_mod = target_period % red_range + 1
    red_numbers.add(period_mod if period_mod <= red_range else period_mod % red_range + 1)

    # 补齐红球
    while len(red_numbers) < red_count:
        n = rng.randint(1, red_range)
        red_numbers.add(n)

    # 蓝球：用坎卦(水)和艮卦(山)数
    ti = gua.ti_gua
    yong = gua.yong_gua
    ti_wx = BAGUA_WUXING.get(ti, "土")
    for i in range(blue_count):
        b = (WUXING_NUM.get(ti_wx, 5) + i * 4 + gua.moving_line + variation * 3) % blue_range + 1
        blue_numbers.add(b)

    # 补齐蓝球
    while len(blue_numbers) < blue_count:
        n = rng.randint(1, blue_range)
        blue_numbers.add(n)

    red_result = sorted(list(red_numbers))[:red_count]
    blue_result = sorted(list(blue_numbers))[:blue_count]

    variation_names = ["本卦主象", "互卦过程", "外应灵数"]
    interpretation = (
        f"【{variation_names[variation] if variation < 3 else '本卦主象'}】\n"
        + "\n".join(interpretation_parts)
    )

    return red_result, blue_result, interpretation


if __name__ == '__main__':
    info = {'mode': 'time', 'year': 2026, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30}
    for v in range(3):
        r, b, interp = predict('ssq', 33, 16, 6, 1, info, target_period=26110, variation=v)
        print(f"\n=== 第{v+1}组 ===")
        print(f"红球: {r}")
        print(f"蓝球: {b}")
        print(interp)