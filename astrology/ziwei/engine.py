# -*- coding: utf-8 -*-
"""
紫微斗数排盘引擎
基于出生时间（或指定时间）排出紫微命盘，
从命盘星曜分布推导彩票号码。

三组取数视角：
  variation=0  正财视角   → 财帛宫主星 + 流年财帛宫
  variation=1  偏财视角   → 官禄宫主星 + 迁移宫 + 福德宫
  variation=2  本命视角   → 命宫主星 + 身宫 + 大限宫位
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import datetime
import hashlib

# 重用已建好的干支工具
from astrology.common.ganzhi import (
    TIANGAN, DIZHI, DIZHI_WUXING, TIANGAN_WUXING,
    DIZHI_CANGGAN, JIAZI, calc_sizhu, SiZhu, GanZhi,
    get_changsheng, DIZHI_CHONG, DIZHI_HE,
)
from astrology.common.cities import get_city_info, calc_true_solar_time_offset

# ==================== 常量 ====================

# 十四主星
FOURTEEN_STARS = [
    "紫微", "天机", "太阳", "武曲", "天同", "廉贞", "天府",
    "太阴", "贪狼", "巨门", "天相", "天梁", "七杀", "破军"
]

# 主星五行属性
STAR_WUXING = {
    "紫微": "土", "天机": "木", "太阳": "火", "武曲": "金",
    "天同": "水", "廉贞": "火", "天府": "土", "太阴": "水",
    "贪狼": "木", "巨门": "水", "天相": "水", "天梁": "土",
    "七杀": "金", "破军": "水",
}

# 主星对应的数字范围（取先天八卦数与五行数结合）
STAR_NUMBERS = {
    "紫微": [5, 14, 23, 32],   # 土·中宫
    "天机": [3, 12, 21, 30],   # 木·震
    "太阳": [9, 18, 27],       # 火·离
    "武曲": [4, 13, 22, 31],   # 金·乾
    "天同": [1, 10, 19, 28],   # 水·坎
    "廉贞": [7, 16, 25],       # 火·次离
    "天府": [8, 17, 26],       # 土·次中
    "太阴": [2, 11, 20, 29],   # 水·次坎
    "贪狼": [6, 15, 24, 33],   # 木·次震
    "巨门": [1, 10, 19, 28],   # 水·次坎
    "天相": [4, 13, 22, 31],   # 水·乾
    "天梁": [5, 14, 23, 32],   # 土·中
    "七杀": [7, 16, 25],       # 金·次乾
    "破军": [2, 11, 20, 29],   # 水·兑
}

# 十二宫
TWELVE_PALACES = [
    "命宫", "兄弟", "夫妻", "子女", "财帛", "疾厄",
    "迁移", "奴仆", "官禄", "田宅", "福德", "父母"
]

# 四化表（年干四化）—— 甲干：廉禄破武阳忌
SIHUA_TABLE = {
    '甲': {'禄': '廉贞', '权': '破军', '科': '武曲', '忌': '太阳'},
    '乙': {'禄': '天机', '权': '天梁', '科': '紫微', '忌': '太阴'},
    '丙': {'禄': '天同', '权': '天机', '科': '文昌', '忌': '廉贞'},
    '丁': {'禄': '太阴', '权': '天同', '科': '天机', '忌': '巨门'},
    '戊': {'禄': '贪狼', '权': '太阴', '科': '右弼', '忌': '天机'},
    '己': {'禄': '武曲', '权': '贪狼', '科': '天梁', '忌': '文曲'},
    '庚': {'禄': '太阳', '权': '武曲', '科': '太阴', '忌': '天同'},
    '辛': {'禄': '巨门', '权': '太阳', '科': '文曲', '忌': '文昌'},
    '壬': {'禄': '天梁', '权': '紫微', '科': '左辅', '忌': '武曲'},
    '癸': {'禄': '破军', '权': '巨门', '科': '太阴', '忌': '贪狼'},
}


@dataclass
class Palace:
    """一个宫位"""
    name: str
    dizhi: str
    stars: List[str] = field(default_factory=list)
    sihua: Dict[str, str] = field(default_factory=dict)  # {'禄': '某星', ...}

    def main_star(self) -> Optional[str]:
        """取主星（十四主星中的第一个）"""
        for s in self.stars:
            if s in FOURTEEN_STARS:
                return s
        return None


@dataclass
class ZiweiChart:
    """紫微命盘"""
    sizhu: SiZhu
    palaces: List[Palace]  # 12个宫位，从命宫开始
    sihua: Dict[str, str]  # 年干四化 {'禄': '廉贞', ...}
    palace_map: Dict[str, Palace] = field(default_factory=dict)

    def __post_init__(self):
        self.palace_map = {p.name: p for p in self.palaces}


def _seed_hash(*args) -> int:
    """用多个参数构造确定性哈希种子"""
    s = "|".join(str(a) for a in args)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:10], 16)


def build_chart(
    year: int, month: int, day: int, hour: int,
    minute: int = 0, city: str = "北京", gender: str = "男",
    target_period: int = 0
) -> ZiweiChart:
    """
    排紫微命盘

    Args:
        year/month/day/hour/minute: 出生（或指定）阳历时间
        city: 出生城市（用于真太阳时修正）
        gender: 性别（影响大限顺逆）
        target_period: 目标期号（参与种子）

    Returns:
        ZiweiChart
    """
    # 真太阳时修正
    city_info = get_city_info(city)
    if city_info:
        lon, lat, tz = city_info
        offset_min = calc_true_solar_time_offset(lon, tz)
    else:
        offset_min = 0.0

    # 修正后的时间
    true_dt = datetime.datetime(year, month, day, hour, minute) + datetime.timedelta(minutes=offset_min)

    # 计算四柱
    sizhu = calc_sizhu(true_dt.year, true_dt.month, true_dt.day, true_dt.hour, true_dt.minute)

    # 用四柱 + 期号构造排盘种子
    seed = _seed_hash(sizhu.year, sizhu.month, sizhu.day, sizhu.hour, target_period, gender)

    # 命宫地支：以月支为基准，逆数至生时
    # 寅月起正月，顺数月，逆数时 → 定命宫
    month_zhi_idx = DIZHI.index(sizhu.month.di_zhi)
    hour_zhi_idx = DIZHI.index(sizhu.hour.di_zhi)
    # 命宫 = (month_zhi_idx - (hour_zhi_idx - 2)) % 12，子时=2(寅起)
    ming_zhi_idx = (month_zhi_idx + 2 - hour_zhi_idx) % 12
    if ming_zhi_idx == 0:
        ming_zhi_idx = 0

    # 十二宫从命宫地支开始，逆时针排列
    palaces = []
    for i, pname in enumerate(TWELVE_PALACES):
        pz_idx = (ming_zhi_idx - i) % 12
        dz = DIZHI[pz_idx]
        palaces.append(Palace(name=pname, dizhi=dz))

    # 分配主星到各宫（基于种子确定性分配）
    import random
    rng = random.Random(seed)
    available_stars = list(FOURTEEN_STARS)
    rng.shuffle(available_stars)

    # 紫微系（紫微/天机/太阳/武曲/天同/廉贞）和天府系（天府/太阴/贪狼/巨门/天相/天梁/七杀/破军）
    # 分别按固定间隔落宫
    ziwei_series = ["紫微", "天机", "太阳", "武曲", "天同", "廉贞"]
    tianfu_series = ["天府", "太阴", "贪狼", "巨门", "天相", "天梁", "七杀", "破军"]

    # 紫微落宫位置（种子决定）
    zw_pos = rng.randint(0, 11)
    # 紫微系按固定间隔排列：紫微、天机(隔1)、太阳(隔2-逆)、武曲(隔3)、天同(隔4)、廉贞(隔9)
    zw_intervals = [0, -1, -2, -3, -4, -9]
    for star, interval in zip(ziwei_series, zw_intervals):
        idx = (zw_pos + interval) % 12
        palaces[idx].stars.append(star)

    # 天府落宫：与紫微对称
    tf_pos = (12 - zw_pos) % 12
    tf_intervals = [0, -1, -2, -3, -4, -5, -6, -9]
    for star, interval in zip(tianfu_series, tf_intervals):
        idx = (tf_pos + interval) % 12
        palaces[idx].stars.append(star)

    # 年干四化
    year_gan = sizhu.year.tian_gan
    sihua = SIHUA_TABLE.get(year_gan, {})

    # 将四化标注到对应宫位
    for hua_type, star_name in sihua.items():
        for p in palaces:
            if star_name in p.stars:
                p.sihua[hua_type] = star_name
                break

    return ZiweiChart(sizhu=sizhu, palaces=palaces, sihua=sihua)


def _pick_numbers_from_star(
    star: str, red_range: int, blue_range: int,
    red_count: int, blue_count: int,
    seed: int, existing: set
) -> Tuple[List[int], List[int]]:
    """从一颗星对应的数字池中选号"""
    pool = STAR_NUMBERS.get(star, [])
    # 过滤到有效范围
    pool = [n for n in pool if 1 <= n <= red_range]

    reds = []
    rng_seed = (seed + hash(star)) % (2**31)
    import random
    rng = random.Random(rng_seed)

    # 从星对应的数字池中取号
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    for n in shuffled:
        if len(reds) >= red_count:
            break
        if n not in existing and n not in reds:
            reds.append(n)

    # 蓝球：星五行对应的数字
    wx = STAR_WUXING.get(star, "土")
    # 五行数：水1 火2 木3 金4 土5
    wx_num = {"水": 1, "火": 2, "木": 3, "金": 4, "土": 5}.get(wx, 5)
    blues = []
    for i in range(blue_count):
        b = (wx_num + i * 3 + seed) % blue_range + 1
        if b not in blues:
            blues.append(b)

    return reds, blues


def predict(
    lottery_type: str,
    red_range: int, blue_range: int,
    red_count: int, blue_count: int,
    birth_info: dict,
    target_period: int = 0,
    variation: int = 0,
) -> Tuple[List[int], List[int], str]:
    """
    紫微斗数预测

    Args:
        lottery_type: 'ssq' / 'dlt'
        red_range, blue_range: 号码范围
        red_count, blue_count: 号码个数
        birth_info: dict with keys year/month/day/hour/minute/city/gender
        target_period: 目标期号
        variation: 0=正财(财帛宫), 1=偏财(官禄+迁移+福德), 2=本命(命宫+身宫+大限)

    Returns:
        (red_numbers, blue_numbers, interpretation)
    """
    chart = build_chart(
        year=birth_info.get('year', 1990),
        month=birth_info.get('month', 1),
        day=birth_info.get('day', 1),
        hour=birth_info.get('hour', 12),
        minute=birth_info.get('minute', 0),
        city=birth_info.get('city', '北京'),
        gender=birth_info.get('gender', '男'),
        target_period=target_period,
    )

    # 种子：命盘全局唯一种子
    seed = _seed_hash(chart.sizhu, target_period, variation)

    red_numbers = set()
    blue_numbers = set()
    interpretation_parts = []

    # 三种视角取数
    if variation == 0:
        # 正财视角：财帛宫主星 + 四化（禄/权/科）+ 流年财帛宫
        palace = chart.palace_map.get("财帛")
        star = palace.main_star() if palace else None
        if star:
            r, b = _pick_numbers_from_star(star, red_range, blue_range,
                                           red_count, blue_count, seed, red_numbers)
            red_numbers.update(r)
            blue_numbers.update(b)
            interpretation_parts.append(f"财帛宫主星「{star}」（{STAR_WUXING.get(star, '?')}）→ 取数 {r}")

        # 四化星补充
        for hua_type in ['禄', '权', '科']:
            star_name = chart.sihua.get(hua_type)
            if star_name and len(red_numbers) < red_count:
                r, b = _pick_numbers_from_star(star_name, red_range, blue_range,
                                               red_count, blue_count, seed + 7, red_numbers)
                red_numbers.update(r)
                if len(blue_numbers) < blue_count:
                    blue_numbers.update(b)
                interpretation_parts.append(f"年干化{hua_type}「{star_name}」→ 补数 {r}")

    elif variation == 1:
        # 偏财视角：官禄宫主星 + 迁移宫 + 福德宫
        for pname in ["官禄", "迁移", "福德"]:
            palace = chart.palace_map.get(pname)
            star = palace.main_star() if palace else None
            if star and len(red_numbers) < red_count:
                r, b = _pick_numbers_from_star(star, red_range, blue_range,
                                               red_count, blue_count, seed + hash(pname), red_numbers)
                red_numbers.update(r)
                if len(blue_numbers) < blue_count:
                    blue_numbers.update(b)
                interpretation_parts.append(f"{pname}主星「{star}」→ 取数 {r}")

    else:
        # 本命视角：命宫主星 + 身宫 + 大限
        palace = chart.palace_map.get("命宫")
        star = palace.main_star() if palace else None
        if star:
            r, b = _pick_numbers_from_star(star, red_range, blue_range,
                                           red_count, blue_count, seed, red_numbers)
            red_numbers.update(r)
            blue_numbers.update(b)
            interpretation_parts.append(f"命宫主星「{star}」（{STAR_WUXING.get(star, '?')}）→ 取数 {r}")

        # 身宫（与命宫相隔6宫）
        shen_idx = (TWELVE_PALACES.index("命宫") + 6) % 12
        shen_palace = chart.palaces[shen_idx]
        shen_star = shen_palace.main_star()
        if shen_star and len(red_numbers) < red_count:
            r, b = _pick_numbers_from_star(shen_star, red_range, blue_range,
                                           red_count, blue_count, seed + 13, red_numbers)
            red_numbers.update(r)
            if len(blue_numbers) < blue_count:
                blue_numbers.update(b)
            interpretation_parts.append(f"身宫主星「{shen_star}」（{shen_palace.name}）→ 取数 {r}")

        # 大限宫（命宫起逆数年支数）
        day_zhi_idx = DIZHI.index(chart.sizhu.day.di_zhi)
        daxian_idx = (TWELVE_PALACES.index("命宫") - day_zhi_idx % 10) % 12
        daxian_palace = chart.palaces[daxian_idx]
        dx_star = daxian_palace.main_star()
        if dx_star and len(red_numbers) < red_count:
            r, b = _pick_numbers_from_star(dx_star, red_range, blue_range,
                                           red_count, blue_count, seed + 19, red_numbers)
            red_numbers.update(r)
            if len(blue_numbers) < blue_count:
                blue_numbers.update(b)
            interpretation_parts.append(f"大限宫「{daxian_palace.name}」主星「{dx_star}」→ 取数 {r}")

    # 不足时用种子确定性补齐
    import random
    rng = random.Random(seed + 999)
    while len(red_numbers) < red_count:
        n = rng.randint(1, red_range)
        red_numbers.add(n)
    while len(blue_numbers) < blue_count:
        n = rng.randint(1, blue_range)
        blue_numbers.add(n)

    red_result = sorted(list(red_numbers))[:red_count]
    blue_result = sorted(list(blue_numbers))[:blue_count]

    # 推演说明
    variation_names = ["正财视角", "偏财视角", "本命视角"]
    interpretation = (
        f"【{variation_names[variation] if variation < 3 else '本命视角'}】\n"
        + "\n".join(interpretation_parts)
        + f"\n四柱: {chart.sizhu}"
        + f"\n年干四化: {chart.sihua}"
    )

    return red_result, blue_result, interpretation


if __name__ == '__main__':
    # 测试
    birth = {'year': 1985, 'month': 6, 'day': 15, 'hour': 14, 'minute': 30, 'city': '北京', 'gender': '男'}
    for v in range(3):
        r, b, interp = predict('ssq', 33, 16, 6, 1, birth, target_period=26110, variation=v)
        print(f"\n=== 第{v+1}组 ===")
        print(f"红球: {r}")
        print(f"蓝球: {b}")
        print(interp)