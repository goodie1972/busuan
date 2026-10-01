# -*- coding: utf-8 -*-
"""
干支、五行、纳音、冲合会等基础数据与算法
纯 Python 实现，无外部依赖
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# ==================== 基础常量 ====================

# 十天干
TIANGAN = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
TIANGAN_YINYANG = ['阳', '阴', '阳', '阴', '阳', '阴', '阳', '阴', '阳', '阴']  # 甲阳乙阴...
TIANGAN_WUXING = ['木', '木', '火', '火', '土', '土', '金', '金', '水', '水']

# 十二地支
DIZHI = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
DIZHI_YINYANG = ['阳', '阴', '阳', '阴', '阳', '阴', '阳', '阴', '阳', '阴', '阳', '阴']
DIZHI_WUXING = ['水', '土', '木', '木', '土', '火', '火', '土', '金', '金', '土', '水']

# 地支藏干（人元法）
DIZHI_CANGGAN = {
    '子': [('癸', '水', 10)],
    '丑': [('己', '土', 6), ('癸', '水', 3), ('辛', '金', 1)],
    '寅': [('戊', '土', 3), ('丙', '火', 7), ('甲', '木', 10)],
    '卯': [('乙', '木', 10)],
    '辰': [('戊', '土', 6), ('乙', '木', 3), ('癸', '水', 1)],
    '巳': [('戊', '土', 3), ('庚', '金', 7), ('丙', '火', 10)],
    '午': [('己', '土', 3), ('丁', '火', 7), ('戊', '土', 10)],
    '未': [('己', '土', 6), ('丁', '火', 3), ('乙', '木', 1)],
    '申': [('庚', '金', 7), ('壬', '水', 3), ('戊', '土', 10)],
    '酉': [('辛', '金', 10)],
    '戌': [('戊', '土', 6), ('辛', '金', 3), ('丁', '火', 1)],
    '亥': [('壬', '水', 7), ('甲', '木', 3)],
}

# 六十甲子
JIAZI = [f"{TIANGAN[i%10]}{DIZHI[i%12]}" for i in range(60)]

# 五行生克
WUXING_SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}  # 我生谁
WUXING_KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}   # 我克谁

# 纳音五行（六十甲子）
NAYIN = {
    '甲子': '海中金', '乙丑': '海中金', '丙寅': '炉中火', '丁卯': '炉中火',
    '戊辰': '大林木', '己巳': '大林木', '庚午': '路旁土', '辛未': '路旁土',
    '壬申': '剑锋金', '癸酉': '剑锋金', '甲戌': '山头火', '乙亥': '山头火',
    '丙子': '涧下水', '丁丑': '涧下水', '戊寅': '城头土', '己卯': '城头土',
    '庚辰': '白蜡金', '辛巳': '白蜡金', '壬午': '杨柳木', '癸未': '杨柳木',
    '甲申': '泉中水', '乙酉': '泉中水', '丙戌': '屋上土', '丁亥': '屋上土',
    '戊子': '霹雳火', '己丑': '霹雳火', '庚寅': '松柏木', '辛卯': '松柏木',
    '壬辰': '常流水', '癸巳': '常流水', '甲午': '砂中金', '乙未': '砂中金',
    '丙申': '山下火', '丁酉': '山下火', '戊戌': '平地木', '己亥': '平地木',
    '庚子': '壁上土', '辛丑': '壁上土', '壬寅': '金箔金', '癸卯': '金箔金',
    '甲辰': '覆灯火', '乙巳': '覆灯火', '丙午': '天河水', '丁未': '天河水',
    '戊申': '大驿土', '己酉': '大驿土', '庚戌': '钗钏金', '辛亥': '钗钏金',
    '壬子': '桑柘木', '癸丑': '桑柘木', '甲寅': '大溪水', '乙卯': '大溪水',
    '丙辰': '沙中土', '丁巳': '沙中土', '戊午': '天上火', '己未': '天上火',
    '庚申': '石榴木', '辛酉': '石榴木', '壬戌': '大海水', '癸亥': '大海水',
}

# 十二长生
CHANGSHENG = {
    '木': ['亥', '子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌'],  # 长生到养
    '火': ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑'],
    '土': ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑'],
    '金': ['巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑', '寅', '卯', '辰'],
    '水': ['申', '酉', '戌', '亥', '子', '丑', '寅', '卯', '辰', '巳', '午', '未'],
}

# 地支六冲
DIZHI_CHONG = {
    '子': '午', '午': '子',
    '丑': '未', '未': '丑',
    '寅': '申', '申': '寅',
    '卯': '酉', '酉': '卯',
    '辰': '戌', '戌': '辰',
    '巳': '亥', '亥': '巳',
}

# 地支六合
DIZHI_HE = {
    '子': '丑', '丑': '子',
    '寅': '亥', '亥': '寅',
    '卯': '戌', '戌': '卯',
    '辰': '酉', '酉': '辰',
    '巳': '申', '申': '巳',
    '午': '未', '未': '午',
}

# 地支三合
DIZHI_SANHE = {
    '申子辰': '水', '子': '申子辰', '辰': '申子辰', '申': '申子辰',
    '亥卯未': '木', '亥': '亥卯未', '卯': '亥卯未', '未': '亥卯未',
    '寅午戌': '火', '寅': '寅午戌', '午': '寅午戌', '戌': '寅午戌',
    '巳酉丑': '金', '巳': '巳酉丑', '酉': '巳酉丑', '丑': '巳酉丑',
}

# 地支三会
DIZHI_SANHUI = {
    '亥子丑': '水', '子': '亥子丑', '丑': '亥子丑', '亥': '亥子丑',
    '寅卯辰': '木', '寅': '寅卯辰', '卯': '寅卯辰', '辰': '寅卯辰',
    '巳午未': '火', '巳': '巳午未', '午': '巳午未', '未': '巳午未',
    '申酉戌': '金', '申': '申酉戌', '酉': '申酉戌', '戌': '申酉戌',
}

# 方位对应
DIZHI_FANGWEI = {
    '子': '北', '丑': '东北', '寅': '东北', '卯': '东',
    '辰': '东南', '巳': '东南', '午': '南', '未': '西南',
    '申': '西南', '酉': '西', '戌': '西北', '亥': '西北',
}

# ==================== 数据类 ====================

@dataclass
class GanZhi:
    """干支对象"""
    tian_gan: str
    di_zhi: str
    
    def __str__(self):
        return f"{self.tian_gan}{self.di_zhi}"
    
    def __repr__(self):
        return f"GanZhi('{self.tian_gan}{self.di_zhi}')"
    
    @property
    def wuxing(self) -> str:
        return TIANGAN_WUXING[TIANGAN.index(self.tian_gan)]
    
    @property
    def yinyang(self) -> str:
        return TIANGAN_YINYANG[TIANGAN.index(self.tian_gan)]
    
    @property
    def nayin(self) -> str:
        return NAYIN.get(str(self), '')
    
    @property
    def canggan(self) -> List[Tuple[str, str, int]]:
        return DIZHI_CANGGAN.get(self.di_zhi, [])

@dataclass
class SiZhu:
    """四柱八字"""
    year: GanZhi
    month: GanZhi
    day: GanZhi
    hour: GanZhi
    
    def __str__(self):
        return f"{self.year} {self.month} {self.day} {self.hour}"
    
    def to_list(self) -> List[GanZhi]:
        return [self.year, self.month, self.day, self.hour]


# ==================== 核心算法函数 ====================

def get_jiazi_index(ganzhi: str) -> int:
    """获取干支在六十甲子中的索引 (0-59)"""
    try:
        return JIAZI.index(ganzhi)
    except ValueError:
        return -1

def ganzhi_from_index(idx: int) -> GanZhi:
    """从索引获取干支"""
    idx = idx % 60
    return GanZhi(TIANGAN[idx % 10], DIZHI[idx % 12])

def get_year_ganzhi(year: int) -> GanZhi:
    """计算年干支（以1984年甲子年为基准）"""
    # 1984年是甲子年
    offset = (year - 1984) % 60
    return ganzhi_from_index(offset)

def get_month_ganzhi(year_gan: str, month: int) -> GanZhi:
    """根据年干推月干支（正月从寅月开始）"""
    # 年干决定月干起点：甲己之年丙作首，乙庚之年戊为头...
    year_gan_idx = TIANGAN.index(year_gan)
    # 月干起始索引表：甲年丙寅(2), 乙年戊寅(4), 丙年庚寅(6)...
    month_gan_start = (year_gan_idx * 2 + 2) % 10
    month_gan_idx = (month_gan_start + month - 1) % 10  # 正月=1=寅月
    month_zhi_idx = (month + 1) % 12  # 1月=寅(2), 12月=丑(1)
    return GanZhi(TIANGAN[month_gan_idx], DIZHI[month_zhi_idx])

def get_day_ganzhi(date_obj) -> GanZhi:
    """计算日干支（以1900-01-01为甲戌日为基准）"""
    import datetime
    if isinstance(date_obj, str):
        date_obj = datetime.datetime.strptime(date_obj, '%Y-%m-%d').date()
    elif isinstance(date_obj, datetime.datetime):
        date_obj = date_obj.date()
    
    base_date = datetime.date(1900, 1, 1)  # 甲戌日
    days_diff = (date_obj - base_date).days
    idx = (days_diff + 10) % 60  # 甲戌是第10个（索引9）
    return ganzhi_from_index(idx)

def get_hour_ganzhi(day_gan: str, hour: int, minute: int = 0) -> GanZhi:
    """根据日干推时干支"""
    # 时辰：子时23-1点，丑时1-3点... 23点归属当日子时
    if hour == 23:
        hour_zhi = '子'
    else:
        hour_zhi = DIZHI[(hour + 1) // 2 % 12]
    
    # 日干决定时干：甲己日甲子时，乙庚日丙子时...
    day_gan_idx = TIANGAN.index(day_gan)
    hour_gan_start = (day_gan_idx * 2) % 10  # 甲日甲子(0), 乙日丙子(2)...
    hour_gan_idx = (hour_gan_start + DIZHI.index(hour_zhi)) % 10
    return GanZhi(TIANGAN[hour_gan_idx], hour_zhi)

def calc_sizhu(year: int, month: int, day: int, hour: int, minute: int = 0) -> SiZhu:
    """计算四柱八字"""
    import datetime
    date_obj = datetime.date(year, month, day)
    
    year_gz = get_year_ganzhi(year)
    month_gz = get_month_ganzhi(year_gz.tian_gan, month)
    day_gz = get_day_ganzhi(date_obj)
    hour_gz = get_hour_ganzhi(day_gz.tian_gan, hour, minute)
    
    return SiZhu(year_gz, month_gz, day_gz, hour_gz)

def get_na_yin(ganzhi: str) -> str:
    """获取纳音"""
    return NAYIN.get(ganzhi, '')

def get_changsheng(wuxing: str, dizhi: str) -> str:
    """获取十二长生状态"""
    if wuxing not in CHANGSHENG:
        return ''
    idx = CHANGSHENG[wuxing].index(dizhi) if dizhi in CHANGSHENG[wuxing] else -1
    names = ['长生', '沐浴', '冠带', '临官', '帝旺', '衰', '病', '死', '墓', '绝', '胎', '养']
    return names[idx] if idx >= 0 else ''

def get_shensha(sizhu: SiZhu) -> Dict[str, List[str]]:
    """计算常用神煞（简化版）"""
    result = {}
    day_gan = sizhu.day.tian_gan
    day_zhi = sizhu.day.di_zhi
    year_zhi = sizhu.year.di_zhi
    
    # 禄神
    lu_map = {'甲': '寅', '乙': '卯', '丙': '巳', '丁': '午', '戊': '巳',
              '己': '午', '庚': '申', '辛': '酉', '壬': '亥', '癸': '子'}
    if lu_map.get(day_gan) in [sizhu.year.di_zhi, sizhu.month.di_zhi, sizhu.day.di_zhi, sizhu.hour.di_zhi]:
        result['禄神'] = [lu_map[day_gan]]
    
    # 羊刃
    yangren_map = {'甲': '卯', '乙': '寅', '丙': '午', '丁': '巳', '戊': '午',
                   '己': '巳', '庚': '酉', '辛': '申', '壬': '子', '癸': '亥'}
    if yangren_map.get(day_gan) in [sizhu.year.di_zhi, sizhu.month.di_zhi, sizhu.day.di_zhi, sizhu.hour.di_zhi]:
        result['羊刃'] = [yangren_map[day_gan]]
    
    # 桃花
    taohua_map = {'申子辰': '酉', '寅午戌': '卯', '巳酉丑': '午', '亥卯未': '子'}
    for group, pos in taohua_map.items():
        if day_zhi in group and pos in [sizhu.year.di_zhi, sizhu.month.di_zhi, sizhu.hour.di_zhi]:
            result.setdefault('桃花', []).append(pos)
    
    return result


# ==================== 测试 ====================

if __name__ == '__main__':
    # 测试四柱计算
    sizhu = calc_sizhu(1990, 5, 15, 14, 30)
    print(f"四柱: {sizhu}")
    print(f"年柱纳音: {sizhu.year.nayin}")
    print(f"月柱藏干: {sizhu.month.canggan}")
    print(f"日柱长生: {get_changsheng(sizhu.day.wuxing, sizhu.day.di_zhi)}")
    print(f"神煞: {get_shensha(sizhu)}")
    
    # 测试六十甲子
    print(f"\n六十甲子前10: {JIAZI[:10]}")
    print(f"甲子索引: {get_jiazi_index('甲子')}")
    print(f"索引30: {ganzhi_from_index(30)}")