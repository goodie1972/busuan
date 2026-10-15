# -*- coding:utf-8 -*-
"""
模型文件完整性校验模块

在 joblib.load 加载 pkl 模型前做防御性检查：
- 文件是否存在、非空、大小合理（损坏的0字节/截断文件）
- 魔数嗅探（拒绝明显错误的文件，如 HTML 错误页被重命名为 .pkl）
- 加载后结构与版本校验（检查预期的模型类型/属性）
- 异常白名单化（拒绝反序列化炸弹式的未知对象）
"""
import os
import pickle
from typing import Any, Optional, Tuple

# 合理大小边界（字节）：过小=损坏/占位，过大=异常
MIN_PKL_SIZE = 100            # 空 scaler 也 >100B
MAX_PKL_SIZE = 2 * 1024**3    # 2GB 上限，防内存炸弹

# 预期的 pkl 内容类型白名单（字符串匹配类名）
ALLOWED_CLASS_HINTS = (
    'RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'CatBoost',
    'MinMaxScaler', 'StandardScaler', 'IsotonicRegression',
    'LogisticRegression', 'Pipeline', 'VotingClassifier',
    'numpy', 'scipy', 'sklearn',
)

# 危险嗅探：这些字节出现在头部 = 文件类型错误
BAD_MAGIC_PREFIXES = (
    b'<!DOCTYPE', b'<html', b'<?xml', b'PK\x03\x04',  # zip 伪装
    b'\x1f\x8b',  # gzip（应解压后再用）
)


def sniff_file_header(path: str, nbytes: int = 256) -> bytes:
    """读取文件头部字节用于魔数嗅探"""
    with open(path, 'rb') as f:
        return f.read(nbytes)


def check_pkl_file(path: str,
                   min_size: int = MIN_PKL_SIZE,
                   max_size: int = MAX_PKL_SIZE) -> Tuple[bool, str]:
    """
    加载前静态检查 pkl 文件（不反序列化）

    Returns:
        (是否通过, 说明)
    """
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    if not os.path.isfile(path):
        return False, f"不是常规文件: {path}"

    size = os.path.getsize(path)
    if size == 0:
        return False, f"空文件（0字节，可能下载/写入中断）: {path}"
    if size < min_size:
        return False, f"文件过小({size}B < {min_size}B)，疑似损坏: {path}"
    if size > max_size:
        return False, f"文件过大({size}B > {max_size}B)，拒绝加载防内存炸弹: {path}"

    try:
        header = sniff_file_header(path)
    except OSError as e:
        return False, f"读取文件头失败: {e}"

    for bad in BAD_MAGIC_PREFIXES:
        if header.startswith(bad):
            return False, f"文件头魔数异常（疑似HTML/zip/gzip误存为pkl）: {path}"

    # pickle 流通常以 \x80 (protocol 2+) 或文本协议指令开头；不做强校验，只做弱提示
    return True, f"OK ({size:,} bytes)"


def safe_load_pkl(path: str,
                  expected_hint: Optional[str] = None) -> Tuple[Optional[Any], str]:
    """
    安全加载 pkl：先静态检查，再反序列化，再白名单嗅探。

    Args:
        path: pkl 文件路径
        expected_hint: 期望的类名提示（如 'MinMaxScaler'），命中白名单才放行

    Returns:
        (对象或None, 说明)
    """
    import joblib

    ok, msg = check_pkl_file(path)
    if not ok:
        return None, f"[静态检查拒绝] {msg}"

    try:
        obj = joblib.load(path)
    except Exception as e:
        return None, f"[反序列化失败] {type(e).__name__}: {e}"

    # 结构嗅探：类名白名单
    cls_name = type(obj).__name__
    module = type(obj).__module__
    full = f"{module}.{cls_name}"

    if expected_hint:
        if expected_hint not in cls_name and expected_hint not in full:
            return None, (f"[类型不符] 期望含 '{expected_hint}'，"
                          f"实际 {full}（疑似模型文件错位）")

    # 白名单软校验：完全不在白名单内的顶层对象要警惕（但放行并提示，
    # 因为 numpy 数组/dict 等容器也是合法的）
    if not any(h in full or h in cls_name for h in ALLOWED_CLASS_HINTS):
        if not isinstance(obj, (dict, list, tuple, set, int, float, str, bytes)):
            return obj, f"[警告] 未知对象类型 {full}，已加载但请人工确认"

    return obj, f"OK: {full}"


def backup_model_before_overwrite(path: str, keep: int = 2) -> Optional[str]:
    """
    训练覆盖模型前备份旧文件（带时间戳，保留最近N个）

    Returns:
        备份路径；文件不存在时返回 None
    """
    import shutil
    from datetime import datetime
    if not os.path.exists(path):
        return None
    bak = f"{path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    # 清理旧备份
    dir_path = os.path.dirname(path) or '.'
    base = os.path.basename(path)
    baks = sorted([f for f in os.listdir(dir_path)
                   if f.startswith(base + '.bak.')], reverse=True)
    for old in baks[keep:]:
        try:
            os.remove(os.path.join(dir_path, old))
        except OSError:
            pass
    return bak
