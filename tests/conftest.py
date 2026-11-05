# -*- coding:utf-8 -*-
"""pytest 全局配置。

Windows 下 PyQt5 的 DLL 与 torch 的 c10.dll 存在初始化顺序冲突：
先导入 PyQt5 再导入 torch 会触发 WinError 1114。GUI 自身已保证 torch
先于 Qt 导入，这里同样在测试收集前预加载 torch，并固定 Qt 使用
offscreen 平台，避免无显示环境启动失败。

另外把 SQLite 库切到临时沙盒：生产 busuan.db 里的预测记录会被集成模型
的权重回测消费，测试假预测写进去就是不可逆污染（见 tests/db_sandbox.py）。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db_sandbox  # noqa: E402

db_sandbox.activate()

try:
    import torch  # noqa: F401
except Exception:
    pass
