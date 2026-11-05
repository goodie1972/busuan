# -*- coding:utf-8 -*-
"""测试用数据库沙盒。

生产库 busuan.db 里的预测记录会被集成模型的权重回测当样本消费，所以任何
测试/自检脚本都不能往里写假预测（历史上 test_integration.py 写进过
latest_period=1000、test_db_adversarial.py 甚至 os.remove 过生产库）。

用法：在 import db_access / prediction_records 之前调用 activate()。
"""
import sqlite3
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROD_DB = PROJECT_DIR / 'busuan.db'
SCHEMA_FILE = PROJECT_DIR / 'schema.sql'


def make_session_db(target: Path) -> Path:
    """建出沙盒库：优先一致性复制生产库，否则按 schema 空建。

    复制用 sqlite3 backup API 而不是拷文件——生产库开着 WAL，直接拷主文件
    可能拿到没有最近提交的中间态。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if PROD_DB.exists():
        try:
            src = sqlite3.connect(f"file:{PROD_DB.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error:
            src = None
        if src is not None:
            dst = sqlite3.connect(str(target))
            try:
                src.backup(dst)
                return target
            except sqlite3.Error:
                pass
            finally:
                dst.close()
                src.close()
    conn = sqlite3.connect(str(target))
    try:
        conn.executescript(SCHEMA_FILE.read_text(encoding='utf-8'))
        conn.commit()
    finally:
        conn.close()
    return target


def activate() -> str:
    """把 BUSUAN_DB 指向临时沙盒库，返回该路径（已设置则原样返回）。"""
    import os
    current = os.environ.get('BUSUAN_DB')
    if current:
        return current
    sandbox = Path(tempfile.mkdtemp(prefix='busuan_sandbox_')) / 'busuan.db'
    os.environ['BUSUAN_DB'] = str(make_session_db(sandbox))
    return os.environ['BUSUAN_DB']
