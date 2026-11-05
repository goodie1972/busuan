# -*- coding:utf-8 -*-
"""数据库隔离测试。

背景：prediction_records 的存档写入 db_access 的 SQLite 库。库路径曾是模块常量
`DB_PATH = "busuan.db"`（相对当前工作目录），任何脚本/测试从项目根运行都会把
假预测（如 latest_period=1000）写进生产库，再被集成模型的权重回测当作真实样本。

这里锁住两件事：
1. 库路径可在运行期用 BUSUAN_DB 环境变量切换；
2. 测试会话内的一切存档都落在临时库，生产库文件零写入。
"""
import sqlite3
from pathlib import Path

import pytest

import db_access
import db_backup
import prediction_records

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROD_DB = PROJECT_DIR / 'busuan.db'
SCHEMA_FILE = PROJECT_DIR / 'schema.sql'


def _init_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_FILE.read_text(encoding='utf-8'))
    conn.commit()
    conn.close()


def _count_predictions(path: Path) -> int:
    if not path.exists():
        return 0
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()


def _latest_periods(path: Path):
    """库内预测的期号集合，用来判断某个库有没有被动过。"""
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return sorted(r[0] for r in conn.execute(
            "SELECT latest_period FROM predictions"))
    finally:
        conn.close()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """把 BUSUAN_DB 指向临时库，并把 CWD 换到含同名生产库的临时目录。"""
    prod = tmp_path / 'busuan.db'
    _init_db(prod)
    alt = tmp_path / 'isolated.db'
    _init_db(alt)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BUSUAN_DB', str(alt))
    return alt, prod


def test_busuan_db_env_overrides_relative_production_path(isolated_db):
    """设了 BUSUAN_DB 时，存档必须落在该路径而不是 CWD 下的 busuan.db。"""
    alt, prod = isolated_db
    prediction_records.add_prediction_record(
        'ssq', 'ensemble', 26118, [1, 2, 3, 4, 5, 6], [7],
        [0.1, 0.2, 0.15, 0.1, 0.08, 0.07], [0.3])

    assert _count_predictions(alt) == 1
    assert _count_predictions(prod) == 0, "临时生产库被写脏，说明库路径不可隔离"


def test_pytest_session_db_is_not_the_repo_database():
    """整个测试会话默认使用临时库，绝不指向项目根的 busuan.db。"""
    current = Path(db_access.get_db_path()).resolve()
    assert current != PROD_DB.resolve(), "测试会话正在直连生产库"
    assert PROJECT_DIR not in current.parents, f"测试库仍在项目目录内: {current}"


def test_archiving_still_stores_every_record(isolated_db):
    """约束回归：每注预测都必须入库（含中间环节），供后续权重回测消费。"""
    alt, _ = isolated_db
    batch = [([r, r + 1, r + 2, r + 3, r + 4, r + 5], [1], None, None)
             for r in range(1, 6)]
    total = prediction_records.add_prediction_records_batch(
        'ssq', 'ensemble', 26118, batch)
    assert total == 5

    stored = db_access.get_predictions(limit=None)
    assert len(stored) == 5
    assert {tuple(p['red_numbers']) for p in stored} == {
        tuple(x[0]) for x in batch}


def test_session_db_is_created_with_full_schema():
    """会话库由 conftest 建好，核验/权重回测所需的表都在。"""
    path = Path(db_access.get_db_path())
    assert path.exists(), f"会话测试库不存在: {path}"
    conn = sqlite3.connect(str(path))
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {'predictions', 'lottery_draws', 'model_weights_history'} <= tables


@pytest.fixture
def env_db(tmp_path, monkeypatch):
    """BUSUAN_DB 指向临时库；CWD 放一个同名"生产库"，用于证明备份没拷错文件。"""
    prod = tmp_path / 'busuan.db'
    _init_db(prod)
    _seed(prod, [('ensemble', 26118)])
    db = tmp_path / 'sub' / 'session.db'
    db.parent.mkdir()
    _init_db(db)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BUSUAN_DB', str(db))
    backup_dir = tmp_path / 'backup'
    backup_dir.mkdir()
    monkeypatch.setattr(db_backup, 'BACKUP_DIR', backup_dir)
    return db, prod


def _seed(path: Path, rows):
    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT INTO predictions (predict_time, lottery_type, model_type, "
        "latest_period, red_numbers, blue_numbers) VALUES (?, 'ssq', ?, ?, '[1,2,3,4,5,6]', '[7]')",
        [('2026-11-05T00:00:00', mt, lp) for mt, lp in rows])
    conn.commit()
    conn.close()


def test_backup_follows_busuan_db_env_not_the_repo_file(env_db):
    """备份必须备的是当前生效库，而不是项目根的 busuan.db。"""
    db, prod = env_db
    _seed(db, [('ensemble', 26119)])
    backup_file = db_backup.backup_database(note='unittest')
    assert backup_file, "备份失败"
    assert _count_predictions(Path(backup_file)) == 1, "备的是项目根库，不是生效库"


def test_backup_includes_uncheckpointed_wal_data(env_db):
    """WAL 模式下未 checkpoint 的提交也必须在备份里（shutil.copy 会漏）。"""
    db, _ = env_db
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute(
            "INSERT INTO predictions (predict_time, lottery_type, model_type, "
            "latest_period, red_numbers, blue_numbers) "
            "VALUES ('2026-11-05T00:00:00','ssq','ensemble',26120,'[1,2,3,4,5,6]','[7]')")
        conn.commit()
        # 不关闭连接，数据仍停在 -wal 里
        backup_file = db_backup.backup_database(note='wal')
    finally:
        conn.close()
    assert backup_file
    assert _count_predictions(Path(backup_file)) == 1, "备份漏掉了 WAL 中未落盘的记录"


def test_restore_follows_busuan_db_env(env_db):
    """恢复同样只作用于生效库，绝不覆盖项目根的生产库。"""
    db, prod = env_db
    _seed(db, [('ensemble', 26121)])
    backup_file = db_backup.backup_database(note='pointintime')
    _seed(db, [('ensemble', 26122)])
    assert db_backup.restore_backup(Path(backup_file).name, confirm=True)
    assert _latest_periods(db) == [26121]
    assert _latest_periods(prod) == [26118], "恢复动作写到了项目根的生产库"

