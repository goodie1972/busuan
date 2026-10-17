# -*- coding:utf-8 -*-
"""
数据库自动备份模块
================

功能：
- 定时备份 busuan.db 到 backup/ 目录
- 保留最近 N 天的备份（默认30天）
- 支持手动触发/定时任务调用
- 备份格式：backup/busuan_YYYYMMDD_HHMMSS.db

使用：
1. 手动备份：python db_backup.py
2. 定时任务：每天 03:00 自动运行（通过 Windows 任务计划程序）
3. GUI 调用：在"系统设置"页添加"立即备份"按钮
"""
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

# 数据库文件路径
DB_FILE = Path(__file__).parent / 'busuan.db'
BACKUP_DIR = Path(__file__).parent / 'backup'
BACKUP_DIR.mkdir(exist_ok=True)

# 备份保留天数
RETENTION_DAYS = 30


def backup_database(note: str = '') -> str:
    """
    备份数据库到 backup/ 目录
    
    Args:
        note: 备份标注（可选，用于标识备份目的）
    
    Returns:
        str: 备份文件路径；失败返回空字符串
    """
    if not DB_FILE.exists():
        print(f"[ERROR] Database not found: {DB_FILE}")
        return ''
    
    try:
        # 生成备份文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        note_suffix = f'_{note}' if note else ''
        backup_file = BACKUP_DIR / f'busuan_{timestamp}{note_suffix}.db'
        
        # 复制文件
        shutil.copy2(DB_FILE, backup_file)
        
        file_size = backup_file.stat().st_size
        print(f"[OK] Backup created: {backup_file.name} ({file_size / 1024:.1f} KB)")
        
        return str(backup_file)
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        return ''


def clean_old_backups(retention_days: int = RETENTION_DAYS) -> int:
    """
    清理超过保留期的备份文件
    
    Args:
        retention_days: 保留天数（默认30天）
    
    Returns:
        int: 清理的文件数
    """
    if not BACKUP_DIR.exists():
        return 0
    
    cutoff_time = time.time() - retention_days * 86400
    deleted = 0
    
    for backup_file in BACKUP_DIR.glob('busuan_*.db'):
        if backup_file.stat().st_mtime < cutoff_time:
            try:
                backup_file.unlink()
                print(f"[Cleanup] Deleted old backup: {backup_file.name}")
                deleted += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete {backup_file.name}: {e}")
    
    return deleted


def list_backups() -> list:
    """
    列出所有备份文件（按时间降序）
    
    Returns:
        list: [(文件名, 大小KB, 修改时间ISO), ...]
    """
    if not BACKUP_DIR.exists():
        return []
    
    backups = []
    for backup_file in BACKUP_DIR.glob('busuan_*.db'):
        stat = backup_file.stat()
        backups.append((
            backup_file.name,
            stat.st_size / 1024,  # KB
            datetime.fromtimestamp(stat.st_mtime).isoformat()
        ))
    
    # 按修改时间降序
    backups.sort(key=lambda x: x[2], reverse=True)
    return backups


def restore_backup(backup_name: str, confirm: bool = False) -> bool:
    """
    从备份恢复数据库（危险操作，需要确认）
    
    Args:
        backup_name: 备份文件名（如 'busuan_20261016_030000.db'）
        confirm: 是否确认恢复（必须显式传 True）
    
    Returns:
        bool: 是否成功恢复
    """
    if not confirm:
        print("[WARNING] Restore requires explicit confirmation (confirm=True)")
        return False
    
    backup_file = BACKUP_DIR / backup_name
    if not backup_file.exists():
        print(f"[ERROR] Backup not found: {backup_name}")
        return False
    
    try:
        # 先备份当前数据库（防误操作）
        if DB_FILE.exists():
            emergency_backup = backup_database(note='pre_restore')
            print(f"[Safety] Current DB backed up to: {Path(emergency_backup).name}")
        
        # 恢复
        shutil.copy2(backup_file, DB_FILE)
        print(f"[OK] Database restored from: {backup_name}")
        return True
    except Exception as e:
        print(f"[ERROR] Restore failed: {e}")
        return False


def schedule_daily_backup():
    """
    创建 Windows 任务计划程序任务（每天 03:00 备份）
    
    注意：需要管理员权限
    """
    import subprocess
    import sys
    
    python_exe = sys.executable
    script_path = Path(__file__).absolute()
    task_name = "BusuanDatabaseBackup"
    
    # 检查任务是否已存在
    check_cmd = f'schtasks /Query /TN "{task_name}"'
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"[INFO] Task '{task_name}' already exists")
        return
    
    # 创建任务（每天 03:00）
    create_cmd = f'''schtasks /Create /TN "{task_name}" /TR "\"{python_exe}\" \"{script_path}\"" /SC DAILY /ST 03:00 /F'''
    
    try:
        subprocess.run(create_cmd, shell=True, check=True)
        print(f"[OK] Scheduled task '{task_name}' created (runs daily at 03:00)")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create scheduled task: {e}")
        print("[HINT] Try running as administrator or create manually:")
        print(f"       Task name: {task_name}")
        print(f"       Command: {python_exe} {script_path}")
        print(f"       Schedule: Daily at 03:00")


if __name__ == "__main__":
    print("=== Database Backup Tool ===")
    print(f"Database: {DB_FILE}")
    print(f"Backup directory: {BACKUP_DIR}")
    print()
    
    # 执行备份
    print("Creating backup...")
    backup_file = backup_database(note='auto')
    
    if backup_file:
        # 清理旧备份
        print(f"\nCleaning backups older than {RETENTION_DAYS} days...")
        deleted = clean_old_backups()
        print(f"Deleted {deleted} old backup(s)")
        
        # 列出现有备份
        print(f"\nCurrent backups:")
        backups = list_backups()
        for name, size, mtime in backups[:5]:  # 只显示最近5个
            print(f"  {name} ({size:.1f} KB, {mtime})")
        if len(backups) > 5:
            print(f"  ... and {len(backups) - 5} more")
        
        print(f"\n[OK] Backup completed successfully!")
    else:
        print(f"\n[FAILED] Backup failed!")
        exit(1)
