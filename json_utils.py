"""
JSON文件安全读写工具模块

提供原子写入、自动备份、损坏恢复、线程安全的JSON文件操作
"""
import json
import os
import shutil
import threading
from typing import Any, Optional
from datetime import datetime

# 文件锁字典：{文件路径: threading.Lock()}
_file_locks = {}
_lock_dict_mutex = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    """
    获取文件专属锁（线程安全）
    
    Args:
        path: 文件路径
        
    Returns:
        该文件的专属锁对象
    """
    with _lock_dict_mutex:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def safe_json_write(file_path: str, data: Any, backup: bool = True) -> bool:
    """
    原子写入JSON + 自动备份
    
    特性：
    - 先写入临时文件，再原子替换（防止写入一半时崩溃）
    - 写入前自动备份旧文件（带时间戳）
    - 线程安全（文件级锁）
    - 自动清理旧备份（保留最近3个）
    
    Args:
        file_path: 目标文件路径
        data: 要写入的数据（可JSON序列化）
        backup: 是否在写入前备份旧文件
    
    Returns:
        bool: 写入是否成功
    """
    lock = _get_file_lock(file_path)
    tmp_path = f"{file_path}.tmp"
    
    with lock:
        try:
            # 1. 备份旧文件
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                # 保留最近3个备份
                _cleanup_old_backups(file_path, keep=3)
            
            # 2. 写入临时文件
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 3. 原子替换（Windows/Linux通用）
            os.replace(tmp_path, file_path)
            return True
            
        except Exception as e:
            print(f"[ERROR] safe_json_write({file_path}) 失败: {e}")
            # 清理临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            return False


def safe_json_load(file_path: str, default: Any = None, 
                   try_backup: bool = True) -> Any:
    """
    读取JSON + 损坏自动恢复
    
    特性：
    - 主文件损坏时自动从最新备份恢复
    - 恢复成功后自动覆盖损坏的主文件
    - 线程安全
    
    Args:
        file_path: 文件路径
        default: 文件不存在/所有备份均损坏时的默认值
        try_backup: 损坏时是否尝试从备份恢复
    
    Returns:
        解析后的数据，失败返回 default
    """
    lock = _get_file_lock(file_path)
    
    with lock:
        # 1. 尝试加载主文件
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARNING] {file_path} JSON损坏: {e}")
                
                # 2. 尝试从最新备份恢复
                if try_backup:
                    dir_path = os.path.dirname(file_path) or '.'
                    base_name = os.path.basename(file_path)
                    backup_files = sorted([
                        f for f in os.listdir(dir_path)
                        if f.startswith(base_name + '.bak.')
                    ], reverse=True)  # 最新的在前
                    
                    for backup_name in backup_files:
                        backup_path = os.path.join(dir_path, backup_name)
                        try:
                            with open(backup_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            print(f"[INFO] 从备份恢复: {backup_name}")
                            # 将备份覆盖到主文件
                            shutil.copy2(backup_path, file_path)
                            return data
                        except:
                            continue
                    
                    print(f"[ERROR] 所有备份均损坏，返回默认值")
        
        # 3. 文件不存在或所有恢复失败
        return default


def _cleanup_old_backups(file_path: str, keep: int = 3):
    """
    清理旧备份（保留最新N个）
    
    Args:
        file_path: 原文件路径
        keep: 保留最近N个备份
    """
    try:
        dir_path = os.path.dirname(file_path) or '.'
        base_name = os.path.basename(file_path)
        backup_files = sorted([
            f for f in os.listdir(dir_path)
            if f.startswith(base_name + '.bak.')
        ], reverse=True)
        
        # 删除超出保留数量的备份
        for old_backup in backup_files[keep:]:
            old_path = os.path.join(dir_path, old_backup)
            os.remove(old_path)
            print(f"[INFO] 清理旧备份: {old_backup}")
    except Exception as e:
        print(f"[WARNING] 清理备份失败: {e}")


def validate_json_structure(data: Any, schema: dict) -> bool:
    """
    验证JSON数据结构（简化版schema校验）
    
    Args:
        data: 要校验的数据
        schema: 期望的结构，格式:
                {'type': 'list'/'dict', 'required_keys': [...], 'item_type': ...}
    
    Returns:
        bool: 是否符合schema
    
    Example:
        schema = {
            'type': 'list',
            'item_type': 'dict',
            'required_keys': ['lottery_type', 'model_type', 'target_period']
        }
    """
    try:
        # 检查顶层类型
        if schema.get('type') == 'list':
            if not isinstance(data, list):
                return False
            # 检查列表项
            if 'item_type' in schema and len(data) > 0:
                if schema['item_type'] == 'dict':
                    for item in data:
                        if not isinstance(item, dict):
                            return False
                        # 检查必需字段
                        if 'required_keys' in schema:
                            for key in schema['required_keys']:
                                if key not in item:
                                    return False
        elif schema.get('type') == 'dict':
            if not isinstance(data, dict):
                return False
            # 检查必需字段
            if 'required_keys' in schema:
                for key in schema['required_keys']:
                    if key not in data:
                        return False
        return True
    except:
        return False
