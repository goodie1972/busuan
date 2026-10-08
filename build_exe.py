# -*- coding: utf-8 -*-
"""
Lottery Prophet — 打包为独立可执行文件 (PyInstaller)
用法: python build_exe.py
"""

import os
import sys
import shutil
import subprocess

# ==================== 配置 ====================
APP_NAME = "busuan"
ENTRY_POINT = "main.py"
ICON_FILE = None  # 如果有 .ico 图标文件，可设置路径
SINGLE_FILE = False  # True=单个 exe, False=目录模式（更可靠）

# 需要额外包含的数据目录 (src -> dst)
# 注意: scripts/ 下的 .py 文件（如 fetch_dlt_data.py、data_analysis.py 等）
# 需要通过 import 分析打包，但为确保运行时 sys.path.append 能找到它们，
# 将整个 scripts/ 目录作为数据文件复制
DATA_DIRS = [
    ("scripts", "scripts"),
]

# 需要在打包后创建的空目录
EMPTY_DIRS = [
    "model/dlt",
    "model/ssq",
    "data/dlt",
    "data/ssq",
]

# 隐式导入 (PyInstaller 可能检测不到的)
HIDDEN_IMPORTS = [
    # 脚本层动态导入
    "scripts.dlt.fetch_dlt_data",
    "scripts.ssq.fetch_ssq_data",
    "scripts.data_analysis",
    "scripts.advanced_statistics",
    # PyTorch CRF
    "TorchCRF",
    # ML 模型
    "sklearn.ensemble",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.tree",
    "sklearn.preprocessing",
    "sklearn.model_selection",
    "sklearn.metrics",
    "xgboost",
    "lightgbm",
    "catboost",
    # PyQt5
    "PyQt5.sip",
    # 其他
    "joblib",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "loguru",
    "requests",
    "bs4",
]

# ==================== 打包过程 ====================

def main():
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print(f"项目根目录: {project_root}")

    # 清理之前的打包产物
    for d in ["build", "dist"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"已清理 {d}/")

    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", APP_NAME,
    ]

    # 模式选择: 单文件 vs 目录模式
    if SINGLE_FILE:
        cmd.append("--onefile")
        cmd.append("--console")  # 单文件模式下保留控制台以便调试
    else:
        cmd.append("--windowed")  # 目录模式无控制台窗口
        cmd.append("--console")  # 保留控制台方便查看错误

    # 收集 ML 库的动态库和数据文件 (DLL/VERSION 等)，避免运行时找不到
    for pkg in ["xgboost", "lightgbm", "catboost"]:
        cmd.extend(["--collect-all", pkg])

    # 添加运行时 hook（解决 torch DLL 加载问题）
    runtime_hook = os.path.join(project_root, "runtime_hook.py")
    if os.path.exists(runtime_hook):
        cmd.extend(["--runtime-hook", runtime_hook])
        print(f"添加运行时 hook: {runtime_hook}")

    # 添加数据目录
    for src, dst in DATA_DIRS:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
            print(f"添加数据目录: {src} -> {dst}")
        else:
            print(f"（数据目录不存在: {src}")

    # 添加隐藏导入
    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])

    # 图标
    if ICON_FILE and os.path.exists(ICON_FILE):
        cmd.extend(["--icon", ICON_FILE])

    # 入口文件
    cmd.append(ENTRY_POINT)

    # 执行打包
    print(f"\n开始打包...")
    print(f"命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print("\n!!! 打包失败，返回码:", result.returncode)
        sys.exit(1)

    # 单文件模式下，所有数据打包在 exe 内部，不需要在 dist 目录创建额外目录
    if not SINGLE_FILE:
        dist_dir = os.path.join("dist", APP_NAME)
        if os.path.exists(dist_dir):
            # 将源码中的 scripts/ model/ data/ 复制到 exe 同级目录
            # （应用内部大量使用 ./scripts/ ./model/ 等相对路径，
            #  且 fetch/训练功能需要写 CSV 和模型文件，不能放只读的 _internal）
            for d in ["scripts", "model", "data"]:
                src_dir = os.path.join(project_root, d)
                if os.path.isdir(src_dir):
                    dst_dir = os.path.join(dist_dir, d)
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("__pycache__"))
                    print(f"复制数据目录: {d}/")
            for d in EMPTY_DIRS:
                full_path = os.path.join(dist_dir, d)
                os.makedirs(full_path, exist_ok=True)
                print(f"创建目录: {full_path}")
            # 复制 README
            for f in ["README.md", "README_EN.md"]:
                if os.path.exists(f):
                    shutil.copy2(f, os.path.join(dist_dir, f))
                    print(f"复制文件: {f}")

    print("\n== 打包完成! ==")
    print("输出目录:", os.path.join(project_root, 'dist', APP_NAME))
    print("可执行文件:", os.path.join(project_root, 'dist', APP_NAME, APP_NAME + '.exe'))

    # 显示大小
    exe_path = os.path.join(project_root, "dist", APP_NAME + ".exe") if SINGLE_FILE else os.path.join(project_root, "dist", APP_NAME, APP_NAME + ".exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"exe 大小: {size_mb:.0f} MB")
        if not SINGLE_FILE:
            # 统计整个 dist 目录大小
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dist_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            print(f"总大小: {total_size / (1024 * 1024):.0f} MB")

if __name__ == "__main__":
    main()