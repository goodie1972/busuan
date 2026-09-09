# -*- coding: utf-8 -*-
"""
PyInstaller 运行时 hook — 在导入 torch 之前正确加载所有 DLL
"""
import os
import sys
import ctypes

# 获取解压后的基础路径
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

torch_lib_path = os.path.join(base_path, 'torch', 'lib')

if os.path.isdir(torch_lib_path):
    # 1. 先将 torch/lib 加到 PATH 中，确保 DLL 依赖能正确解析
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")

    # 2. 设置 OpenMP/MKL 环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_INIT_AT_FORK"] = "FALSE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 3. 按正确顺序预加载 torch DLL（依赖顺序）
    dll_order = [
        "libiomp5md.dll",       # Intel OpenMP — 无依赖
        "c10.dll",              # 基础 torch 库
        "torch_cpu.dll",        # CPU 后端
        "torch_python.dll",     # Python 绑定
        "torch.dll",            # 主 torch DLL
        "torch_global_deps.dll",# 全局依赖
        "uv.dll",               # 工具库
        "shm.dll",              # 共享内存
    ]

    loaded = []
    for dll_name in dll_order:
        dll_path = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_path):
            try:
                # 使用 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                # 但先尝试普通 LoadLibraryW
                handle = ctypes.CDLL(dll_path)
                loaded.append(dll_name)
            except Exception as e:
                # 尝试用 Windows API 加载
                try:
                    handle = ctypes.windll.kernel32.LoadLibraryW(dll_path)
                    if handle:
                        loaded.append(dll_name)
                except Exception:
                    pass  # 让 torch 自己去处理

    # 4. 修改 torch 的 DLL 加载行为，使其更宽容
    # 在 torch 导入前，设置环境变量让 torch 使用更宽松的加载方式
    os.environ["TORCH_DLL_LOAD_FLAGS"] = "0"