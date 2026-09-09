# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('scripts', 'scripts')]
binaries = []
hiddenimports = ['scripts.dlt.fetch_dlt_data', 'scripts.ssq.fetch_ssq_data', 'scripts.data_analysis', 'scripts.advanced_statistics', 'TorchCRF', 'sklearn.ensemble', 'sklearn.svm', 'sklearn.neighbors', 'sklearn.tree', 'sklearn.preprocessing', 'sklearn.model_selection', 'sklearn.metrics', 'xgboost', 'lightgbm', 'catboost', 'PyQt5.sip', 'joblib', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'loguru', 'requests', 'bs4']
tmp_ret = collect_all('xgboost')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lightgbm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('catboost')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['D:\\bobo\\python\\lottoprophet-master\\runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LotteryProphet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LotteryProphet',
)
