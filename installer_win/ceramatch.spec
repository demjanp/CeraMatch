# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, copy_metadata


args = sys.argv[2:]
if len(args) != 1:
    raise Exception("Invalid number of arguments.\nCorrect use: pyinstaller installer_win\\ceramatch.spec <deposit dir>")

deposit_path, = args
deposit_path = deposit_path.strip("\"")
deposit_path = os.path.normpath(os.path.abspath(deposit_path))

if os.path.split(deposit_path)[-1] != "deposit_gui":
    raise Exception("Invalid deposit dir: path must end with deposit_gui")

block_cipher = None
lap_data_hiddenimports = collect_submodules("lap_data")
lap_data_metadata = copy_metadata("lap_data")

path = "installer_win\\hiddenimports.py"
with open(path, "r", encoding="utf-8") as file:
    file_content = file.read()
imports = []
exec(file_content)

a = Analysis(
    ["..\\bin\\start_cm.py"],
    pathex=["..\\src", "..\\.venv\\Lib\\site-packages"],
    binaries=[],
    datas=[
        ("..\\src\\ceramatch\\res", "ceramatch\\res"),
        ("..\\src\\ceramatch\\THIRDPARTY.TXT", "ceramatch"),
        ("..\\cm_icon.ico", "."),
        (os.path.join(deposit_path, "dist", "deposit_gui"), "deposit_gui"),
        (os.path.join(deposit_path, "dist", "deposit_gui", "deposit_gui"), "deposit_gui"),
    ] + lap_data_metadata,
    hiddenimports=[
        "ceramatch",
        "deposit_gui",
        "lap_data",
        "networkit.base",
        "networkit.helpers",
        "networkit.traversal",
        "scipy.io",
        "skimage.filters.edges",
        "skimage.filters.rank.core_cy_3d",
        "sklearn.neighbors.quad_tree",
        "sklearn.neighbors.typedefs",
        "sklearn.tree._utils",
        "sklearn.utils._cython_blas",
        "sklearn.utils._weight_vector",
    ] + lap_data_hiddenimports + imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tcl", "tk", "wx", "h5py", "matplotlib", "sip", "PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ceramatch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="..\\cm_icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ceramatch",
    distpath="dist/ceramatch",
)
