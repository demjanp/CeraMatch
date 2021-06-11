# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['cm.py'],
             pathex=['C:\\documents_synced\\CeraMatch'],
             binaries=[],
             datas=[],
             hiddenimports=['pywt', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['h5py', 'matplotlib', 'sip', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='cm',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='cm_icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='cm')
