# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['photogeoalign.py'],
    pathex=[],
    binaries=[],
    datas=[('resources/logo.png', '.')],
    hiddenimports=[
        'modules.core.utils',
        'modules.core.micmac', 
        'modules.core.geodetic',
        'modules.gui.main_window',
        'modules.gui.dialogs',
        'modules.workers.pipeline_thread',
        'modules.workers.geodetic_thread',
        'modules.workers.utils'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtNetwork'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='photogeoalign_windows',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/logo.png'
) 