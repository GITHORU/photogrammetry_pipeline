# -*- mode: python ; coding: utf-8 -*-


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
        'modules.workers.utils',
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.managers',
        'multiprocessing.synchronize',
        'multiprocessing.heap',
        'scipy._lib._ccallback_c',
        'scipy._cyutility',
        'scipy.interpolate._bsplines',
        'scipy.interpolate._fitpack',
        'scipy.interpolate._fitpack2',
        'scipy.sparse.csgraph._validation',
        'scipy.special._ufuncs_cxx',
        # Open3D - géré par le hook personnalisé
        'open3d',
        # Rasterio - modules manquants forcés
        'rasterio',
        'rasterio.sample',
        'rasterio.vrt',
        'rasterio._features', 
        'rasterio.coords',
        'rasterio.transform',
        'rasterio.warp',
        'rasterio.io',
        'rasterio.crs',
        'rasterio._base',
        'rasterio._env',
        'rasterio._err',
        'rasterio._filepath',
        'rasterio._version',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtNetwork'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='photogeoalign',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Optimisation taille
    upx=False,   # UPX peut causer des problèmes GLIBC
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console pour debugging cluster
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Retirer icon pour éviter dépendances graphiques
    # icon=['resources/logo.png'],
)