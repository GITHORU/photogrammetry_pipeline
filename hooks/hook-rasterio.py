# Hook personnalisé pour rasterio
# Force l'emballage de TOUS les modules rasterio

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collecter TOUS les modules rasterio
datas, binaries, hiddenimports = collect_all('rasterio')

# Ajouter explicitement les modules manquants
hiddenimports += [
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
]

# Collecter tous les sous-modules rasterio
hiddenimports += collect_submodules('rasterio')

# Forcer l'emballage des données rasterio
datas += [
    ('rasterio', 'rasterio'),
]

print(f"HOOK RASTERIO: {len(hiddenimports)} modules cachés, {len(datas)} données")
