# Hook personnalisé pour rasterio
# Force l'emballage de TOUS les modules rasterio

# Pas de collect_all car rasterio peut ne pas être importable encore
# On force directement les imports nécessaires

# Modules rasterio essentiels
hiddenimports = [
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
    'rasterio._io',
    'rasterio._shim',
    'rasterio._warp',
    'rasterio._features',
    'rasterio._env',
    'rasterio._err',
    'rasterio._filepath',
    'rasterio._version',
    'rasterio._base',
    'rasterio._io',
    'rasterio._shim',
    'rasterio._warp',
]

# Données rasterio
datas = []

print(f"HOOK RASTERIO SIMPLIFIÉ: {len(hiddenimports)} modules cachés")
