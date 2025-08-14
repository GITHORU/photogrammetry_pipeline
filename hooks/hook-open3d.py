# Hook personnalisé pour Open3D
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collecte des données Open3D
datas = collect_data_files('open3d')

# Collecte des bibliothèques dynamiques
binaries = collect_dynamic_libs('open3d')

# Imports cachés critiques
hiddenimports = [
    'open3d',
    'open3d.cpu',
    'open3d.cpu.pybind',
    'open3d.geometry',
    'open3d.io',
    'open3d.utility',
    'open3d.visualization',
    'open3d.core',
    'open3d.pipelines',
    'open3d.pipelines.registration',
    'open3d.pipelines.odometry',
    'open3d.pipelines.color_map',
    'open3d.pipelines.integration',
    'open3d.pipelines.slam',
    'open3d.pipelines.structure',
    'open3d.pipelines.t',
    'open3d.pipelines.t.io',
    'open3d.pipelines.t.geometry',
    'open3d.pipelines.t.utility',
    'open3d.pipelines.t.visualization',
    'open3d.pipelines.t.pipelines',
    'open3d.pipelines.t.pipelines.registration',
    'open3d.pipelines.t.pipelines.odometry',
    'open3d.pipelines.t.pipelines.color_map',
    'open3d.pipelines.t.pipelines.integration',
    'open3d.pipelines.t.pipelines.slam',
    'open3d.pipelines.t.pipelines.structure',
]
