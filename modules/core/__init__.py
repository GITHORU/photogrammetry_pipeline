"""
Module core pour les opérations géodésiques.
Contient tous les modules de base pour la photogrammétrie.
"""

# Import des modules principaux
from . import geodetic_core
from . import geodetic_processing
from . import geodetic_orthoimage_basic
# from . import geodetic_orthoimage_fusion  # Temporairement commenté - rasterio manquant
from . import geodetic_utils
from . import analysis

# Import des fonctions principales pour faciliter l'utilisation
from .geodetic_core import (
    patch_rasterio_essentials,
    process_single_cloud_add_offset,
    process_single_cloud_itrf_to_enu,
    process_single_cloud_deform,
    process_single_cloud_orthoimage
)

from .geodetic_processing import (
    add_offset_to_clouds,
    convert_itrf_to_enu,
    deform_clouds,
    process_single_cloud_for_unified,
    individual_zone_equalization
)

from .geodetic_orthoimage_basic import (
    create_orthoimage_from_pointcloud,
    create_unified_orthoimage_and_dtm,
    merge_orthoimages_and_dtm
)

# from .geodetic_orthoimage_fusion import (
#     process_zone_with_orthos,
#     unified_ortho_mnt_fusion
# )

from .geodetic_utils import (
    calculate_global_histogram_and_quantiles,
    equalize_zone_to_global_quantiles,
    simple_mnt_assembly,
    simple_ortho_assembly
)

from .analysis import (
    run_analysis_pipeline,
    analyze_mnt_comparison,
    analyze_ortho_comparison,
    generate_analysis_report
)

__all__ = [
    # Core functions
    'patch_rasterio_essentials',
    'process_single_cloud_add_offset',
    'process_single_cloud_itrf_to_enu',
    'process_single_cloud_deform',
    'process_single_cloud_orthoimage',
    
    # Processing functions
    'add_offset_to_clouds',
    'convert_itrf_to_enu',
    'deform_clouds',
    'process_single_cloud_for_unified',
    'individual_zone_equalization',
    
    # Basic orthoimage functions
    'create_orthoimage_from_pointcloud',
    'create_unified_orthoimage_and_dtm',
    'merge_orthoimages_and_dtm',
    
    # Fusion orthoimage functions (temporairement commentées)
    # 'process_zone_with_orthos',
    # 'unified_ortho_mnt_fusion',
    
    # Utils functions
    'calculate_global_histogram_and_quantiles',
    'equalize_zone_to_global_quantiles',
    'simple_mnt_assembly',
    'simple_ortho_assembly',
    
    # Analysis functions
    'run_analysis_pipeline',
    'analyze_mnt_comparison',
    'analyze_ortho_comparison',
    'generate_analysis_report'
] 