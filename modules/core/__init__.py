# Package core pour PhotoGeoAlign

# Module 1: Transformations géodésiques de base
from .geodetic_core import (
    add_offset_to_clouds,
    convert_itrf_to_enu,
    deform_clouds
)

# Module 2: Traitement des nuages de points
from .geodetic_processing import (
    process_single_cloud_add_offset,
    process_single_cloud_itrf_to_enu,
    process_single_cloud_deform,
    process_single_cloud_for_unified
)

# Module 3: Création et fusion d'orthoimages
from .geodetic_orthoimage import (
    create_orthoimage_from_pointcloud,
    process_single_cloud_orthoimage,
    create_unified_orthoimage_and_dtm,
    merge_orthoimages_and_dtm,
    process_zone_with_orthos,
    unified_ortho_mnt_fusion
)

# Module 4: Utilitaires et patches
from .geodetic_utils import (
    patch_rasterio_essentials,
    calculate_global_histogram_and_quantiles,
    equalize_zone_to_global_quantiles,
    individual_zone_equalization,
    simple_ortho_assembly,
    simple_mnt_assembly
)

# Liste de toutes les fonctions exportées
__all__ = [
    # Module 1: Transformations géodésiques de base
    'add_offset_to_clouds',
    'convert_itrf_to_enu',
    'deform_clouds',
    
    # Module 2: Traitement des nuages de points
    'process_single_cloud_add_offset',
    'process_single_cloud_itrf_to_enu',
    'process_single_cloud_deform',
    'process_single_cloud_for_unified',
    
    # Module 3: Création et fusion d'orthoimages
    'create_orthoimage_from_pointcloud',
    'process_single_cloud_orthoimage',
    'create_unified_orthoimage_and_dtm',
    'merge_orthoimages_and_dtm',
    'process_zone_with_orthos',
    'unified_ortho_mnt_fusion',
    
    # Module 4: Utilitaires et patches
    'patch_rasterio_essentials',
    'calculate_global_histogram_and_quantiles',
    'equalize_zone_to_global_quantiles',
    'individual_zone_equalization',
    'simple_ortho_assembly',
    'simple_mnt_assembly'
] 