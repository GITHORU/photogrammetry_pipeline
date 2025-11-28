#!/usr/bin/env python3
"""
Module d'analyse pour PhotoGeoAlign
Contient les fonctions de calcul et d'analyse pour les images MNT et ortho
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
import itertools
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr, kendalltau
import cv2  # requis (opencv-python[-headless])
try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

def load_raster_data(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Charge les données raster depuis un fichier
    
    Args:
        file_path: Chemin vers le fichier raster
        
    Returns:
        Tuple contenant les données et les métadonnées
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Lecture de la première bande
            
            # Gestion des nodata : convertir en NaN
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
                logger.info(f"Nodata {src.nodata} convertis en NaN pour {os.path.basename(file_path)}")
            
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'nodata': src.nodata,
                'dtype': src.dtypes[0]
            }
            logger.info(f"Données chargées depuis {file_path}: {data.shape}")
            return data, metadata
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {file_path}: {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
        raise

def create_common_grid_and_reproject(image1_path: str, image2_path: str, 
                                   resolution: float, output_dir: str) -> Tuple[str, str]:
    """
    Crée une grille commune aux deux images et les reprojette dans cette grille
    
    Args:
        image1_path: Chemin vers la première image
        image2_path: Chemin vers la deuxième image
        resolution: Résolution de la grille commune en mètres
        output_dir: Dossier de sortie
        
    Returns:
        Tuple des chemins vers les images reprojetées
    """
    logger.info(f"Création d'une grille commune à la résolution {resolution} m")
    
    # Chargement des métadonnées des deux images
    with rasterio.open(image1_path) as src1:
        bounds1 = src1.bounds
        crs1 = src1.crs
        transform1 = src1.transform
    
    with rasterio.open(image2_path) as src2:
        bounds2 = src2.bounds
        crs2 = src2.crs
        transform2 = src2.transform
    
    # Vérification que les deux images ont le même CRS
    if crs1 != crs2:
        logger.warning(f"CRS différents : {crs1} vs {crs2}")
        logger.info("Reprojection de l'image 2 vers le CRS de l'image 1")
    
    # Calcul de l'étendue commune (union des bounds)
    min_x = float(min(bounds1.left, bounds2.left))
    min_y = float(min(bounds1.bottom, bounds2.bottom))
    max_x = float(max(bounds1.right, bounds2.right))
    max_y = float(max(bounds1.top, bounds2.top))
    
    logger.info(f"Étendue commune : X[{float(min_x):.2f}, {float(max_x):.2f}], Y[{float(min_y):.2f}, {float(max_y):.2f}]")
    
    # Calcul des dimensions de la grille commune
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    
    logger.info(f"Dimensions de la grille commune : {width} x {height} pixels")
    
    # Création de la transform pour la grille commune
    common_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Chemins de sortie
    image1_reprojected = os.path.join(output_dir, "image1_common_grid.tif")
    image2_reprojected = os.path.join(output_dir, "image2_common_grid.tif")
    
    # Reprojection de l'image 1
    logger.info("Reprojection de l'image 1 vers la grille commune")
    with rasterio.open(image1_path) as src:
        # Vérifier le nombre de bandes
        num_bands = src.count
        logger.info(f"Image 1 : {num_bands} bande(s)")
        
        if num_bands >= 3:
            # Image RGB - traiter les 3 canaux
            reprojected_data1 = np.zeros((3, height, width), dtype=np.float32)
            for band_idx in range(1, 4):  # Bandes 1, 2, 3 (R, G, B)
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=reprojected_data1[band_idx-1],
                    src_crs=crs1,
                    dst_crs=crs1,
                    dst_transform=common_transform,
                    dst_width=width,
                    dst_height=height,
                    resampling=Resampling.bilinear
                )
            # Sauvegarde du résultat RGB
            with rasterio.open(image1_reprojected, 'w', driver='GTiff',
                              width=width, height=height, count=3,
                              dtype=reprojected_data1.dtype, crs=crs1, transform=common_transform) as dst:
                dst.write(reprojected_data1)
        else:
            # Image en niveaux de gris - traiter une seule bande
            reprojected_data1 = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data1,
                src_crs=crs1,
                dst_crs=crs1,
                dst_transform=common_transform,
                dst_width=width,
                dst_height=height,
                resampling=Resampling.bilinear
            )
            # Sauvegarde du résultat en niveaux de gris
            with rasterio.open(image1_reprojected, 'w', driver='GTiff',
                              width=width, height=height, count=1,
                              dtype=reprojected_data1.dtype, crs=crs1, transform=common_transform) as dst:
                dst.write(reprojected_data1, 1)
    
    # Reprojection de l'image 2
    logger.info("Reprojection de l'image 2 vers la grille commune")
    with rasterio.open(image2_path) as src:
        # Vérifier le nombre de bandes
        num_bands = src.count
        logger.info(f"Image 2 : {num_bands} bande(s)")
        
        if num_bands >= 3:
            # Image RGB - traiter les 3 canaux
            reprojected_data2 = np.zeros((3, height, width), dtype=np.float32)
            for band_idx in range(1, 4):  # Bandes 1, 2, 3 (R, G, B)
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=reprojected_data2[band_idx-1],
                    src_crs=crs2,
                    dst_crs=crs1,  # Utilise le CRS de l'image 1
                    dst_transform=common_transform,
                    dst_width=width,
                    dst_height=height,
                    resampling=Resampling.bilinear
                )
            # Sauvegarde du résultat RGB
            with rasterio.open(image2_reprojected, 'w', driver='GTiff',
                              width=width, height=height, count=3,
                              dtype=reprojected_data2.dtype, crs=crs1, transform=common_transform) as dst:
                dst.write(reprojected_data2)
        else:
            # Image en niveaux de gris - traiter une seule bande
            reprojected_data2 = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data2,
                src_crs=crs2,
                dst_crs=crs1,  # Utilise le CRS de l'image 1
                dst_transform=common_transform,
                dst_width=width,
                dst_height=height,
                resampling=Resampling.bilinear
            )
            # Sauvegarde du résultat en niveaux de gris
            with rasterio.open(image2_reprojected, 'w', driver='GTiff',
                              width=width, height=height, count=1,
                              dtype=reprojected_data2.dtype, crs=crs1, transform=common_transform) as dst:
                dst.write(reprojected_data2, 1)
    
    logger.info(f"Images reprojetées sauvegardées :")
    logger.info(f"  - Image 1 : {image1_reprojected}")
    logger.info(f"  - Image 2 : {image2_reprojected}")
    
    return image1_reprojected, image2_reprojected

def resample_to_common_resolution(data1: np.ndarray, data2: np.ndarray, 
                                 metadata1: Dict[str, Any], metadata2: Dict[str, Any],
                                 target_resolution: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Remet à l'échelle deux rasters à la même résolution (pour MNT)
    
    Args:
        data1: Données du premier raster
        data2: Données du deuxième raster
        metadata1: Métadonnées du premier raster
        metadata2: Métadonnées du deuxième raster
        target_resolution: Résolution cible en mètres
        
    Returns:
        Tuple des deux rasters remis à l'échelle et des métadonnées communes
    """
    logger.info(f"Remise à l'échelle à la résolution {target_resolution} m")
    
    # Calcul des nouvelles dimensions
    bounds1 = rasterio.transform.array_bounds(data1.shape[0], data1.shape[1], metadata1['transform'])
    bounds2 = rasterio.transform.array_bounds(data2.shape[0], data2.shape[1], metadata2['transform'])
    
    # Union des bounds
    min_x = min(bounds1[0], bounds2[0])
    min_y = min(bounds1[1], bounds2[1])
    max_x = max(bounds1[2], bounds2[2])
    max_y = max(bounds1[3], bounds2[3])
    
    # Nouvelle transform
    width = int((max_x - min_x) / target_resolution)
    height = int((max_y - min_y) / target_resolution)
    new_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Remise à l'échelle avec nodata par défaut (pas de 0 artificiels)
    resampled1 = np.full((height, width), np.nan, dtype=np.float32)
    resampled2 = np.full((height, width), np.nan, dtype=np.float32)
    
    # Reprojection avec average (moyenne pondérée, avec nodata explicites)
    reproject(
        source=data1,
        destination=resampled1,
        src_transform=metadata1['transform'],
        src_crs=metadata1['crs'],
        dst_transform=new_transform,
        dst_crs=metadata1['crs'],
        dst_width=width,
        dst_height=height,
        resampling=Resampling.average,
        src_nodata=metadata1.get('nodata', np.nan),  # Nodata source explicite
        dst_nodata=np.nan  # Nodata destination explicite
    )
    
    reproject(
        source=data2,
        destination=resampled2,
        src_transform=metadata2['transform'],
        src_crs=metadata2['crs'],
        dst_transform=new_transform,
        dst_crs=metadata2['crs'],
        dst_width=width,
        dst_height=height,
        resampling=Resampling.average,
        src_nodata=metadata2.get('nodata', np.nan),  # Nodata source explicite
        dst_nodata=np.nan  # Nodata destination explicite
    )
    
    # Métadonnées communes
    common_metadata = {
        'transform': new_transform,
        'crs': metadata1['crs'],  # On utilise le CRS du premier raster
        'width': width,
        'height': height,
        'resolution': target_resolution,
        'bounds': (min_x, min_y, max_x, max_y)
    }
    
    logger.info(f"Rasters remis à l'échelle: {resampled1.shape}")
    return resampled1, resampled2, common_metadata

def save_resampled_mnts(resampled1: np.ndarray, resampled2: np.ndarray, 
                       common_metadata: Dict[str, Any], image1_path: str, 
                       image2_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Sauvegarde les MNTs remis à l'échelle sur la grille commune
    
    Args:
        resampled1: Premier MNT remis à l'échelle
        resampled2: Deuxième MNT remis à l'échelle
        common_metadata: Métadonnées communes
        image1_path: Chemin du premier MNT original
        image2_path: Chemin du deuxième MNT original
        output_dir: Dossier de sortie
        
    Returns:
        Tuple des chemins des fichiers sauvegardés
    """
    logger.info("Sauvegarde des MNTs remis à l'échelle...")
    
    # Noms des fichiers de sortie (ajouter _1 et _2 pour éviter les conflits)
    base1 = os.path.splitext(os.path.basename(image1_path))[0]
    base2 = os.path.splitext(os.path.basename(image2_path))[0]
    
    # Si les noms de base sont identiques, utiliser un suffixe commun avec _1 et _2
    if base1 == base2:
        base_common = base1
        output1_path = os.path.join(output_dir, f"{base_common}_resampled_{common_metadata['resolution']}m_1.tif")
        output2_path = os.path.join(output_dir, f"{base_common}_resampled_{common_metadata['resolution']}m_2.tif")
    else:
        output1_path = os.path.join(output_dir, f"{base1}_resampled_{common_metadata['resolution']}m.tif")
        output2_path = os.path.join(output_dir, f"{base2}_resampled_{common_metadata['resolution']}m.tif")
    
    # Sauvegarde du premier MNT avec nodata standard
    with rasterio.open(
        output1_path,
        'w',
        driver='GTiff',
        height=common_metadata['height'],
        width=common_metadata['width'],
        count=1,
        dtype=resampled1.dtype,
        crs=common_metadata['crs'],
        transform=common_metadata['transform'],
        nodata=-9999.0
    ) as dst:
        # Convertir NaN en valeur nodata standard
        data_to_write = np.where(np.isnan(resampled1), -9999.0, resampled1)
        dst.write(data_to_write, 1)
    
    # Sauvegarde du deuxième MNT avec nodata standard
    with rasterio.open(
        output2_path,
        'w',
        driver='GTiff',
        height=common_metadata['height'],
        width=common_metadata['width'],
        count=1,
        dtype=resampled2.dtype,
        crs=common_metadata['crs'],
        transform=common_metadata['transform'],
        nodata=-9999.0
    ) as dst:
        # Convertir NaN en valeur nodata standard
        data_to_write = np.where(np.isnan(resampled2), -9999.0, resampled2)
        dst.write(data_to_write, 1)
    
    logger.info(f"MNTs sauvegardés:")
    logger.info(f"  - {output1_path}")
    logger.info(f"  - {output2_path}")
    
    return output1_path, output2_path

def create_vertical_displacement_map(mnt1: np.ndarray, mnt2: np.ndarray, 
                                   common_metadata: Dict[str, Any], 
                                   image1_path: str, image2_path: str, 
                                   output_dir: str) -> str:
    """
    Crée une carte de déplacement vertical entre deux MNTs
    
    Args:
        mnt1: Premier MNT (référence)
        mnt2: Deuxième MNT (comparaison)
        common_metadata: Métadonnées communes
        image1_path: Chemin du premier MNT original
        image2_path: Chemin du deuxième MNT original
        output_dir: Dossier de sortie
        
    Returns:
        Chemin vers la carte de déplacement vertical
    """
    logger.info("Création de la carte de déplacement vertical...")
    
    # Calcul des déplacements verticaux (mnt2 - mnt1)
    # Positif = élévation, Négatif = subsidence
    displacement = mnt2 - mnt1
    
    # Masque des valeurs valides (exclure les nodata)
    valid_mask = ~(np.isnan(mnt1) | np.isnan(mnt2))
    displacement_clean = np.where(valid_mask, displacement, np.nan)
    
    # Statistiques des déplacements
    valid_displacements = displacement_clean[valid_mask]
    if len(valid_displacements) > 0:
        min_disp = np.min(valid_displacements)
        max_disp = np.max(valid_displacements)
        mean_disp = np.mean(valid_displacements)
        std_disp = np.std(valid_displacements)
        
        logger.info(f"Statistiques des déplacements verticaux:")
        logger.info(f"  - Minimum: {min_disp:.3f} m")
        logger.info(f"  - Maximum: {max_disp:.3f} m")
        logger.info(f"  - Moyenne: {mean_disp:.3f} m")
        logger.info(f"  - Écart-type: {std_disp:.3f} m")
        logger.info(f"  - Nombre de pixels valides: {len(valid_displacements)}")
    
    # Nom du fichier de sortie
    base1 = os.path.splitext(os.path.basename(image1_path))[0]
    base2 = os.path.splitext(os.path.basename(image2_path))[0]
    displacement_path = os.path.join(output_dir, f"vertical_displacement_{base1}_vs_{base2}_{common_metadata['resolution']}m.tif")
    
    # Sauvegarde de la carte de déplacement
    with rasterio.open(
        displacement_path,
        'w',
        driver='GTiff',
        height=common_metadata['height'],
        width=common_metadata['width'],
        count=1,
        dtype=displacement_clean.dtype,
        crs=common_metadata['crs'],
        transform=common_metadata['transform'],
        nodata=-9999.0
    ) as dst:
        # Convertir NaN en valeur nodata standard
        data_to_write = np.where(np.isnan(displacement_clean), -9999.0, displacement_clean)
        dst.write(data_to_write, 1)
        
        # Ajouter des métadonnées descriptives
        dst.update_tags(
            Title="Carte de Déplacement Vertical",
            Description=f"Déplacement vertical entre {base1} et {base2}",
            Source1=os.path.basename(image1_path),
            Source2=os.path.basename(image2_path),
            Resolution=f"{common_metadata['resolution']}m",
            Unit="meters",
            Positive="élévation",
            Negative="subsidence",
            Min_Displacement=f"{min_disp:.3f}m" if len(valid_displacements) > 0 else "N/A",
            Max_Displacement=f"{max_disp:.3f}m" if len(valid_displacements) > 0 else "N/A",
            Mean_Displacement=f"{mean_disp:.3f}m" if len(valid_displacements) > 0 else "N/A",
            Std_Displacement=f"{std_disp:.3f}m" if len(valid_displacements) > 0 else "N/A",
            Valid_Pixels=str(len(valid_displacements)) if len(valid_displacements) > 0 else "0"
        )
    
    logger.info(f"Carte de déplacement vertical sauvegardée: {displacement_path}")
    return displacement_path

def analyze_mnt_comparison(mnt1: np.ndarray, mnt2: np.ndarray, 
                          resolution: float) -> Dict[str, Any]:
    """
    Analyse comparative de deux MNT
    
    Args:
        mnt1: Premier MNT
        mnt2: Deuxième MNT
        resolution: Résolution en mètres
        
    Returns:
        Dictionnaire contenant les résultats d'analyse
    """
    logger.info("Début de l'analyse comparative MNT")
    
    # Masque des valeurs valides (exclut seulement les nodata)
    valid_mask = ~(np.isnan(mnt1) | np.isnan(mnt2) | (mnt1 == -9999.0) | (mnt2 == -9999.0))
    
    if not np.any(valid_mask):
        logger.warning("Aucune donnée valide trouvée pour l'analyse")
        return {}
    
    mnt1_valid = mnt1[valid_mask]
    mnt2_valid = mnt2[valid_mask]
    
    # Calcul des statistiques de base (MNT2 - MNT1 comme indiqué dans le rapport)
    diff = mnt2_valid - mnt1_valid
    abs_diff = np.abs(diff)
    
    results = {
        'mean_diff': np.mean(diff),
        'median_diff': np.median(diff),
        'std_diff': np.std(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mae': np.mean(abs_diff),
        'max_diff': np.max(diff),
        'min_diff': np.min(diff),
        'correlation_pearson': pearsonr(mnt1_valid, mnt2_valid)[0],
        'correlation_spearman': spearmanr(mnt1_valid, mnt2_valid)[0],
        'correlation_kendall': kendalltau(mnt1_valid, mnt2_valid)[0],
        'n_points': len(mnt1_valid),
        'resolution': resolution
    }
    
    # Calcul des percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    results['percentiles_diff'] = {f'p{p}': np.percentile(diff, p) for p in percentiles}
    
    logger.info(f"Analyse MNT terminée: RMSE={float(results['rmse']):.3f}m, Corrélation={float(results['correlation_pearson']):.3f}")
    return results

def adapt_farneback_params(resolution: float, base_config: dict, base_resolution: float = 0.01) -> dict:
    """
    Adapte les paramètres Farneback en fonction de la résolution.
    
    Args:
        resolution: Résolution actuelle en mètres
        base_config: Configuration de référence optimisée pour base_resolution
        base_resolution: Résolution de référence (défaut: 0.01m)
    
    Returns:
        dict: Paramètres adaptés avec winsize calculé dynamiquement
    """
    ratio = base_resolution / resolution
    
    adapted_config = {
        'pyr_scale': base_config['pyr_scale'],  # Constant: 0.8 (structure optimale)
        'levels': base_config['levels'],        # Constant: 5 (robustesse)
        'winsize': max(3, int(base_config['winsize'] * ratio)),  # Adapté selon la résolution
        'iterations': base_config['iterations'],  # Constant: 10
        'poly_n': base_config['poly_n'],        # Constant: 7
        'poly_sigma': base_config['poly_sigma']  # Constant: 1.2
    }
    
    # S'assurer que winsize est impair (requis par OpenCV)
    if adapted_config['winsize'] % 2 == 0:
        adapted_config['winsize'] += 1
    
    return adapted_config


def calculate_displacements_farneback(data1: np.ndarray, data2: np.ndarray,
                                     resolution: float, output_dir: str,
                                     logger: logging.Logger, farneback_params: dict = None,
                                     valid_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    # OpenCV requis
    """
    Calcule les déplacements entre deux images en utilisant la méthode de Farneback
    
    Args:
        data1: Première image (peut être RGB ou niveaux de gris)
        data2: Deuxième image (peut être RGB ou niveaux de gris)
        resolution: Résolution en mètres par pixel
        output_dir: Dossier de sortie
        logger: Logger pour les messages
        farneback_params: Paramètres configurables pour Farneback
        
    Returns:
        Dictionnaire contenant les résultats des déplacements
    """
    logger.info("Calcul des déplacements avec la méthode de Farneback")
    
    try:
        # Conversion en niveaux de gris si nécessaire
        if len(data1.shape) == 3:  # Image RGB
            # Conversion RGB vers niveaux de gris (moyenne pondérée)
            gray1 = cv2.cvtColor(data1.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(data2.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            logger.info("Images RGB converties en niveaux de gris pour Farneback")
        else:  # Image déjà en niveaux de gris
            gray1 = data1.astype(np.uint8)
            gray2 = data2.astype(np.uint8)
            logger.info("Images en niveaux de gris utilisées directement")
        
        # Normalisation des images (0-255)
        gray1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray2 = cv2.normalize(gray2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Paramètres pour Farneback (avec valeurs par défaut)
        default_params = {
            'pyr_scale': 0.5,
            'levels': 1,
            'winsize': 21,
            'iterations': 5,
            'poly_n': 7,
            'poly_sigma': 1.5
        }
        
        # Utilisation des paramètres fournis ou des valeurs par défaut
        params = farneback_params or {}
        pyr_scale = params.get('pyr_scale', default_params['pyr_scale'])
        levels = params.get('levels', default_params['levels'])
        winsize = params.get('winsize', default_params['winsize'])
        iterations = params.get('iterations', default_params['iterations'])
        poly_n = params.get('poly_n', default_params['poly_n'])
        poly_sigma = params.get('poly_sigma', default_params['poly_sigma'])
        flags = 0
        
        logger.info(f"Paramètres Farneback: pyr_scale={pyr_scale}, levels={levels}, winsize={winsize}")
        
        # Calcul du flux optique avec Farneback
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        )
        
        # Séparation des composantes X et Y
        flow_x = flow[:, :, 0]  # Déplacements en X (pixels)
        flow_y = flow[:, :, 1]  # Déplacements en Y (pixels)
        
        # Conversion des déplacements de pixels vers mètres
        displacement_x_m = flow_x * resolution  # mètres
        displacement_y_m = flow_y * resolution  # mètres
        
        # Calcul de l'amplitude 2D des déplacements (X, Y seulement)
        displacement_magnitude = np.sqrt(displacement_x_m**2 + displacement_y_m**2)
        
        # Statistiques des déplacements (exclure nodata et zones 0 des orthos si fourni)
        nan_mask = ~(np.isnan(displacement_x_m) | np.isnan(displacement_y_m))
        if valid_mask is not None:
            combined_valid = nan_mask & valid_mask
        else:
            combined_valid = nan_mask
        valid_displacements_x = displacement_x_m[combined_valid]
        valid_displacements_y = displacement_y_m[combined_valid]
        valid_magnitudes = displacement_magnitude[combined_valid]
        
        # Sauvegarde des cartes de déplacement
        displacement_x_path = os.path.join(output_dir, "displacement_x.tif")
        displacement_y_path = os.path.join(output_dir, "displacement_y.tif")
        displacement_magnitude_path = os.path.join(output_dir, "displacement_magnitude.tif")
        
        # Utiliser les métadonnées de l'image originale pour la sauvegarde
        with rasterio.open(os.path.join(output_dir, "image1_common_grid.tif")) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1)
        
        # Sauvegarde de la carte de déplacement X
        with rasterio.open(displacement_x_path, 'w', **profile) as dst:
            dst.write(displacement_x_m.astype(rasterio.float32), 1)
        
        # Sauvegarde de la carte de déplacement Y
        with rasterio.open(displacement_y_path, 'w', **profile) as dst:
            dst.write(displacement_y_m.astype(rasterio.float32), 1)
        
        # Sauvegarde de la carte d'amplitude
        with rasterio.open(displacement_magnitude_path, 'w', **profile) as dst:
            dst.write(displacement_magnitude.astype(rasterio.float32), 1)
        
        logger.info(f"Cartes de déplacement sauvegardées:")
        logger.info(f"  - Déplacement X: {displacement_x_path}")
        logger.info(f"  - Déplacement Y: {displacement_y_path}")
        logger.info(f"  - Amplitude: {displacement_magnitude_path}")
        
        # Calcul des statistiques
        results = {
            'displacement_x_path': displacement_x_path,
            'displacement_y_path': displacement_y_path,
            'displacement_magnitude_path': displacement_magnitude_path,
            'mean_displacement_x': np.mean(valid_displacements_x),
            'mean_displacement_y': np.mean(valid_displacements_y),
            'mean_displacement_magnitude': np.mean(valid_magnitudes),
            'median_displacement_x': np.median(valid_displacements_x),
            'median_displacement_y': np.median(valid_displacements_y),
            'median_displacement_magnitude': np.median(valid_magnitudes),
            'std_displacement_x': np.std(valid_displacements_x),
            'std_displacement_y': np.std(valid_displacements_y),
            'std_displacement_magnitude': np.std(valid_magnitudes),
            'max_displacement_x': np.max(valid_displacements_x),
            'max_displacement_y': np.max(valid_displacements_y),
            'max_displacement_magnitude': np.max(valid_magnitudes),
            'min_displacement_x': np.min(valid_displacements_x),
            'min_displacement_y': np.min(valid_displacements_y),
            'min_displacement_magnitude': np.min(valid_magnitudes),
            'n_valid_displacements': int(np.sum(combined_valid)),
            'resolution_m_per_pixel': resolution
        }
        
        logger.info(f"Statistiques des déplacements:")
        logger.info(f"  - Déplacement X moyen: {results['mean_displacement_x']:.3f} m")
        logger.info(f"  - Déplacement Y moyen: {results['mean_displacement_y']:.3f} m")
        logger.info(f"  - Amplitude moyenne: {results['mean_displacement_magnitude']:.3f} m")
        logger.info(f"  - Amplitude max: {results['max_displacement_magnitude']:.3f} m")
        logger.info(f"  - Points valides: {results['n_valid_displacements']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des déplacements Farneback: {str(e)}")
        raise

def analyze_ortho_comparison(ortho1_path: str, ortho2_path: str,
                           resolution: float, output_dir: str, farneback_params: dict = None) -> Dict[str, Any]:
    """
    Analyse comparative de deux orthoimages avec création de grille commune
    
    Args:
        ortho1_path: Chemin vers la première orthoimage
        ortho2_path: Chemin vers la deuxième orthoimage
        resolution: Résolution d'analyse en mètres
        output_dir: Dossier de sortie
        farneback_params: Paramètres pour la méthode de Farneback
        
    Returns:
        Dictionnaire contenant les résultats d'analyse
    """
    logger.info("Début de l'analyse comparative orthoimage")
    
    try:
        # Étape 1 : Création de la grille commune et reprojection
        logger.info("Étape 1 : Création de la grille commune")
        ortho1_common, ortho2_common = create_common_grid_and_reproject(
            ortho1_path, ortho2_path, resolution, output_dir
        )
        
        # Étape 2 : Chargement des images reprojetées
        logger.info("Étape 2 : Chargement des images reprojetées")
        with rasterio.open(ortho1_common) as src:
            num_bands1 = src.count
            if num_bands1 >= 3:
                # Image RGB - lire les 3 canaux
                data1 = src.read()  # Lit toutes les bandes (3, height, width)
                logger.info(f"Image 1 chargée : {num_bands1} canaux, forme {data1.shape}")
            else:
                # Image en niveaux de gris
                data1 = src.read(1)  # Lit seulement la première bande
                logger.info(f"Image 1 chargée : {num_bands1} canal, forme {data1.shape}")
            transform1 = src.transform
        
        with rasterio.open(ortho2_common) as src:
            num_bands2 = src.count
            if num_bands2 >= 3:
                # Image RGB - lire les 3 canaux
                data2 = src.read()  # Lit toutes les bandes (3, height, width)
                logger.info(f"Image 2 chargée : {num_bands2} canaux, forme {data2.shape}")
            else:
                # Image en niveaux de gris
                data2 = src.read(1)  # Lit seulement la première bande
                logger.info(f"Image 2 chargée : {num_bands2} canal, forme {data2.shape}")
            transform2 = src.transform
        
        # Vérification que les transforms sont identiques
        if transform1 != transform2:
            logger.warning("Transforms différents après reprojection")
        
        # Vérifier et corriger les tailles si nécessaire
        if len(data1.shape) == 3:
            h1, w1 = data1.shape[1], data1.shape[2]
            h2, w2 = data2.shape[1], data2.shape[2]
        else:
            h1, w1 = data1.shape[0], data1.shape[1]
            h2, w2 = data2.shape[0], data2.shape[1]
        
        if (h1 != h2) or (w1 != w2):
            logger.warning(f"Tailles différentes après reprojection: {data1.shape} vs {data2.shape}")
            logger.info("Rééchantillonnage pour harmoniser les tailles...")
            # Utiliser la taille minimale pour les deux dimensions
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            
            if len(data1.shape) == 3:
                # Image RGB
                from scipy.ndimage import zoom
                zoom_h1 = target_h / h1
                zoom_w1 = target_w / w1
                zoom_h2 = target_h / h2
                zoom_w2 = target_w / w2
                
                data1_resized = np.zeros((data1.shape[0], target_h, target_w), dtype=data1.dtype)
                data2_resized = np.zeros((data2.shape[0], target_h, target_w), dtype=data2.dtype)
                
                for i in range(data1.shape[0]):
                    data1_resized[i] = zoom(data1[i], (zoom_h1, zoom_w1), order=1)
                for i in range(data2.shape[0]):
                    data2_resized[i] = zoom(data2[i], (zoom_h2, zoom_w2), order=1)
                
                data1 = data1_resized
                data2 = data2_resized
            else:
                # Image en niveaux de gris
                from scipy.ndimage import zoom
                zoom_h1 = target_h / h1
                zoom_w1 = target_w / w1
                zoom_h2 = target_h / h2
                zoom_w2 = target_w / w2
                
                data1 = zoom(data1, (zoom_h1, zoom_w1), order=1)
                data2 = zoom(data2, (zoom_h2, zoom_w2), order=1)
            
            logger.info(f"Tailles harmonisées: {data1.shape} et {data2.shape}")
        
        # Masque des valeurs valides (adapté pour RGB et niveaux de gris)
        if len(data1.shape) == 3:  # Image RGB (3, height, width)
            # Pour RGB, vérifier que tous les canaux sont valides
            valid_mask = ~(np.isnan(data1).any(axis=0) | np.isnan(data2).any(axis=0) |
                          (data1 == 0).all(axis=0) | (data2 == 0).all(axis=0))
            grid_width, grid_height = data1.shape[2], data1.shape[1]
        else:  # Image en niveaux de gris (height, width)
            valid_mask = ~(np.isnan(data1) | np.isnan(data2) |
                           (data1 == 0) | (data2 == 0))
            grid_width, grid_height = data1.shape[1], data1.shape[0]
        
        if not np.any(valid_mask):
            logger.warning("Aucune donnée valide trouvée pour l'analyse")
            return {}
        
        # Statistiques de base sur les images reprojetées
        results = {
            'image1_original': ortho1_path,
            'image2_original': ortho2_path,
            'image1_common_grid': ortho1_common,
            'image2_common_grid': ortho2_common,
            'resolution': resolution,
            'grid_width': grid_width,
            'grid_height': grid_height,
            'n_valid_pixels': np.sum(valid_mask),
            'transform_common': transform1,
            'image1_channels': num_bands1,
            'image2_channels': num_bands2
        }
        
        logger.info(f"Grille commune créée : {grid_width} x {grid_height} pixels")
        logger.info(f"Pixels valides : {np.sum(valid_mask)} / {valid_mask.size}")
        logger.info(f"Image 1 : {num_bands1} canal(aux), Image 2 : {num_bands2} canal(aux)")
        
        # Étape 3 : Calcul des déplacements avec Farneback
        logger.info("Étape 3 : Calcul des déplacements avec Farneback")
        displacement_results = calculate_displacements_farneback(
            data1, data2, resolution, output_dir, logger, farneback_params, valid_mask
        )
        
        # Fusion des résultats
        results.update(displacement_results)
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse ortho : {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
        raise

def generate_analysis_report(results: Dict[str, Any], analysis_type: str,
                           image1_path: str, image2_path: str,
                           output_dir: str) -> str:
    """
    Génère un rapport d'analyse
    
    Args:
        results: Résultats de l'analyse
        analysis_type: Type d'analyse ('mnt' ou 'ortho')
        image1_path: Chemin vers la première image
        image2_path: Chemin vers la deuxième image
        output_dir: Dossier de sortie
        
    Returns:
        Chemin vers le rapport généré
    """
    logger.info("Génération du rapport d'analyse")
    
    report_path = os.path.join(output_dir, f"analysis_report_{analysis_type}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'ANALYSE PHOTOGEOALIGN\n")
        f.write("=" * 80 + "\n\n")
        
        # En-tête avec informations générales
        f.write("INFORMATIONS GÉNÉRALES\n")
        f.write("-" * 25 + "\n")
        f.write(f"Type d'analyse: {analysis_type.upper()}\n")
        f.write(f"Date d'analyse: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Résolution d'analyse: {results.get('resolution', 'N/A')} m\n")
        n_points = results.get('n_points', 'N/A')
        n_points_str = f"{n_points:,}" if isinstance(n_points, (int, float)) else str(n_points)
        f.write(f"Nombre de points analysés: {n_points_str}\n\n")
        
        # Fichiers sources
        f.write("FICHIERS SOURCES\n")
        f.write("-" * 18 + "\n")
        f.write(f"Image 1 (référence): {os.path.basename(image1_path)}\n")
        f.write(f"Image 2 (comparaison): {os.path.basename(image2_path)}\n")
        if analysis_type == 'mnt':
            f.write(f"MNT 1 remis à l'échelle: {os.path.basename(results.get('resampled1_path', 'N/A'))}\n")
            f.write(f"MNT 2 remis à l'échelle: {os.path.basename(results.get('resampled2_path', 'N/A'))}\n")
            f.write(f"Carte de déplacement vertical: {os.path.basename(results.get('displacement_map_path', 'N/A'))}\n")
        elif analysis_type == 'mnt_ortho':
            # Pour mnt_ortho, on affiche aussi les MNTs si fournis
            mnt1_src = results.get('mnt1_path', '')
            mnt2_src = results.get('mnt2_path', '')
            if mnt1_src:
                f.write(f"MNT 1 (source): {os.path.basename(mnt1_src)}\n")
            if mnt2_src:
                f.write(f"MNT 2 (source): {os.path.basename(mnt2_src)}\n")
        f.write("\n")
        
        if analysis_type == 'mnt':
            # Section spécifique aux MNTs
            f.write("ANALYSE DES MODÈLES NUMÉRIQUES DE TERRAIN (MNT)\n")
            f.write("-" * 55 + "\n")
            f.write("Cette analyse compare deux MNTs pour identifier les changements topographiques.\n")
            f.write("Les déplacements verticaux sont calculés comme: MNT2 - MNT1\n")
            f.write("• Valeurs positives = Élévation du terrain\n")
            f.write("• Valeurs négatives = Subsidence du terrain\n\n")
            
            f.write("STATISTIQUES DES DÉPLACEMENTS VERTICAUX\n")
            f.write("-" * 40 + "\n")
            f.write("(Calculées sur les données valides, excluant les nodata)\n")
            f.write(f"Moyenne: {results.get('mean_diff', 'N/A')} m\n")
            f.write(f"Médiane: {results.get('median_diff', 'N/A')} m\n")
            f.write(f"Écart-type: {results.get('std_diff', 'N/A')} m\n")
            f.write(f"RMSE: {results.get('rmse', 'N/A')} m\n")
            f.write(f"MAE: {results.get('mae', 'N/A')} m\n")
            f.write(f"Maximum: {results.get('max_diff', 'N/A')} m\n")
            f.write(f"Minimum: {results.get('min_diff', 'N/A')} m\n\n")
            
            # Percentiles
            f.write("PERCENTILES DES DÉPLACEMENTS\n")
            f.write("-" * 30 + "\n")
            percentiles = results.get('percentiles_diff', {})
            if percentiles:
                f.write(f"P1: {percentiles.get('p1', 'N/A')} m\n")
                f.write(f"P5: {percentiles.get('p5', 'N/A')} m\n")
                f.write(f"P10: {percentiles.get('p10', 'N/A')} m\n")
                f.write(f"P25: {percentiles.get('p25', 'N/A')} m\n")
                f.write(f"P50: {percentiles.get('p50', 'N/A')} m\n")
                f.write(f"P75: {percentiles.get('p75', 'N/A')} m\n")
                f.write(f"P90: {percentiles.get('p90', 'N/A')} m\n")
                f.write(f"P95: {percentiles.get('p95', 'N/A')} m\n")
                f.write(f"P99: {percentiles.get('p99', 'N/A')} m\n\n")
            
            f.write("CORRÉLATIONS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Pearson: {results.get('correlation_pearson', 'N/A')}\n")
            f.write(f"Spearman: {results.get('correlation_spearman', 'N/A')}\n")
            f.write(f"Kendall: {results.get('correlation_kendall', 'N/A')}\n\n")
            
        elif analysis_type == 'mnt_ortho':
            # Section MNT (verticaux)
            vertical = results.get('vertical_results', {})
            f.write("ANALYSE MNT (VERTICAL)\n")
            f.write("-" * 25 + "\n")
            f.write("(Calculées sur les données valides, excluant les nodata)\n")
            f.write(f"Moyenne: {vertical.get('mean_diff', 'N/A')} m\n")
            f.write(f"Médiane: {vertical.get('median_diff', 'N/A')} m\n")
            f.write(f"Écart-type: {vertical.get('std_diff', 'N/A')} m\n")
            f.write(f"RMSE: {vertical.get('rmse', 'N/A')} m\n")
            f.write(f"MAE: {vertical.get('mae', 'N/A')} m\n")
            f.write(f"Maximum: {vertical.get('max_diff', 'N/A')} m\n")
            f.write(f"Minimum: {vertical.get('min_diff', 'N/A')} m\n\n")
            # Percentiles
            vperc = vertical.get('percentiles_diff', {})
            if vperc:
                f.write("PERCENTILES (MNT)\n")
                f.write("-" * 20 + "\n")
                f.write(f"P1: {vperc.get('p1', 'N/A')} m\n")
                f.write(f"P5: {vperc.get('p5', 'N/A')} m\n")
                f.write(f"P10: {vperc.get('p10', 'N/A')} m\n")
                f.write(f"P25: {vperc.get('p25', 'N/A')} m\n")
                f.write(f"P50: {vperc.get('p50', 'N/A')} m\n")
                f.write(f"P75: {vperc.get('p75', 'N/A')} m\n")
                f.write(f"P90: {vperc.get('p90', 'N/A')} m\n")
                f.write(f"P95: {vperc.get('p95', 'N/A')} m\n")
                f.write(f"P99: {vperc.get('p99', 'N/A')} m\n\n")
            f.write("CORRÉLATIONS (MNT)\n")
            f.write("-" * 22 + "\n")
            f.write(f"Pearson: {vertical.get('correlation_pearson', 'N/A')}\n")
            f.write(f"Spearman: {vertical.get('correlation_spearman', 'N/A')}\n")
            f.write(f"Kendall: {vertical.get('correlation_kendall', 'N/A')}\n\n")
            
            # Section Ortho (horizontaux)
            horizontal = results.get('horizontal_results', {})
            f.write("ANALYSE ORTHO (HORIZONTAL)\n")
            f.write("-" * 28 + "\n")
            if 'mean_displacement_x' in horizontal:
                f.write(f"Déplacement X moyen: {horizontal.get('mean_displacement_x', 'N/A')} m\n")
                f.write(f"Déplacement Y moyen: {horizontal.get('mean_displacement_y', 'N/A')} m\n")
                f.write(f"Amplitude moyenne: {horizontal.get('mean_displacement_magnitude', 'N/A')} m\n")
                f.write(f"Amplitude max: {horizontal.get('max_displacement_magnitude', 'N/A')} m\n")
                f.write(f"Écart-type X: {horizontal.get('std_displacement_x', 'N/A')} m\n")
                f.write(f"Écart-type Y: {horizontal.get('std_displacement_y', 'N/A')} m\n")
                f.write(f"Points de déplacement valides: {horizontal.get('n_valid_displacements', 'N/A')}\n\n")
            else:
                # Fallback au format différences si présent
                f.write(f"Différence moyenne: {horizontal.get('mean_diff', 'N/A')}\n")
                f.write(f"Écart-type: {horizontal.get('std_diff', 'N/A')}\n")
                f.write(f"RMSE: {horizontal.get('rmse', 'N/A')}\n")
                f.write(f"MAE: {horizontal.get('mae', 'N/A')}\n")
                f.write(f"Différence max: {horizontal.get('max_diff', 'N/A')}\n")
                f.write(f"Différence min: {horizontal.get('min_diff', 'N/A')}\n\n")
            # Corrélations (Afficher seulement si présentes)
            if any(k in horizontal for k in ('correlation_pearson','correlation_spearman','correlation_kendall')):
                f.write("CORRÉLATIONS (ORTHO)\n")
                f.write("-" * 22 + "\n")
                f.write(f"Pearson: {horizontal.get('correlation_pearson', 'N/A')}\n")
                f.write(f"Spearman: {horizontal.get('correlation_spearman', 'N/A')}\n")
                f.write(f"Kendall: {horizontal.get('correlation_kendall', 'N/A')}\n\n")
            
            # Bloc 3D (composantes moyennes + médianes + norme)
            f.write("SYNTHÈSE 3D\n")
            f.write("-" * 10 + "\n")
            f.write(f"dx (moyen): {results.get('displacement_x', 'N/A')} m\n")
            f.write(f"dy (moyen): {results.get('displacement_y', 'N/A')} m\n")
            f.write(f"dz (moyen): {results.get('displacement_z', 'N/A')} m\n")
            f.write(f"Norme 3D (moyenne): {results.get('displacement_3d', 'N/A')} m\n\n")
            f.write(f"dx (médian): {results.get('median_displacement_x', 'N/A')} m\n")
            f.write(f"dy (médian): {results.get('median_displacement_y', 'N/A')} m\n")
            f.write(f"dz (médian): {results.get('median_displacement_z', 'N/A')} m\n")
            f.write(f"Norme 3D (médiane): {results.get('median_displacement_3d', 'N/A')} m\n\n")
            
        else:
            # Section pour les orthoimages
            f.write("ANALYSE DES ORTHOIMAGES\n")
            f.write("-" * 25 + "\n")
            f.write("Cette analyse compare deux orthoimages pour identifier les déplacements.\n\n")
            
            f.write("STATISTIQUES DE DIFFÉRENCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Différence moyenne: {results.get('mean_diff', 'N/A')}\n")
            f.write(f"Écart-type: {results.get('std_diff', 'N/A')}\n")
            f.write(f"RMSE: {results.get('rmse', 'N/A')}\n")
            f.write(f"MAE: {results.get('mae', 'N/A')}\n")
            f.write(f"Différence max: {results.get('max_diff', 'N/A')}\n")
            f.write(f"Différence min: {results.get('min_diff', 'N/A')}\n\n")
            
            f.write("CORRÉLATIONS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Pearson: {results.get('correlation_pearson', 'N/A')}\n")
            f.write(f"Spearman: {results.get('correlation_spearman', 'N/A')}\n")
            f.write(f"Kendall: {results.get('correlation_kendall', 'N/A')}\n\n")
        

        
        # Statistiques de déplacement Farneback (pour orthoimages)
        if 'mean_displacement_x' in results:
            f.write("STATISTIQUES DES DÉPLACEMENTS (FARNEBACK)\n")
            f.write("-" * 45 + "\n")
            f.write(f"Déplacement X moyen: {results.get('mean_displacement_x', 'N/A')} m\n")
            f.write(f"Déplacement Y moyen: {results.get('mean_displacement_y', 'N/A')} m\n")
            f.write(f"Amplitude moyenne: {results.get('mean_displacement_magnitude', 'N/A')} m\n")
            f.write(f"Amplitude max: {results.get('max_displacement_magnitude', 'N/A')} m\n")
            f.write(f"Écart-type X: {results.get('std_displacement_x', 'N/A')} m\n")
            f.write(f"Écart-type Y: {results.get('std_displacement_y', 'N/A')} m\n")
            f.write(f"Points de déplacement valides: {results.get('n_valid_displacements', 'N/A')}\n\n")
        
        # Informations complémentaires
        f.write("INFORMATIONS COMPLÉMENTAIRES\n")
        f.write("-" * 30 + "\n")
        if analysis_type == 'mnt':
            f.write("• Les déplacements verticaux sont calculés comme: MNT2 - MNT1\n")
            f.write("• Valeurs positives: élévation du terrain\n")
            f.write("• Valeurs négatives: subsidence du terrain\n")
            f.write("• Examiner la carte de déplacement vertical pour l'analyse spatiale\n")
        else:
            f.write("• Examiner les cartes de déplacement pour l'analyse spatiale\n")
            f.write("• Vérifier la cohérence des résultats avec les observations terrain\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Rapport généré par PhotoGeoAlign\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Rapport généré: {report_path}")
    return report_path

def analyze_3d_displacement(ortho1_path: str, ortho2_path: str, mnt1_path: str, mnt2_path: str,
                           resolution: float, output_dir: str, farneback_params: dict = None) -> Dict[str, Any]:
    """
    Analyse 3D combinant déplacements horizontaux (orthos) et verticaux (MNTs)
    
    Args:
        ortho1_path: Chemin vers la première orthoimage
        ortho2_path: Chemin vers la deuxième orthoimage
        mnt1_path: Chemin vers le premier MNT
        mnt2_path: Chemin vers le deuxième MNT
        resolution: Résolution d'analyse en mètres
        output_dir: Dossier de sortie
        farneback_params: Paramètres Farneback
        
    Returns:
        Dictionnaire contenant les résultats de l'analyse 3D
    """
    logger.info("Début de l'analyse 3D (MNT + Ortho)")
    
    # Analyse des déplacements horizontaux (orthos)
    logger.info("Analyse des déplacements horizontaux...")
    horizontal_results = run_analysis_pipeline(ortho1_path, ortho2_path, 'ortho', resolution, output_dir, farneback_params, None, None, generate_report=False)
    
    # Analyse des déplacements verticaux (MNTs)
    logger.info("Analyse des déplacements verticaux...")
    vertical_results = run_analysis_pipeline(mnt1_path, mnt2_path, 'mnt', resolution, output_dir, farneback_params, None, None, generate_report=False)
    
    # Combinaison des résultats 3D
    logger.info("Combinaison des résultats 3D...")
    
    # Calcul des déplacements 3D (hypothénuse des déplacements X, Y, Z)
    if 'mean_displacement_x' in horizontal_results and 'mean_displacement_y' in horizontal_results and 'mean_diff' in vertical_results:
        dx = horizontal_results['mean_displacement_x']
        dy = horizontal_results['mean_displacement_y']
        dz = vertical_results['mean_diff']
        
        # Déplacement 3D total
        displacement_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        results_3d = {
            'displacement_3d': displacement_3d,
            'displacement_x': dx,
            'displacement_y': dy,
            'displacement_z': dz,
            'horizontal_results': horizontal_results,
            'vertical_results': vertical_results
        }
        # Médianes des composantes
        results_3d['median_displacement_z'] = vertical_results.get('median_diff', 'N/A')
        results_3d['median_displacement_x'] = horizontal_results.get('median_displacement_x', 'N/A')
        results_3d['median_displacement_y'] = horizontal_results.get('median_displacement_y', 'N/A')
        # Norme 3D médiane (si les trois composantes sont des scalaires numériques)
        mdx = results_3d.get('median_displacement_x')
        mdy = results_3d.get('median_displacement_y')
        mdz = results_3d.get('median_displacement_z')
        def _is_scalar_number(v):
            try:
                float(v)
                return True
            except Exception:
                return False
        if _is_scalar_number(mdx) and _is_scalar_number(mdy) and _is_scalar_number(mdz):
            mdx_f = float(mdx)
            mdy_f = float(mdy)
            mdz_f = float(mdz)
            results_3d['median_displacement_3d'] = float(np.sqrt(mdx_f**2 + mdy_f**2 + mdz_f**2))
        else:
            results_3d['median_displacement_3d'] = 'N/A'
        # Propager résolution et nombre de points pour le rapport
        results_3d['resolution'] = resolution
        # n_points: prioriser MNT (pixels valides), sinon déplacements valides Farneback
        results_3d['n_points'] = (
            vertical_results.get('n_points')
            if isinstance(vertical_results, dict) and 'n_points' in vertical_results
            else horizontal_results.get('n_valid_displacements', 'N/A')
        )
        
        logger.info(f"Déplacement 3D total: {displacement_3d:.3f} m")
        
        return results_3d
    else:
        logger.warning("Impossible de combiner les résultats 3D - données manquantes")
        return {
            'horizontal_results': horizontal_results,
            'vertical_results': vertical_results
        }

def run_analysis_pipeline(image1_path: str, image2_path: str, 
                         analysis_type: str, resolution: float,
                         output_dir: str, farneback_params: dict = None,
                         mnt1_path: str = None, mnt2_path: str = None,
                         generate_report: bool = True) -> Dict[str, Any]:
    """
    Pipeline principal d'analyse
    
    Args:
        image1_path: Chemin vers la première image
        image2_path: Chemin vers la deuxième image
        analysis_type: Type d'analyse ('mnt', 'ortho' ou 'mnt_ortho')
        resolution: Résolution d'analyse en mètres
        output_dir: Dossier de sortie
        farneback_params: Paramètres pour la méthode de Farneback
        mnt1_path: Chemin vers le premier MNT (pour mode mnt_ortho)
        mnt2_path: Chemin vers le deuxième MNT (pour mode mnt_ortho)
        
    Returns:
        Dictionnaire contenant tous les résultats
    """
    logger.info(f"Début du pipeline d'analyse {analysis_type}")
    
    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Configuration de référence optimisée pour 0.01m
        base_config = {
            'pyr_scale': 0.8,
            'levels': 5,
            'winsize': 101,
            'iterations': 10,
            'poly_n': 7,
            'poly_sigma': 1.2
        }
        
        # Adaptation automatique des paramètres selon la résolution
        # Farneback est utilisé pour 'ortho' et 'mnt_ortho' (pour la partie ortho)
        if analysis_type in ('ortho', 'mnt_ortho'):
            if farneback_params is None:
                # Utiliser la configuration de base adaptée
                adapted_params = adapt_farneback_params(resolution, base_config)
                logger.info(f"=== PARAMÈTRES FARNEBACK UTILISÉS ===")
                logger.info(f"Configuration optimisée adaptée automatiquement pour résolution {resolution}m:")
                logger.info(f"  - pyr_scale: {adapted_params['pyr_scale']} (constant)")
                logger.info(f"  - levels: {adapted_params['levels']} (constant)")
                ratio = 0.01 / resolution
                logger.info(f"  - winsize: {adapted_params['winsize']} (adapté: {base_config['winsize']} * {ratio:.2f} = {base_config['winsize'] * ratio:.0f})")
                logger.info(f"  - iterations: {adapted_params['iterations']} (constant)")
                logger.info(f"  - poly_n: {adapted_params['poly_n']} (constant)")
                logger.info(f"  - poly_sigma: {adapted_params['poly_sigma']} (constant)")
                logger.info(f"=====================================")
            else:
                # Utiliser les paramètres fournis mais adapter winsize
                adapted_params = farneback_params.copy()
                adapted_winsize = adapt_farneback_params(resolution, base_config)['winsize']
                adapted_params['winsize'] = adapted_winsize
                logger.info(f"=== PARAMÈTRES FARNEBACK UTILISÉS ===")
                logger.info(f"Paramètres personnalisés avec winsize adapté automatiquement:")
                logger.info(f"  - pyr_scale: {adapted_params['pyr_scale']}")
                logger.info(f"  - levels: {adapted_params['levels']}")
                logger.info(f"  - winsize: {adapted_winsize} (adapté pour résolution {resolution}m)")
                logger.info(f"  - iterations: {adapted_params['iterations']}")
                logger.info(f"  - poly_n: {adapted_params['poly_n']}")
                logger.info(f"  - poly_sigma: {adapted_params['poly_sigma']}")
                logger.info(f"=====================================")
        else:
            # Pour 'mnt', pas besoin de Farneback
            adapted_params = None
        
        # Chargement des données
        logger.info("Chargement des données...")
        data1, metadata1 = load_raster_data(image1_path)
        data2, metadata2 = load_raster_data(image2_path)
        
        # Remise à l'échelle à la même résolution (pour MNT seulement)
        if analysis_type == 'mnt':
            logger.info("Remise à l'échelle des données...")
            resampled1, resampled2, common_metadata = resample_to_common_resolution(
                data1, data2, metadata1, metadata2, resolution
            )
            
            # Sauvegarde des MNTs remis à l'échelle
            resampled1_path, resampled2_path = save_resampled_mnts(
                resampled1, resampled2, common_metadata, 
                image1_path, image2_path, output_dir
            )
        
        # Analyse selon le type
        if analysis_type == 'mnt_ortho':
            # Analyse 3D combinant MNTs et orthos
            logger.info("Lancement de l'analyse 3D...")
            results = analyze_3d_displacement(image1_path, image2_path, mnt1_path, mnt2_path, 
                                            resolution, output_dir, adapted_params)
            # Exposer les chemins des MNTs pour le rapport
            results['mnt1_path'] = mnt1_path
            results['mnt2_path'] = mnt2_path
        elif analysis_type == 'mnt':
            # Création de la carte de déplacement vertical
            displacement_path = create_vertical_displacement_map(
                resampled1, resampled2, common_metadata, 
                image1_path, image2_path, output_dir
            )
            
            # Analyse comparative
            results = analyze_mnt_comparison(resampled1, resampled2, resolution)
            
            # Ajouter les chemins des fichiers sauvegardés
            results['resampled1_path'] = resampled1_path
            results['resampled2_path'] = resampled2_path
            results['displacement_map_path'] = displacement_path
        elif analysis_type == 'ortho':
            results = analyze_ortho_comparison(image1_path, image2_path, resolution, output_dir, adapted_params)
        else:
            raise ValueError(f"Type d'analyse non supporté: {analysis_type}")
        
        # Génération du rapport (optionnelle)
        if results and generate_report:
            report_path = generate_analysis_report(
                results, analysis_type, image1_path, image2_path, output_dir
            )
            results['report_path'] = report_path
        
        logger.info("Pipeline d'analyse terminé avec succès")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du pipeline d'analyse: {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
        raise

def create_valid_mask_pairwise(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    """
    Crée un masque valide pour une paire d'images.
    Un pixel est valide si les deux images ont des données (pas [0,0,0] ou 0).
    
    Args:
        data1: Première image (peut être RGB ou niveaux de gris)
        data2: Deuxième image (peut être RGB ou niveaux de gris)
        
    Returns:
        Masque booléen (True = valide, False = non valide)
    """
    if len(data1.shape) == 3:  # Image RGB
        # Vérifier que tous les canaux ne sont pas [0,0,0]
        mask1 = ~(data1[0] == 0) | ~(data1[1] == 0) | ~(data1[2] == 0)
        mask2 = ~(data2[0] == 0) | ~(data2[1] == 0) | ~(data2[2] == 0)
    else:  # Image en niveaux de gris
        mask1 = data1 != 0
        mask2 = data2 != 0
    
    # Masque valide = intersection (les deux images doivent avoir des données)
    valid_mask = mask1 & mask2
    
    return valid_mask

def calculate_pairwise_metrics_on_mask(displacement_x: np.ndarray, displacement_y: np.ndarray,
                                     displacement_z: np.ndarray, valid_mask: np.ndarray,
                                     resolution: float) -> Dict[str, Any]:
    """
    Calcule toutes les métriques sur le masque valide.
    
    Args:
        displacement_x: Déplacements en X (m)
        displacement_y: Déplacements en Y (m)
        displacement_z: Déplacements en Z (m)
        valid_mask: Masque valide (True = valide)
        resolution: Résolution en mètres
        
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    # Appliquer le masque
    dx_valid = displacement_x[valid_mask]
    dy_valid = displacement_y[valid_mask]
    dz_valid = displacement_z[valid_mask]
    
    # Logs de débogage
    logger.info(f"Nombre de pixels valides: {len(dx_valid)}")
    logger.info(f"Taille du masque: {valid_mask.shape}, nombre de True: {np.sum(valid_mask)}")
    logger.info(f"Taille des déplacements: X={displacement_x.shape}, Y={displacement_y.shape}, Z={displacement_z.shape}")
    logger.info(f"Nombre de NaN dans dx_valid: {np.sum(np.isnan(dx_valid))}")
    logger.info(f"Nombre de NaN dans dy_valid: {np.sum(np.isnan(dy_valid))}")
    logger.info(f"Nombre de NaN dans dz_valid: {np.sum(np.isnan(dz_valid))}")
    logger.info(f"Valeurs min/max dx_valid: {np.nanmin(dx_valid):.4f} / {np.nanmax(dx_valid):.4f}")
    logger.info(f"Valeurs min/max dy_valid: {np.nanmin(dy_valid):.4f} / {np.nanmax(dy_valid):.4f}")
    logger.info(f"Valeurs min/max dz_valid: {np.nanmin(dz_valid):.4f} / {np.nanmax(dz_valid):.4f}")
    
    # Vérifier qu'il y a des données valides
    if len(dx_valid) == 0:
        logger.warning("Aucun pixel valide dans le masque pour le calcul des métriques")
        # Retourner des métriques avec des valeurs NaN
        return {
            'displacement_3d_mean': float('nan'),
            'displacement_3d_median': float('nan'),
            'displacement_3d_std': float('nan'),
            'displacement_3d_max': float('nan'),
            'displacement_3d_min': float('nan'),
            'displacement_3d_p95': float('nan'),
            'displacement_3d_p99': float('nan'),
            'displacement_x_mean': float('nan'),
            'displacement_x_median': float('nan'),
            'displacement_x_std': float('nan'),
            'displacement_y_mean': float('nan'),
            'displacement_y_median': float('nan'),
            'displacement_y_std': float('nan'),
            'displacement_z_mean': float('nan'),
            'displacement_z_median': float('nan'),
            'displacement_z_std': float('nan'),
            'rmse_3d': float('nan'),
            'mae_3d': float('nan'),
            'correlation_pearson_x': float('nan'),
            'correlation_pearson_y': float('nan'),
            'correlation_pearson_z': float('nan'),
            'correlation_spearman_x': float('nan'),
            'correlation_spearman_y': float('nan'),
            'correlation_spearman_z': float('nan'),
            'n_valid_pixels': 0,
            'coverage_ratio': 0.0,
        }
    
    # Filtrer les NaN avant de calculer les métriques
    # Créer un masque pour les valeurs non-NaN
    valid_data_mask = ~(np.isnan(dx_valid) | np.isnan(dy_valid) | np.isnan(dz_valid))
    
    if np.sum(valid_data_mask) == 0:
        logger.warning("Tous les déplacements sont NaN après application du masque")
        # Retourner des métriques avec des valeurs NaN
        return {
            'displacement_3d_mean': float('nan'),
            'displacement_3d_median': float('nan'),
            'displacement_3d_std': float('nan'),
            'displacement_3d_max': float('nan'),
            'displacement_3d_min': float('nan'),
            'displacement_3d_p95': float('nan'),
            'displacement_3d_p99': float('nan'),
            'displacement_x_mean': float('nan'),
            'displacement_x_median': float('nan'),
            'displacement_x_std': float('nan'),
            'displacement_y_mean': float('nan'),
            'displacement_y_median': float('nan'),
            'displacement_y_std': float('nan'),
            'displacement_z_mean': float('nan'),
            'displacement_z_median': float('nan'),
            'displacement_z_std': float('nan'),
            'rmse_3d': float('nan'),
            'mae_3d': float('nan'),
            'correlation_pearson_x': float('nan'),
            'correlation_pearson_y': float('nan'),
            'correlation_pearson_z': float('nan'),
            'correlation_spearman_x': float('nan'),
            'correlation_spearman_y': float('nan'),
            'correlation_spearman_z': float('nan'),
            'n_valid_pixels': 0,
            'coverage_ratio': 0.0,
        }
    
    # Filtrer les NaN
    dx_clean = dx_valid[valid_data_mask]
    dy_clean = dy_valid[valid_data_mask]
    dz_clean = dz_valid[valid_data_mask]
    
    logger.info(f"Nombre de pixels valides après filtrage NaN: {len(dx_clean)}")
    
    # Calcul de la norme 3D
    displacement_3d = np.sqrt(dx_clean**2 + dy_clean**2 + dz_clean**2)
    
    # Métriques de déplacement 3D (scalaires)
    metrics = {
        'displacement_3d_mean': float(np.mean(displacement_3d)),
        'displacement_3d_median': float(np.median(displacement_3d)),
        'displacement_3d_std': float(np.std(displacement_3d)),
        'displacement_3d_max': float(np.max(displacement_3d)),
        'displacement_3d_min': float(np.min(displacement_3d)),
        'displacement_3d_p95': float(np.percentile(displacement_3d, 95)),
        'displacement_3d_p99': float(np.percentile(displacement_3d, 99)),
    }
    
    # Composantes X
    metrics.update({
        'displacement_x_mean': float(np.mean(dx_clean)),
        'displacement_x_median': float(np.median(dx_clean)),
        'displacement_x_std': float(np.std(dx_clean)),
    })
    
    # Composantes Y
    metrics.update({
        'displacement_y_mean': float(np.mean(dy_clean)),
        'displacement_y_median': float(np.median(dy_clean)),
        'displacement_y_std': float(np.std(dy_clean)),
    })
    
    # Composantes Z
    metrics.update({
        'displacement_z_mean': float(np.mean(dz_clean)),
        'displacement_z_median': float(np.median(dz_clean)),
        'displacement_z_std': float(np.std(dz_clean)),
    })
    
    # Métriques de qualité
    diff_3d = displacement_3d
    metrics.update({
        'rmse_3d': float(np.sqrt(np.mean(diff_3d**2))),
        'mae_3d': float(np.mean(np.abs(diff_3d))),
    })
    
    # Métriques de corrélation
    # Note: Dans le contexte d'une analyse de déplacement paire par paire,
    # les corrélations sont moins pertinentes. On les calcule entre les composantes
    # pour mesurer la cohérence spatiale des déplacements.
    if len(dx_clean) > 1:
        try:
            # Corrélations entre composantes X et Y (cohérence horizontale)
            if np.std(dx_clean) > 0 and np.std(dy_clean) > 0:
                corr_xy = float(pearsonr(dx_clean, dy_clean)[0])
                spearman_xy = float(spearmanr(dx_clean, dy_clean)[0])
            else:
                corr_xy = spearman_xy = 0.0
            
            # Pour X, Y, Z individuellement, on calcule la corrélation avec la norme 3D
            # comme mesure de cohérence
            corr_x = float(pearsonr(dx_clean, displacement_3d)[0]) if np.std(dx_clean) > 0 else 0.0
            corr_y = float(pearsonr(dy_clean, displacement_3d)[0]) if np.std(dy_clean) > 0 else 0.0
            corr_z = float(pearsonr(dz_clean, displacement_3d)[0]) if np.std(dz_clean) > 0 else 0.0
            spearman_x = float(spearmanr(dx_clean, displacement_3d)[0]) if np.std(dx_clean) > 0 else 0.0
            spearman_y = float(spearmanr(dy_clean, displacement_3d)[0]) if np.std(dy_clean) > 0 else 0.0
            spearman_z = float(spearmanr(dz_clean, displacement_3d)[0]) if np.std(dz_clean) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Erreur lors du calcul des corrélations: {e}")
            corr_x = corr_y = corr_z = 0.0
            spearman_x = spearman_y = spearman_z = 0.0
        
        metrics.update({
            'correlation_pearson_x': corr_x,
            'correlation_pearson_y': corr_y,
            'correlation_pearson_z': corr_z,
            'correlation_spearman_x': spearman_x,
            'correlation_spearman_y': spearman_y,
            'correlation_spearman_z': spearman_z,
        })
    else:
        metrics.update({
            'correlation_pearson_x': 0.0,
            'correlation_pearson_y': 0.0,
            'correlation_pearson_z': 0.0,
            'correlation_spearman_x': 0.0,
            'correlation_spearman_y': 0.0,
            'correlation_spearman_z': 0.0,
        })
    
    # Métriques de couverture
    n_valid = int(np.sum(valid_mask))
    n_total = int(valid_mask.size)
    metrics.update({
        'n_valid_pixels': n_valid,
        'coverage_ratio': float(n_valid / n_total) if n_total > 0 else 0.0,
    })
    
    return metrics

def run_pairwise_analysis_pipeline(model_paths: list, analysis_type: str, resolution: float,
                                  output_dir: str, farneback_params: dict = None,
                                  mnt_paths: list = None, parallel: bool = True, max_workers: int = None) -> Dict[str, Any]:
    """
    Pipeline d'analyse paire par paire pour N modèles.
    
    Args:
        model_paths: Liste des chemins vers les modèles (orthos ou MNTs selon analysis_type)
        analysis_type: Type d'analyse ('mnt', 'ortho' ou 'mnt_ortho')
        resolution: Résolution d'analyse en mètres
        output_dir: Dossier de sortie
        farneback_params: Paramètres pour la méthode de Farneback
        mnt_paths: Liste des chemins vers les MNTs (requis si analysis_type='mnt_ortho')
        parallel: Utiliser le parallélisme (défaut: True)
        max_workers: Nombre maximum de workers parallèles (défaut: None = utilise os.cpu_count())
        
    Returns:
        Dictionnaire contenant tous les résultats
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logger.info(f"Début du pipeline d'analyse paire par paire")
    logger.info(f"Nombre de modèles: {len(model_paths)}")
    logger.info(f"Type d'analyse: {analysis_type}")
    logger.info(f"Résolution: {resolution} m")
    
    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Génération de toutes les paires
    n_models = len(model_paths)
    pairs = list(itertools.combinations(range(n_models), 2))
    n_comparisons = len(pairs)
    logger.info(f"Nombre de comparaisons: {n_comparisons}")
    
    # Fonction pour traiter une paire
    def process_pair(pair_idx, i, j):
        model1_path = model_paths[i]
        model2_path = model_paths[j]
        
        # Utiliser les indices au lieu des noms de fichiers
        pair_output_dir = os.path.join(comparisons_dir, f"{i}_vs_{j}")
        os.makedirs(pair_output_dir, exist_ok=True)
        
        logger.info(f"Traitement de la paire {pair_idx+1}/{n_comparisons}: {i} vs {j}")
        
        try:
            # Lancer l'analyse pour cette paire
            if analysis_type == 'mnt_ortho':
                mnt1_path = mnt_paths[i] if mnt_paths else None
                mnt2_path = mnt_paths[j] if mnt_paths else None
                results = run_analysis_pipeline(
                    model1_path, model2_path, analysis_type, resolution,
                    pair_output_dir, farneback_params, mnt1_path, mnt2_path,
                    generate_report=False
                )
            else:
                results = run_analysis_pipeline(
                    model1_path, model2_path, analysis_type, resolution,
                    pair_output_dir, farneback_params, None, None,
                    generate_report=False
                )
            
            # Charger les données pour créer le masque valide
            # Pour l'analyse paire par paire, on utilise les images reprojetées depuis les résultats
            # car elles sont déjà sur la même grille commune
            if analysis_type == 'mnt_ortho':
                # Pour mnt_ortho, utiliser les orthos reprojetées depuis les résultats
                horizontal_results = results.get('horizontal_results', {})
                ortho1_common = horizontal_results.get('image1_common_grid')
                ortho2_common = horizontal_results.get('image2_common_grid')
                
                if ortho1_common and ortho2_common and os.path.exists(ortho1_common) and os.path.exists(ortho2_common):
                    with rasterio.open(ortho1_common) as src1:
                        data1 = src1.read()
                    with rasterio.open(ortho2_common) as src2:
                        data2 = src2.read()
                else:
                    # Fallback : utiliser les fichiers originaux
                    with rasterio.open(model1_path) as src1:
                        data1 = src1.read()
                    with rasterio.open(model2_path) as src2:
                        data2 = src2.read()
            elif analysis_type == 'ortho':
                # Utiliser les images reprojetées depuis les résultats
                ortho1_common = results.get('image1_common_grid')
                ortho2_common = results.get('image2_common_grid')
                
                if ortho1_common and ortho2_common and os.path.exists(ortho1_common) and os.path.exists(ortho2_common):
                    with rasterio.open(ortho1_common) as src1:
                        data1 = src1.read()
                    with rasterio.open(ortho2_common) as src2:
                        data2 = src2.read()
                else:
                    # Fallback : utiliser les fichiers originaux
                    with rasterio.open(model1_path) as src1:
                        data1 = src1.read()
                    with rasterio.open(model2_path) as src2:
                        data2 = src2.read()
            else:  # mnt
                # Pour MNT, utiliser les fichiers remis à l'échelle depuis les résultats
                resampled1_path = results.get('resampled1_path')
                resampled2_path = results.get('resampled2_path')
                
                if resampled1_path and resampled2_path and os.path.exists(resampled1_path) and os.path.exists(resampled2_path):
                    data1, _ = load_raster_data(resampled1_path)
                    data2, _ = load_raster_data(resampled2_path)
                else:
                    # Fallback : utiliser les fichiers originaux
                    data1, _ = load_raster_data(model1_path)
                    data2, _ = load_raster_data(model2_path)
                
                # Convertir en format compatible (ajouter dimension si nécessaire)
                if len(data1.shape) == 2:
                    data1 = data1[np.newaxis, :, :]
                if len(data2.shape) == 2:
                    data2 = data2[np.newaxis, :, :]
            
            # Vérifier et harmoniser les tailles si nécessaire
            if len(data1.shape) == 3:
                h1, w1 = data1.shape[1], data1.shape[2]
                h2, w2 = data2.shape[1], data2.shape[2]
            else:
                h1, w1 = data1.shape[0], data1.shape[1]
                h2, w2 = data2.shape[0], data2.shape[1]
            
            if (h1 != h2) or (w1 != w2):
                logger.warning(f"Tailles différentes pour le masque: {data1.shape} vs {data2.shape}")
                logger.info("Rééchantillonnage pour harmoniser les tailles...")
                target_h = min(h1, h2)
                target_w = min(w1, w2)
                
                if len(data1.shape) == 3:
                    from scipy.ndimage import zoom
                    zoom_h1 = target_h / h1
                    zoom_w1 = target_w / w1
                    zoom_h2 = target_h / h2
                    zoom_w2 = target_w / w2
                    
                    data1_resized = np.zeros((data1.shape[0], target_h, target_w), dtype=data1.dtype)
                    data2_resized = np.zeros((data2.shape[0], target_h, target_w), dtype=data2.dtype)
                    
                    for i in range(data1.shape[0]):
                        data1_resized[i] = zoom(data1[i], (zoom_h1, zoom_w1), order=1)
                    for i in range(data2.shape[0]):
                        data2_resized[i] = zoom(data2[i], (zoom_h2, zoom_w2), order=1)
                    
                    data1 = data1_resized
                    data2 = data2_resized
                else:
                    from scipy.ndimage import zoom
                    zoom_h1 = target_h / h1
                    zoom_w1 = target_w / w1
                    zoom_h2 = target_h / h2
                    zoom_w2 = target_w / w2
                    
                    data1 = zoom(data1, (zoom_h1, zoom_w1), order=1)
                    data2 = zoom(data2, (zoom_h2, zoom_w2), order=1)
            
            # Créer le masque valide
            valid_mask = create_valid_mask_pairwise(data1, data2)
            
            # Extraire les déplacements depuis les résultats
            if analysis_type == 'mnt_ortho':
                # Charger les cartes de déplacement depuis les résultats
                horizontal_results = results.get('horizontal_results', {})
                vertical_results = results.get('vertical_results', {})
                
                displacement_x_path = horizontal_results.get('displacement_x_path')
                displacement_y_path = horizontal_results.get('displacement_y_path')
                
                logger.info(f"Paire {i}-{j}: horizontal_results keys: {list(horizontal_results.keys())}")
                logger.info(f"Paire {i}-{j}: vertical_results keys: {list(vertical_results.keys())}")
                
                if displacement_x_path and os.path.exists(displacement_x_path):
                    with rasterio.open(displacement_x_path) as src:
                        displacement_x = src.read(1)
                    logger.info(f"Paire {i}-{j}: displacement_x chargé depuis {displacement_x_path}, shape={displacement_x.shape}")
                else:
                    logger.warning(f"Paire {i}-{j}: displacement_x_path non trouvé ou invalide: {displacement_x_path}")
                    displacement_x = np.zeros_like(valid_mask, dtype=np.float32)
                
                if displacement_y_path and os.path.exists(displacement_y_path):
                    with rasterio.open(displacement_y_path) as src:
                        displacement_y = src.read(1)
                    logger.info(f"Paire {i}-{j}: displacement_y chargé depuis {displacement_y_path}, shape={displacement_y.shape}")
                else:
                    logger.warning(f"Paire {i}-{j}: displacement_y_path non trouvé ou invalide: {displacement_y_path}")
                    displacement_y = np.zeros_like(valid_mask, dtype=np.float32)
                
                # Pour Z, utiliser les résultats MNT
                resampled1_path = vertical_results.get('resampled1_path')
                resampled2_path = vertical_results.get('resampled2_path')
                
                logger.info(f"Paire {i}-{j}: resampled1_path = {resampled1_path}")
                logger.info(f"Paire {i}-{j}: resampled2_path = {resampled2_path}")
                
                if resampled1_path and resampled2_path and os.path.exists(resampled1_path) and os.path.exists(resampled2_path):
                    mnt1_data, _ = load_raster_data(resampled1_path)
                    mnt2_data, _ = load_raster_data(resampled2_path)
                    displacement_z = mnt2_data - mnt1_data
                    logger.info(f"Paire {i}-{j}: displacement_z calculé, shape={displacement_z.shape}, min={np.nanmin(displacement_z):.4f}, max={np.nanmax(displacement_z):.4f}, mean={np.nanmean(displacement_z):.4f}")
                else:
                    logger.warning(f"Paire {i}-{j}: chemins MNT non trouvés ou invalides. resampled1_path existe: {os.path.exists(resampled1_path) if resampled1_path else False}, resampled2_path existe: {os.path.exists(resampled2_path) if resampled2_path else False}")
                    displacement_z = np.zeros_like(valid_mask, dtype=np.float32)
            elif analysis_type == 'ortho':
                displacement_x_path = results.get('displacement_x_path')
                displacement_y_path = results.get('displacement_y_path')
                
                if displacement_x_path and os.path.exists(displacement_x_path):
                    with rasterio.open(displacement_x_path) as src:
                        displacement_x = src.read(1)
                else:
                    displacement_x = np.zeros_like(valid_mask, dtype=np.float32)
                
                if displacement_y_path and os.path.exists(displacement_y_path):
                    with rasterio.open(displacement_y_path) as src:
                        displacement_y = src.read(1)
                else:
                    displacement_y = np.zeros_like(valid_mask, dtype=np.float32)
                
                displacement_z = np.zeros_like(valid_mask, dtype=np.float32)
            else:  # mnt
                # Pour MNT, on a seulement Z
                resampled1_path = results.get('resampled1_path')
                resampled2_path = results.get('resampled2_path')
                
                if resampled1_path and resampled2_path:
                    mnt1_data, _ = load_raster_data(resampled1_path)
                    mnt2_data, _ = load_raster_data(resampled2_path)
                    displacement_z = mnt2_data - mnt1_data
                else:
                    displacement_z = np.zeros_like(valid_mask, dtype=np.float32)
                
                displacement_x = np.zeros_like(valid_mask, dtype=np.float32)
                displacement_y = np.zeros_like(valid_mask, dtype=np.float32)
            
            # S'assurer que les déplacements ont la même taille que le masque
            # (peut nécessiter un rééchantillonnage si les grilles diffèrent)
            if displacement_x.shape != valid_mask.shape or displacement_y.shape != valid_mask.shape or displacement_z.shape != valid_mask.shape:
                logger.warning(f"Tailles différentes entre masque et déplacements: masque={valid_mask.shape}, dx={displacement_x.shape}, dy={displacement_y.shape}, dz={displacement_z.shape}")
                logger.info("Rééchantillonnage des déplacements pour correspondre au masque...")
                
                target_h, target_w = valid_mask.shape
                
                # Rééchantillonner les déplacements pour correspondre au masque
                from scipy.ndimage import zoom
                
                if displacement_x.shape != valid_mask.shape:
                    zoom_h = target_h / displacement_x.shape[0]
                    zoom_w = target_w / displacement_x.shape[1]
                    displacement_x = zoom(displacement_x, (zoom_h, zoom_w), order=1)
                
                if displacement_y.shape != valid_mask.shape:
                    zoom_h = target_h / displacement_y.shape[0]
                    zoom_w = target_w / displacement_y.shape[1]
                    displacement_y = zoom(displacement_y, (zoom_h, zoom_w), order=1)
                
                if displacement_z.shape != valid_mask.shape:
                    zoom_h = target_h / displacement_z.shape[0]
                    zoom_w = target_w / displacement_z.shape[1]
                    displacement_z = zoom(displacement_z, (zoom_h, zoom_w), order=1)
                
                logger.info(f"Tailles harmonisées: masque={valid_mask.shape}, dx={displacement_x.shape}, dy={displacement_y.shape}, dz={displacement_z.shape}")
            
            # Calculer les métriques sur le masque valide
            metrics = calculate_pairwise_metrics_on_mask(
                displacement_x, displacement_y, displacement_z,
                valid_mask, resolution
            )
            
            # Sauvegarder le masque valide avec les bonnes métadonnées géospatiales
            # Utiliser les métadonnées de l'image reprojetée sur la grille commune
            mask_path = os.path.join(pair_output_dir, 'valid_mask.tif')
            
            # Déterminer quelle image reprojetée utiliser comme référence pour les métadonnées
            reference_image_path = None
            if analysis_type == 'mnt_ortho':
                horizontal_results = results.get('horizontal_results', {})
                reference_image_path = horizontal_results.get('image1_common_grid')
            elif analysis_type == 'ortho':
                reference_image_path = results.get('image1_common_grid')
            elif analysis_type == 'mnt':
                # Pour MNT, utiliser le premier MNT remis à l'échelle
                reference_image_path = results.get('resampled1_path')
            
            # Si on a une image reprojetée, utiliser son profil
            if reference_image_path and os.path.exists(reference_image_path):
                with rasterio.open(reference_image_path) as src:
                    profile = src.profile.copy()
                    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
                    # S'assurer que les dimensions correspondent au masque
                    if profile['height'] != valid_mask.shape[0] or profile['width'] != valid_mask.shape[1]:
                        logger.warning(f"Dimensions du masque ({valid_mask.shape}) ne correspondent pas au profil ({profile['height']}, {profile['width']}). Ajustement...")
                        profile['height'] = valid_mask.shape[0]
                        profile['width'] = valid_mask.shape[1]
                    with rasterio.open(mask_path, 'w', **profile) as dst:
                        dst.write(valid_mask.astype(rasterio.uint8), 1)
                logger.info(f"Masque valide sauvegardé avec les métadonnées de {reference_image_path}")
            else:
                # Fallback : utiliser model1_path (mais avec un avertissement)
                logger.warning(f"Image reprojetée non trouvée, utilisation de model1_path pour les métadonnées. Le masque pourrait ne pas être aligné correctement.")
                with rasterio.open(model1_path) as src:
                    profile = src.profile.copy()
                    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
                    # Ajuster les dimensions si nécessaire
                    if profile['height'] != valid_mask.shape[0] or profile['width'] != valid_mask.shape[1]:
                        profile['height'] = valid_mask.shape[0]
                        profile['width'] = valid_mask.shape[1]
                    with rasterio.open(mask_path, 'w', **profile) as dst:
                        dst.write(valid_mask.astype(rasterio.uint8), 1)
            
            # Sauvegarder les cartes de déplacement avec les mêmes métadonnées que le masque
            # Déterminer le profil à utiliser (même que pour le masque)
            if reference_image_path and os.path.exists(reference_image_path):
                with rasterio.open(reference_image_path) as src:
                    displacement_profile = src.profile.copy()
                    displacement_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
                    if displacement_profile['height'] != valid_mask.shape[0] or displacement_profile['width'] != valid_mask.shape[1]:
                        displacement_profile['height'] = valid_mask.shape[0]
                        displacement_profile['width'] = valid_mask.shape[1]
            else:
                # Fallback : utiliser model1_path
                with rasterio.open(model1_path) as src:
                    displacement_profile = src.profile.copy()
                    displacement_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
                    if displacement_profile['height'] != valid_mask.shape[0] or displacement_profile['width'] != valid_mask.shape[1]:
                        displacement_profile['height'] = valid_mask.shape[0]
                        displacement_profile['width'] = valid_mask.shape[1]
            
            # Sauvegarder displacement_x.tif
            displacement_x_path = os.path.join(pair_output_dir, 'displacement_x.tif')
            with rasterio.open(displacement_x_path, 'w', **displacement_profile) as dst:
                dst.write(displacement_x.astype(rasterio.float32), 1)
            
            # Sauvegarder displacement_y.tif
            displacement_y_path = os.path.join(pair_output_dir, 'displacement_y.tif')
            with rasterio.open(displacement_y_path, 'w', **displacement_profile) as dst:
                dst.write(displacement_y.astype(rasterio.float32), 1)
            
            # Sauvegarder displacement_z.tif
            displacement_z_path = os.path.join(pair_output_dir, 'displacement_z.tif')
            with rasterio.open(displacement_z_path, 'w', **displacement_profile) as dst:
                dst.write(displacement_z.astype(rasterio.float32), 1)
            
            # Calculer et sauvegarder la magnitude 3D (X, Y, Z)
            displacement_magnitude_3d = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
            displacement_magnitude_path = os.path.join(pair_output_dir, 'displacement_magnitude_3d.tif')
            with rasterio.open(displacement_magnitude_path, 'w', **displacement_profile) as dst:
                dst.write(displacement_magnitude_3d.astype(rasterio.float32), 1)
            
            logger.info(f"Cartes de déplacement sauvegardées pour la paire {i}-{j}")
            
            # Ajouter les métriques aux résultats
            results['pairwise_metrics'] = metrics
            results['valid_mask_path'] = mask_path
            results['displacement_x_path'] = displacement_x_path
            results['displacement_y_path'] = displacement_y_path
            results['displacement_z_path'] = displacement_z_path
            results['displacement_magnitude_path'] = displacement_magnitude_path
            results['model1_index'] = i
            results['model2_index'] = j
            results['model1_path'] = os.path.abspath(model1_path)
            results['model2_path'] = os.path.abspath(model2_path)
            
            return {
                'pair_index': pair_idx,
                'model1_index': i,
                'model2_index': j,
                'model1_path': os.path.abspath(model1_path),
                'model2_path': os.path.abspath(model2_path),
                'results': results,
                'metrics': metrics,
                'output_dir': pair_output_dir
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la paire {i}-{j}: {str(e)}")
            return {
                'pair_index': pair_idx,
                'model1_index': i,
                'model2_index': j,
                'error': str(e)
            }
    
    # Traitement des paires (parallèle ou séquentiel)
    all_comparisons = []
    if parallel:
        if max_workers is None:
            max_workers = min(os.cpu_count() or 1, n_comparisons)
        else:
            max_workers = min(max_workers, n_comparisons)  # Ne pas dépasser le nombre de comparaisons
        logger.info(f"Traitement parallèle avec {max_workers} workers (sur {n_comparisons} comparaisons)")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pair, idx, i, j): (idx, i, j) 
                      for idx, (i, j) in enumerate(pairs)}
            
            for future in as_completed(futures):
                result = future.result()
                all_comparisons.append(result)
                pair_idx, i, j = futures[future]
                logger.info(f"Paire {pair_idx+1}/{n_comparisons} terminée: {i}-{j}")
    else:
        logger.info("Traitement séquentiel")
        for idx, (i, j) in enumerate(pairs):
            result = process_pair(idx, i, j)
            all_comparisons.append(result)
    
    # Trier par index de paire
    all_comparisons.sort(key=lambda x: x['pair_index'])
    
    # Créer le fichier de mapping numéro -> chemin absolu
    import json
    mapping_file = os.path.join(output_dir, 'model_mapping.json')
    model_mapping = {}
    for idx, path in enumerate(model_paths):
        abs_path = os.path.abspath(path)
        model_mapping[str(idx)] = abs_path
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(model_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"Fichier de mapping créé: {mapping_file}")
    
    # Créer aussi un fichier CSV pour faciliter la lecture
    mapping_csv = os.path.join(output_dir, 'model_mapping.csv')
    with open(mapping_csv, 'w', encoding='utf-8') as f:
        f.write("index,chemin_absolu\n")
        for idx, path in enumerate(model_paths):
            abs_path = os.path.abspath(path)
            f.write(f"{idx},{abs_path}\n")
    logger.info(f"Fichier de mapping CSV créé: {mapping_csv}")
    
    # Agrégation des résultats
    logger.info("Agrégation des résultats...")
    
    # Créer les matrices de déplacements pour différentes métriques
    displacement_matrix_mean = np.full((n_models, n_models), np.nan)
    displacement_matrix_median = np.full((n_models, n_models), np.nan)
    displacement_matrix_std = np.full((n_models, n_models), np.nan)
    displacement_matrix_max = np.full((n_models, n_models), np.nan)
    displacement_matrix_p95 = np.full((n_models, n_models), np.nan)
    displacement_matrix_p99 = np.full((n_models, n_models), np.nan)
    
    # Matrices pour les composantes individuelles (moyennes)
    displacement_x_matrix_mean = np.full((n_models, n_models), np.nan)
    displacement_y_matrix_mean = np.full((n_models, n_models), np.nan)
    displacement_z_matrix_mean = np.full((n_models, n_models), np.nan)
    
    # Matrices pour les composantes individuelles (médianes)
    displacement_x_matrix_median = np.full((n_models, n_models), np.nan)
    displacement_y_matrix_median = np.full((n_models, n_models), np.nan)
    displacement_z_matrix_median = np.full((n_models, n_models), np.nan)
    
    # Matrices pour les composantes individuelles (écart-type)
    displacement_x_matrix_std = np.full((n_models, n_models), np.nan)
    displacement_y_matrix_std = np.full((n_models, n_models), np.nan)
    displacement_z_matrix_std = np.full((n_models, n_models), np.nan)
    
    # Matrices de qualité
    rmse_matrix = np.full((n_models, n_models), np.nan)
    mae_matrix = np.full((n_models, n_models), np.nan)
    
    # Matrice de couverture
    coverage_matrix = np.full((n_models, n_models), np.nan)
    
    for comp in all_comparisons:
        if 'error' not in comp:
            i = comp['model1_index']
            j = comp['model2_index']
            # Les métriques sont stockées directement dans 'metrics'
            metrics = comp.get('metrics', {})
            
            # Si les métriques ne sont pas dans 'metrics', chercher dans 'results'['pairwise_metrics']
            if not metrics and 'results' in comp:
                metrics = comp['results'].get('pairwise_metrics', {})
            
            if not metrics:
                logger.warning(f"Aucune métrique trouvée pour la paire {i}-{j}. Clés disponibles: {list(comp.keys())}")
                if 'results' in comp:
                    logger.warning(f"Clés dans results: {list(comp['results'].keys())}")
                continue
            
            logger.info(f"Paire {i}-{j}: métriques disponibles: {list(metrics.keys())[:10]}...")  # Afficher les 10 premières clés
            
            # Vérifier que les métriques ne sont pas toutes NaN
            if metrics:
                sample_key = list(metrics.keys())[0]
                sample_value = metrics.get(sample_key)
                if sample_value is None or (isinstance(sample_value, float) and np.isnan(sample_value)):
                    logger.warning(f"Paire {i}-{j}: les métriques semblent être NaN ou None. Exemple: {sample_key}={sample_value}")
            
            # Adapter selon le type d'analyse
            if 'displacement_3d_mean' in metrics:
                # Analyse mnt_ortho : utiliser le déplacement 3D
                mean_val = metrics.get('displacement_3d_mean')
                median_val = metrics.get('displacement_3d_median')
                std_val = metrics.get('displacement_3d_std')
                max_val = metrics.get('displacement_3d_max')
                p95_val = metrics.get('displacement_3d_p95')
                p99_val = metrics.get('displacement_3d_p99')
                
                # Remplir les matrices 3D
                if not (isinstance(mean_val, float) and np.isnan(mean_val)):
                    displacement_matrix_mean[i, j] = mean_val
                    displacement_matrix_mean[j, i] = mean_val
                    logger.info(f"Paire {i}-{j}: déplacement 3D moyen = {mean_val:.4f} m")
                
                if not (isinstance(median_val, float) and np.isnan(median_val)):
                    displacement_matrix_median[i, j] = median_val
                    displacement_matrix_median[j, i] = median_val
                
                if not (isinstance(std_val, float) and np.isnan(std_val)):
                    displacement_matrix_std[i, j] = std_val
                    displacement_matrix_std[j, i] = std_val
                
                if not (isinstance(max_val, float) and np.isnan(max_val)):
                    displacement_matrix_max[i, j] = max_val
                    displacement_matrix_max[j, i] = max_val
                
                if not (isinstance(p95_val, float) and np.isnan(p95_val)):
                    displacement_matrix_p95[i, j] = p95_val
                    displacement_matrix_p95[j, i] = p95_val
                
                if not (isinstance(p99_val, float) and np.isnan(p99_val)):
                    displacement_matrix_p99[i, j] = p99_val
                    displacement_matrix_p99[j, i] = p99_val
                
                # Remplir les matrices des composantes (moyennes)
                dx_mean = metrics.get('displacement_x_mean')
                dy_mean = metrics.get('displacement_y_mean')
                dz_mean = metrics.get('displacement_z_mean')
                
                if not (isinstance(dx_mean, float) and np.isnan(dx_mean)):
                    displacement_x_matrix_mean[i, j] = dx_mean
                    displacement_x_matrix_mean[j, i] = dx_mean
                
                if not (isinstance(dy_mean, float) and np.isnan(dy_mean)):
                    displacement_y_matrix_mean[i, j] = dy_mean
                    displacement_y_matrix_mean[j, i] = dy_mean
                
                if not (isinstance(dz_mean, float) and np.isnan(dz_mean)):
                    displacement_z_matrix_mean[i, j] = dz_mean
                    displacement_z_matrix_mean[j, i] = dz_mean
                    logger.info(f"Paire {i}-{j}: déplacement Z moyen = {dz_mean:.4f} m")
                else:
                    logger.warning(f"Paire {i}-{j}: displacement_z_mean est NaN ou invalide")
                
                # Remplir les matrices des composantes (médianes)
                dx_median = metrics.get('displacement_x_median')
                dy_median = metrics.get('displacement_y_median')
                dz_median = metrics.get('displacement_z_median')
                
                if not (isinstance(dx_median, float) and np.isnan(dx_median)):
                    displacement_x_matrix_median[i, j] = dx_median
                    displacement_x_matrix_median[j, i] = dx_median
                
                if not (isinstance(dy_median, float) and np.isnan(dy_median)):
                    displacement_y_matrix_median[i, j] = dy_median
                    displacement_y_matrix_median[j, i] = dy_median
                
                if not (isinstance(dz_median, float) and np.isnan(dz_median)):
                    displacement_z_matrix_median[i, j] = dz_median
                    displacement_z_matrix_median[j, i] = dz_median
                
                # Remplir les matrices des composantes (écart-type)
                dx_std = metrics.get('displacement_x_std')
                dy_std = metrics.get('displacement_y_std')
                dz_std = metrics.get('displacement_z_std')
                
                if not (isinstance(dx_std, float) and np.isnan(dx_std)):
                    displacement_x_matrix_std[i, j] = dx_std
                    displacement_x_matrix_std[j, i] = dx_std
                
                if not (isinstance(dy_std, float) and np.isnan(dy_std)):
                    displacement_y_matrix_std[i, j] = dy_std
                    displacement_y_matrix_std[j, i] = dy_std
                
                if not (isinstance(dz_std, float) and np.isnan(dz_std)):
                    displacement_z_matrix_std[i, j] = dz_std
                    displacement_z_matrix_std[j, i] = dz_std
                
                # Remplir les matrices de qualité
                rmse_val = metrics.get('rmse_3d')
                mae_val = metrics.get('mae_3d')
                
                if not (isinstance(rmse_val, float) and np.isnan(rmse_val)):
                    rmse_matrix[i, j] = rmse_val
                    rmse_matrix[j, i] = rmse_val
                
                if not (isinstance(mae_val, float) and np.isnan(mae_val)):
                    mae_matrix[i, j] = mae_val
                    mae_matrix[j, i] = mae_val
                
                # Remplir la matrice de couverture
                coverage_val = metrics.get('coverage_ratio')
                if not (isinstance(coverage_val, float) and np.isnan(coverage_val)):
                    coverage_matrix[i, j] = coverage_val
                    coverage_matrix[j, i] = coverage_val
            elif analysis_type == 'ortho':
                # Analyse ortho : calculer le déplacement 2D (horizontal)
                if 'displacement_x_mean' in metrics and 'displacement_y_mean' in metrics:
                    # Calculer la norme 2D moyenne
                    dx_mean = metrics.get('displacement_x_mean', 0.0)
                    dy_mean = metrics.get('displacement_y_mean', 0.0)
                    displacement_2d_mean = np.sqrt(dx_mean**2 + dy_mean**2)
                    
                    dx_median = metrics.get('displacement_x_median', 0.0)
                    dy_median = metrics.get('displacement_y_median', 0.0)
                    displacement_2d_median = np.sqrt(dx_median**2 + dy_median**2)
                    
                    displacement_matrix_mean[i, j] = displacement_2d_mean
                    displacement_matrix_mean[j, i] = displacement_2d_mean
                    displacement_matrix_median[i, j] = displacement_2d_median
                    displacement_matrix_median[j, i] = displacement_2d_median
                    logger.debug(f"Paire {i}-{j}: déplacement 2D moyen = {displacement_2d_mean:.4f} m")
                else:
                    logger.warning(f"Paire {i}-{j}: métriques X/Y manquantes pour analyse ortho")
            elif analysis_type == 'mnt':
                # Analyse MNT : utiliser le déplacement vertical (Z)
                if 'displacement_z_mean' in metrics:
                    displacement_matrix_mean[i, j] = abs(metrics['displacement_z_mean'])
                    displacement_matrix_mean[j, i] = abs(metrics['displacement_z_mean'])
                    displacement_matrix_median[i, j] = abs(metrics['displacement_z_median'])
                    displacement_matrix_median[j, i] = abs(metrics['displacement_z_median'])
                    logger.debug(f"Paire {i}-{j}: déplacement Z moyen = {abs(metrics['displacement_z_mean']):.4f} m")
                else:
                    logger.warning(f"Paire {i}-{j}: métrique Z manquante pour analyse MNT")
            else:
                logger.warning(f"Paire {i}-{j}: type d'analyse non reconnu: {analysis_type}")
        else:
            logger.warning(f"Paire {comp.get('model1_index', '?')}-{comp.get('model2_index', '?')}: erreur détectée")
    
    # Sauvegarder les matrices
    matrices_dir = os.path.join(output_dir, 'aggregated_results', 'matrices')
    os.makedirs(matrices_dir, exist_ok=True)
    
    # Utiliser les indices comme noms de lignes/colonnes
    index_names = [str(i) for i in range(n_models)]
    
    # Fonction helper pour sauvegarder une matrice
    def save_matrix(matrix, filename, description):
        matrix_path = os.path.join(matrices_dir, filename)
        if pd is not None:
            df = pd.DataFrame(matrix, index=index_names, columns=index_names)
            df.to_csv(matrix_path)
            logger.info(f"{description} sauvegardée: {matrix_path}")
        else:
            # Fallback sans pandas
            with open(matrix_path, 'w') as f:
                f.write(',' + ','.join(index_names) + '\n')
                for i in range(n_models):
                    row = [str(matrix[i, j]) if not np.isnan(matrix[i, j]) else '' 
                           for j in range(n_models)]
                    f.write(str(i) + ',' + ','.join(row) + '\n')
            logger.info(f"{description} sauvegardée (sans pandas): {matrix_path}")
    
    # Sauvegarder toutes les matrices
    save_matrix(displacement_matrix_mean, 'displacement_matrix_mean_3d.csv', 'Matrice de déplacements 3D moyens')
    save_matrix(displacement_matrix_median, 'displacement_matrix_median_3d.csv', 'Matrice de déplacements 3D médians')
    save_matrix(displacement_matrix_std, 'displacement_matrix_std_3d.csv', 'Matrice de déplacements 3D écart-type')
    save_matrix(displacement_matrix_max, 'displacement_matrix_max_3d.csv', 'Matrice de déplacements 3D maximum')
    save_matrix(displacement_matrix_p95, 'displacement_matrix_p95_3d.csv', 'Matrice de déplacements 3D 95e percentile')
    save_matrix(displacement_matrix_p99, 'displacement_matrix_p99_3d.csv', 'Matrice de déplacements 3D 99e percentile')
    
    save_matrix(displacement_x_matrix_mean, 'displacement_x_matrix_mean.csv', 'Matrice de déplacements X moyens')
    save_matrix(displacement_y_matrix_mean, 'displacement_y_matrix_mean.csv', 'Matrice de déplacements Y moyens')
    save_matrix(displacement_z_matrix_mean, 'displacement_z_matrix_mean.csv', 'Matrice de déplacements Z moyens')
    
    save_matrix(displacement_x_matrix_median, 'displacement_x_matrix_median.csv', 'Matrice de déplacements X médians')
    save_matrix(displacement_y_matrix_median, 'displacement_y_matrix_median.csv', 'Matrice de déplacements Y médians')
    save_matrix(displacement_z_matrix_median, 'displacement_z_matrix_median.csv', 'Matrice de déplacements Z médians')
    
    save_matrix(displacement_x_matrix_std, 'displacement_x_matrix_std.csv', 'Matrice de déplacements X écart-type')
    save_matrix(displacement_y_matrix_std, 'displacement_y_matrix_std.csv', 'Matrice de déplacements Y écart-type')
    save_matrix(displacement_z_matrix_std, 'displacement_z_matrix_std.csv', 'Matrice de déplacements Z écart-type')
    
    save_matrix(rmse_matrix, 'rmse_matrix_3d.csv', 'Matrice RMSE 3D')
    save_matrix(mae_matrix, 'mae_matrix_3d.csv', 'Matrice MAE 3D')
    save_matrix(coverage_matrix, 'coverage_matrix.csv', 'Matrice de couverture')
    
    # Résultats finaux
    final_results = {
        'n_models': n_models,
        'n_comparisons': n_comparisons,
        'model_paths': model_paths,
        'model_mapping': model_mapping,
        'comparisons': all_comparisons,
        'displacement_matrix_mean': displacement_matrix_mean.tolist(),
        'displacement_matrix_median': displacement_matrix_median.tolist(),
        'displacement_matrix_std': displacement_matrix_std.tolist(),
        'displacement_matrix_max': displacement_matrix_max.tolist(),
        'displacement_matrix_p95': displacement_matrix_p95.tolist(),
        'displacement_matrix_p99': displacement_matrix_p99.tolist(),
        'displacement_x_matrix_mean': displacement_x_matrix_mean.tolist(),
        'displacement_y_matrix_mean': displacement_y_matrix_mean.tolist(),
        'displacement_z_matrix_mean': displacement_z_matrix_mean.tolist(),
        'displacement_x_matrix_median': displacement_x_matrix_median.tolist(),
        'displacement_y_matrix_median': displacement_y_matrix_median.tolist(),
        'displacement_z_matrix_median': displacement_z_matrix_median.tolist(),
        'displacement_x_matrix_std': displacement_x_matrix_std.tolist(),
        'displacement_y_matrix_std': displacement_y_matrix_std.tolist(),
        'displacement_z_matrix_std': displacement_z_matrix_std.tolist(),
        'displacement_x_matrix_std': displacement_x_matrix_std.tolist(),
        'displacement_y_matrix_std': displacement_y_matrix_std.tolist(),
        'displacement_z_matrix_std': displacement_z_matrix_std.tolist(),
        'rmse_matrix': rmse_matrix.tolist(),
        'mae_matrix': mae_matrix.tolist(),
        'coverage_matrix': coverage_matrix.tolist(),
        'output_dir': output_dir,
        'mapping_file': mapping_file,
        'mapping_csv': mapping_csv,
        'matrices_dir': matrices_dir
    }
    
    logger.info("Pipeline d'analyse paire par paire terminé avec succès")
    return final_results
