#!/usr/bin/env python3
"""
Module d'analyse pour PhotoGeoAlign
Contient les fonctions de calcul et d'analyse pour les images MNT et ortho
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
import cv2

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
    
    # Noms des fichiers de sortie
    base1 = os.path.splitext(os.path.basename(image1_path))[0]
    base2 = os.path.splitext(os.path.basename(image2_path))[0]
    
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
    
    # Masque des valeurs valides
    valid_mask = ~(np.isnan(mnt1) | np.isnan(mnt2) | (mnt1 == 0) | (mnt2 == 0))
    
    if not np.any(valid_mask):
        logger.warning("Aucune donnée valide trouvée pour l'analyse")
        return {}
    
    mnt1_valid = mnt1[valid_mask]
    mnt2_valid = mnt2[valid_mask]
    
    # Calcul des statistiques de base
    diff = mnt1_valid - mnt2_valid
    abs_diff = np.abs(diff)
    
    results = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mae': np.mean(abs_diff),
        'max_diff': np.max(diff),
        'min_diff': np.min(diff),
        'correlation_pearson': pearsonr(mnt1_valid, mnt2_valid)[0],
        'correlation_spearman': spearmanr(mnt1_valid, mnt2_valid)[0],
        'n_points': len(mnt1_valid),
        'resolution': resolution
    }
    
    # Calcul des percentiles
    percentiles = [5, 25, 50, 75, 95]
    results['percentiles_diff'] = {f'p{p}': np.percentile(diff, p) for p in percentiles}
    results['percentiles_abs_diff'] = {f'p{p}': np.percentile(abs_diff, p) for p in percentiles}
    
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
                                     logger: logging.Logger, farneback_params: dict = None) -> Dict[str, Any]:
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
        
        # Calcul de l'amplitude des déplacements
        displacement_magnitude = np.sqrt(displacement_x_m**2 + displacement_y_m**2)
        
        # Statistiques des déplacements
        valid_mask = ~(np.isnan(displacement_x_m) | np.isnan(displacement_y_m))
        valid_displacements_x = displacement_x_m[valid_mask]
        valid_displacements_y = displacement_y_m[valid_mask]
        valid_magnitudes = displacement_magnitude[valid_mask]
        
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
            'std_displacement_x': np.std(valid_displacements_x),
            'std_displacement_y': np.std(valid_displacements_y),
            'std_displacement_magnitude': np.std(valid_magnitudes),
            'max_displacement_x': np.max(valid_displacements_x),
            'max_displacement_y': np.max(valid_displacements_y),
            'max_displacement_magnitude': np.max(valid_magnitudes),
            'min_displacement_x': np.min(valid_displacements_x),
            'min_displacement_y': np.min(valid_displacements_y),
            'min_displacement_magnitude': np.min(valid_magnitudes),
            'n_valid_displacements': len(valid_displacements_x),
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
        
        # Masque des valeurs valides (adapté pour RGB et niveaux de gris)
        if len(data1.shape) == 3:  # Image RGB (3, height, width)
            # Pour RGB, vérifier que tous les canaux sont valides
            valid_mask = ~(np.isnan(data1).any(axis=0) | np.isnan(data2).any(axis=0) | 
                          (data1 == 0).all(axis=0) | (data2 == 0).all(axis=0))
            grid_width, grid_height = data1.shape[2], data1.shape[1]
        else:  # Image en niveaux de gris (height, width)
            valid_mask = ~(np.isnan(data1) | np.isnan(data2) | (data1 == 0) | (data2 == 0))
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
            data1, data2, resolution, output_dir, logger, farneback_params
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
        f.write("=" * 60 + "\n")
        f.write("RAPPORT D'ANALYSE PHOTOGEOALIGN\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Type d'analyse: {analysis_type.upper()}\n")
        f.write(f"Image 1: {image1_path}\n")
        f.write(f"Image 2: {image2_path}\n")
        f.write(f"Résolution: {results.get('resolution', 'N/A')} m\n")
        f.write(f"Nombre de points: {results.get('n_points', 'N/A')}\n\n")
        
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
        f.write(f"Spearman: {results.get('correlation_spearman', 'N/A')}\n\n")
        
        f.write("PERCENTILES DES DIFFÉRENCES\n")
        f.write("-" * 30 + "\n")
        percentiles = results.get('percentiles_diff', {})
        for p, value in percentiles.items():
            f.write(f"{p}: {value}\n")
        
        f.write("\nPERCENTILES DES DIFFÉRENCES ABSOLUES\n")
        f.write("-" * 40 + "\n")
        abs_percentiles = results.get('percentiles_abs_diff', {})
        for p, value in abs_percentiles.items():
            f.write(f"{p}: {value}\n")
        
        # Ajout des statistiques de déplacement si disponibles
        if 'mean_displacement_x' in results:
            f.write("\nSTATISTIQUES DES DÉPLACEMENTS (FARNEBACK)\n")
            f.write("-" * 45 + "\n")
            f.write(f"Déplacement X moyen: {results.get('mean_displacement_x', 'N/A')} m\n")
            f.write(f"Déplacement Y moyen: {results.get('mean_displacement_y', 'N/A')} m\n")
            f.write(f"Amplitude moyenne: {results.get('mean_displacement_magnitude', 'N/A')} m\n")
            f.write(f"Amplitude max: {results.get('max_displacement_magnitude', 'N/A')} m\n")
            f.write(f"Écart-type X: {results.get('std_displacement_x', 'N/A')} m\n")
            f.write(f"Écart-type Y: {results.get('std_displacement_y', 'N/A')} m\n")
            f.write(f"Points de déplacement valides: {results.get('n_valid_displacements', 'N/A')}\n")
    
    logger.info(f"Rapport généré: {report_path}")
    return report_path

def run_analysis_pipeline(image1_path: str, image2_path: str, 
                         analysis_type: str, resolution: float,
                         output_dir: str, farneback_params: dict = None) -> Dict[str, Any]:
    """
    Pipeline principal d'analyse
    
    Args:
        image1_path: Chemin vers la première image
        image2_path: Chemin vers la deuxième image
        analysis_type: Type d'analyse ('mnt' ou 'ortho')
        resolution: Résolution d'analyse en mètres
        output_dir: Dossier de sortie
        farneback_params: Paramètres pour la méthode de Farneback
        
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
        if analysis_type == 'ortho':
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
            adapted_params = farneback_params
        
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
        if analysis_type == 'mnt':
            results = analyze_mnt_comparison(resampled1, resampled2, resolution)
            # Ajouter les chemins des fichiers sauvegardés
            results['resampled1_path'] = resampled1_path
            results['resampled2_path'] = resampled2_path
        elif analysis_type == 'ortho':
            results = analyze_ortho_comparison(image1_path, image2_path, resolution, output_dir, adapted_params)
        else:
            raise ValueError(f"Type d'analyse non supporté: {analysis_type}")
        
        # Génération du rapport
        if results:
            report_path = generate_analysis_report(
                results, analysis_type, image1_path, image2_path, output_dir
            )
            results['report_path'] = report_path
        
        logger.info("Pipeline d'analyse terminé avec succès")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du pipeline d'analyse: {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
        raise
