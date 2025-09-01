"""
Module d'utilitaires géodésiques.
Contient les fonctions d'aide pour le traitement des orthoimages.
"""

import os
import numpy as np
import logging
import rasterio

def calculate_global_histogram_and_quantiles(zones_output_dir, logger):
    """
    CALCUL GLOBAL : Histogramme et quantiles de référence sur TOUTES les zones
    Exclut les pixels noirs ET très sombres (seuil réaliste)
    
    Args:
        zones_output_dir: Répertoire contenant toutes les zones
        logger: Logger pour les messages
    
    Returns:
        dict: Quantiles globaux de référence par bande
    """
    import numpy as np
    import os
    import rasterio
    
    logger.info("🌍 CALCUL GLOBAL : Histogramme et quantiles de référence sur toutes les zones...")
    
    # ÉTAPE 1 : Trouver tous les fichiers de zones
    zone_files = []
    for file in os.listdir(zones_output_dir):
        if file.endswith('_fused_color_median_harmonized.tif'):
            zone_files.append(os.path.join(zones_output_dir, file))
    
    if not zone_files:
        logger.warning("⚠️ Aucune zone trouvée pour le calcul global")
        return None
    
    logger.info(f"  📁 Zones trouvées : {len(zone_files)}")
    
    # ÉTAPE 2 : Calculer l'histogramme global de toutes les zones
    logger.info("  📊 Calcul de l'histogramme global...")
    global_histogram = np.zeros((3, 256))  # R, G, B, 0-255
    
    total_pixels_processed = 0
    total_valid_pixels = 0
    
    for zone_file in zone_files:
        try:
            with rasterio.open(zone_file) as src:
                zone_image = src.read()  # (3, H, W)
                height, width = src.height, src.width
                
                # Convertir en format (H, W, 3) pour le traitement
                zone_image_rgb = zone_image.transpose(1, 2, 0)
                
                # Créer un masque pour exclure UNIQUEMENT les pixels noirs (0,0,0)
                black_pixels_mask = np.all(zone_image_rgb == [0, 0, 0], axis=2)
                valid_pixels_mask = ~black_pixels_mask
                
                valid_pixels_count = np.sum(valid_pixels_mask)
                total_pixels_processed += height * width
                total_valid_pixels += valid_pixels_count
                
                # Ajouter à l'histogramme global (pixels valides uniquement)
                for band_idx in range(3):
                    band = zone_image_rgb[:, :, band_idx]
                    valid_band_values = band[valid_pixels_mask]
                    
                    if len(valid_band_values) > 0:
                        hist, _ = np.histogram(valid_band_values, bins=256, range=(0, 256))
                        global_histogram[band_idx] += hist
                
                logger.info(f"    ✅ {os.path.basename(zone_file)}: {valid_pixels_count}/{height*width} pixels valides")
                
        except Exception as e:
            logger.warning(f"    ⚠️ Erreur lors du traitement de {os.path.basename(zone_file)}: {e}")
            continue
    
    logger.info(f"  📊 Total global : {total_valid_pixels}/{total_pixels_processed} pixels valides ({total_valid_pixels/total_pixels_processed*100:.1f}%)")
    
    # ÉTAPE 3 : Calculer les quantiles de référence globaux
    logger.info("  🎯 Calcul des quantiles de référence globaux...")
    global_quantiles = {}
    
    for band_idx in range(3):
        cumulative_hist = np.cumsum(global_histogram[band_idx])
        total_pixels_band = cumulative_hist[-1]
        
        if total_pixels_band == 0:
            logger.warning(f"    ⚠️ Bande {['Rouge', 'Vert', 'Bleu'][band_idx]} : aucun pixel valide")
            global_quantiles[band_idx] = {'q25': 128, 'q50': 128, 'q75': 128}
            continue
        
        # Trouver les indices des quantiles
        q25_idx = np.searchsorted(cumulative_hist, total_pixels_band * 0.25)
        q50_idx = np.searchsorted(cumulative_hist, total_pixels_band * 0.50)
        q75_idx = np.searchsorted(cumulative_hist, total_pixels_band * 0.75)
        
        global_quantiles[band_idx] = {
            'q25': int(q25_idx),
            'q50': int(q50_idx), 
            'q75': int(q75_idx)
        }
        
        logger.info(f"    {['Rouge', 'Vert', 'Bleu'][band_idx]}: Q25={q25_idx}, Q50={q50_idx}, Q75={q75_idx}")
    
    logger.info("  ✅ Quantiles globaux calculés avec succès")
    return global_quantiles

def equalize_zone_to_global_quantiles(zone_ortho_path, global_quantiles, logger):
    """
    ÉGALISATION GLOBALE : Égalise une zone vers les quantiles globaux de référence
    PRÉSERVE LA GÉOSPATIALITÉ (CRS + géotransformation)
    
    Args:
        zone_ortho_path: Chemin vers l'ortho de zone à égaliser
        global_quantiles: Quantiles globaux de référence
        logger: Logger pour les messages
    
    Returns:
        str: Chemin vers la zone égalisée
    """
    import numpy as np
    import os
    import rasterio
    
    logger.info(f"🎨 ÉGALISATION GLOBALE DE LA ZONE : {os.path.basename(zone_ortho_path)}")
    
    try:
        # ÉTAPE 1 : Lire l'image avec rasterio pour préserver la géospatialité
        logger.info(f"  📖 Lecture de la zone avec rasterio...")
        with rasterio.open(zone_ortho_path) as src:
            zone_image = src.read()  # (3, H, W)
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            bounds = src.bounds
            
            logger.info(f"  📏 Dimensions : {width} × {height} pixels")
            logger.info(f"  🌍 CRS : {crs}")
            logger.info(f"  📍 Bounds : {bounds}")
        
        # Convertir en format (H, W, 3) pour le traitement
        zone_image_rgb = zone_image.transpose(1, 2, 0)  # (H, W, 3)
        
        # ÉTAPE 2 : Créer un masque pour exclure UNIQUEMENT les pixels noirs (0,0,0)
        logger.info(f"  🔍 Création du masque des pixels valides...")
        black_pixels_mask = np.all(zone_image_rgb == [0, 0, 0], axis=2)
        valid_pixels_mask = ~black_pixels_mask
        
        valid_pixels_count = np.sum(valid_pixels_mask)
        total_pixels = height * width
        
        logger.info(f"  🔍 Pixels valides : {valid_pixels_count}/{total_pixels} ({valid_pixels_count/total_pixels*100:.1f}%)")
        
        if valid_pixels_count == 0:
            logger.warning(f"  ⚠️ Aucun pixel valide trouvé dans la zone")
            return zone_ortho_path
        
        # ÉTAPE 3 : Égalisation vers les quantiles globaux
        logger.info(f"  🔧 Application de l'égalisation vers les quantiles globaux...")
        
        # Créer une image de sortie
        equalized_image = np.zeros_like(zone_image_rgb)
        
        # Égalisation par bande vers les quantiles globaux
        equalization_factors = []
        
        for band_idx in range(3):
            band = zone_image_rgb[:, :, band_idx]
            valid_band_values = band[valid_pixels_mask]
            
            if len(valid_band_values) == 0:
                equalized_image[:, :, band_idx] = band
                equalization_factors.append(1.0)
                continue
            
            # Calculer les quantiles actuels de la zone
            current_q25 = np.percentile(valid_band_values, 25)
            current_q50 = np.percentile(valid_band_values, 50)
            current_q75 = np.percentile(valid_band_values, 75)
            
            # Quantiles globaux de référence
            target_q25 = global_quantiles[band_idx]['q25']
            target_q50 = global_quantiles[band_idx]['q50']
            target_q75 = global_quantiles[band_idx]['q75']
            
            logger.info(f"    {['Rouge', 'Vert', 'Bleu'][band_idx]}:")
            logger.info(f"      Actuel: Q25={current_q25:.1f}, Q50={current_q50:.1f}, Q75={current_q75:.1f}")
            logger.info(f"      Cible: Q25={target_q25}, Q50={target_q50}, Q75={target_q75}")
            
            # ÉTAPE 4 : Égalisation par transformation de quantiles
            # Utiliser une transformation linéaire basée sur les quantiles
            if current_q75 > current_q25:  # Éviter division par zéro
                # Transformation linéaire : (x - q25) / (q75 - q25) * (target_q75 - target_q25) + target_q25
                scale_factor = (target_q75 - target_q25) / (current_q75 - current_q25)
                offset = target_q25 - current_q25 * scale_factor
                
                # Appliquer la transformation
                equalized_band = np.clip(band * scale_factor + offset, 0, 255).astype(np.uint8)
                
                # Calculer le facteur d'égalisation pour les logs
                factor = scale_factor
            else:
                # Si pas de variation, appliquer un décalage simple
                offset = target_q50 - current_q50
                equalized_band = np.clip(band + offset, 0, 255).astype(np.uint8)
                factor = 1.0
            
            equalized_image[:, :, band_idx] = equalized_band
            equalization_factors.append(factor)
        
        logger.info(f"  📊 Facteurs d'égalisation: R={equalization_factors[0]:.3f}, G={equalization_factors[1]:.3f}, B={equalization_factors[2]:.3f}")
        
        # ÉTAPE 5 : Sauvegarder avec rasterio pour préserver la géospatialité
        logger.info(f"  💾 Sauvegarde avec rasterio (géospatialité préservée)...")
        
        # Créer le nom de fichier de sortie
        base_name = os.path.splitext(os.path.basename(zone_ortho_path))[0]
        output_path = zone_ortho_path.replace(
            base_name, 
            f"{base_name}_equalized_global_quantiles"
        )
        
        # Sauvegarder avec rasterio en préservant CRS et géotransformation
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=crs,  # ✅ PRÉSERVE LE CRS
            transform=transform,  # ✅ PRÉSERVE LA GÉOTRANSFORMATION
            photometric='rgb'
        ) as dst:
            # Écrire les 3 bandes RGB (format rasterio : (3, H, W))
            equalized_image_rasterio = equalized_image.transpose(2, 0, 1)  # (H, W, 3) → (3, H, W)
            dst.write(equalized_image_rasterio)
            
            # Métadonnées
            dst.update_tags(
                Software='PhotoGeoAlign Global Quantile Equalization',
                Method='Global quantile transformation',
                Global_Q25=f"{[global_quantiles[i]['q25'] for i in range(3)]}",
                Global_Q50=f"{[global_quantiles[i]['q50'] for i in range(3)]}",
                Global_Q75=f"{[global_quantiles[i]['q75'] for i in range(3)]}",
                Equalization_Factors=f"{equalization_factors}",
                Pixels_Excluded=str(total_pixels - valid_pixels_count)
            )
        
        logger.info(f"  ✅ Zone égalisée sauvegardée : {os.path.basename(output_path)}")
        logger.info(f"  📋 Métadonnées d'égalisation:")
        logger.info(f"    - Méthode: Égalisation vers quantiles globaux")
        logger.info(f"    - Pixels exclus: {total_pixels - valid_pixels_count} (noirs [0,0,0])")
        logger.info(f"    - Facteurs R/G/B: {equalization_factors}")
        logger.info(f"    - Géospatialité: CRS et géotransformation préservés ✅")
        
        return output_path
        
    except Exception as e:
        logger.error(f"  ❌ Erreur lors de l'égalisation de la zone : {e}")
        return None

def simple_mnt_assembly(zones_output_dir, final_resolution, logger):
    """
    ASSEMBLAGE SIMPLE DES MNT : Place les MNT de zones côte à côte sans fusion
    Utilise la même logique de géoréférencement que simple_ortho_assembly
    
    Args:
        zones_output_dir: Répertoire contenant les zones avec leurs MNT
        final_resolution: Résolution finale en mètres par pixel (si None, détection automatique)
        logger: Logger pour les messages
    
    Returns:
        str: Chemin vers le MNT unifié final
    """
    import os
    import numpy as np
    import rasterio
    import re
    from rasterio.coords import BoundingBox
    from rasterio.transform import Affine
    
    logger.info("🔧 ASSEMBLAGE SIMPLE DES MNT DE ZONES (pas de fusion)")
    logger.info(f"📁 Répertoire des zones : {zones_output_dir}")
    
    # 🔧 CORRECTION : Détecter automatiquement la résolution des MNT unitaires
    if final_resolution is None:
        logger.info("🔍 Détection automatique de la résolution des MNT unitaires...")
        # Lire le premier MNT pour obtenir sa résolution
        first_mnt = None
        for file in os.listdir(zones_output_dir):
            if file.endswith('_fused_height.tif'):
                first_mnt = os.path.join(zones_output_dir, file)
                break
        
        if first_mnt:
            with rasterio.open(first_mnt) as src:
                # Calculer la résolution à partir de la transformation affine
                transform = src.transform
                # La résolution est la valeur absolue de a (largeur) et e (hauteur) de la transformation
                pixel_width = abs(transform.a)
                pixel_height = abs(transform.e)
                # Prendre la moyenne des deux résolutions
                final_resolution = (pixel_width + pixel_height) / 2
                logger.info(f"  ✅ Résolution détectée : {final_resolution:.6f}m/pixel")
                logger.info(f"  📏 Largeur pixel : {pixel_width:.6f}m, Hauteur pixel : {pixel_height:.6f}m")
        else:
            logger.warning("⚠️ Aucun MNT trouvé, utilisation de la résolution par défaut 0.1m")
            final_resolution = 0.1
    
    logger.info(f"📏 Résolution finale utilisée : {final_resolution}m")
    
    # ÉTAPE 1 : Lire les MNT de zones
    logger.info("📖 Lecture des MNT de zones...")
    
    zone_mnt_files = []
    for file in os.listdir(zones_output_dir):
        if file.endswith('_fused_height.tif'):
            file_path = os.path.join(zones_output_dir, file)
            try:
                with rasterio.open(file_path) as src:
                    # Extraire les informations de la zone
                    zone_bounds = src.bounds
                    zone_data = {
                        'file_path': file_path,
                        'bounds': zone_bounds,
                        'width': src.width,
                        'height': src.height,
                        'crs': src.crs,
                        'transform': src.transform
                    }
                    
                    # Extraire l'ID de zone du nom de fichier
                    zone_id_match = re.search(r'zone_(\d+)_', file)
                    if zone_id_match:
                        zone_data['zone_id'] = int(zone_id_match.group(1))
                    else:
                        zone_data['zone_id'] = len(zone_mnt_files)
                    
                    zone_mnt_files.append(zone_data)
                    logger.info(f"  ✅ {file}: {src.width}×{src.height} pixels, CRS: {src.crs}")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Erreur lors de la lecture de {file}: {e}")
                continue
    
    if not zone_mnt_files:
        logger.error("❌ Aucun MNT de zone trouvé !")
        return None
    
    logger.info(f"📊 MNT de zones trouvés : {len(zone_mnt_files)}")
    
    # ÉTAPE 2 : Calculer les bornes globales
    logger.info("🌍 Calcul des bornes globales...")
    
    # Initialiser avec la première zone
    global_bounds = zone_mnt_files[0]['bounds']
    
    for zone_data in zone_mnt_files[1:]:
        zone_bounds = zone_data['bounds']
        global_bounds = BoundingBox(
            left=min(global_bounds.left, zone_bounds.left),
            bottom=min(global_bounds.bottom, zone_bounds.bottom),
            right=max(global_bounds.right, zone_bounds.right),
            top=max(global_bounds.top, zone_bounds.top)
        )
    
    logger.info(f"📏 Bornes globales : {global_bounds}")
    
    # ÉTAPE 3 : Calculer les dimensions de la grille finale
    logger.info("📐 Calcul des dimensions de la grille finale...")
    
    # Calculer la taille en pixels (arrondir pour éviter les décalages)
    final_width = round((global_bounds.right - global_bounds.left) / final_resolution)
    final_height = round((global_bounds.top - global_bounds.bottom) / final_resolution)
    
    logger.info(f"📏 Grille finale : {final_width} × {final_height} pixels")
    logger.info(f"📏 Étendue : {(global_bounds.right - global_bounds.left):.3f}m × {(global_bounds.top - global_bounds.bottom):.3f}m")
    
    # Créer la transformation affine finale
    # IMPORTANT : Pour les MNT, on inverse l'axe Y car rasterio attend que Y augmente vers le bas
    # mais les coordonnées géographiques augmentent vers le haut
    final_transform = Affine.translation(global_bounds.left, global_bounds.top) * Affine.scale(final_resolution, -final_resolution)
    
    # ÉTAPE 4 : Créer la grille finale et placer les MNT
    logger.info("🔧 Placement des MNT de zones dans la grille finale...")
    
    # Grille finale pour les MNT (1 bande, données float32)
    final_mnt = np.full((final_height, final_width), np.nan, dtype=np.float32)
    
    zones_placed = 0
    
    for zone_data in zone_mnt_files:
        try:
            logger.info(f"  🔄 Placement Zone {zone_data['zone_id']}...")
            
            # Lire le MNT de la zone
            with rasterio.open(zone_data['file_path']) as src:
                zone_mnt = src.read(1)  # 1 bande de hauteur
            
            zone_bounds = zone_data['bounds']
            
            # Calculer la position dans la grille finale (même logique que pour les orthos)
            x_pos = (zone_bounds.left - global_bounds.left) / final_resolution
            y_pos = (global_bounds.top - zone_bounds.top) / final_resolution
            
            # Arrondir pour éviter les décalages de pixels
            start_x = round(x_pos)
            start_y = round(y_pos)
            end_x = start_x + zone_data['width']
            end_y = start_y + zone_data['height']
            
            # DEBUG : Vérifier les arrondis de position
            logger.info(f"    🔍 DEBUG Position calculée:")
            logger.info(f"      x_pos brute: {x_pos:.6f} -> arrondie: {start_x}")
            logger.info(f"      y_pos brute: {y_pos:.6f} -> arrondie: {start_y}")
            logger.info(f"      décalage x: {abs(x_pos - start_x):.6f} pixels")
            logger.info(f"      décalage y: {abs(y_pos - start_y):.6f} pixels")
            
            # DEBUG : Afficher les calculs détaillés pour vérifier l'alignement
            logger.info(f"    🔍 DEBUG Alignement Y:")
            logger.info(f"      global_bounds.top: {global_bounds.top:.6f}m")
            logger.info(f"      zone_bounds.top: {zone_bounds.top:.6f}m")
            logger.info(f"      diff_y: {global_bounds.top - zone_bounds.top:.6f}m")
            logger.info(f"      start_y pixels: {start_y}")
            logger.info(f"      zone height: {zone_data['height']} pixels")
            
            logger.info(f"    📍 Position dans la grille : ({start_x}, {start_y}) à ({end_x}, {end_y})")
            
            # Vérifier les limites
            if (start_x < 0 or start_y < 0 or 
                end_x > final_width or end_y > final_height):
                logger.warning(f"    ⚠️ Zone {zone_data['zone_id']} dépasse les limites de la grille finale")
                continue
            
            # Placer le MNT de la zone dans la grille finale
            final_mnt[start_y:end_y, start_x:end_x] = zone_mnt
            
            zones_placed += 1
            logger.info(f"    ✅ Zone {zone_data['zone_id']} placée avec succès")
            
        except Exception as e:
            logger.error(f"    ❌ Erreur lors du placement de la Zone {zone_data['zone_id']}: {e}")
            continue
    
    logger.info(f"📊 Zones placées : {zones_placed}/{len(zone_mnt_files)}")
    
    # ÉTAPE 5 : Sauvegarder le MNT unifié final
    logger.info("💾 Sauvegarde du MNT unifié final...")
    
    output_path = os.path.join(zones_output_dir, "mnt_unified_final.tif")
    
    # Récupérer le CRS de la première zone (toutes devraient avoir le même)
    reference_crs = zone_mnt_files[0]['crs']
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=final_height,
        width=final_width,
        count=1,  # 1 bande pour les MNT
        dtype=np.float32,  # Données de hauteur en float32
        crs=reference_crs,
        transform=final_transform,
        nodata=np.nan  # Valeur nodata pour les pixels sans données
    ) as dst:
        # Écrire la bande de hauteur
        dst.write(final_mnt, 1)
        
        # Métadonnées
        dst.update_tags(
            Software='PhotoGeoAlign Simple MNT Assembly',
            Resolution=f'{final_resolution}m per pixel',
            Zones_Processed=str(zones_placed),
            Assembly_Method='Simple placement without fusion',
            Data_Type='Height values in meters',
            Nodata_Value='NaN'
        )
    
    logger.info(f"🎉 MNT UNIFIÉ CRÉÉ : {output_path}")
    logger.info(f"   📏 Dimensions : {final_width} × {final_height} pixels")
    logger.info(f"   📏 Étendue : {(global_bounds.right - global_bounds.left):.3f}m × {(global_bounds.top - global_bounds.bottom):.3f}m")
    logger.info(f"   🎯 Zones assemblées : {zones_placed}")
    
    # Statistiques sur les données
    valid_pixels = np.sum(~np.isnan(final_mnt))
    if valid_pixels > 0:
        height_min = np.nanmin(final_mnt)
        height_max = np.nanmax(final_mnt)
        logger.info(f"   📊 Hauteurs : {height_min:.3f}m à {height_max:.3f}m")
        logger.info(f"   📊 Pixels valides : {valid_pixels}/{final_width * final_height}")
    
    return output_path

def simple_ortho_assembly(zones_output_dir, logger, final_resolution=None):
    """
    Simple assemblage des orthos de zones (pas de fusion)
    
    Args:
        zones_output_dir: Répertoire contenant les orthos fusionnées par zones
        logger: Logger pour les messages
        final_resolution: Résolution finale en mètres (si None, utilise la résolution des orthos unitaires)
    
    Returns:
        str: Chemin vers l'ortho unifiée créée
    """
    import os
    import numpy as np
    import rasterio
    from rasterio.coords import BoundingBox
    from rasterio.transform import from_origin
    
    logger.info("🔧 ASSEMBLAGE SIMPLE DES ORTHOS DE ZONES (pas de fusion)")
    logger.info(f"📁 Répertoire des zones : {zones_output_dir}")
    
    # 🔧 CORRECTION : Détecter automatiquement la résolution des orthos unitaires
    if final_resolution is None:
        logger.info("🔍 Détection automatique de la résolution des orthos unitaires...")
        # Lire la première ortho pour obtenir sa résolution
        first_ortho = None
        for file in os.listdir(zones_output_dir):
            if file.endswith('_fused_color_median_harmonized.tif'):
                first_ortho = os.path.join(zones_output_dir, file)
                break
        
        if first_ortho:
            with rasterio.open(first_ortho) as src:
                # Calculer la résolution à partir de la transformation affine
                transform = src.transform
                # La résolution est la valeur absolue de a (largeur) et e (hauteur) de la transformation
                pixel_width = abs(transform.a)
                pixel_height = abs(transform.e)
                # Prendre la moyenne des deux résolutions
                final_resolution = (pixel_width + pixel_height) / 2
                logger.info(f"  ✅ Résolution détectée : {final_resolution:.6f}m/pixel")
                logger.info(f"  📏 Largeur pixel : {pixel_width:.6f}m, Hauteur pixel : {pixel_height:.6f}m")
        else:
            logger.warning("⚠️ Aucune ortho trouvée, utilisation de la résolution par défaut 0.1m")
            final_resolution = 0.1
    
    logger.info(f"📏 Résolution finale utilisée : {final_resolution}m")
    
    # ÉTAPE 1 : Lire toutes les orthos de zones
    logger.info("📖 Lecture des orthos de zones...")
    
    zone_ortho_files = []
    zone_bounds_list = []
    
    # Utiliser directement les zones originales (sans égalisation)
    logger.info("  🔄 Lecture des zones originales (sans égalisation)...")
    for file in os.listdir(zones_output_dir):
        if file.endswith('_fused_color_median_harmonized.tif'):
            file_path = os.path.join(zones_output_dir, file)
            
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    transform = src.transform
                    width = src.width
                    height = src.height
                    crs = src.crs
                    
                    # Extraire l'ID de zone du nom de fichier
                    zone_id = int(file.split('_')[1])
                    
                    zone_ortho_files.append({
                        'file_path': file_path,
                        'zone_id': zone_id,
                        'bounds': bounds,
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'crs': crs
                    })
                    
                    zone_bounds_list.append(bounds)
                    
                    logger.info(f"  ✅ Zone {zone_id}: {width}×{height} pixels, bounds: {bounds}")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Impossible de lire {file}: {e}")
                continue
    
    if not zone_ortho_files:
        logger.error("❌ Aucune ortho de zone trouvée !")
        return None
    
    logger.info(f"📊 Total : {len(zone_ortho_files)} orthos de zones trouvées")
    
    # ÉTAPE 2 : Calculer la grille finale unifiée
    logger.info("🧮 Calcul de la grille finale unifiée...")
    
    # Calculer l'étendue globale
    global_left = min(bounds.left for bounds in zone_bounds_list)
    global_bottom = min(bounds.bottom for bounds in zone_bounds_list)
    global_right = max(bounds.right for bounds in zone_bounds_list)
    global_top = max(bounds.top for bounds in zone_bounds_list)
    
    global_bounds = BoundingBox(
        left=global_left,
        bottom=global_bottom,
        right=global_right,
        top=global_top
    )
    
    logger.info(f"🌍 Étendue globale : {global_bounds}")
    logger.info(f"  📏 Largeur : {global_bounds.right - global_bounds.left:.3f}m")
    logger.info(f"  📏 Hauteur : {global_bounds.top - global_bounds.bottom:.3f}m")
    
    # Calculer les dimensions de la grille finale
    # CORRECTION : Utiliser round() au lieu de int() pour éviter les lignes noires
    width_pixels = (global_bounds.right - global_bounds.left) / final_resolution
    height_pixels = (global_bounds.top - global_bounds.bottom) / final_resolution
    
    final_width = round(width_pixels)
    final_height = round(height_pixels)
    
    logger.info(f"🔍 DEBUG Dimensions calculées:")
    logger.info(f"  Largeur brute: {width_pixels:.6f} pixels -> arrondie: {final_width}")
    logger.info(f"  Hauteur brute: {height_pixels:.6f} pixels -> arrondie: {final_height}")
    logger.info(f"  Différence largeur: {abs(width_pixels - final_width):.6f} pixels")
    logger.info(f"  Différence hauteur: {abs(height_pixels - final_height):.6f} pixels")
    
    logger.info(f"🖼️ Grille finale : {final_width} × {final_height} pixels")
    
    # ÉTAPE 3 : Créer la grille finale vide
    logger.info("🎨 Création de la grille finale vide...")
    
    # Initialiser avec des valeurs nodata (noir)
    final_ortho = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    
    # Créer le transform pour la grille finale
    # CORRECTION : Utiliser from_bounds pour un alignement parfait
    final_transform = rasterio.transform.from_bounds(
        global_bounds.left, global_bounds.bottom, 
        global_bounds.right, global_bounds.top, 
        final_width, final_height
    )
    
    logger.info(f"🔍 DEBUG Transform final:")
    logger.info(f"  from_origin: left={global_bounds.left:.6f}, top={global_bounds.top:.6f}")
    logger.info(f"  from_bounds: {final_transform}")
    
    # ÉTAPE 4 : Placer chaque ortho de zone à sa position exacte
    logger.info("🔧 Placement des orthos de zones...")
    
    zones_placed = 0
    
    for zone_data in zone_ortho_files:
        try:
            logger.info(f"  🔄 Placement Zone {zone_data['zone_id']}...")
            
            # Lire l'ortho de la zone
            with rasterio.open(zone_data['file_path']) as src:
                zone_image = src.read([1, 2, 3])  # RGB
                zone_bounds = zone_data['bounds']
            
            # Calculer la position dans la grille finale
            # CORRECTION COMPLÈTE : Utiliser round() pour éviter les décalages d'un pixel
            x_pos = (zone_bounds.left - global_bounds.left) / final_resolution
            y_pos = (global_bounds.top - zone_bounds.top) / final_resolution
            
            start_x = round(x_pos)
            start_y = round(y_pos)
            end_x = start_x + zone_data['width']
            end_y = start_y + zone_data['height']
            
            # DEBUG : Vérifier les arrondis de position
            logger.info(f"    🔍 DEBUG Position calculée:")
            logger.info(f"      x_pos brute: {x_pos:.6f} -> arrondie: {start_x}")
            logger.info(f"      y_pos brute: {y_pos:.6f} -> arrondie: {start_y}")
            logger.info(f"      décalage x: {abs(x_pos - start_x):.6f} pixels")
            logger.info(f"      décalage y: {abs(y_pos - start_y):.6f} pixels")
            
            # DEBUG : Afficher les calculs détaillés pour vérifier l'alignement
            logger.info(f"    🔍 DEBUG Alignement Y:")
            logger.info(f"      global_bounds.top: {global_bounds.top:.6f}m")
            logger.info(f"      zone_bounds.top: {zone_bounds.top:.6f}m")
            logger.info(f"      diff_y: {global_bounds.top - zone_bounds.top:.6f}m")
            logger.info(f"      start_y pixels: {start_y}")
            logger.info(f"      zone height: {zone_data['height']} pixels")
            
            logger.info(f"    📍 Position dans la grille : ({start_x}, {start_y}) à ({end_x}, {end_y})")
            
            # Vérifier les limites
            if (start_x < 0 or start_y < 0 or 
                end_x > final_width or end_y > final_height):
                logger.warning(f"    ⚠️ Zone {zone_data['zone_id']} dépasse les limites de la grille finale")
                continue
            
            # Placer l'ortho de la zone dans la grille finale
            # Note : zone_image est (3, height, width), on transpose pour (height, width, 3)
            zone_image_rgb = zone_image.transpose(1, 2, 0)
            
            final_ortho[start_y:end_y, start_x:end_x] = zone_image_rgb
            
            zones_placed += 1
            logger.info(f"    ✅ Zone {zone_data['zone_id']} placée avec succès")
            
        except Exception as e:
            logger.error(f"    ❌ Erreur lors du placement de la Zone {zone_data['zone_id']}: {e}")
            continue
    
    logger.info(f"📊 Zones placées : {zones_placed}/{len(zone_ortho_files)}")
    
    # ÉTAPE 5 : Sauvegarder l'ortho unifiée finale
    logger.info("💾 Sauvegarde de l'ortho unifiée finale...")
    
    output_path = os.path.join(zones_output_dir, "ortho_unified_final.tif")
    
    # Récupérer le CRS de la première zone (toutes devraient avoir le même)
    reference_crs = zone_ortho_files[0]['crs']
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=final_height,
        width=final_width,
        count=3,
        dtype=np.uint8,
        crs=reference_crs,
        transform=final_transform,
        photometric='rgb'
    ) as dst:
        # Écrire les 3 bandes RGB
        dst.write(final_ortho[:, :, 0], 1)  # Rouge
        dst.write(final_ortho[:, :, 1], 2)  # Vert
        dst.write(final_ortho[:, :, 2], 3)  # Bleu
        
        # Métadonnées
        dst.update_tags(
            Software='PhotoGeoAlign Simple Ortho Assembly',
            Resolution=f'{final_resolution}m per pixel',
            Zones_Processed=str(zones_placed),
            Assembly_Method='Simple placement without fusion'
        )
    
    logger.info(f"🎉 ORTHO UNIFIÉE CRÉÉE : {output_path}")
    logger.info(f"   📏 Dimensions : {final_width} × {final_height} pixels")
    logger.info(f"   📏 Étendue : {(global_bounds.right - global_bounds.left):.3f}m × {(global_bounds.top - global_bounds.bottom):.3f}m")
    logger.info(f"   🎯 Zones assemblées : {zones_placed}")
    
    return output_path
