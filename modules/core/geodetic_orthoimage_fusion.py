"""
Module de fusion et assemblage d'orthoimages.
Contient les fonctions avancées pour la fusion d'orthoimages existantes.
"""

import os
import numpy as np
import logging
import rasterio
from rasterio import warp, transform, enums
from multiprocessing import Pool, cpu_count

# Import des fonctions d'assemblage depuis le module utils
from .geodetic_utils import (
    simple_ortho_assembly,
    simple_mnt_assembly
)

def process_zone_with_orthos(zone_data):
    """
    Fonction globale exécutée par chaque processus pour traiter une zone
    UNIQUEMENT avec l'égalisation par médiane superposée
    """
    
    zone_id = zone_data['zone_id']
    assigned_files = zone_data['assigned_orthos']
    
    # SÉPARER LES ORTHOS ET LES MNT
    ortho_color_files = [f for f in assigned_files if f.endswith('_color.tif')]
    ortho_height_files = [f for f in assigned_files if f.endswith('_height.tif')]
    
    print(f"🔄 PROCESSUS {os.getpid()}: Traitement Zone {zone_id}")
    print(f"   🎨 Orthos couleur ({len(ortho_color_files)}): {ortho_color_files}")
    print(f"   📏 MNT hauteur ({len(ortho_height_files)}): {ortho_height_files}")
    
    # TRAITEMENT RÉEL DES ORTHOS ET MNT
    zone_bounds = zone_data['bounds']
    zone_size_meters = zone_data['zone_size_meters']
    final_resolution = zone_data['final_resolution']
    
    # Calculer les dimensions de la zone en pixels
    zone_width_px = max(1, round(zone_size_meters / final_resolution))
    zone_height_px = max(1, round(zone_size_meters / final_resolution))
    
    results = {}
    
    # 1. FUSION DES ORTHOS COULEUR (UNIQUEMENT ÉGALISATION PAR MÉDIANE SUPERPOSÉE)
    if ortho_color_files:
        print(f"   🎨 Fusion de {len(ortho_color_files)} orthos couleur...")
        try:
            # Lire et reprojeter chaque ortho couleur
            ortho_arrays = []
            nodata_values = []
            for ortho_file in ortho_color_files:
                full_path = os.path.join(zone_data['input_dir'], ortho_file)
                with rasterio.open(full_path) as src:
                    # LECTURE DES NODATA
                    nodata_value = src.nodata
                    nodata_values.append(nodata_value)
                    
                    # Reprojection vers la zone (SANS interpolation pour éviter les contours sombres)
                    ortho_reprojected, _ = warp.reproject(
                        source=rasterio.band(src, [1, 2, 3]),  # 3 bandes RGB
                        destination=np.zeros((3, zone_height_px, zone_width_px), dtype=np.float32),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform.from_bounds(*zone_bounds, zone_width_px, zone_height_px),
                        dst_crs=src.crs,
                        resampling=enums.Resampling.nearest,  # ✅ PLUS PROCHE VOISIN (pas d'interpolation)
                        nodata=nodata_value  # Préserver la valeur nodata originale
                    )
                    
                    ortho_arrays.append(ortho_reprojected)
                    print(f"         ✅ {ortho_file}: reprojeté avec nodata géré")
            
            # ÉGALISATION UNIQUEMENT PAR MÉDIANE SUPERPOSÉE
            if ortho_arrays:
                print(f"         🎨 Égalisation par médiane superposée...")
                
                # 1. CRÉER UN MASQUE COMMUN DES PIXELS VALIDES
                print(f"            🔍 Création du masque commun des pixels valides...")
                common_valid_mask = np.zeros((zone_height_px, zone_width_px), dtype=bool)
                
                for ortho_array in ortho_arrays:
                    # Un pixel est valide s'il n'est pas (0,0,0) dans au moins une bande
                    ortho_valid = ~np.all(ortho_array == 0.0, axis=0)
                    common_valid_mask |= ortho_valid
                
                valid_pixels_count = np.sum(common_valid_mask)
                print(f"            📊 Masque commun : {valid_pixels_count}/{zone_height_px * zone_width_px} pixels valides")
                
                if valid_pixels_count > 0:
                    # 2. HARMONISATION DIRECTE DES COULEURS PAR ANALYSE DES DIFFÉRENCES
                    print(f"            🎨 Harmonisation directe des couleurs par analyse des différences...")
                    
                    if len(ortho_arrays) > 1:
                        # 1. SÉLECTION DE L'ORTHO DE RÉFÉRENCE (plus de pixels valides)
                        ortho_pixel_counts = []
                        for i, ortho in enumerate(ortho_arrays):
                            valid_pixels = np.sum(np.any(ortho > 0, axis=0))
                            ortho_pixel_counts.append(valid_pixels)
                            print(f"               📊 Ortho {i+1} - Pixels valides: {valid_pixels}")
                        
                        ref_index = np.argmax(ortho_pixel_counts)
                        print(f"            🏆 Ortho de référence: Ortho {ref_index + 1} ({ortho_pixel_counts[ref_index]} pixels valides)")
                        
                        # 2. CALCUL DES DIFFÉRENCES ET FACTEURS CORRECTEURS
                        harmonized_orthos = []
                        
                        for i, ortho in enumerate(ortho_arrays):
                            if i == ref_index:
                                # L'ortho de référence reste inchangée
                                harmonized_orthos.append(np.copy(ortho))
                                print(f"               ✅ Ortho {i+1} (référence) - Aucune modification")
                            else:
                                # Calculer les différences avec l'ortho de référence
                                print(f"               🔄 Harmonisation Ortho {i+1} vers la référence...")
                                
                                # Créer le masque des pixels de chevauchement
                                ref_valid = np.any(ortho_arrays[ref_index] > 0, axis=0)
                                ortho_valid = np.any(ortho > 0, axis=0)
                                overlap_mask = ref_valid & ortho_valid
                                
                                if np.sum(overlap_mask) > 0:
                                    # Calculer les différences par bande (avec signe)
                                    diff_R = ortho_arrays[ref_index][0, overlap_mask] - ortho[0, overlap_mask]
                                    diff_G = ortho_arrays[ref_index][1, overlap_mask] - ortho[1, overlap_mask]
                                    diff_B = ortho_arrays[ref_index][2, overlap_mask] - ortho[2, overlap_mask]
                                    
                                    # 🆕 HARMONISATION MULTI-QUANTILES : Q25, Q50 (médiane), Q75
                                    print(f"                  📊 Analyse multi-quantiles des différences...")
                                    
                                    # Calculer les quantiles des différences pour chaque bande
                                    diff_Q25_R = np.percentile(diff_R, 25)
                                    diff_Q50_R = np.percentile(diff_R, 50)  # Médiane
                                    diff_Q75_R = np.percentile(diff_R, 75)
                                    
                                    diff_Q25_G = np.percentile(diff_G, 25)
                                    diff_Q50_G = np.percentile(diff_G, 50)  # Médiane
                                    diff_Q75_G = np.percentile(diff_G, 75)
                                    
                                    diff_Q25_B = np.percentile(diff_B, 25)
                                    diff_Q50_B = np.percentile(diff_B, 50)  # Médiane
                                    diff_Q75_B = np.percentile(diff_B, 75)
                                    
                                    print(f"                  📈 Bande R - Q25={diff_Q25_R:.1f}, Q50={diff_Q50_R:.1f}, Q75={diff_Q75_R:.1f}")
                                    print(f"                  📈 Bande G - Q25={diff_Q25_G:.1f}, Q50={diff_Q50_G:.1f}, Q75={diff_Q75_G:.1f}")
                                    print(f"                  📈 Bande B - Q25={diff_Q25_B:.1f}, Q50={diff_Q50_B:.1f}, Q75={diff_Q75_B:.1f}")
                                    
                                    # Calculer les facteurs correcteurs multi-quantiles
                                    # Facteur = 1 + (Q50 / ref_mean) + (Q75 - Q25) / (2 * ref_mean)
                                    # Q50 : correction centrale, (Q75-Q25) : correction de la dispersion
                                    ref_mean_R = np.mean(ortho_arrays[ref_index][0, overlap_mask])
                                    ref_mean_G = np.mean(ortho_arrays[ref_index][1, overlap_mask])
                                    ref_mean_B = np.mean(ortho_arrays[ref_index][2, overlap_mask])
                                    
                                    if ref_mean_R != 0:
                                        # Correction centrale + correction de dispersion
                                        facteur_R = 1 + (diff_Q50_R / ref_mean_R) + (diff_Q75_R - diff_Q25_R) / (2 * ref_mean_R)
                                    else:
                                        facteur_R = 1.0
                                        
                                    if ref_mean_G != 0:
                                        facteur_G = 1 + (diff_Q50_G / ref_mean_G) + (diff_Q75_G - diff_Q25_G) / (2 * ref_mean_G)
                                    else:
                                        facteur_G = 1.0
                                        
                                    if ref_mean_B != 0:
                                        facteur_B = 1 + (diff_Q50_B / ref_mean_B) + (diff_Q75_B - diff_Q25_B) / (2 * ref_mean_B)
                                    else:
                                        facteur_B = 1.0
                                    
                                    print(f"                  🎯 Facteurs multi-quantiles: R={facteur_R:.3f}, G={facteur_G:.3f}, B={facteur_B:.3f}")
                                    
                                    # Appliquer les facteurs correcteurs à toute l'ortho
                                    harmonized_ortho = np.copy(ortho)
                                    harmonized_ortho[0] *= facteur_R  # Bande R
                                    harmonized_ortho[1] *= facteur_G  # Bande G
                                    harmonized_ortho[2] *= facteur_B  # Bande B
                                    
                                    # Clipper pour éviter les valeurs > 255
                                    harmonized_ortho = np.clip(harmonized_ortho, 0, 255)
                                    
                                    harmonized_orthos.append(harmonized_ortho)
                                    print(f"               ✅ Ortho {i+1} harmonisée")
                                else:
                                    # Pas de chevauchement, pas d'harmonisation possible
                                    harmonized_orthos.append(np.copy(ortho))
                                    print(f"               ⚠️ Ortho {i+1} - Pas de chevauchement avec la référence")
                        
                        # Remplacer les orthos par les versions harmonisées
                        equalized_orthos = harmonized_orthos
                        print(f"            ✅ Harmonisation des couleurs terminée")
                    else:
                        # Une seule ortho, pas d'harmonisation possible
                        equalized_orthos = ortho_arrays
                        print(f"            ⚠️ Pas d'harmonisation possible - une seule ortho")
                    
                    # 3. FUSION PAR MOYENNE DES ORTHOS HARMONISÉES
                    print(f"            🎨 Fusion par moyenne des orthos harmonisées...")
                    fused_color = np.zeros((3, zone_height_px, zone_width_px), dtype=np.float32)
                    
                    # Pour chaque pixel, calculer la moyenne des orthos harmonisées
                    for y in range(zone_height_px):
                        for x in range(zone_width_px):
                            for band in range(3):  # RGB
                                # Collecter les valeurs valides pour ce pixel et cette bande
                                valid_values = []
                                
                                for equalized_ortho in equalized_orthos:
                                    pixel_value = equalized_ortho[band, y, x]
                                    # Un pixel est valide s'il n'est pas 0 (après harmonisation)
                                    if pixel_value > 0:
                                        valid_values.append(pixel_value)
                                
                                # Si on a des valeurs valides, calculer la moyenne
                                if valid_values:
                                    fused_color[band, y, x] = np.mean(valid_values)
                                # Sinon, le pixel reste à 0 (pas de données)
                    
                    # Clip et convertir en uint8
                    fused_color = np.clip(fused_color, 0, 255).astype(np.uint8)
                    
                    # Sauvegarder l'ortho fusionnée finale
                    color_filename = f"zone_{zone_id}_fused_color_median_harmonized.tif"
                    color_path = os.path.join(zone_data['output_dir'], color_filename)
                    
                    with rasterio.open(
                        color_path, 'w',
                        driver='GTiff',
                        height=zone_height_px,
                        width=zone_width_px,
                        count=3,
                        dtype=np.uint8,
                        crs=src.crs,
                        transform=transform.from_bounds(*zone_bounds, zone_width_px, zone_height_px),
                        nodata=None
                    ) as dst:
                        dst.write(fused_color, [1, 2, 3])
                    
                    print(f"            ✅ Ortho fusionnée finale sauvegardée : {color_filename}")
                    results['color_fused_final'] = color_path
                    
                else:
                    print(f"            ⚠️ Aucun pixel valide trouvé, pas de fusion possible")
                
        except Exception as e:
            print(f"   ❌ Erreur fusion orthos couleur : {e}")
            results['color_error'] = str(e)
    
    # 2. FUSION DES MNT HAUTEUR (MAXIMUM)
    if ortho_height_files:
        print(f"   📏 Fusion de {len(ortho_height_files)} MNT hauteur...")
        try:
            # Lire et reprojeter chaque MNT hauteur
            height_arrays = []
            nodata_values = []
            
            for height_file in ortho_height_files:
                full_path = os.path.join(zone_data['input_dir'], height_file)
                with rasterio.open(full_path) as src:
                    # LECTURE DES NODATA
                    nodata_value = src.nodata
                    nodata_values.append(nodata_value)
                    
                    # REPROJECTION COHÉRENTE VERS LA ZONE (COMME POUR LES ORTHOS COULEUR)
                    dst_transform = transform.from_bounds(*zone_bounds, zone_width_px, zone_height_px)
                    
                    # Reprojection avec plus proche voisin pour préserver les valeurs exactes
                    height_reprojected, _ = warp.reproject(
                        source=rasterio.band(src, 1),  # Bande 1 pour la hauteur
                        destination=np.zeros((zone_height_px, zone_width_px), dtype=np.float32),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        resampling=enums.Resampling.nearest,  # Plus proche voisin pour préserver les valeurs exactes
                        nodata=nodata_value  # Préserver la valeur nodata originale
                    )
                    
                    # POST-TRAITEMENT : Remplacer les pixels hors limites par NaN
                    if nodata_value is not None:
                        valid_mask = height_reprojected != nodata_value
                        height_reprojected[(height_reprojected == 0.0) & valid_mask] = np.nan
                    else:
                        height_reprojected[height_reprojected == 0.0] = np.nan
                    
                    height_arrays.append(height_reprojected)
                    print(f"         ✅ {height_file}: reprojeté vers {height_reprojected.shape} avec transform cohérent et nodata géré")
            
            # Fusion par maximum des hauteurs (IGNORANT LES VALEURS NODATA)
            if height_arrays:
                # Initialiser avec NaN (pas de valeur initiale)
                fused_height = np.full((zone_height_px, zone_width_px), np.nan, dtype=np.float32)
                
                # Pour chaque pixel de la zone
                for y in range(zone_height_px):
                    for x in range(zone_width_px):
                        # Collecter toutes les valeurs valides pour ce pixel
                        valid_values = []
                        
                        for height_array in height_arrays:
                            pixel_value = height_array[y, x]
                            
                            # Vérifier si la valeur est valide (pas nodata, pas NaN, pas hors limites)
                            if not np.isnan(pixel_value):  # Ignorer les NaN (pixels hors limites)
                                valid_values.append(pixel_value)
                        
                        # Si on a des valeurs valides, prendre le maximum
                        if valid_values:
                            fused_height[y, x] = max(valid_values)
                        # Sinon, le pixel reste NaN (pas de données)
                
                # Compter les pixels avec des valeurs valides
                valid_pixels = ~np.isnan(fused_height)
                print(f"         🎯 Fusion terminée : {np.sum(valid_pixels)}/{fused_height.size} pixels avec valeurs valides")
                
                # Sauvegarder le MNT fusionné
                height_filename = f"zone_{zone_id}_fused_height.tif"
                height_path = os.path.join(zone_data['output_dir'], height_filename)
                
                with rasterio.open(
                    height_path, 'w',
                    driver='GTiff',
                    height=zone_height_px,
                    width=zone_width_px,
                    count=1,
                    dtype=np.float32,
                    crs=src.crs,
                    transform=transform.from_bounds(*zone_bounds, zone_width_px, zone_height_px),
                    nodata=None
                ) as dst:
                    dst.write(fused_height, 1)
                
                print(f"   ✅ MNT hauteur fusionné sauvegardé : {height_filename}")
                results['height_fused'] = height_path
                
        except Exception as e:
            print(f"   ❌ Erreur fusion MNT hauteur : {e}")
            results['height_error'] = str(e)
    
    return {
        'zone_id': zone_id,
        'status': 'success',
        'orthos_color_processed': len(ortho_color_files),
        'orthos_height_processed': len(ortho_height_files),
        'message': f"Zone {zone_id}: {len(ortho_color_files)} orthos couleur + {len(ortho_height_files)} MNT hauteur"
    }

def unified_ortho_mnt_fusion(input_dir, logger, output_dir, final_resolution=None, zone_size_meters=5.0, max_workers=None):
    """
    🎯 FUSION FINALE : Assemblage des orthoimages et MNT unifiés
    Objectif : Fusionner les orthoimages unitaires en une orthoimage finale unifiée
    
    Args:
        input_dir: Répertoire d'entrée contenant les orthoimages unitaires
        logger: Logger pour les messages
        output_dir: Répertoire de sortie
        final_resolution: Résolution finale en mètres (défaut: 0.003m = 3mm)
        zone_size_meters: Taille de chaque zone en mètres (défaut: 5.0m)
        max_workers: Nombre maximum de processus parallèles
    """
    from rasterio.coords import BoundingBox
    from rasterio.transform import from_origin
    from multiprocessing import Pool
    
    logger.info("🎯 FUSION FINALE : Assemblage des orthoimages et MNT unifiés")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # ÉTAPE 1.1 : Analyser les orthoimages unitaires pour déterminer l'étendue globale
    logger.info("Analyse des orthoimages unitaires...")
    
    # Chercher tous les fichiers .tif dans le dossier d'entrée
    ortho_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            ortho_files.append(os.path.join(input_dir, file))
    
    if not ortho_files:
        logger.warning("⚠️ Aucune orthoimage .tif trouvée, impossible de continuer sans données")
        raise ValueError("Aucune orthoimage .tif trouvée dans le dossier d'entrée")
    else:
        logger.info(f"Trouvé {len(ortho_files)} orthoimage(s) unitaire(s)")
        
        # Analyser la première orthoimage pour obtenir la résolution et les dimensions
        first_ortho = ortho_files[0]
        logger.info(f"Analyse de l'orthoimage de référence : {os.path.basename(first_ortho)}")
        
        try:
            with rasterio.open(first_ortho) as src:
                bounds = src.bounds
                transform = src.transform
                width = src.width
                height = src.height
                crs = src.crs
                
                logger.info(f"  CRS : {crs}")
                logger.info(f"  Dimensions : {width} × {height} pixels")
                logger.info(f"  Transform : {transform}")
                logger.info(f"  Bounds : {bounds}")
                
                # Calculer la résolution réelle
                pixel_size_x = abs(transform.a)
                pixel_size_y = abs(transform.e)
                logger.info(f"  Résolution pixel : {pixel_size_x:.6f}m × {pixel_size_y:.6f}m")
                
                # Si la résolution finale n'est pas spécifiée, utiliser celle de l'ortho
                if final_resolution is None:
                    final_resolution = min(pixel_size_x, pixel_size_y)
                    logger.info(f"  Résolution finale automatique : {final_resolution:.6f}m")
                
                # S'assurer que la résolution finale est valide
                if final_resolution <= 0:
                    logger.warning(f"  ⚠️ Résolution invalide détectée : {final_resolution}, utilisation de 0.1m")
                    final_resolution = 0.1
                
                # Calculer l'étendue globale en analysant toutes les orthos
                all_bounds = [bounds]
                for ortho_file in ortho_files[1:]:
                    try:
                        with rasterio.open(ortho_file) as src2:
                            all_bounds.append(src2.bounds)
                    except Exception as e:
                        logger.warning(f"  ⚠️ Impossible de lire {os.path.basename(ortho_file)} : {e}")
                
                # Calculer l'étendue globale
                global_left = min(b.left for b in all_bounds)
                global_bottom = min(b.bottom for b in all_bounds)
                global_right = max(b.right for b in all_bounds)
                global_top = max(b.top for b in all_bounds)
                
                # Vérifier que les bounds sont valides
                if global_left >= global_right or global_bottom >= global_top:
                    logger.error(f"❌ Bounds invalides détectés : left={global_left}, right={global_right}, bottom={global_bottom}, top={global_top}")
                    raise ValueError("Bounds invalides - l'étendue des orthoimages est incorrecte")
                
                logger.info(f"  Étendue des orthos : {global_left:.2f}m à {global_right:.2f}m (X), {global_bottom:.2f}m à {global_top:.2f}m (Y)")
                
                # Si la taille de grille n'est pas spécifiée, l'utiliser pour contraindre
                # 🎯 GRID AUTOMATIQUE : Utiliser l'étendue réelle des orthos
                global_bounds = BoundingBox(
                    left=global_left, bottom=global_bottom,
                    right=global_right, top=global_top
                )
                
                logger.info(f"Étendue globale calculée : {global_bounds}")
                logger.info(f"  Largeur : {global_bounds.right - global_bounds.left:.2f}m")
                logger.info(f"  Hauteur : {global_bounds.top - global_bounds.bottom:.2f}m")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse de l'orthoimage : {e}")
            raise RuntimeError(f"Impossible d'analyser les orthoimages : {e}")
    
    # S'assurer que la résolution finale est valide
    if final_resolution is None or final_resolution <= 0:
        logger.warning(f"⚠️ Résolution finale invalide : {final_resolution}, utilisation de 0.1m")
        final_resolution = 0.1
    
    # S'assurer que la taille des zones est valide
    if zone_size_meters <= 0:
        logger.warning(f"⚠️ Taille des zones invalide : {zone_size_meters}, utilisation de 5.0m")
        zone_size_meters = 5.0
    
    logger.info(f"Résolution finale : {final_resolution}m")
    logger.info(f"Taille des zones : {zone_size_meters}m × {zone_size_meters}m")
    
    # 🔧 CORRECTION CRITIQUE : Adapter la taille des zones à la résolution exacte des pixels
    # Calculer combien de pixels correspondent exactement à la taille demandée
    pixels_per_zone = zone_size_meters / final_resolution
    logger.info(f"🔍 Calcul pixels par zone : {zone_size_meters}m ÷ {final_resolution}m = {pixels_per_zone:.6f} pixels")
    
    # Adapter la taille des zones pour qu'elles correspondent à un nombre entier de pixels
    # OPTION 1 : Troncature (zones légèrement plus petites)
    zone_size_adjusted_down = int(pixels_per_zone) * final_resolution
    # OPTION 2 : Arrondi (zones légèrement plus grandes)
    zone_size_adjusted_up = round(pixels_per_zone) * final_resolution
    
    logger.info(f"🔍 Option 1 (troncature) : {int(pixels_per_zone)} pixels × {final_resolution}m = {zone_size_adjusted_down:.6f}m")
    logger.info(f"🔍 Option 2 (arrondi) : {round(pixels_per_zone)} pixels × {final_resolution}m = {zone_size_adjusted_up:.6f}m")
    
    # Choisir l'option qui minimise le décalage avec la taille demandée
    if abs(zone_size_adjusted_down - zone_size_meters) <= abs(zone_size_adjusted_up - zone_size_meters):
        zone_size = zone_size_adjusted_down
        logger.info(f"✅ Choix : Option 1 (troncature) - Taille finale des zones : {zone_size:.6f}m × {zone_size:.6f}m")
        logger.info(f"   → Décalage : {zone_size - zone_size_meters:.6f}m ({((zone_size - zone_size_meters) * 1000):.3f}mm)")
    else:
        zone_size = zone_size_adjusted_up
        logger.info(f"✅ Choix : Option 2 (arrondi) - Taille finale des zones : {zone_size:.6f}m × {zone_size:.6f}m")
        logger.info(f"   → Décalage : {zone_size - zone_size_meters:.6f}m ({((zone_size - zone_size_meters) * 1000):.3f}mm)")
    
    # ÉTAPE 1.2 : Créer des zones paramétrables parfaitement alignées
    zones = []
    
    # OPTION 2 : Aligner la grille directement sur les coordonnées des orthos
    # Sauvegarder les bounds originaux des orthos AVANT modification
    original_ortho_bounds = BoundingBox(
        left=global_bounds.left,
        bottom=global_bounds.bottom,
        right=global_bounds.right,
        top=global_bounds.top
    )
    
    aligned_left = global_bounds.left
    aligned_bottom = global_bounds.bottom
    
    # Étendre légèrement la grille en bas et à droite pour couvrir complètement
    # CORRECTION : Utiliser zone_size (ajustée) au lieu de zone_size_meters
    aligned_right = global_bounds.right + (zone_size - (global_bounds.right - global_bounds.left) % zone_size) % zone_size
    aligned_top = global_bounds.top + (zone_size - (global_bounds.top - global_bounds.bottom) % zone_size) % zone_size
    
    # S'assurer que l'extension ne dépasse pas une zone complète
    if aligned_right > global_bounds.right + zone_size:
        aligned_right = global_bounds.right + zone_size
    if aligned_top > global_bounds.top + zone_size:
        aligned_top = global_bounds.top + zone_size
    
    logger.info(f"Coordonnées des orthos (sans arrondi) : left={aligned_left:.6f}, bottom={aligned_bottom:.6f}, right={aligned_right:.6f}, top={aligned_top:.6f}")
    logger.info(f"  → Grille alignée directement sur les orthos (pas de décalage de résolution)")
    logger.info(f"  → Extension en bas/droite pour couvrir complètement les zones de {zone_size_meters}m")
    
    # Créer les zones
    zone_id = 0
    x = aligned_left
    while x < aligned_right:
        y = aligned_bottom
        while y < aligned_top:
            zone = {
                'id': zone_id,
                'bounds': (x, y, x + zone_size, y + zone_size),
                'center': (x + zone_size/2, y + zone_size/2),
                'color': zone_id % 8  # 8 couleurs différentes pour identifier les zones
            }
            zones.append(zone)
            zone_id += 1
            y += zone_size
        x += zone_size
    
    logger.info(f"Créé {len(zones)} zones de {zone_size}m × {zone_size}m")
    
    # ÉTAPE 1.2.5 : ASSIGNER LES ORTHOS AUX ZONES QUI LES CHEVAUCHENT
    logger.info("Assignation des orthos aux zones qui les chevauchent...")
    
    def zones_overlap(zone_bounds, ortho_bounds):
        """
        Vérifie si une zone et une ortho se chevauchent
        zone_bounds: (x1, y1, x2, y2) - coordonnées de la zone
        ortho_bounds: (left, bottom, right, top) - bounds de l'ortho
        """
        x1, y1, x2, y2 = zone_bounds
        left, bottom, right, top = ortho_bounds
        
        # Pas de chevauchement si une zone est complètement à gauche, droite, haut ou bas
        return not (x2 < left or x1 > right or y2 < bottom or y1 > top)
    
    # Assigner les orthos aux zones
    zone_assignments = {}
    for zone in zones:
        zone_id = zone['id']
        zone_bounds = zone['bounds']
        overlapping_orthos = []
        
        for ortho_file in ortho_files:
            try:
                with rasterio.open(ortho_file) as src:
                    ortho_bounds = src.bounds
                    if zones_overlap(zone_bounds, ortho_bounds):
                        overlapping_orthos.append(os.path.basename(ortho_file))
                        logger.debug(f"Zone {zone_id} ← {os.path.basename(ortho_file)} (chevauchement)")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de lire {os.path.basename(ortho_file)} : {e}")
        
        zone_assignments[zone_id] = overlapping_orthos
        logger.info(f"Zone {zone_id}: {len(overlapping_orthos)} ortho(s) assignée(s) : {overlapping_orthos}")
    
    # Ajouter les assignations aux zones
    for zone in zones:
        zone['assigned_orthos'] = zone_assignments[zone['id']]
    
    logger.info(f"✅ Assignation terminée : {len(zone_assignments)} zones avec orthos assignées")
    
    # Résumé des assignations
    total_orthos_assigned = sum(len(orthos) for orthos in zone_assignments.values())
    zones_with_ortho = sum(1 for orthos in zone_assignments.values() if len(orthos) > 0)
    logger.info(f"📊 Résumé : {total_orthos_assigned} assignations totales, {zones_with_ortho}/{len(zones)} zones avec orthos")
    
    # ÉTAPE 1.3 : PARALLÉLISATION - TRAITEMENT DES ZONES AVEC ORTHOS RÉELLES
    logger.info("🚀 DÉMARRAGE DE LA PARALLÉLISATION - Traitement des zones avec orthos réelles...")
    
    # Préparer les données pour chaque processus
    process_data = []
    for zone in zones:
        process_data.append({
            'zone_id': zone['id'],
            'assigned_orthos': zone['assigned_orthos'],
            'bounds': zone['bounds'],
            'zone_size_meters': zone_size_meters,
            'final_resolution': final_resolution,
            'input_dir': input_dir,  # Chemin des orthos d'entrée
            'output_dir': output_dir  # Dossier de sortie pour les résultats
        })
    
    logger.info(f"📋 Données préparées pour {len(process_data)} processus")
    
    # Configuration de la parallélisation
    if max_workers is None:
        max_workers = min(4, len(process_data))  # Maximum 4 processus par défaut
    else:
        max_workers = min(max_workers, len(process_data))  # Respecter la limite demandée
    logger.info(f"🔧 Lancement de {max_workers} processus parallèles...")
    
    try:
        with Pool(processes=max_workers) as pool:
            logger.info(f"🚀 Pool de processus créé avec {max_workers} workers")
            
            # Lancer le traitement parallèle
            results = pool.map(process_zone_with_orthos, process_data)
            
            # Analyser les résultats
            logger.info("📊 Résultats du traitement parallèle :")
            total_color = sum(result['orthos_color_processed'] for result in results)
            total_height = sum(result['orthos_height_processed'] for result in results)
            
            for result in results:
                logger.info(f"  ✅ {result['message']}")
            
            logger.info(f"🎉 PARALLÉLISATION TERMINÉE : {len(results)} zones traitées")
            logger.info(f"📊 TOTAL : {total_color} orthos couleur + {total_height} MNT hauteur traités")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la parallélisation : {e}")
        logger.warning("⚠️ Fallback vers le traitement séquentiel...")
        
        # Fallback séquentiel
        results = []
        for zone_data in process_data:
            try:
                result = process_zone_with_orthos(zone_data)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Erreur zone {zone_data['zone_id']} : {e}")
        
        logger.info(f"🔄 Traitement séquentiel terminé : {len(results)} zones traitées")
    
    logger.info(f"✅ FUSION FINALE TERMINÉE : Assemblage des orthoimages et MNT unifiés terminé")
    logger.info(f"Résultat attendu : {len(results)} zones traitées avec orthos fusionnées")
    
    # 🆕 ÉTAPE 2 : ÉGALISATION DÉSACTIVÉE POUR LE MOMENT
    logger.info("⏸️ ÉGALISATION COLORIMÉTRIQUE DÉSACTIVÉE - Utilisation des zones originales")
    
    # Utiliser directement les zones fusionnées sans égalisation
    equalized_zones = []
    for file in os.listdir(output_dir):
        if file.endswith('_fused_color_median_harmonized.tif'):
            equalized_zones.append(os.path.join(output_dir, file))
    
    logger.info(f"📁 Zones utilisées (sans égalisation) : {len(equalized_zones)}")
    
    # 🆕 ÉTAPE 3 : ASSEMBLAGE AUTOMATIQUE DES ORTHOS UNIFIÉES
    logger.info("🚀 LANCEMENT AUTOMATIQUE DE L'ASSEMBLAGE DES ORTHOS...")
    
    try:
        # Récupérer la résolution finale depuis les paramètres
        final_resolution = None  # Détection automatique
        
        # Lancer l'assemblage automatique des orthos ET des MNT
        logger.info("🚀 LANCEMENT DE L'ASSEMBLAGE AUTOMATIQUE : Orthos + MNT...")
        
        # Assemblage des orthos
        unified_ortho_path = simple_ortho_assembly(
            zones_output_dir=output_dir,
            logger=logger,
            final_resolution=final_resolution
        )
        
        # Assemblage des MNT
        unified_mnt_path = simple_mnt_assembly(
            zones_output_dir=output_dir,
            logger=logger,
            final_resolution=final_resolution
        )
        
        if unified_ortho_path and unified_mnt_path:
            logger.info(f"🎉 ASSEMBLAGE AUTOMATIQUE COMPLET RÉUSSI !")
            logger.info(f"📁 Ortho unifiée créée : {unified_ortho_path}")
            logger.info(f"📁 MNT unifié créé : {unified_mnt_path}")
            logger.info(f"✅ PIPELINE COMPLET TERMINÉ : Zones + Orthos + MNT unifiés")
        elif unified_ortho_path:
            logger.warning(f"⚠️ Ortho créée mais MNT échoué : {unified_ortho_path}")
        elif unified_mnt_path:
            logger.warning(f"⚠️ MNT créé mais ortho échouée : {unified_mnt_path}")
        else:
            logger.warning(f"⚠️ Assemblage automatique échoué, mais zones créées avec succès")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'assemblage automatique : {e}")
        logger.warning(f"⚠️ Les zones ont été créées, mais l'assemblage a échoué")
    
    return output_dir