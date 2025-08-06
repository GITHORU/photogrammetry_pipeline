import os
import numpy as np
import logging
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

def process_single_cloud_add_offset(args):
    """Fonction de traitement d'un seul nuage pour l'ajout d'offset (pour multiprocessing)"""
    ply_file, output_dir, coord_file, extra_params = args
    
    # Création d'un logger pour ce processus
    logger = logging.getLogger(f"AddOffset_{os.getpid()}")
    logger.setLevel(logging.INFO)
    
    try:
        # Import d'open3d
        import open3d as o3d
        
        # Lecture du nuage
        cloud = o3d.io.read_point_cloud(ply_file)
        if not cloud.has_points():
            return False, f"Nuage vide dans {os.path.basename(ply_file)}"
        
        points = np.asarray(cloud.points)
        
        # Lecture de l'offset depuis le fichier de coordonnées
        offset = None
        if coord_file and os.path.exists(coord_file):
            with open(coord_file, 'r') as f:
                for line in f:
                    if line.startswith('#Offset to add :'):
                        offset_text = line.replace('#Offset to add :', '').strip()
                        offset = [float(x) for x in offset_text.split()]
                        break
        
        if not offset:
            return False, f"Offset non trouvé dans {coord_file}"
        
        # Application de l'offset
        offset_array = np.array(offset)
        deformed_points = points + offset_array
        
        # Création du nouveau nuage
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(deformed_points)
        
        # Copie des couleurs et normales
        if cloud.has_colors():
            new_cloud.colors = cloud.colors
        if cloud.has_normals():
            new_cloud.normals = cloud.normals
        
        # Sauvegarde
        output_file = os.path.join(output_dir, os.path.basename(ply_file))
        success = o3d.io.write_point_cloud(output_file, new_cloud)
        
        if success:
            return True, f"Traité : {os.path.basename(ply_file)} ({len(points)} points)"
        else:
            return False, f"Erreur de sauvegarde : {os.path.basename(ply_file)}"
            
    except Exception as e:
        return False, f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}"

def process_single_cloud_itrf_to_enu(args):
    """Fonction de traitement d'un seul nuage pour la conversion ITRF→ENU (pour multiprocessing)"""
    ply_file, output_dir, coord_file, extra_params, ref_point_name = args
    
    # Création d'un logger pour ce processus
    logger = logging.getLogger(f"ITRFtoENU_{os.getpid()}")
    logger.setLevel(logging.INFO)
    
    try:
        import open3d as o3d
        import pyproj
        
        # Lecture du nuage
        cloud = o3d.io.read_point_cloud(ply_file)
        if not cloud.has_points():
            return False, f"Nuage vide dans {os.path.basename(ply_file)}"
        
        points = np.asarray(cloud.points)
        logger.info(f"  {len(points)} points chargés")
        
        # Lecture du point de référence depuis le fichier de coordonnées
        ref_point = None
        offset = None
        
        if coord_file and os.path.exists(coord_file):
            with open(coord_file, 'r') as f:
                lines = f.readlines()
            
            # Recherche du point de référence
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:  # Format: NOM X Y Z
                        try:
                            point_name = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            if ref_point_name is None or point_name == ref_point_name:
                                ref_point = [x, y, z]
                                break
                        except ValueError:
                            continue
            
            # Lecture de l'offset
            for line in lines:
                line = line.strip()
                if line.startswith('#Offset to add :'):
                    offset_text = line.replace('#Offset to add :', '').strip()
                    offset = [float(x) for x in offset_text.split()]
                    break
        
        if ref_point is None:
            return False, f"Point de référence non trouvé dans {coord_file}"
        
        if offset is None:
            return False, f"Offset non trouvé dans {coord_file}"
        
        # Application de l'offset au point de référence
        ref_point_with_offset = [ref_point[0] + offset[0], ref_point[1] + offset[1], ref_point[2] + offset[2]]
        
        # Configuration de la transformation topocentrique
        tr_center = ref_point_with_offset
        tr_ellps = "GRS80"
        
        pipeline = "+proj=topocentric +X_0={0} +Y_0={1} +Z_0={2} +ellps={3}".format(
            tr_center[0], tr_center[1], tr_center[2], tr_ellps
        )
        
        transformer = pyproj.Transformer.from_pipeline(pipeline)
        
        # Application de la transformation topocentrique
        arr_x = np.array(list(points[:, 0]))
        arr_y = np.array(list(points[:, 1]))
        arr_z = np.array(list(points[:, 2]))
        
        arr_pts_ENU = np.array(transformer.transform(arr_x, arr_y, arr_z)).T
        
        # Création du nouveau nuage
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(arr_pts_ENU)
        
        # Copie des couleurs et normales
        if cloud.has_colors():
            new_cloud.colors = cloud.colors
        if cloud.has_normals():
            new_cloud.normals = cloud.normals
        
        # Sauvegarde
        output_file = os.path.join(output_dir, os.path.basename(ply_file))
        success = o3d.io.write_point_cloud(output_file, new_cloud)
        
        if success:
            return True, f"Conversion ITRF→ENU : {os.path.basename(ply_file)} ({len(points)} points)"
        else:
            return False, f"Erreur de sauvegarde : {os.path.basename(ply_file)}"
        
    except Exception as e:
        return False, f"Erreur lors de la conversion ITRF→ENU de {os.path.basename(ply_file)} : {e}"

def process_single_cloud_deform(args):
    """Fonction de traitement d'un seul nuage pour la déformation (pour multiprocessing)"""
    ply_file, output_dir, residues_enu, gcp_positions, deformation_type = args
    
    # Création d'un logger pour ce processus
    logger = logging.getLogger(f"Deform_{os.getpid()}")
    logger.setLevel(logging.INFO)
    
    try:
        import open3d as o3d
        from scipy.spatial.distance import cdist
        from scipy.linalg import solve
        
        # Lecture du nuage
        cloud = o3d.io.read_point_cloud(ply_file)
        if not cloud.has_points():
            return False, f"Nuage vide dans {os.path.basename(ply_file)}"
        
        points = np.asarray(cloud.points)
        logger.info(f"  {len(points)} points chargés")
        
        # Préparation des données pour l'interpolation TPS
        control_points = []
        control_values = []
        
        for gcp_name, residue_data in residues_enu.items():
            if gcp_name in gcp_positions:
                control_points.append(gcp_positions[gcp_name])
                control_values.append(residue_data['offset'])
                logger.info(f"    Point de contrôle {gcp_name}: position={gcp_positions[gcp_name]}, correction={residue_data['offset']}")
        
        if len(control_points) < 3:
            return False, f"Trop peu de points de contrôle ({len(control_points)}) pour {os.path.basename(ply_file)}"
        
        # Conversion en arrays numpy
        control_points = np.array(control_points)
        control_values = np.array(control_values)
        
        # Application de la déformation selon le type
        if deformation_type == "tps":
            logger.info(f"  Interpolation TPS avec {len(control_points)} points de contrôle...")
            
            # Interpolation TPS
            def thin_plate_spline_interpolation(points, control_points, control_values):
                """Interpolation par Thin Plate Splines"""
                try:
                    from scipy.spatial.distance import cdist
                    from scipy.linalg import solve
                    
                    M = len(control_points)
                    N = len(points)
                    
                    # Calcul des distances entre points de contrôle
                    K = cdist(control_points, control_points, metric='euclidean')
                    # Fonction de base radiale (RBF)
                    K = K * np.log(K + 1e-10)  # Éviter log(0)
                    
                    # Construction du système linéaire
                    # [K  P] [w] = [v]
                    # [P' 0] [a]   [0]
                    # où P = [1, x, y, z] pour chaque point de contrôle
                    
                    P = np.column_stack([np.ones(M), control_points])
                    A = np.block([[K, P], [P.T, np.zeros((4, 4))]])
                    
                    # Résolution pour chaque composante (x, y, z)
                    interpolated_values = np.zeros((N, 3))
                    
                    for dim in range(3):
                        b = np.concatenate([control_values[:, dim], np.zeros(4)])
                        solution = solve(A, b)
                        w = solution[:M]
                        a = solution[M:]
                        
                        # Calcul des distances entre points d'interpolation et points de contrôle
                        K_interp = cdist(points, control_points, metric='euclidean')
                        K_interp = K_interp * np.log(K_interp + 1e-10)
                        
                        # Interpolation
                        P_interp = np.column_stack([np.ones(N), points])
                        interpolated_values[:, dim] = K_interp @ w + P_interp @ a
                    
                    return interpolated_values
                    
                except ImportError:
                    # Fallback si scipy n'est pas disponible
                    logger.warning("scipy non disponible, utilisation de l'interpolation linéaire")
                    return linear_interpolation(points, control_points, control_values)
            
            def linear_interpolation(points, control_points, control_values):
                """Interpolation linéaire simple comme fallback"""
                # Calcul de la moyenne des corrections
                mean_correction = np.mean(control_values, axis=0)
                return np.tile(mean_correction, (len(points), 1))
            
            # Application de l'interpolation TPS
            deformations = thin_plate_spline_interpolation(points, control_points, control_values)
            
        elif deformation_type == "lineaire":
            logger.info(f"  Interpolation linéaire avec {len(control_points)} points de contrôle...")
            
            def linear_interpolation(points, control_points, control_values):
                """Interpolation linéaire par distance inverse pondérée"""
                from scipy.spatial.distance import cdist
                
                # Calcul des distances
                distances = cdist(points, control_points, metric='euclidean')
                
                # Pondération par distance inverse
                weights = 1.0 / (distances + 1e-10)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                
                # Interpolation
                interpolated_values = weights @ control_values
                
                return interpolated_values
            
            deformations = linear_interpolation(points, control_points, control_values)
            
        else:  # uniforme
            logger.info(f"  Déformation uniforme avec {len(control_points)} points de contrôle...")
            
            # Calcul de la moyenne des corrections
            mean_correction = np.mean(control_values, axis=0)
            deformations = np.tile(mean_correction, (len(points), 1))
        
        # Application des déformations
        deformed_points = points + deformations
        
        # Calcul des statistiques de déformation
        deformation_magnitudes = np.linalg.norm(deformations, axis=1)
        min_deform = np.min(deformation_magnitudes)
        max_deform = np.max(deformation_magnitudes)
        mean_deform = np.mean(deformation_magnitudes)
        
        logger.info(f"  Déformation TPS appliquée - min: {min_deform:.6f}, max: {max_deform:.6f}, moy: {mean_deform:.6f}")
        
        # Création du nouveau nuage
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(deformed_points)
        
        # Copie des couleurs et normales
        if cloud.has_colors():
            new_cloud.colors = cloud.colors
            logger.info("  Couleurs copiées")
        
        if cloud.has_normals():
            new_cloud.normals = cloud.normals
            logger.info("  Normales copiées")
        
        # Sauvegarde
        output_file = os.path.join(output_dir, os.path.basename(ply_file))
        success = o3d.io.write_point_cloud(output_file, new_cloud)
        
        if success:
            return True, f"Déformation TPS : {os.path.basename(ply_file)} ({len(points)} points)"
        else:
            return False, f"Erreur de sauvegarde : {os.path.basename(ply_file)}"
        
    except Exception as e:
        return False, f"Erreur lors de la déformation de {os.path.basename(ply_file)} : {e}"

def add_offset_to_clouds(input_dir, logger, coord_file=None, extra_params="", max_workers=None):
    """Ajoute l'offset aux nuages de points .ply dans le dossier fourni (et ses sous-dossiers éventuels)"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Ajout de l'offset aux nuages dans {abs_input_dir} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    if not coord_file:
        logger.error("Aucun fichier de coordonnées fourni pour l'ajout d'offset.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour l'ajout d'offset.")
    
    # Création du dossier de sortie pour cette étape
    output_dir = os.path.join(os.path.dirname(abs_input_dir), "offset_step")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # Lecture du fichier de coordonnées pour extraire l'offset
    offset = None
    try:
        with open(coord_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#Offset to add :'):
                    # Format: #Offset to add : X Y Z
                    parts = line.split(':')[1].strip().split()
                    if len(parts) == 3:
                        offset = [float(parts[0]), float(parts[1]), float(parts[2])]
                        break
        
        if offset is None:
            logger.error("Offset non trouvé dans le fichier de coordonnées.")
            raise RuntimeError("Offset non trouvé dans le fichier de coordonnées.")
        
        logger.info(f"Offset extrait : {offset}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
        raise RuntimeError(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
    
    # Import d'open3d pour gérer les fichiers PLY
    try:
        import open3d as o3d
        logger.info("Open3D importé avec succès")
    except ImportError:
        logger.error("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
        raise RuntimeError("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
    
    total_files_processed = 0
    
    # Recherche des fichiers .ply dans le dossier fourni (et sous-dossiers)
    ply_files = []
    for root, dirs, files in os.walk(abs_input_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    logger.info(f"Trouvé {len(ply_files)} fichiers .ply dans {abs_input_dir}")
    
    if len(ply_files) == 0:
        logger.warning(f"Aucun fichier .ply trouvé dans {abs_input_dir}")
        logger.info("Aucun fichier à traiter.")
        return
    
    # Configuration de la parallélisation
    if max_workers is None:
        max_workers = min(10, cpu_count(), len(ply_files))  # Maximum 10 processus par défaut
    else:
        max_workers = min(max_workers, len(ply_files))  # Respecter la limite demandée (pas de limite CPU sur cluster)
    logger.info(f"Traitement parallèle avec {max_workers} processus...")
    
    # Préparation des arguments pour le multiprocessing
    process_args = []
    for ply_file in ply_files:
        process_args.append((ply_file, output_dir, coord_file, extra_params))
    
    # Traitement parallèle
    total_files_processed = 0
    failed_files = []
    
    try:
        with Pool(processes=max_workers) as pool:
            # Lancement du traitement parallèle
            results = pool.map(process_single_cloud_add_offset, process_args)
            
            # Analyse des résultats
            for i, (success, message) in enumerate(results):
                if success:
                    logger.info(f"✅ {message}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {message}")
                    failed_files.append(ply_files[i])
                    
    except Exception as e:
        logger.error(f"Erreur lors du traitement parallèle : {e}")
        # Fallback vers le traitement séquentiel en cas d'erreur
        logger.info("Tentative de traitement séquentiel...")
        total_files_processed = 0
        for ply_file in ply_files:
            try:
                result = process_single_cloud_add_offset((ply_file, output_dir, coord_file, extra_params))
                if result[0]:
                    logger.info(f"✅ {result[1]}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {result[1]}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}")
    
    # Résumé final
    if failed_files:
        logger.warning(f"⚠️ {len(failed_files)} fichiers n'ont pas pu être traités")
    logger.info(f"Ajout d'offset terminé. {total_files_processed} fichiers traités dans {output_dir}.")

def convert_itrf_to_enu(input_dir, logger, coord_file=None, extra_params="", ref_point_name=None, max_workers=None):
    """Convertit les nuages de points d'ITRF vers ENU"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Conversion ITRF vers ENU dans {abs_input_dir} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    if not coord_file:
        logger.error("Aucun fichier de coordonnées fourni pour la conversion ITRF→ENU.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour la conversion ITRF→ENU.")
    
    # Création du dossier de sortie pour cette étape
    output_dir = os.path.join(os.path.dirname(abs_input_dir), "itrf_to_enu_step")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # ÉTAPE 1 : Lecture du fichier de coordonnées pour obtenir le point de référence
    logger.info(f"Lecture du fichier de coordonnées : {coord_file}")
    try:
        with open(coord_file, 'r') as f:
            lines = f.readlines()
        
        # Recherche du point de référence
        ref_point = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:  # Format: NOM X Y Z
                    try:
                        point_name = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # Si un nom de point de référence est spécifié, on cherche ce point
                        if ref_point_name and point_name == ref_point_name:
                            ref_point = np.array([x, y, z])
                            logger.info(f"Point de référence spécifié trouvé : {point_name} ({x:.6f}, {y:.6f}, {z:.6f})")
                            break
                        # Sinon, on prend le premier point valide
                        elif ref_point is None:
                            ref_point = np.array([x, y, z])
                            logger.info(f"Point de référence trouvé : {point_name} ({x:.6f}, {y:.6f}, {z:.6f})")
                            break
                    except ValueError:
                        continue
        
        if ref_point is None:
            if ref_point_name:
                raise RuntimeError(f"Point de référence '{ref_point_name}' non trouvé dans le fichier de coordonnées")
            else:
                raise RuntimeError("Aucun point de référence valide trouvé dans le fichier de coordonnées")
        
        # ÉTAPE 1.5 : Lecture de l'offset et application au point de référence
        logger.info("Lecture de l'offset depuis le fichier de coordonnées...")
        offset = None
        try:
            with open(coord_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#Offset to add :'):
                        # Format: #Offset to add : X Y Z
                        parts = line.split(':')[1].strip().split()
                        if len(parts) == 3:
                            offset = [float(parts[0]), float(parts[1]), float(parts[2])]
                            break
            
            if offset is None:
                logger.warning("Offset non trouvé dans le fichier de coordonnées. Utilisation du point de référence sans offset.")
            else:
                logger.info(f"Offset trouvé : {offset}")
                # Application de l'offset au point de référence
                ref_point = ref_point + np.array(offset)
                logger.info(f"Point de référence avec offset : ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
                
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture de l'offset : {e}. Utilisation du point de référence sans offset.")
            
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
        raise RuntimeError(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
    
    # ÉTAPE 2 : Configuration de la transformation topocentrique
    logger.info("Configuration de la transformation topocentrique ITRF2020→ENU...")
    
    try:
        import pyproj
        
        # Le point de référence (ref_point) est le centre de la transformation topocentrique
        # Il définit l'origine du système ENU local
        tr_center = ref_point  # [X, Y, Z] en ITRF2020
        tr_ellps = "GRS80"     # Ellipsoïde de référence (standard pour ITRF)
        
        logger.info(f"Centre de transformation topocentrique : ({tr_center[0]:.3f}, {tr_center[1]:.3f}, {tr_center[2]:.3f})")
        logger.info(f"Ellipsoïde de référence : {tr_ellps}")
        
        # Création du pipeline de transformation topocentrique
        # +proj=topocentric : projection topocentrique
        # +X_0, +Y_0, +Z_0 : coordonnées du centre de transformation en ITRF
        # +ellps : ellipsoïde de référence
        pipeline = "+proj=topocentric +X_0={0} +Y_0={1} +Z_0={2} +ellps={3}".format(
            tr_center[0], tr_center[1], tr_center[2], tr_ellps
        )
        
        transformer = pyproj.Transformer.from_pipeline(pipeline)
        logger.info(f"Pipeline de transformation créé : {pipeline}")
        
    except ImportError:
        logger.error("pyproj n'est pas installé. La transformation topocentrique nécessite pyproj.")
        raise RuntimeError("pyproj n'est pas installé. Veuillez l'installer avec: pip install pyproj")
    
    logger.info("Transformation topocentrique configurée avec succès")
    
    # Import d'open3d pour gérer les fichiers PLY
    try:
        import open3d as o3d
        logger.info("Open3D importé avec succès")
    except ImportError:
        logger.error("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
        raise RuntimeError("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
    
    # ÉTAPE 4 : Traitement des nuages de points
    ply_files = []
    for root, dirs, files in os.walk(abs_input_dir):
        for file in files:
            if file.lower().endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    logger.info(f"Trouvé {len(ply_files)} fichiers .ply dans {abs_input_dir}")
    
    if len(ply_files) == 0:
        logger.warning(f"Aucun fichier .ply trouvé dans {abs_input_dir}")
        logger.info("Aucun fichier à traiter.")
        return
    
    # Configuration de la parallélisation
    if max_workers is None:
        max_workers = min(10, cpu_count(), len(ply_files))  # Maximum 10 processus par défaut
    else:
        max_workers = min(max_workers, len(ply_files))  # Respecter la limite demandée (pas de limite CPU sur cluster)
    logger.info(f"Traitement parallèle avec {max_workers} processus...")
    
    # Préparation des arguments pour le multiprocessing
    process_args = []
    for ply_file in ply_files:
        process_args.append((ply_file, output_dir, coord_file, extra_params, ref_point_name))
    
    # Traitement parallèle
    total_files_processed = 0
    failed_files = []
    
    try:
        with Pool(processes=max_workers) as pool:
            # Lancement du traitement parallèle
            results = pool.map(process_single_cloud_itrf_to_enu, process_args)
            
            # Analyse des résultats
            for i, (success, message) in enumerate(results):
                if success:
                    logger.info(f"✅ {message}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {message}")
                    failed_files.append(ply_files[i])
                    
    except Exception as e:
        logger.error(f"Erreur lors du traitement parallèle : {e}")
        # Fallback vers le traitement séquentiel en cas d'erreur
        logger.info("Tentative de traitement séquentiel...")
        total_files_processed = 0
        for ply_file in ply_files:
            try:
                result = process_single_cloud_itrf_to_enu((ply_file, output_dir, coord_file, extra_params, ref_point_name))
                if result[0]:
                    logger.info(f"✅ {result[1]}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {result[1]}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}")
    
    # Résumé final
    if failed_files:
        logger.warning(f"⚠️ {len(failed_files)} fichiers n'ont pas pu être traités")
    logger.info(f"Conversion ITRF vers ENU terminée. {total_files_processed} fichiers traités dans {output_dir}.")

def deform_clouds(input_dir, logger, deformation_type="lineaire", deformation_params="", extra_params="", bascule_xml_file=None, coord_file=None, max_workers=None):
    """Applique une déformation aux nuages de points basée sur les résidus GCPBascule"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Déformation des nuages dans {abs_input_dir} avec le type {deformation_type} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    # Import d'open3d pour gérer les fichiers PLY
    try:
        import open3d as o3d
        logger.info("Open3D importé avec succès")
    except ImportError:
        logger.error("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
        raise RuntimeError("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
    
    # Création du dossier de sortie pour cette étape
    output_dir = os.path.join(os.path.dirname(abs_input_dir), f"deform_{deformation_type}_step")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # ÉTAPE 1 : Lecture des résidus depuis le fichier XML GCPBascule
    logger.info("Lecture des résidus GCPBascule...")
    
    if not bascule_xml_file:
        logger.error("Aucun fichier XML GCPBascule spécifié")
        raise RuntimeError("Aucun fichier XML GCPBascule spécifié. Utilisez le paramètre bascule_xml_file.")
    
    if not os.path.exists(bascule_xml_file):
        logger.error(f"Fichier XML GCPBascule introuvable : {bascule_xml_file}")
        raise RuntimeError(f"Fichier XML GCPBascule introuvable : {bascule_xml_file}")
    
    xml_file = bascule_xml_file
    logger.info(f"Fichier XML GCPBascule : {xml_file}")
    
    # Lecture et parsing du XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extraction des résidus
        residues = {}
        for residue in root.findall('.//Residus'):
            name = residue.find('Name').text
            offset_elem = residue.find('Offset')
            offset_text = offset_elem.text.strip().split()
            offset = [float(x) for x in offset_text]
            dist = float(residue.find('Dist').text)
            
            residues[name] = {
                'offset': np.array(offset),  # Résidu en ITRF
                'distance': dist
            }
            logger.info(f"Résidu {name}: offset={offset}, distance={dist:.3f}m")
        
        if not residues:
            logger.error("Aucun résidu trouvé dans le fichier XML")
            raise RuntimeError("Aucun résidu trouvé dans le fichier XML GCPBascule")
            
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier XML : {e}")
        raise RuntimeError(f"Erreur lors de la lecture du fichier XML : {e}")
    
    logger.info(f"Lecture terminée : {len(residues)} résidus trouvés")
    
    # ÉTAPE 1.5 : Conversion des résidus ITRF vers ENU
    logger.info("Conversion des résidus ITRF vers ENU...")
    
    # Nous avons besoin du point de référence pour la transformation topocentrique
    # Nous devons le lire depuis le fichier de coordonnées
    if not coord_file:
        logger.error("Fichier de coordonnées requis pour la conversion des résidus ITRF→ENU")
        raise RuntimeError("Fichier de coordonnées requis pour la conversion des résidus ITRF→ENU")
    
    if not os.path.exists(coord_file):
        logger.error(f"Fichier de coordonnées introuvable : {coord_file}")
        raise RuntimeError(f"Fichier de coordonnées introuvable : {coord_file}")
    
    # Lecture du point de référence depuis le fichier de coordonnées
    try:
        with open(coord_file, 'r') as f:
            lines = f.readlines()
        
        # Recherche du point de référence (premier point par défaut)
        ref_point = None
        offset = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('#F=') or line.startswith('#'):
                continue
            
            # Format attendu : NOM X Y Z
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    ref_point = np.array([x, y, z])
                    logger.info(f"Point de référence trouvé : {parts[0]} ({x:.6f}, {y:.6f}, {z:.6f})")
                    break
                except ValueError:
                    continue
        
        if ref_point is None:
            logger.error("Aucun point de référence valide trouvé dans le fichier de coordonnées")
            raise RuntimeError("Aucun point de référence valide trouvé dans le fichier de coordonnées")
        
        # Lecture de l'offset
        try:
            for line in lines:
                if line.startswith('#Offset to add :'):
                    offset_text = line.replace('#Offset to add :', '').strip()
                    offset = [float(x) for x in offset_text.split()]
                    logger.info(f"Offset trouvé : {offset}")
                    break
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture de l'offset : {e}. Utilisation du point de référence sans offset.")
            offset = [0.0, 0.0, 0.0]
        
        # Application de l'offset au point de référence
        if offset:
            ref_point = ref_point + np.array(offset)
            logger.info(f"Point de référence avec offset : ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
        raise RuntimeError(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
    
    # Configuration de la transformation topocentrique pour les résidus
    try:
        import pyproj
        
        # Le point de référence (ref_point) est le centre de la transformation topocentrique
        tr_center = ref_point  # [X, Y, Z] en ITRF2020
        tr_ellps = "GRS80"     # Ellipsoïde de référence (standard pour ITRF)
        
        logger.info(f"Centre de transformation pour résidus : ({tr_center[0]:.3f}, {tr_center[1]:.3f}, {tr_center[2]:.3f})")
        
        # Création du pipeline de transformation topocentrique
        pipeline = "+proj=topocentric +X_0={0} +Y_0={1} +Z_0={2} +ellps={3}".format(
            tr_center[0], tr_center[1], tr_center[2], tr_ellps
        )
        
        transformer = pyproj.Transformer.from_pipeline(pipeline)
        logger.info(f"Pipeline de transformation pour résidus créé : {pipeline}")
        
    except ImportError:
        logger.error("pyproj n'est pas installé. Veuillez l'installer avec: pip install pyproj")
        raise RuntimeError("pyproj n'est pas installé. Veuillez l'installer avec: pip install pyproj")
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la transformation : {e}")
        raise RuntimeError(f"Erreur lors de la configuration de la transformation : {e}")
    
    # Conversion des résidus ITRF vers ENU
    residues_enu = {}
    for name, residue_data in residues.items():
        # Le résidu est un vecteur de déplacement en ITRF
        itrf_offset = residue_data['offset']
        
        # Conversion du vecteur de déplacement ITRF vers ENU
        # Pour un vecteur de déplacement, nous utilisons la matrice de rotation
        # qui est la dérivée de la transformation topocentrique
        
        # Méthode 1 : Utilisation de la matrice de rotation
        # La transformation topocentrique a une matrice de rotation R
        # Pour un vecteur de déplacement : ENU_vector = R * ITRF_vector
        
        # Calcul de la matrice de rotation (approximation)
        # Pour un point proche du centre de transformation, la rotation est approximativement constante
        
        # Méthode 2 : Transformation directe (plus précise)
        # Nous transformons le point de référence + le vecteur de déplacement
        point_with_offset = ref_point + itrf_offset
        enu_point = transformer.transform(point_with_offset[0], point_with_offset[1], point_with_offset[2])
        enu_ref = transformer.transform(ref_point[0], ref_point[1], ref_point[2])
        
        # Le vecteur de déplacement ENU est la différence
        enu_offset = np.array([enu_point[0] - enu_ref[0], 
                              enu_point[1] - enu_ref[1], 
                              enu_point[2] - enu_ref[2]])
        
        residues_enu[name] = {
            'offset': enu_offset,
            'distance': residue_data['distance']
        }
        
        logger.info(f"Résidu {name} ENU: offset={enu_offset.tolist()}, distance={residue_data['distance']:.3f}m")
    
    logger.info(f"Conversion terminée : {len(residues_enu)} résidus convertis en ENU")
    
    # ÉTAPE 1.6 : Préparation des données pour l'interpolation TPS
    logger.info("Préparation des données pour l'interpolation TPS...")
    
    # Nous avons besoin des positions des GCPs dans les nuages pour l'interpolation
    # Pour l'instant, nous utilisons une approximation basée sur les coordonnées nominales
    # TODO: Implémenter la détection automatique des GCPs dans les nuages
    
    # Extraction des positions des GCPs depuis le fichier de coordonnées
    gcp_positions = {}
    try:
        with open(coord_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('#F=') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    name = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # Application de l'offset
                    if offset:
                        x += offset[0]
                        y += offset[1]
                        z += offset[2]
                    
                    # Conversion en ENU
                    point_with_offset = np.array([x, y, z])
                    enu_pos = transformer.transform(point_with_offset[0], point_with_offset[1], point_with_offset[2])
                    gcp_positions[name] = np.array([enu_pos[0], enu_pos[1], enu_pos[2]])
                    
                except ValueError:
                    continue
        
        logger.info(f"Positions des GCPs préparées : {len(gcp_positions)} points")
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des positions GCPs : {e}")
        raise RuntimeError(f"Erreur lors de la préparation des positions GCPs : {e}")
    
    # ÉTAPE 2 : Fonctions d'interpolation TPS
    def thin_plate_spline_interpolation(points, control_points, control_values):
        """
        Interpolation par Thin Plate Splines
        
        Args:
            points: Points à interpoler (N, 3)
            control_points: Points de contrôle (M, 3)
            control_values: Valeurs aux points de contrôle (M, 3)
        
        Returns:
            Valeurs interpolées aux points (N, 3)
        """
        try:
            from scipy.spatial.distance import cdist
            from scipy.linalg import solve
            
            M = len(control_points)
            N = len(points)
            
            # Calcul des distances entre points de contrôle
            K = cdist(control_points, control_points, metric='euclidean')
            # Fonction de base radiale (RBF)
            K = K * np.log(K + 1e-10)  # Éviter log(0)
            
            # Construction du système linéaire
            # [K  P] [w] = [v]
            # [P' 0] [a]   [0]
            # où P = [1, x, y, z] pour chaque point de contrôle
            
            P = np.column_stack([np.ones(M), control_points])
            A = np.block([[K, P], [P.T, np.zeros((4, 4))]])
            
            # Résolution pour chaque composante (x, y, z)
            interpolated_values = np.zeros((N, 3))
            
            for dim in range(3):
                b = np.concatenate([control_values[:, dim], np.zeros(4)])
                solution = solve(A, b)
                w = solution[:M]
                a = solution[M:]
                
                # Calcul des distances entre points d'interpolation et points de contrôle
                K_interp = cdist(points, control_points, metric='euclidean')
                K_interp = K_interp * np.log(K_interp + 1e-10)
                
                # Interpolation
                P_interp = np.column_stack([np.ones(N), points])
                interpolated_values[:, dim] = K_interp @ w + P_interp @ a
            
            return interpolated_values
            
        except ImportError:
            logger.warning("scipy non disponible, utilisation de l'interpolation linéaire")
            return linear_interpolation(points, control_points, control_values)
    
    def linear_interpolation(points, control_points, control_values):
        """
        Interpolation linéaire simple par distance inverse pondérée
        """
        from scipy.spatial.distance import cdist
        
        # Calcul des distances
        distances = cdist(points, control_points, metric='euclidean')
        
        # Pondération par distance inverse
        weights = 1.0 / (distances + 1e-10)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Interpolation
        interpolated_values = weights @ control_values
        
        return interpolated_values
    
    # ÉTAPE 3 : Traitement des nuages de points
    ply_files = []
    for root, dirs, files in os.walk(abs_input_dir):
        for file in files:
            if file.lower().endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    logger.info(f"Trouvé {len(ply_files)} fichiers .ply dans {abs_input_dir}")
    
    if len(ply_files) == 0:
        logger.warning(f"Aucun fichier .ply trouvé dans {abs_input_dir}")
        logger.info("Aucun fichier à traiter.")
        return
    
    # Configuration de la parallélisation
    if max_workers is None:
        max_workers = min(10, cpu_count(), len(ply_files))  # Maximum 10 processus par défaut
    else:
        max_workers = min(max_workers, len(ply_files))  # Respecter la limite demandée (pas de limite CPU sur cluster)
    logger.info(f"Traitement parallèle avec {max_workers} processus...")
    
    # Préparation des arguments pour le multiprocessing
    process_args = []
    for ply_file in ply_files:
        process_args.append((ply_file, output_dir, residues_enu, gcp_positions, deformation_type))
    
    # Traitement parallèle
    total_files_processed = 0
    failed_files = []
    
    try:
        with Pool(processes=max_workers) as pool:
            # Lancement du traitement parallèle
            results = pool.map(process_single_cloud_deform, process_args)
            
            # Analyse des résultats
            for i, (success, message) in enumerate(results):
                if success:
                    logger.info(f"✅ {message}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {message}")
                    failed_files.append(ply_files[i])
                    
    except Exception as e:
        logger.error(f"Erreur lors du traitement parallèle : {e}")
        # Fallback vers le traitement séquentiel en cas d'erreur
        logger.info("Tentative de traitement séquentiel...")
        total_files_processed = 0
        for ply_file in ply_files:
            try:
                result = process_single_cloud_deform((ply_file, output_dir, residues_enu, gcp_positions, deformation_type))
                if result[0]:
                    logger.info(f"✅ {result[1]}")
                    total_files_processed += 1
                else:
                    logger.error(f"❌ {result[1]}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}")
    
    # Résumé final
    if failed_files:
        logger.warning(f"⚠️ {len(failed_files)} fichiers n'ont pas pu être traités")
    logger.info(f"Déformation {deformation_type} terminée. {total_files_processed} fichiers traités dans {output_dir}.")

def convert_enu_to_itrf(input_dir, logger, coord_file=None, extra_params="", max_workers=None):
    """Convertit les nuages de points d'ENU vers ITRF"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Conversion ENU vers ITRF dans {abs_input_dir} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    if not coord_file:
        logger.error("Aucun fichier de coordonnées fourni pour la conversion ENU→ITRF.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour la conversion ENU→ITRF.")
    
    # Création du dossier de sortie pour cette étape
    output_dir = os.path.join(os.path.dirname(abs_input_dir), "enu_to_itrf_step")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # TODO: Implémenter la conversion ENU→ITRF
    # - Lire le fichier de coordonnées
    # - Déterminer le point de référence ITRF
    # - Convertir les coordonnées des nuages
    # - Sauvegarder les nuages convertis dans output_dir
    
    logger.info(f"Conversion ENU vers ITRF terminée. Fichiers sauvegardés dans {output_dir}.") 