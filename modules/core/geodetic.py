import os
import numpy as np
import logging
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

# PATCH RASTERIO CIBLÉ - Seulement les modules vraiment nécessaires
def patch_rasterio_essentials():
    """Patch ciblé pour les modules rasterio essentiels"""
    import types
    
    # Modules vraiment utilisés dans le code
    essential_modules = [
        'rasterio.sample',    # Utilisé dans process_single_cloud_orthoimage
        'rasterio.vrt',       # Utilisé dans process_single_cloud_orthoimage  
        'rasterio._features', # Erreur actuelle
        'rasterio.coords',    # Utilisé pour BoundingBox
    ]
    
    for module_name in essential_modules:
        try:
            __import__(module_name)
        except ImportError:
            # Créer un module minimal avec seulement ce qui est nécessaire
            module = types.ModuleType(module_name)
            
            # Cas spéciaux pour certains modules
            if module_name == 'rasterio.coords':
                class BoundingBox:
                    def __init__(self, left, bottom, right, top):
                        self.left = left
                        self.bottom = bottom
                        self.right = right
                        self.top = top
                module.BoundingBox = BoundingBox
                logging.getLogger(__name__).warning(f"PATCH: {module_name}.BoundingBox créé")
            
            # Injecter le module dans rasterio
            module_parts = module_name.split('.')
            if len(module_parts) == 2:
                parent_name, child_name = module_parts
                if parent_name in globals():
                    parent = globals()[parent_name]
                    setattr(parent, child_name, module)
                    logging.getLogger(__name__).warning(f"PATCH: {module_name} créé (module minimal)")

# Appliquer le patch au démarrage
patch_rasterio_essentials()

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
        
        # Application de la transformation topocentrique (optimisée)
        # Traitement par chunks pour éviter surcharge mémoire avec gros nuages
        chunk_size = 100000  # 100k points par chunk
        total_points = len(points)
        arr_pts_ENU = np.zeros_like(points)
        
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            chunk_points = points[i:end_idx]
            
            # Extraction directe sans conversion list inutile
            arr_x = chunk_points[:, 0]
            arr_y = chunk_points[:, 1] 
            arr_z = chunk_points[:, 2]
            
            # Transformation du chunk
            chunk_enu = np.array(transformer.transform(arr_x, arr_y, arr_z)).T
            arr_pts_ENU[i:end_idx] = chunk_enu
        
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
        # Limiter par CPU disponibles ET fichiers pour éviter surcharge cluster
        max_workers = min(max_workers, cpu_count(), len(ply_files))
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
        # Limiter par CPU disponibles ET fichiers pour éviter surcharge cluster
        max_workers = min(max_workers, cpu_count(), len(ply_files))
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
        # Limiter par CPU disponibles ET fichiers pour éviter surcharge cluster
        max_workers = min(max_workers, cpu_count(), len(ply_files))
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

def create_orthoimage_from_pointcloud(input_dir, logger, output_dir=None, resolution=0.1, height_field="z", color_field="rgb", max_workers=None):
    """Crée une orthoimage à partir des nuages de points .ply dans le dossier fourni"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Création d'orthoimage à partir des nuages dans {abs_input_dir} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    # Création du dossier de sortie pour cette étape
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(abs_input_dir), "orthoimage_step")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # Import d'open3d pour gérer les fichiers PLY
    try:
        import open3d as o3d
        logger.info("Open3D importé avec succès")
    except ImportError:
        logger.error("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
        raise RuntimeError("Open3D n'est pas installé. Veuillez l'installer avec: pip install open3d")
    
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
        max_workers = min(max_workers, len(ply_files))  # Respecter la limite demandée
    logger.info(f"Traitement parallèle avec {max_workers} processus...")
    
    # Préparation des arguments pour le multiprocessing
    process_args = []
    for ply_file in ply_files:
        process_args.append((ply_file, output_dir, resolution, height_field, color_field))
    
    # Traitement parallèle
    total_files_processed = 0
    failed_files = []
    
    try:
        with Pool(processes=max_workers) as pool:
            # Lancement du traitement parallèle
            results = pool.map(process_single_cloud_orthoimage, process_args)
            
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
                result = process_single_cloud_orthoimage((ply_file, output_dir, resolution, height_field, color_field))
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
    logger.info(f"Création d'orthoimage terminée. {total_files_processed} fichiers traités dans {output_dir}.")

def process_single_cloud_orthoimage(args):
    """Fonction de traitement d'un seul nuage pour la création d'orthoimage (pour multiprocessing)"""
    ply_file, output_dir, resolution, height_field, color_field = args
    
    # Création d'un logger pour ce processus
    logger = logging.getLogger(f"Orthoimage_{os.getpid()}")
    logger.setLevel(logging.INFO)
    
    try:
        import open3d as o3d
        import numpy as np
        from PIL import Image
        
        # DEBUG: Import rasterio avec gestion d'erreur détaillée
        try:
            print("DEBUG: Tentative d'import rasterio...")
            import rasterio
            print(f"DEBUG: rasterio importé avec succès, version: {rasterio.__version__}")
            print(f"DEBUG: Modules rasterio disponibles: {[x for x in dir(rasterio) if not x.startswith('_')]}")
        except ImportError as e:
            print(f"DEBUG: Erreur import rasterio: {e}")
            raise
        except Exception as e:
            print(f"DEBUG: Erreur inattendue import rasterio: {e}")
            raise
        
        try:
            print("DEBUG: Tentative d'import from_origin...")
            from rasterio.transform import from_origin
            print("DEBUG: from_origin importé avec succès")
        except ImportError as e:
            print(f"DEBUG: Erreur import from_origin: {e}")
            raise
        except Exception as e:
            print(f"DEBUG: Erreur inattendue import from_origin: {e}")
            raise
        
        # PATCH: Modules rasterio gérés globalement par patch_rasterio_essentials()
        
        # Lecture du nuage
        cloud = o3d.io.read_point_cloud(ply_file)
        if not cloud.has_points():
            return False, f"Nuage vide dans {os.path.basename(ply_file)}"
        
        points = np.asarray(cloud.points)
        logger.info(f"  {len(points)} points chargés")
        
        # Calcul des limites du nuage
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Création de la grille pour l'orthoimage avec résolution précise
        # CORRECTION: Inverser l'axe Y pour éviter la rotation de 180°
        # IMPORTANT: Rasterio attend une grille où l'origine (0,0) est en haut à gauche
        x_range = np.arange(min_coords[0], max_coords[0] + resolution, resolution)
        y_range = np.arange(max_coords[1], min_coords[1] - resolution, -resolution)  # Y inversé !
        
        logger.info(f"  Grille créée : {len(x_range)} x {len(y_range)} pixels")
        logger.info(f"  Résolution : {resolution}m par pixel")
        logger.info(f"  Étendue : X[{min_coords[0]:.3f}, {max_coords[0]:.3f}], Y[{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
        
        # Création des matrices pour l'orthoimage
        height_image = np.full((len(y_range), len(x_range)), np.nan)
        color_image = np.zeros((len(y_range), len(x_range), 3), dtype=np.uint8)
        
        # Compteurs pour les statistiques
        points_processed = 0
        points_outside_grid = 0
        
        # Rasterisation des points
        for i, point in enumerate(points):
            # Conversion des coordonnées en indices de grille
            # CORRECTION: Ajuster pour la grille Y inversée
            x_idx = int((point[0] - min_coords[0]) / resolution)
            y_idx = int((max_coords[1] - point[1]) / resolution)  # Y inversé !
            
            # Vérification des limites
            if 0 <= x_idx < len(x_range) and 0 <= y_idx < len(y_range):
                points_processed += 1
                # Mise à jour de la hauteur (prendre la plus haute si plusieurs points)
                if np.isnan(height_image[y_idx, x_idx]) or point[2] > height_image[y_idx, x_idx]:
                    height_image[y_idx, x_idx] = point[2]
                
                # Mise à jour de la couleur
                if cloud.has_colors():
                    colors = np.asarray(cloud.colors)
                    color = colors[i]
                    # Conversion de [0,1] vers [0,255]
                    color_255 = (color * 255).astype(np.uint8)
                    color_image[y_idx, x_idx] = color_255
            else:
                points_outside_grid += 1
        
        logger.info(f"  Points traités : {points_processed}/{len(points)}")
        if points_outside_grid > 0:
            logger.warning(f"  Points hors grille : {points_outside_grid}")
        
        # Création de l'image de hauteur
        # CORRECTION: Garder les hauteurs réelles en mètres, pas de normalisation !
        # height_image contient déjà les hauteurs réelles en mètres
        # Pas de normalisation, hauteurs réelles conservées
        
        # Calcul du géoréférencement avec rasterio
        # IMPORTANT: Les coordonnées sont dans le repère local ENU (East-North-Up)
        # - X = East (Est) en mètres par rapport au point de référence
        # - Y = North (Nord) en mètres par rapport au point de référence
        # - Z = Up (Haut) en mètres par rapport au point de référence
        
        # PROBLÈME IDENTIFIÉ: Chaque orthoimage a une origine différente
        # SOLUTION: Utiliser une origine de référence commune pour toutes les orthoimages
        
        # Origine de référence commune (à définir selon votre point de référence ENU)
        # TODO: Récupérer cette origine depuis la fonction d'appel
        reference_origin_east = 0.0  # Point de référence ENU (East)
        reference_origin_north = 0.0  # Point de référence ENU (North)
        
        # Origine géographique de cette orthoimage spécifique
        # IMPORTANT: Coordonnées absolues dans le repère ENU, pas relatives au nuage
        # CORRECTION: Avec la grille Y inversée, l'origine est maintenant correcte
        origin_x = min_coords[0]  # Coordonnée East minimale (absolue)
        origin_y = max_coords[1]  # Coordonnée North maximale (absolue) = coin supérieur de la grille
        
        # Création de la transformation affine
        # Note: from_origin(x, y, pixel_width, pixel_height) où x,y sont les coordonnées du coin supérieur gauche
        # MAINTENANT CORRECT: L'origine Y correspond au coin supérieur de la grille inversée
        transform = from_origin(origin_x, origin_y, resolution, resolution)
        
        logger.info(f"  Géoréférencement ENU: origine East={origin_x:.3f}m, North={origin_y:.3f}m, résolution {resolution}m")
        logger.info(f"  Grille: X[{min_coords[0]:.3f}, {max_coords[0]:.3f}], Y[{max_coords[1]:.3f} → {min_coords[1]:.3f}] (Y inversé)")
        logger.info(f"  Point de référence ENU: East={reference_origin_east:.3f}m, North={reference_origin_north:.3f}m")
        logger.info(f"  Coordonnées absolues: cette orthoimage va de East {min_coords[0]:.3f}m à {max_coords[0]:.3f}m")
        logger.info(f"  Coordonnées absolues: cette orthoimage va de North {min_coords[1]:.3f}m à {max_coords[1]:.3f}m")
        
        # Vérification de l'orientation de la grille (CORRIGÉE)
        logger.info(f"  Orientation grille CORRIGÉE:")
        logger.info(f"    pixel (0,0) = East={min_coords[0]:.3f}m, North={max_coords[1]:.3f}m (coin supérieur gauche)")
        logger.info(f"    pixel (0,{len(y_range)-1}) = East={min_coords[0]:.3f}m, North={min_coords[1]:.3f}m (coin inférieur gauche)")
        logger.info(f"    pixel ({len(x_range)-1},0) = East={max_coords[0]:.3f}m, North={max_coords[1]:.3f}m (coin supérieur droit)")
        logger.info(f"    pixel ({len(x_range)-1},{len(y_range)-1}) = East={max_coords[0]:.3f}m, North={min_coords[1]:.3f}m (coin inférieur droit)")
        
        # Métadonnées détaillées
        metadata = {
            'Software': 'PhotoGeoAlign Orthoimage Generator',
            'Resolution': f'{resolution}m per pixel',
            'Origin_X': f'{origin_x:.6f}',
            'Origin_Y': f'{origin_y:.6f}',
            'Extent_X': f'{max_coords[0] - min_coords[0]:.3f}m',
            'Extent_Y': f'{max_coords[1] - min_coords[1]:.3f}m',
            'Points_Processed': str(points_processed),
            'Height_Range': f'{np.nanmin(height_image):.3f}m to {np.nanmax(height_image):.3f}m'
        }
        
        # Sauvegarde de l'image de hauteur en GeoTIFF
        height_filename = os.path.splitext(os.path.basename(ply_file))[0] + "_height.tif"
        height_path = os.path.join(output_dir, height_filename)
        
        # Sauvegarde avec rasterio pour le géoréférencement
        # IMPORTANT: Les coordonnées sont dans un repère local ENU (East-North-Up)
        # Ce repère est centré sur le point de référence après conversion ITRF → ENU
        # Nous utilisons un CRS local qui respecte les coordonnées East-North en mètres
        
        # CRS local ENU (repère local East-North)
        # Ce CRS définit un repère cartésien local où :
        # - X = East (Est) en mètres par rapport au point de référence
        # - Y = North (Nord) en mètres par rapport au point de référence  
        # - Z = Up (Haut) en mètres par rapport au point de référence
        
        # CRS cartésien local simple (pas de projection complexe)
        # Préserve exactement les distances et angles du repère ENU
        crs_string = '+proj=geocent +ellps=WGS84 +units=m +no_defs'
        
        with rasterio.open(
            height_path,
            'w',
            driver='GTiff',
            height=height_image.shape[0],
            width=height_image.shape[1],
            count=1,
            dtype=height_image.dtype,  # Utiliser le type des hauteurs réelles
            crs=crs_string,
            transform=transform,
            nodata=np.nan  # Utiliser NaN pour les pixels sans données
        ) as dst:
            dst.write(height_image, 1)  # Écrire les hauteurs réelles
            # Ajout des métadonnées
            dst.update_tags(**metadata)
        
        # Sauvegarde de l'image couleur en GeoTIFF
        color_filename = os.path.splitext(os.path.basename(ply_file))[0] + "_color.tif"
        color_path = os.path.join(output_dir, color_filename)
        
        with rasterio.open(
            color_path,
            'w',
            driver='GTiff',
            height=color_image.shape[0],
            width=color_image.shape[1],
            count=3,
            dtype=color_image.dtype,
            crs=crs_string,  # Utiliser le même CRS que l'image de hauteur
            transform=transform,
            photometric='rgb'
        ) as dst:
            dst.write(color_image[:,:,0], 1)  # Rouge
            dst.write(color_image[:,:,1], 2)  # Vert
            dst.write(color_image[:,:,2], 3)  # Bleu
            # Ajout des métadonnées
            dst.update_tags(**metadata)
        
        return True, f"Orthoimage créée : {os.path.basename(ply_file)} (hauteur: {height_filename}, couleur: {color_filename})"
        
    except Exception as e:
        return False, f"Erreur lors de la création d'orthoimage de {os.path.basename(ply_file)} : {e}"

def create_unified_orthoimage_and_dtm(input_dir, logger, output_dir=None, resolution=0.1, max_workers=None):
    """Crée une orthoimage et un MNT unifiés à partir de tous les nuages de points .ply dans le dossier fourni"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Création d'orthoimage et MNT unifiés à partir des nuages dans {abs_input_dir} ...")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    # Création du dossier de sortie pour cette étape
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(abs_input_dir), "unified_orthoimage_dtm")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # Import d'open3d pour gérer les fichiers PLY
    try:
        import open3d as o3d
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        
        # PATCH: Modules rasterio gérés globalement par patch_rasterio_essentials()
        
        logger.info("Open3D et Rasterio importés avec succès")
    except ImportError as e:
        logger.error(f"Importation échouée : {e}")
        raise RuntimeError(f"Importation échouée : {e}")
    
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
    
    # ÉTAPE 1 : Calculer l'étendue globale de tous les nuages
    logger.info("Calcul de l'étendue globale de tous les nuages...")
    global_min_coords = None
    global_max_coords = None
    
    for ply_file in ply_files:
        try:
            cloud = o3d.io.read_point_cloud(ply_file)
            if not cloud.has_points():
                continue
            
            points = np.asarray(cloud.points)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            if global_min_coords is None:
                global_min_coords = min_coords
                global_max_coords = max_coords
            else:
                global_min_coords = np.minimum(global_min_coords, min_coords)
                global_max_coords = np.maximum(global_max_coords, max_coords)
                
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture de {os.path.basename(ply_file)} : {e}")
            continue
    
    if global_min_coords is None:
        logger.error("Aucun nuage valide trouvé")
        return
    
    logger.info(f"Étendue globale : X[{global_min_coords[0]:.3f}, {global_max_coords[0]:.3f}], Y[{global_min_coords[1]:.3f}, {global_max_coords[1]:.3f}]")
    
    # Création de la grille globale
    x_range = np.arange(global_min_coords[0], global_max_coords[0] + resolution, resolution)
    y_range = np.arange(global_min_coords[1], global_max_coords[1] + resolution, resolution)
    
    logger.info(f"Grille globale créée : {len(x_range)} x {len(y_range)} pixels")
    logger.info(f"Résolution : {resolution}m par pixel")
    
    # ÉTAPE 2 : Traitement des nuages individuels
    logger.info("Traitement des nuages individuels...")
    
    # Configuration de la parallélisation
    if max_workers is None:
        max_workers = min(10, cpu_count(), len(ply_files))
    else:
        max_workers = min(max_workers, len(ply_files))
    logger.info(f"Traitement parallèle avec {max_workers} processus...")
    
    # Préparation des arguments pour le multiprocessing
    process_args = []
    for ply_file in ply_files:
        process_args.append((ply_file, global_min_coords, global_max_coords, resolution, len(x_range), len(y_range)))
    
    # Traitement parallèle
    individual_rasters = []
    failed_files = []
    
    try:
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_single_cloud_for_unified, process_args)
            
            for i, result in enumerate(results):
                if result[0]:  # Succès
                    individual_rasters.append(result[1])
                    logger.info(f"✅ {os.path.basename(ply_files[i])} traité")
                else:
                    logger.error(f"❌ {result[1]}")
                    failed_files.append(ply_files[i])
                    
    except Exception as e:
        logger.error(f"Erreur lors du traitement parallèle : {e}")
        # Fallback vers le traitement séquentiel
        logger.info("Tentative de traitement séquentiel...")
        for ply_file in ply_files:
            try:
                result = process_single_cloud_for_unified((ply_file, global_min_coords, global_max_coords, resolution, len(x_range), len(y_range)))
                if result[0]:
                    individual_rasters.append(result[1])
                    logger.info(f"✅ {os.path.basename(ply_file)} traité")
                else:
                    logger.error(f"❌ {result[1]}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}")
    
    if not individual_rasters:
        logger.error("Aucun raster individuel créé")
        return
    
    logger.info(f"Rasters individuels créés : {len(individual_rasters)}")
    
    # ÉTAPE 3 : Fusion des rasters
    logger.info("Fusion des rasters individuels...")
    
    # Initialisation des rasters unifiés
    unified_height = np.full((len(y_range), len(x_range)), np.nan)
    unified_color = np.zeros((len(y_range), len(x_range), 3), dtype=np.uint8)
    unified_count = np.zeros((len(y_range), len(x_range)), dtype=np.uint8)
    
    # Fusion des rasters
    for height_raster, color_raster in individual_rasters:
        # Pour la hauteur : prendre le maximum (éviter les occlusions)
        mask = ~np.isnan(height_raster)
        unified_height[mask] = np.maximum(unified_height[mask], height_raster[mask])
        
        # Pour la couleur : moyenne pondérée
        for c in range(3):  # RGB
            color_channel = color_raster[:, :, c]
            valid_mask = (color_channel > 0) & ~np.isnan(height_raster)
            if np.any(valid_mask):
                # Moyenne pondérée par la hauteur
                unified_color[valid_mask, c] = (
                    (unified_color[valid_mask, c] * unified_count[valid_mask] + 
                     color_channel[valid_mask] * height_raster[valid_mask]) / 
                    (unified_count[valid_mask] + height_raster[valid_mask])
                ).astype(np.uint8)
        
        # Compteur pour la moyenne
        unified_count[mask] += 1
    
    # Normalisation finale de la couleur
    valid_color_mask = unified_count > 0
    for c in range(3):
        unified_color[valid_color_mask, c] = (unified_color[valid_color_mask, c] / unified_count[valid_color_mask]).astype(np.uint8)
    
    logger.info(f"Fusion terminée. Pixels valides : {np.sum(~np.isnan(unified_height))}")
    
    # ÉTAPE 4 : Sauvegarde des résultats unifiés
    logger.info("Sauvegarde des résultats unifiés...")
    
    # Calcul du géoréférencement
    origin_x = global_min_coords[0]
    origin_y = global_max_coords[1]  # Y inversé pour l'image
    transform = from_origin(origin_x, origin_y, resolution, resolution)
    
    # Métadonnées
    metadata = {
        'Software': 'PhotoGeoAlign Unified Orthoimage/DTM Generator',
        'Resolution': f'{resolution}m per pixel',
        'Origin_X': f'{origin_x:.6f}',
        'Origin_Y': f'{origin_y:.6f}',
        'Extent_X': f'{global_max_coords[0] - global_min_coords[0]:.3f}m',
        'Extent_Y': f'{global_max_coords[1] - global_min_coords[1]:.3f}m',
        'Source_Files': str(len(ply_files)),
        'Valid_Pixels': str(int(np.sum(~np.isnan(unified_height)))),
        'Height_Range': f'{np.nanmin(unified_height):.3f}m to {np.nanmax(unified_height):.3f}m' if np.sum(~np.isnan(unified_height)) > 0 else 'No valid pixels'
    }
    
    # Sauvegarde de l'orthoimage unifiée
    orthoimage_filename = "unified_orthoimage.tif"
    orthoimage_path = os.path.join(output_dir, orthoimage_filename)
    
    with rasterio.open(
        orthoimage_path,
        'w',
        driver='GTiff',
        height=unified_color.shape[0],
        width=unified_color.shape[1],
        count=3,
        dtype=unified_color.dtype,
        crs='+proj=utm +zone=30 +datum=WGS84',
        transform=transform,
        photometric='rgb'
    ) as dst:
        dst.write(unified_color[:,:,0], 1)  # Rouge
        dst.write(unified_color[:,:,1], 2)  # Vert
        dst.write(unified_color[:,:,2], 3)  # Bleu
        dst.update_tags(**metadata)
    
    # Sauvegarde du MNT unifié
    dtm_filename = "unified_dtm.tif"
    dtm_path = os.path.join(output_dir, dtm_filename)
    
    # Conversion des NaN en nodata
    dtm_data = unified_height.copy()
    dtm_data = np.where(np.isnan(dtm_data), 0, dtm_data)
    
    with rasterio.open(
        dtm_path,
        'w',
        driver='GTiff',
        height=dtm_data.shape[0],
        width=dtm_data.shape[1],
        count=1,
        dtype=dtm_data.dtype,
        crs='+proj=utm +zone=30 +datum=WGS84',
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(dtm_data, 1)
        dst.update_tags(**metadata)
    
    logger.info(f"Orthoimage unifiée sauvegardée : {orthoimage_filename}")
    logger.info(f"MNT unifié sauvegardé : {dtm_filename}")
    logger.info(f"Création d'orthoimage et MNT unifiés terminée dans {output_dir}")

def process_single_cloud_for_unified(args):
    """Fonction de traitement d'un seul nuage pour la création d'orthoimage/MNT unifiés (pour multiprocessing)"""
    ply_file, global_min_coords, global_max_coords, resolution, grid_width, grid_height = args
    
    # Création d'un logger pour ce processus
    logger = logging.getLogger(f"Unified_{os.getpid()}")
    logger.setLevel(logging.INFO)
    
    try:
        import open3d as o3d
        import numpy as np
        
        # Lecture du nuage
        cloud = o3d.io.read_point_cloud(ply_file)
        if not cloud.has_points():
            return False, f"Nuage vide dans {os.path.basename(ply_file)}"
        
        points = np.asarray(cloud.points)
        logger.info(f"  {len(points)} points chargés")
        
        # Création des matrices pour ce nuage
        height_raster = np.full((grid_height, grid_width), np.nan)
        color_raster = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Rasterisation des points
        points_processed = 0
        for i, point in enumerate(points):
            # Conversion des coordonnées en indices de grille globale
            x_idx = int((point[0] - global_min_coords[0]) / resolution)
            y_idx = int((point[1] - global_min_coords[1]) / resolution)
            
            # Vérification des limites
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                points_processed += 1
                # Mise à jour de la hauteur (prendre la plus haute si plusieurs points)
                if np.isnan(height_raster[y_idx, x_idx]) or point[2] > height_raster[y_idx, x_idx]:
                    height_raster[y_idx, x_idx] = point[2]
                
                # Mise à jour de la couleur
                if cloud.has_colors():
                    colors = np.asarray(cloud.colors)
                    color = colors[i]
                    # Conversion de [0,1] vers [0,255]
                    color_255 = (color * 255).astype(np.uint8)
                    color_raster[y_idx, x_idx] = color_255
        
        logger.info(f"  Points traités : {points_processed}/{len(points)}")
        
        return True, (height_raster, color_raster)
        
    except Exception as e:
        return False, f"Erreur lors du traitement de {os.path.basename(ply_file)} : {e}"

def merge_orthoimages_and_dtm(input_dir, logger, output_dir=None, target_resolution=None, max_workers=None, color_fusion_method="average"):
    """Fusionne les orthoimages et MNT individuels déjà générés en orthoimage et MNT unifiés"""
    abs_input_dir = os.path.abspath(input_dir)
    logger.info(f"Fusion des orthoimages et MNT dans {abs_input_dir} ...")
    logger.info(f"Méthode de fusion des couleurs : {color_fusion_method}")
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(abs_input_dir):
        logger.error(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
        raise RuntimeError(f"Le dossier d'entrée n'existe pas : {abs_input_dir}")
    
    if not os.path.isdir(abs_input_dir):
        logger.error(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
        raise RuntimeError(f"Le chemin spécifié n'est pas un dossier : {abs_input_dir}")
    
    # Création du dossier de sortie pour cette étape
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(abs_input_dir), "unified_orthoimage_dtm")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Dossier de sortie créé : {output_dir}")
    
    # Import des bibliothèques nécessaires
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        
        # PATCH: Modules rasterio gérés globalement par patch_rasterio_essentials()
        
        logger.info("Rasterio importé avec succès")
    except ImportError as e:
        logger.error(f"Importation échouée : {e}")
        raise RuntimeError(f"Importation échouée : {e}")
    
    # Recherche des fichiers .tif dans le dossier fourni
    height_files = []
    color_files = []
    
    for file in os.listdir(abs_input_dir):
        if file.endswith('_height.tif'):
            height_files.append(os.path.join(abs_input_dir, file))
        elif file.endswith('_color.tif'):
            color_files.append(os.path.join(abs_input_dir, file))
    
    logger.info(f"Trouvé {len(height_files)} fichiers de hauteur et {len(color_files)} fichiers couleur")
    
    if len(height_files) == 0:
        logger.warning(f"Aucun fichier de hauteur trouvé dans {abs_input_dir}")
        logger.info("Aucun fichier à traiter.")
        return
    
    # ÉTAPE 1 : Calculer l'étendue globale et la résolution
    logger.info("Calcul de l'étendue globale et de la résolution...")
    
    # Lire le premier fichier pour obtenir les métadonnées de référence
    with rasterio.open(height_files[0]) as src:
        reference_transform = src.transform
        reference_crs = src.crs
        reference_resolution = src.res[0]  # Résolution en mètres
    
    logger.info(f"Fichier de référence : {os.path.basename(height_files[0])}")
    logger.info(f"CRS utilisé : {reference_crs}")
    logger.info(f"Transform de référence : {reference_transform}")
    logger.info(f"Résolution de référence : {reference_resolution}m")
    
    # Vérification que nous sommes bien dans un repère local ENU
    if reference_crs and ('geocent' in str(reference_crs) or 'cart' in str(reference_crs)):
        logger.info("✅ Repère local ENU détecté (CRS cartésien local)")
    elif reference_crs and 'tmerc' in str(reference_crs):
        logger.info("✅ Repère local ENU détecté (Transverse Mercator local)")
    else:
        logger.warning("⚠️ CRS non standard détecté - vérifiez que c'est bien un repère local ENU")
    
    # Utiliser la résolution cible si elle est fournie, sinon utiliser la résolution de référence
    if target_resolution is not None:
        final_resolution = target_resolution
        logger.info(f"Résolution cible spécifiée : {final_resolution}m")
        logger.info(f"Résolution finale utilisée : {final_resolution}m")
    else:
        final_resolution = reference_resolution
        logger.info(f"Résolution de référence utilisée : {final_resolution}m")
    
    # Vérifier la cohérence des CRS entre tous les fichiers
    logger.info("Vérification de la cohérence des CRS...")
    crs_issues = []
    for i, height_file in enumerate(height_files):
        with rasterio.open(height_file) as src:
            if src.crs != reference_crs:
                crs_issues.append(f"  - {os.path.basename(height_file)}: {src.crs} (attendu: {reference_crs})")
    
    if crs_issues:
        logger.warning("ATTENTION: Incohérences de CRS détectées:")
        for issue in crs_issues:
            logger.warning(issue)
        logger.warning("La fusion peut échouer ou donner des résultats incorrects!")
    else:
        logger.info("✅ Tous les fichiers ont le même CRS")
    
    # Calculer l'étendue globale en lisant tous les fichiers
    global_bounds = None
    
    for height_file in height_files:
        with rasterio.open(height_file) as src:
            bounds = src.bounds
            if global_bounds is None:
                global_bounds = bounds
            else:
                global_bounds = rasterio.coords.BoundingBox(
                    min(global_bounds.left, bounds.left),
                    min(global_bounds.bottom, bounds.bottom),
                    max(global_bounds.right, bounds.right),
                    max(global_bounds.top, bounds.top)
                )
    
    logger.info(f"Étendue globale : {global_bounds}")
    
    # Calculer les dimensions de la grille unifiée avec la résolution finale
    width = int((global_bounds.right - global_bounds.left) / final_resolution)
    height = int((global_bounds.top - global_bounds.bottom) / final_resolution)
    
    logger.info(f"Grille unifiée : {width} x {height} pixels")
    logger.info(f"Dimensions physiques : {(global_bounds.right - global_bounds.left):.3f}m x {(global_bounds.top - global_bounds.bottom):.3f}m")
    
    # Calcul du géoréférencement unifié avec la résolution finale
    transform = from_origin(global_bounds.left, global_bounds.top, final_resolution, final_resolution)
    
    # ÉTAPE 2 : Fusion des rasters
    logger.info("Fusion des rasters...")
    
    # Initialisation des rasters unifiés
    unified_height = np.full((height, width), np.nan)
    unified_color = np.zeros((height, width, 3), dtype=np.uint8)
    unified_count = np.zeros((height, width), dtype=np.uint8)
    
    # Traitement de chaque fichier
    for i, height_file in enumerate(height_files):
        logger.info(f"Traitement de {os.path.basename(height_file)} ({i+1}/{len(height_files)})")
        
        try:
            # Lecture du raster de hauteur
            with rasterio.open(height_file) as src:
                height_data = src.read(1)
                height_transform = src.transform
                
                # Créer un raster temporaire pour la transformation
                # CORRECTION: Initialiser avec une valeur nodata différente pour détecter les conversions
                temp_height = np.full((height, width), -9999.0)  # Valeur nodata distinctive
                
                # Transformation de la hauteur
                logger.info(f"  Reprojection de {os.path.basename(height_file)}")
                logger.info(f"    Source: {height_data.shape}, transform: {height_transform}")
                logger.info(f"    Destination: {temp_height.shape}, transform: {transform}")
                
                # CORRECTION: Debug des données source avant reprojection
                source_valid = np.sum(~np.isnan(height_data))
                source_nan = np.sum(np.isnan(height_data))
                source_zero = np.sum(height_data == 0)
                source_negative = np.sum(height_data < 0)
                logger.info(f"    Source: {source_valid} pixels valides, {source_nan} pixels NaN")
                logger.info(f"    Source: {source_zero} pixels à 0, {source_negative} pixels négatifs")
                if source_valid > 0:
                    logger.info(f"    Source hauteur min/max: {np.nanmin(height_data):.3f}m / {np.nanmax(height_data):.3f}m")
                
                # CORRECTION: Vérification des valeurs 0 dans la source
                if source_zero > 0:
                    logger.warning(f"    ⚠️ ATTENTION: {source_zero} pixels avec valeur 0 dans la source!")
                    logger.warning(f"    Ces valeurs 0 vont être propagées dans la fusion")
                
                try:
                    # CORRECTION: Reprojection avec gestion explicite des nodata
                    # Utiliser src_nodata et dst_nodata pour préserver les zones sans données
                    
                    # Déterminer la valeur nodata source
                    if np.any(np.isnan(height_data)):
                        src_nodata = np.nan
                    else:
                        # Si pas de NaN, chercher une valeur nodata appropriée
                        src_nodata = height_data.min() - 1.0
                    
                    reproject(
                        height_data,
                        temp_height,
                        src_transform=height_transform,
                        dst_transform=transform,
                        src_crs=reference_crs,
                        dst_crs=reference_crs,
                        src_nodata=src_nodata,
                        dst_nodata=-9999.0,  # Valeur nodata de destination
                        resampling=Resampling.nearest
                    )
                    logger.info(f"  ✅ Reprojection réussie avec gestion des nodata")
                    
                    # CORRECTION: Debug des données après reprojection
                    dest_valid = np.sum(temp_height != -9999.0)
                    dest_nan = np.sum(np.isnan(temp_height))
                    dest_zero = np.sum(temp_height == 0)
                    dest_negative = np.sum(temp_height < 0)
                    dest_nodata = np.sum(temp_height == -9999.0)
                    logger.info(f"    Après reprojection: {dest_valid} pixels valides, {dest_nan} pixels NaN")
                    logger.info(f"    Après reprojection: {dest_zero} pixels à 0, {dest_negative} pixels négatifs, {dest_nodata} pixels nodata")
                    if dest_valid > 0:
                        logger.info(f"    Dest hauteur min/max: {np.nanmin(temp_height[temp_height != -9999.0]):.3f}m / {np.nanmax(temp_height[temp_height != -9999.0]):.3f}m")
                    
                    # CORRECTION: Vérification de la gestion des nodata
                    if dest_nodata > 0:
                        logger.info(f"    ✅ Gestion des nodata réussie: {dest_nodata} pixels nodata préservés")
                    else:
                        logger.warning(f"    ⚠️ ATTENTION: Aucun pixel nodata détecté après reprojection")
                    
                    # CORRECTION: Vérification des valeurs 0 (maintenant suspectes)
                    if dest_zero > 0:
                        logger.warning(f"    ⚠️ ATTENTION: {dest_zero} pixels avec valeur 0 après reprojection!")
                        logger.warning(f"    Ces valeurs 0 peuvent être des vraies hauteurs ou des erreurs")
                    
                    # CORRECTION: Vérification de la perte de données
                    if source_valid > 0 and dest_valid == 0:
                        logger.error(f"  ❌ PROBLÈME: Toutes les données perdues lors de la reprojection!")
                        logger.error(f"    Source: {source_valid} pixels → Destination: {dest_valid} pixels")
                        logger.error(f"    Vérifiez l'alignement des transformations géométriques")
                    elif dest_valid < source_valid * 0.5:  # Perte de plus de 50%
                        logger.warning(f"  ⚠️ Perte importante de données: {source_valid} → {dest_valid} pixels")
                except Exception as e:
                    logger.error(f"  ❌ Erreur de reprojection: {e}")
                    continue
                
                # Mise à jour de la hauteur (prendre le maximum)
                # CORRECTION: Traiter TOUS les pixels, pas seulement les valides
                # Cela permet de combler les trous entre les orthoimages
                # CORRECTION: Adapter le masque pour les nouvelles valeurs nodata
                # Maintenant que la reprojection gère correctement les nodata
                # Exclure les pixels nodata (-9999.0) ET les pixels NaN
                valid_mask = (temp_height != -9999.0) & ~np.isnan(temp_height)
                valid_pixels_count = np.sum(valid_mask)
                logger.info(f"  Pixels valides dans ce raster: {valid_pixels_count}")
                
                # CORRECTION: Logique de fusion des hauteurs améliorée
                pixels_updated = 0
                pixels_new = 0
                pixels_max_updated = 0
                
                for r in range(height):
                    for c_idx in range(width):
                        temp_height_val = temp_height[r, c_idx]
                        
                        # CORRECTION: Ignorer les pixels nodata (-9999.0) pendant la fusion
                        if not np.isnan(temp_height_val) and temp_height_val != -9999.0:
                            # Pixel avec données valides (pas nodata)
                            if np.isnan(unified_height[r, c_idx]):
                                # Premier pixel valide à cette position
                                unified_height[r, c_idx] = temp_height_val
                                unified_count[r, c_idx] = 1
                                pixels_new += 1
                            else:
                                # CORRECTION: Prendre le maximum des hauteurs, mais préserver les vraies valeurs négatives
                                old_height = unified_height[r, c_idx]
                                
                                # Vérifier si l'ancienne valeur est une vraie hauteur ou une valeur par défaut
                                # CORRECTION: Détecter les vraies valeurs par défaut (0 ou valeurs très proches de 0)
                                if (abs(old_height) < 0.001 or old_height == 0) and temp_height_val < 0:
                                    # Si l'ancienne valeur est proche de 0 (par défaut) et la nouvelle est négative,
                                    # prendre la valeur négative (c'est une vraie hauteur)
                                    unified_height[r, c_idx] = temp_height_val
                                    pixels_max_updated += 1
                                    logger.debug(f"    Pixel ({r}, {c_idx}): remplacement de {old_height:.3f}m par {temp_height_val:.3f}m (valeur négative)")
                                else:
                                    # Sinon, prendre le maximum normal
                                    unified_height[r, c_idx] = max(unified_height[r, c_idx], temp_height_val)
                                    if unified_height[r, c_idx] > old_height:
                                        pixels_max_updated += 1
                                
                                unified_count[r, c_idx] += 1
                                pixels_updated += 1
                        # Si temp_height_val est NaN, on ne fait rien (pas de données à cette position)
                
                # CORRECTION: Logs déplacés en dehors de la boucle des pixels
                logger.info(f"  Hauteur min/max de ce raster: {np.nanmin(temp_height):.3f}m / {np.nanmax(temp_height):.3f}m")
                logger.info(f"  Pixels mis à jour: {pixels_updated} (dont {pixels_max_updated} avec hauteur max)")
                logger.info(f"  Nouveaux pixels: {pixels_new}")
                
                # CORRECTION: Debug des valeurs négatives dans ce raster
                negative_pixels_in_raster = np.sum((temp_height < 0) & ~np.isnan(temp_height))
                if negative_pixels_in_raster > 0:
                    logger.info(f"  Pixels avec hauteurs négatives dans ce raster: {negative_pixels_in_raster}")
                    logger.info(f"  Hauteur minimale négative: {np.nanmin(temp_height[temp_height < 0]):.3f}m")
                else:
                    logger.info(f"  Aucune hauteur négative dans ce raster")
                
                # CORRECTION: Vérification de sécurité des hauteurs
                if np.nanmax(temp_height) > 1000:  # Plus de 1km = suspect
                    logger.warning(f"  ⚠️ Hauteur maximale suspecte: {np.nanmax(temp_height):.3f}m")
                    logger.warning(f"  Vérifiez les unités et la géométrie du nuage de points")
                
                if np.nanmin(temp_height) < -1000:  # Moins de -1km = suspect
                    logger.warning(f"  ⚠️ Hauteur minimale suspecte: {np.nanmin(temp_height):.3f}m")
                    logger.warning(f"  Vérifiez les unités et la géométrie du nuage de points")
                
                if valid_pixels_count == 0:
                    logger.warning(f"  Aucun pixel valide dans le raster de hauteur")
                
                # CORRECTION: Debug du traitement des couleurs
                logger.info(f"  Traitement des couleurs avec le même masque que les hauteurs")
                logger.info(f"  Masque de validité: {np.sum(valid_mask)} pixels valides")
                
                # CORRECTION: Debug de l'état de unified_height après ce raster
                current_negative = np.sum((unified_height < 0) & ~np.isnan(unified_height))
                current_positive = np.sum((unified_height > 0) & ~np.isnan(unified_height))
                current_zero = np.sum((unified_height == 0) & ~np.isnan(unified_height))
                current_near_zero = np.sum((abs(unified_height) < 0.001) & ~np.isnan(unified_height))
                logger.info(f"  État unified_height après ce raster:")
                logger.info(f"    Pixels négatifs: {current_negative}, Pixels positifs: {current_positive}, Pixels à zéro: {current_zero}, Pixels proches de zéro: {current_near_zero}")
                if current_negative > 0:
                    logger.info(f"    Plage hauteurs négatives: {np.nanmin(unified_height[unified_height < 0]):.3f}m à {np.nanmax(unified_height[unified_height < 0]):.3f}m")
                if current_zero > 0:
                    logger.warning(f"    ⚠️ ATTENTION: {current_zero} pixels avec valeur exactement 0 (suspect!)")
                    logger.warning(f"    Vérifiez la source de ces valeurs 0")
                
                # Lecture du raster couleur correspondant
                color_file = height_file.replace('_height.tif', '_color.tif')
                if os.path.exists(color_file):
                    with rasterio.open(color_file) as src_color:
                        color_data = src_color.read([1, 2, 3])  # RGB
                        
                        # Transformation de la couleur
                        temp_color = np.zeros((height, width, 3), dtype=np.uint8)
                        for c in range(3):  # RGB
                            reproject(
                                color_data[c],
                                temp_color[:, :, c],
                                src_transform=height_transform,
                                dst_transform=transform,
                                src_crs=reference_crs,
                                dst_crs=reference_crs,
                                resampling=Resampling.average
                            )
                        
                        # Traitement des couleurs selon la méthode sélectionnée
                        if color_fusion_method == "median":
                            # Méthode médiane : plus robuste aux valeurs aberrantes
                            logger.info(f"  Application de la médiane pour les couleurs")
                            
                            # Pour la médiane, on utilise une approche différente :
                            # On garde le pixel avec la valeur la plus proche de la moyenne locale
                            # Cela évite les valeurs aberrantes tout en préservant les détails
                            
                            for c in range(3):  # RGB
                                temp_color_float = temp_color[:, :, c].astype(np.float64)
                                unified_color_float = unified_color[:, :, c].astype(np.float64)
                                
                                for r in range(height):
                                    for c_idx in range(width):
                                        if valid_mask[r, c_idx]:
                                            if unified_count[r, c_idx] == 0:
                                                # Premier pixel à cette position
                                                unified_color_float[r, c_idx] = temp_color_float[r, c_idx]
                                            else:
                                                # Comparer avec le pixel existant
                                                # On garde celui qui est le plus "représentatif"
                                                current_pixel = temp_color_float[r, c_idx]
                                                existing_pixel = unified_color_float[r, c_idx]
                                                
                                                # Calculer la "qualité" basée sur la différence avec la moyenne
                                                # Plus la différence est faible, plus le pixel est représentatif
                                                local_mean = (current_pixel + existing_pixel) / 2.0
                                                current_quality = abs(current_pixel - local_mean)
                                                existing_quality = abs(existing_pixel - local_mean)
                                                
                                                if current_quality < existing_quality:
                                                    # Nouveau pixel a une meilleure qualité
                                                    unified_color_float[r, c_idx] = current_pixel
                                
                                # Conversion finale en uint8
                                unified_color[:, :, c] = np.clip(unified_color_float, 0, 255).astype(np.uint8)
                                
                                logger.debug(f"    Canal {['R', 'G', 'B'][c]}: médiane appliquée")
                        
                        else:
                            # Méthode moyenne (par défaut)
                            logger.info(f"  Application de la moyenne pondérée pour les couleurs")
                            
                            for c in range(3):  # RGB
                                # Utiliser le même masque que pour les hauteurs (cohérence)
                                valid_color_mask = valid_mask  # Pas de condition sur couleur > 0
                                
                                if np.any(valid_color_mask):
                                    # Utiliser des calculs en float64 pour éviter les overflows
                                    temp_color_float = temp_color[:, :, c].astype(np.float64)
                                    unified_color_float = unified_color[:, :, c].astype(np.float64)
                                    
                                    # Mise à jour de la couleur avec moyenne pondérée
                                    for r in range(height):
                                        for c_idx in range(width):
                                            if valid_color_mask[r, c_idx]:
                                                if unified_count[r, c_idx] > 0:
                                                    # Moyenne pondérée par le nombre de contributions
                                                    unified_color_float[r, c_idx] = (
                                                        (unified_color_float[r, c_idx] * (unified_count[r, c_idx] - 1) + temp_color_float[r, c_idx]) / 
                                                        unified_count[r, c_idx]
                                                    )
                                                else:
                                                    unified_color_float[r, c_idx] = temp_color_float[r, c_idx]
                                    
                                    # Conversion finale en uint8 avec clipping
                                    unified_color[:, :, c] = np.clip(unified_color_float, 0, 255).astype(np.uint8)
                                    
                                    logger.debug(f"    Canal {['R', 'G', 'B'][c]}: {np.sum(valid_color_mask)} pixels traités")
                                else:
                                    logger.warning(f"    Canal {['R', 'G', 'B'][c]}: Aucun pixel valide à traiter")
                
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {os.path.basename(height_file)} : {e}")
            continue
    
    # Pas de normalisation finale nécessaire car la couleur est déjà correctement moyennée
    # Le unified_color contient déjà les valeurs finales
    
    # CORRECTION: Pas d'interpolation - on garde les NaN pour les pixels sans données
    # L'objectif est de conserver la valeur la plus haute de chaque pixel
    holes_count = np.sum(np.isnan(unified_height))
    total_pixels = height * width
    logger.info(f"Statistiques finales:")
    logger.info(f"  Pixels totaux: {total_pixels}")
    logger.info(f"  Pixels avec données: {total_pixels - holes_count}")
    logger.info(f"  Pixels sans données (NaN): {holes_count}")
    logger.info(f"  Couverture: {((total_pixels - holes_count) / total_pixels * 100):.1f}%")
    
    # Debug: afficher des informations sur les données unifiées
    valid_pixels = np.sum(~np.isnan(unified_height))
    logger.info(f"Fusion terminée. Pixels valides : {valid_pixels}")
    
    if valid_pixels == 0:
        logger.warning("ATTENTION: Aucun pixel valide trouvé dans l'orthoimage unifiée!")
        logger.warning("Vérifiez que les fichiers d'entrée contiennent des données valides")
        logger.warning(f"Nombre de fichiers traités: {len(height_files)}")
        logger.warning(f"Dimensions de la grille unifiée: {width} x {height}")
        logger.warning(f"Étendue globale: {global_bounds}")
    else:
        logger.info(f"Hauteur min/max: {np.nanmin(unified_height):.3f}m / {np.nanmax(unified_height):.3f}m")
        logger.info(f"Couleur min/max (R): {np.min(unified_color[:,:,0])} / {np.max(unified_color[:,:,0])}")
        logger.info(f"Couleur min/max (G): {np.min(unified_color[:,:,1])} / {np.max(unified_color[:,:,1])}")
        logger.info(f"Couleur min/max (B): {np.min(unified_color[:,:,2])} / {np.max(unified_color[:,:,2])}")
        
        # CORRECTION: Affichage des statistiques de fusion
        logger.info(f"Statistiques de fusion:")
        logger.info(f"  Pixels avec 1 contribution: {np.sum(unified_count == 1)}")
        logger.info(f"  Pixels avec 2+ contributions: {np.sum(unified_count > 1)}")
        logger.info(f"  Hauteur moyenne des pixels multiples: {np.nanmean(unified_height[unified_count > 1]):.3f}m")
        
        # CORRECTION: Debug des valeurs négatives
        negative_pixels = np.sum((unified_height < 0) & ~np.isnan(unified_height))
        if negative_pixels > 0:
            logger.info(f"  Pixels avec hauteurs négatives: {negative_pixels}")
            logger.info(f"  Hauteur minimale (avec négatifs): {np.nanmin(unified_height):.3f}m")
            logger.info(f"  Hauteur maximale: {np.nanmax(unified_height):.3f}m")
        else:
            logger.info(f"  Aucune hauteur négative détectée")
        
        # CORRECTION: Statistiques de couverture pour identifier les trous
        total_pixels = height * width
        coverage_percentage = (valid_pixels / total_pixels) * 100
        logger.info(f"Statistiques de couverture:")
        logger.info(f"  Pixels totaux: {total_pixels}")
        logger.info(f"  Pixels couverts: {valid_pixels}")
        logger.info(f"  Couverture: {coverage_percentage:.1f}%")
        
        if coverage_percentage < 80:
            logger.warning(f"⚠️ Couverture faible ({coverage_percentage:.1f}%) - Vérifiez l'alignement des orthoimages")
        elif coverage_percentage < 95:
            logger.info(f"ℹ️ Couverture correcte ({coverage_percentage:.1f}%) - Quelques trous à combler")
        else:
            logger.info(f"✅ Couverture excellente ({coverage_percentage:.1f}%)")
    
    # ÉTAPE 3 : Sauvegarde des résultats unifiés
    logger.info("Sauvegarde des résultats unifiés...")
    
    # Métadonnées
    metadata = {
        'Software': 'PhotoGeoAlign Unified Orthoimage/DTM Merger',
        'Resolution': f'{reference_resolution}m per pixel',
        'Origin_X': f'{global_bounds.left:.6f}',
        'Origin_Y': f'{global_bounds.top:.6f}',
        'Extent_X': f'{global_bounds.right - global_bounds.left:.3f}m',
        'Extent_Y': f'{global_bounds.top - global_bounds.bottom:.3f}m',
        'Source_Files': str(len(height_files)),
        'Valid_Pixels': str(int(np.sum(~np.isnan(unified_height)))),
        'Height_Range': f'{np.nanmin(unified_height):.3f}m to {np.nanmax(unified_height):.3f}m' if valid_pixels > 0 else 'No valid pixels'
    }
    
    # Sauvegarde de l'orthoimage unifiée
    orthoimage_filename = "unified_orthoimage.tif"
    orthoimage_path = os.path.join(output_dir, orthoimage_filename)
    
    logger.info(f"Sauvegarde de l'orthoimage unifiée avec le CRS : {reference_crs}")
    
    with rasterio.open(
        orthoimage_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=unified_color.dtype,
        crs=reference_crs,  # Utiliser le CRS du fichier de référence
        transform=transform,
        photometric='rgb'
    ) as dst:
        dst.write(unified_color[:,:,0], 1)  # Rouge
        dst.write(unified_color[:,:,1], 2)  # Vert
        dst.write(unified_color[:,:,2], 3)  # Bleu
        dst.update_tags(**metadata)
    
    # Sauvegarde du MNT unifié
    dtm_filename = "unified_dtm.tif"
    dtm_path = os.path.join(output_dir, dtm_filename)
    
    # CORRECTION: Préserver les valeurs négatives et utiliser une valeur nodata appropriée
    dtm_data = unified_height.copy()
    
    # Déterminer une valeur nodata appropriée (en dehors de la plage des hauteurs)
    height_min = np.nanmin(dtm_data)
    height_max = np.nanmax(dtm_data)
    height_range = height_max - height_min
    
    # Utiliser une valeur nodata en dehors de la plage des hauteurs
    if height_min < 0:
        # Si on a des hauteurs négatives, utiliser une valeur très négative
        nodata_value = height_min - height_range - 1.0
        logger.info(f"  Hauteurs négatives détectées: min={height_min:.3f}m, nodata={nodata_value:.3f}m")
    else:
        # Si toutes les hauteurs sont positives, utiliser -9999
        nodata_value = -9999.0
        logger.info(f"  Toutes hauteurs positives: min={height_min:.3f}m, nodata={nodata_value:.3f}m")
    
    # Remplacer les NaN par la valeur nodata temporairement pour la fusion
    dtm_data = np.where(np.isnan(dtm_data), nodata_value, dtm_data)
    
    # CORRECTION: Remplacer les -9999.0 par np.nan juste avant la sauvegarde
    # Pour avoir le même comportement que les MNT unitaires
    # Utiliser une comparaison plus robuste pour les floats
    dtm_data = np.where(np.isclose(dtm_data, nodata_value, atol=0.1), np.nan, dtm_data)
    
    # Debug: vérifier que la conversion a fonctionné
    remaining_nodata = np.sum(np.isclose(dtm_data, nodata_value, atol=0.1))
    if remaining_nodata > 0:
        logger.warning(f"  ⚠️ ATTENTION: {remaining_nodata} pixels nodata n'ont pas été convertis!")
    else:
        logger.info(f"  ✅ Conversion nodata → NaN réussie")
    
    # Debug détaillé des valeurs
    unique_values = np.unique(dtm_data)
    logger.info(f"  Valeurs uniques dans dtm_data après conversion: {unique_values}")
    
    # Vérifier s'il y a encore des valeurs proches de -9999
    near_9999 = np.sum(np.abs(dtm_data + 9999) < 1.0)
    if near_9999 > 0:
        logger.warning(f"  ⚠️ ATTENTION: {near_9999} pixels avec des valeurs proches de -9999!")
    
    # Vérifier le type de données et le convertir si nécessaire
    if dtm_data.dtype != np.float32 and dtm_data.dtype != np.float64:
        dtm_data = dtm_data.astype(np.float32)
        logger.info(f"  Type de données converti en float32 pour préserver les valeurs négatives")
    
    # Statistiques finales avec np.nan
    valid_pixels = np.sum(~np.isnan(dtm_data))
    nodata_pixels = np.sum(np.isnan(dtm_data))
    logger.info(f"  MNT final: {valid_pixels} pixels valides, {nodata_pixels} pixels nodata (NaN)")
    if valid_pixels > 0:
        logger.info(f"  Plage hauteurs: {np.nanmin(dtm_data):.3f}m à {np.nanmax(dtm_data):.3f}m")
    
    with rasterio.open(
        dtm_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtm_data.dtype,
        crs=reference_crs,
        transform=transform,
        nodata=np.nan  # Utiliser NaN comme les MNT unitaires
    ) as dst:
        dst.write(dtm_data, 1)
        dst.update_tags(**metadata)
    
    logger.info(f"Orthoimage unifiée sauvegardée : {orthoimage_filename}")
    logger.info(f"MNT unifié sauvegardé : {dtm_filename}")
    logger.info(f"Fusion des orthoimages et MNT terminée dans {output_dir}")

def process_zone_with_orthos(zone_data):
    """
    Fonction globale exécutée par chaque processus pour traiter une zone
    UNIQUEMENT avec l'égalisation par médiane superposée
    """
    import os
    import numpy as np
    import rasterio
    from rasterio import warp, transform, enums
    
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

def test_zone_fusion_with_borders(input_dir, logger, output_dir, final_resolution=0.003, grid_size_meters=None, zone_size_meters=5.0, max_workers=None):
    """
    TEST ÉTAPE 1 : Création de zones avec orthos réelles et fusion parallèle
    Objectif : Valider la logique de fusion des zones avec process_zone_with_orthos
    
    Args:
        input_dir: Répertoire d'entrée contenant les orthoimages unitaires
        logger: Logger pour les messages
        output_dir: Répertoire de sortie
        final_resolution: Résolution finale en mètres (défaut: 0.003m = 3mm)
        grid_size_meters: Taille de la grille en mètres (si None, calculée automatiquement)
        zone_size_meters: Taille de chaque zone en mètres (défaut: 5.0m)
        max_workers: Nombre maximum de processus parallèles
    """
    import os
    import numpy as np
    import rasterio
    from rasterio.coords import BoundingBox
    from rasterio.transform import from_origin
    from multiprocessing import Pool
    
    logger.info("🧪 TEST ÉTAPE 1 : Création de zones avec orthos réelles et fusion parallèle")
    
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
        logger.warning("⚠️ Aucune orthoimage .tif trouvée, utilisation des paramètres par défaut")
        if grid_size_meters is None:
            grid_size_meters = 20.0
        global_bounds = BoundingBox(
            left=0.0, bottom=0.0, right=float(grid_size_meters), top=float(grid_size_meters)
        )
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
                pixel_size_x = abs(transform[0])
                pixel_size_y = abs(transform[3])
                logger.info(f"  Résolution pixel : {pixel_size_x:.6f}m × {pixel_size_y:.6f}m")
                
                # Si la résolution finale n'est pas spécifiée, utiliser celle de l'ortho
                if final_resolution is None:
                    final_resolution = min(pixel_size_x, pixel_size_y)
                    logger.info(f"  Résolution finale automatique : {final_resolution:.6f}m")
                
                # S'assurer que la résolution finale est valide
                if final_resolution <= 0:
                    logger.warning(f"  ⚠️ Résolution invalide détectée : {final_resolution}, utilisation de 0.003m")
                    final_resolution = 0.003
                
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
                if grid_size_meters is not None:
                    grid_size = grid_size_meters
                    adjusted_left = global_left #round(global_left / grid_size) * grid_size
                    adjusted_bottom = global_bottom #round(global_bottom / grid_size) * grid_size
                    adjusted_right = global_right #round(global_right / grid_size) * grid_size
                    adjusted_top = global_top #round(global_top / grid_size) * grid_size
                    
                    # S'assurer que l'étendue couvre au moins la zone des orthos
                    if adjusted_right < global_right:
                        adjusted_right += grid_size
                    if adjusted_top < global_top:
                        adjusted_top += grid_size
                    
                    global_bounds = BoundingBox(
                        left=adjusted_left, bottom=adjusted_bottom,
                        right=adjusted_right, top=adjusted_top
                    )
                else:
                    # Utiliser l'étendue réelle des orthos (SANS ARRONDI)
                    global_bounds = BoundingBox(
                        left=global_left, bottom=global_bottom,
                        right=global_right, top=global_top
                    )
                
                logger.info(f"Étendue globale calculée : {global_bounds}")
                logger.info(f"  Largeur : {global_bounds.right - global_bounds.left:.2f}m")
                logger.info(f"  Hauteur : {global_bounds.top - global_bounds.bottom:.2f}m")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse de l'orthoimage : {e}")
            logger.warning("⚠️ Utilisation des paramètres par défaut")
            if grid_size_meters is None:
                grid_size_meters = 20.0
            global_bounds = BoundingBox(
                left=0.0, bottom=0.0, right=float(grid_size_meters), top=float(grid_size_meters)
            )
    
    # S'assurer que la résolution finale est valide
    if final_resolution is None or final_resolution <= 0:
        logger.warning(f"⚠️ Résolution finale invalide : {final_resolution}, utilisation de 0.003m")
        final_resolution = 0.003
    
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
    aligned_right = global_bounds.right + (zone_size_meters - (global_bounds.right - global_bounds.left) % zone_size_meters) % zone_size_meters
    aligned_top = global_bounds.top + (zone_size_meters - (global_bounds.top - global_bounds.bottom) % zone_size_meters) % zone_size_meters
    
    # S'assurer que l'extension ne dépasse pas une zone complète
    if aligned_right > global_bounds.right + zone_size_meters:
        aligned_right = global_bounds.right + zone_size_meters
    if aligned_top > global_bounds.top + zone_size_meters:
        aligned_top = global_bounds.top + zone_size_meters
    
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
    
    # 🆕 CRÉATION DE L'IMAGE DE DEBUG AVEC CONTOURS DES ZONES (AVANT traitement parallèle)
    # DÉSACTIVÉE - Problèmes d'affichage et non critique pour le fonctionnement
    # logger.info("🎨 Création de l'image de debug avec contours des zones...")
    
    # try:
    #     # Créer une image de debug avec les contours des zones
    #     debug_image_path = os.path.join(output_dir, "debug_zones_contours.png")
    #     
    #     # Calculer les dimensions de l'image de debug
    #     # IMPORTANT : L'image doit couvrir TOUTES les zones, pas seulement les orthos !
    #     # Les zones s'étendent souvent au-delà des bounds des orthos pour l'alignement
    #     debug_bounds = BoundingBox(
    #         left=aligned_left,      # Coordonnée la plus à gauche des zones
    #         bottom=aligned_bottom,  # Coordonnée la plus basse des zones  
    #         right=aligned_right,    # Coordonnée la plus à droite des zones
    #         top=aligned_top         # Coordonnée la plus haute des zones
    #     )
    #     
    #     # Ajouter des logs pour debug
    #     logger.info(f"🔍 Debug - Bounds originaux des orthos : {original_ortho_bounds}")
    #     logger.info(f"🔍 Debug - Bounds étendus pour toutes les zones : {debug_bounds}")
    #     logger.info(f"🔍 Debug - Résolution finale : {final_resolution}m")
    #     
    #     # Créer une image de debug avec le MÊME géoréférencement que les orthos finales
    #     debug_resolution = final_resolution  # 0.003m/pixel (même que les orthos)
    #     
    #     debug_width = int((debug_bounds.right - debug_bounds.left) / debug_resolution)
    #     debug_height = int((debug_bounds.top - debug_bounds.bottom) / debug_resolution)
    #     
    #     logger.info(f"🔍 Debug - Résolution debug : {debug_resolution}m/pixel (même que les orthos)")
    #     logger.info(f"🔍 Debug - Dimensions image : {debug_width} × {debug_height} pixels")
    #     logger.info(f"🔍 Debug - Dimensions physiques : {(debug_bounds.right - debug_bounds.left):.3f}m × {(debug_bounds.top - debug_bounds.bottom):.3f}m")
    #     
    #     # Créer une image vide (noire)
    #     debug_image = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
    #     
    #     # Dessiner les contours de chaque zone
    #     for zone in zones:
    #         zone_bounds = zone['bounds']  # (x, y, x + zone_size, y + zone_size)
    #         
    #         # Extraire les coordonnées du tuple
    #         zone_left = zone_bounds[0]
    #         zone_bottom = zone_bounds[1]
    #         zone_right = zone_bounds[2]
    #         zone_top = zone_bounds[3]
    #         
    #         # Convertir les coordonnées géographiques en pixels (avec résolution debug)
    #         # Utiliser round() au lieu de int() pour éviter les troncatures
    #         zone_left_px = round((zone_left - debug_bounds.left) / debug_resolution)
    #         zone_right_px = round((zone_right - debug_bounds.left) / debug_resolution)
    #         # CORRECTION : Pas d'inversion Y, utiliser directement les coordonnées
    #         zone_bottom_px = round((zone_bottom - debug_bounds.bottom) / debug_resolution)
    #         zone_top_px = round((zone_top - debug_bounds.bottom) / debug_resolution)
    #         
    #         # Log de debug pour cette zone
    #         logger.info(f"🔍 Zone {zone['id']} - Coord: ({zone_left:.3f}, {zone_bottom:.3f}) à ({zone_right:.3f}, {zone_top:.3f})")
    #         logger.info(f"🔍 Zone {zone['id']} - Pixels: ({zone_left_px}, {zone_top_px}) à ({zone_right_px}, {zone_bottom_px})")
    #         
    #         # Couleur unique pour chaque zone (cycle de couleurs)
    #         colors = [
    #             (255, 0, 0),    # Rouge
    #             (0, 255, 0),    # Vert
    #             (0, 0, 255),    # Bleu
    #             (255, 255, 0),  # Jaune
    #             (255, 0, 255),  # Magenta
    #             (0, 255, 255),  # Cyan
    #             (255, 128, 0),  # Orange
    #             (128, 0, 255),  # Violet
    #             (0, 128, 255),  # Bleu clair
    #             (255, 0, 128),  # Rose
    #             (128, 255, 0),  # Vert clair
    #             (255, 128, 128) # Rose clair
    #         ]
    #         color = colors[zone['id'] % len(colors)]
    #         
    #         # Dessiner le contour de la zone (2 pixels d'épaisseur)
    #         thickness = 2
    #         
    #         # Lignes horizontales
    #         for y in range(max(0, zone_top_px), min(debug_height, zone_bottom_px + 1)):
    #             if 0 <= zone_left_px < debug_width:
    #                 debug_image[y, zone_left_px:min(zone_left_px + thickness, debug_width)] = color
    #             if 0 <= zone_right_px < debug_width:
    #                 debug_image[y, max(0, zone_right_px - thickness):zone_right_px] = color
    #                 
    #         # Lignes verticales
    #         for x in range(max(0, zone_left_px), min(debug_width, zone_right_px + 1)):
    #             if 0 <= zone_top_px < debug_height:
    #                 debug_image[max(0, zone_top_px):min(zone_top_px + thickness, debug_height), x] = color
    #             if 0 <= zone_bottom_px < debug_height:
    #                 debug_image[max(0, zone_bottom_px - thickness):zone_bottom_px, x] = color
    #                 
    #         # Ajouter le numéro de zone au centre
    #         center_x = (zone_left_px + zone_right_px) // 2
    #         center_y = (zone_top_px + zone_bottom_px) // 2
    #         
    #         if 0 <= center_x < debug_width and 0 <= center_y < debug_height:
    #             # Créer un petit carré blanc avec le numéro
    #             label_size = 20
    #             label_left = max(0, center_x - label_size // 2)
    #             label_right = min(debug_width, center_x + label_size // 2)
    #             label_top = max(0, center_y - label_size // 2)
    #             label_bottom = min(debug_height, center_y + label_size // 2)
    #             
    #             debug_image[label_top:label_bottom, label_left:label_right] = [255, 255, 255]  # Blanc
    #         
    #     # Sauvegarder l'image de debug en GeoTIFF avec le bon géoréférencement
    #     import rasterio
    #     from rasterio.transform import from_origin
    #     
    #     # Créer le transform pour le géoréférencement (même que les orthos)
    #     transform = from_origin(debug_bounds.left, debug_bounds.top, debug_resolution, debug_resolution)
    #     
    #     # Récupérer le CRS des orthos unitaires (même que les orthos finales)
    #     ortho_crs = None
    #     try:
    #         with rasterio.open(ortho_files[0]) as src:
    #             ortho_crs = src.crs
    #             logger.info(f"🔍 Debug - CRS des orthos : {ortho_crs}")
    #     except Exception as e:
    #         logger.warning(f"⚠️ Impossible de lire le CRS des orthos : {e}")
    #         ortho_crs = 'EPSG:4326'  # Fallback
    #     
    #     # Sauvegarder en GeoTIFF avec géoréférencement
    #     with rasterio.open(
    #         debug_image_path.replace('.png', '.tif'),
    #         'w',
    #         driver='GTiff',
    #         height=debug_height,
    #         width=debug_width,
    #         count=3,
    #         dtype=debug_image.dtype,
    #         crs=ortho_crs,  # Même CRS que les orthos
    #         transform=transform
    #     ) as dst:
    #         dst.write(debug_image.transpose(2, 0, 1))  # PIL attend (channels, height, width)
    #     
    #     logger.info(f"✅ GeoTIFF de debug créé : {debug_image_path.replace('.png', '.tif')}")
    #     logger.info(f"   📏 Dimensions : {debug_width} × {debug_height} pixels")
    #     logger.info(f"   🎨 {len(zones)} zones avec contours colorés")
    #     logger.info(f"   🗺️ Géoréférencement : {transform}")
    #     logger.info(f"   🗺️ CRS : {ortho_crs}")
    #     
    # except Exception as e:
    #     logger.error(f"⚠️ Erreur lors de la création de l'image de debug : {e}")
    
    logger.info("🎨 Image de debug désactivée - Passage direct à la parallélisation")
    
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
    
    logger.info(f"✅ TEST ÉTAPE 1 TERMINÉ : Fusion parallèle des zones terminée")
    logger.info(f"Résultat attendu : {len(results)} zones traitées avec orthos fusionnées")
    
    # 🆕 ÉTAPE 2 : ASSEMBLAGE AUTOMATIQUE DES ORTHOS UNIFIÉES
    logger.info("🚀 LANCEMENT AUTOMATIQUE DE L'ASSEMBLAGE DES ORTHOS...")
    
    try:
        # Récupérer la résolution finale depuis les paramètres
        final_resolution = 0.003  # Résolution par défaut
        
        # Lancer l'assemblage automatique
        unified_ortho_path = simple_ortho_assembly(
            zones_output_dir=output_dir,
            logger=logger,
            final_resolution=final_resolution
        )
        
        if unified_ortho_path:
            logger.info(f"🎉 ASSEMBLAGE AUTOMATIQUE RÉUSSI !")
            logger.info(f"📁 Ortho unifiée créée : {unified_ortho_path}")
            logger.info(f"✅ PIPELINE COMPLET TERMINÉ : Zones + Assemblage")
        else:
            logger.warning(f"⚠️ Assemblage automatique échoué, mais zones créées avec succès")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'assemblage automatique : {e}")
        logger.warning(f"⚠️ Les zones ont été créées, mais l'assemblage a échoué")
    
    return output_dir

def simple_ortho_assembly(zones_output_dir, logger, final_resolution=0.003):
    """
    Simple assemblage des orthos de zones (pas de fusion)
    
    Args:
        zones_output_dir: Répertoire contenant les orthos fusionnées par zones
        logger: Logger pour les messages
        final_resolution: Résolution finale en mètres (défaut: 0.003m)
    
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
    logger.info(f"📏 Résolution finale : {final_resolution}m")
    
    # ÉTAPE 1 : Lire toutes les orthos de zones
    logger.info("📖 Lecture des orthos de zones...")
    
    zone_ortho_files = []
    zone_bounds_list = []
    
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
    final_width = int((global_bounds.right - global_bounds.left) / final_resolution)
    final_height = int((global_bounds.top - global_bounds.bottom) / final_resolution)
    
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
            start_x = int((zone_bounds.left - global_bounds.left) / final_resolution)
            # CORRECTION : Calculer start_y depuis le bas de la grille (y=0 en haut)
            start_y = int((global_bounds.top - zone_bounds.top) / final_resolution)
            end_x = start_x + zone_data['width']
            end_y = start_y + zone_data['height']
            
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

 