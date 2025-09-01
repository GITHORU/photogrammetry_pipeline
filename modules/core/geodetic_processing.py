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

def convert_itrf_to_enu(input_dir, logger, coord_file=None, extra_params="", ref_point_name=None, max_workers=None, global_ref_point=None, force_global_ref=False):
    # DEBUG : Vérification des paramètres reçus
    logger.info(f"🔍 DEBUG - Paramètres reçus dans convert_itrf_to_enu:")
    logger.info(f"   global_ref_point: {global_ref_point}")
    logger.info(f"   force_global_ref: {force_global_ref}")
    logger.info(f"   ref_point_name: {ref_point_name}")
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
        
        # ÉTAPE 1.5 : Priorisation du point de référence global si forcé
        if force_global_ref and global_ref_point is not None:
            logger.info("🎯 UTILISATION DU POINT DE RÉFÉRENCE GLOBAL FORCÉ")
            logger.info(f"Point global : ({global_ref_point[0]:.6f}, {global_ref_point[1]:.6f}, {global_ref_point[2]:.6f})")
            ref_point = np.array(global_ref_point)
            logger.info("Le point global remplace le point local pour unifier le repère ENU")
            logger.info("⚠️  L'offset ne s'applique PAS au point global (coordonnées absolues préservées)")
        else:
            logger.info("📍 UTILISATION DU POINT DE RÉFÉRENCE LOCAL")
            logger.info(f"Point local : ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
            
            # ÉTAPE 1.6 : Lecture de l'offset et application au point local UNIQUEMENT
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
                    # Application de l'offset au point local UNIQUEMENT
                    ref_point = ref_point + np.array(offset)
                    logger.info(f"Point local avec offset appliqué : ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
                    
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
        # DEBUG : Vérification avant l'appel parallèle
        logger.info(f"🔍 DEBUG - Préparation de l'appel à process_single_cloud_itrf_to_enu:")
        logger.info(f"   Nombre de fichiers PLY: {len(ply_files)}")
        logger.info(f"   Max workers: {max_workers}")
        logger.info(f"   Centre de transformation: ({tr_center[0]:.6f}, {tr_center[1]:.6f}, {tr_center[2]:.6f})")
        logger.info(f"   Premier fichier: {os.path.basename(ply_files[0]) if ply_files else 'Aucun'}")
        
        # Traitement parallèle avec Pool
        with Pool(processes=max_workers) as pool:
            # Lancement du traitement parallèle
            # Ajout des paramètres du point global aux arguments
            extended_process_args = [args + (global_ref_point, force_global_ref) for args in process_args]
            results = pool.map(process_single_cloud_itrf_to_enu, extended_process_args)
            
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

def deform_clouds(input_dir, logger, deformation_type="lineaire", deformation_params="", extra_params="", bascule_xml_file=None, coord_file=None, max_workers=None, global_ref_point=None, force_global_ref=False):
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
        
        # ÉTAPE 1.5 : Priorisation du point de référence global si forcé
        if force_global_ref and global_ref_point is not None:
            logger.info("🎯 UTILISATION DU POINT DE RÉFÉRENCE GLOBAL FORCÉ (déformation)")
            logger.info(f"Point global : ({global_ref_point[0]:.6f}, {global_ref_point[1]:.6f}, {global_ref_point[2]:.6f})")
            ref_point = np.array(global_ref_point)
            logger.info("Le point global remplace le point local pour la déformation")
            logger.info("⚠️  L'offset ne s'applique PAS au point global (coordonnées absolues préservées)")
        else:
            logger.info("📍 UTILISATION DU POINT DE RÉFÉRENCE LOCAL (déformation)")
            
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
                        logger.info(f"Point local trouvé : {parts[0]} ({x:.6f}, {y:.6f}, {z:.6f})")
                        break
                    except ValueError:
                        continue
            
            if ref_point is None:
                logger.error("Aucun point de référence valide trouvé dans le fichier de coordonnées")
                raise RuntimeError("Aucun point de référence valide trouvé dans le fichier de coordonnées")
            
            # Lecture de l'offset et application au point local UNIQUEMENT
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
            
            # Application de l'offset au point local UNIQUEMENT
            if offset:
                ref_point = ref_point + np.array(offset)
                logger.info(f"Point local avec offset appliqué : ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
        
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
        
        # DEBUG : Vérification des coordonnées de transformation
        logger.info(f"🔍 DEBUG - Transformation résidu {name}:")
        logger.info(f"   Point de référence ITRF: ({ref_point[0]:.6f}, {ref_point[1]:.6f}, {ref_point[2]:.6f})")
        logger.info(f"   Point avec offset ITRF: ({point_with_offset[0]:.6f}, {point_with_offset[1]:.6f}, {point_with_offset[2]:.6f})")
        logger.info(f"   Point de référence ENU: ({enu_ref[0]:.6f}, {enu_ref[1]:.6f}, {enu_ref[2]:.6f})")
        logger.info(f"   Point avec offset ENU: ({enu_point[0]:.6f}, {enu_point[1]:.6f}, {enu_point[2]:.6f})")
        logger.info(f"   Vecteur déplacement ENU: ({enu_offset[0]:.6f}, {enu_offset[1]:.6f}, {enu_offset[2]:.6f})")
        
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
        
        # IMPORTANT : Relecture de l'offset du fichier de coordonnées pour les GCPs
        coord_offset = None
        try:
            for line in lines:
                if line.startswith('#Offset to add :'):
                    offset_text = line.replace('#Offset to add :', '').strip()
                    coord_offset = [float(x) for x in offset_text.split()]
                    logger.info(f"Offset du fichier de coordonnées pour GCPs : {coord_offset}")
                    break
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture de l'offset pour GCPs : {e}")
            coord_offset = [0.0, 0.0, 0.0]
        
        for line in lines:
            line = line.strip()
            if line.startswith('#F=') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    name = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # IMPORTANT : L'offset doit TOUJOURS être appliqué pour obtenir les vraies coordonnées ITRF
                    # Le point global sert seulement de centre de transformation ENU
                    if coord_offset is not None:
                        x += coord_offset[0]
                        y += coord_offset[1]
                        z += coord_offset[2]
                        logger.info(f"GCP {name} : offset appliqué ({coord_offset[0]:.6f}, {coord_offset[1]:.6f}, {coord_offset[2]:.6f})")
                    else:
                        logger.warning(f"GCP {name} : pas d'offset disponible, coordonnées relatives utilisées")
                    
                    point_itrf = np.array([x, y, z])
                    
                    if force_global_ref and global_ref_point is not None:
                        logger.info(f"GCP {name} : transformation ENU avec le point global comme centre")
                    else:
                        logger.info(f"GCP {name} : transformation ENU avec le point local comme centre")
                    
                    # Conversion en ENU avec le MÊME transformer
                    enu_pos = transformer.transform(point_itrf[0], point_itrf[1], point_itrf[2])
                    gcp_positions[name] = np.array([enu_pos[0], enu_pos[1], enu_pos[2]])
                    
                    logger.info(f"GCP {name}: ITRF({point_itrf[0]:.6f}, {point_itrf[1]:.6f}, {point_itrf[2]:.6f}) → ENU({enu_pos[0]:.6f}, {enu_pos[1]:.6f}, {enu_pos[2]:.6f})")
                    
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

def individual_zone_equalization(zone_ortho_path, logger):
    """
    ÉGALISATION GLOBALE : Interface principale pour l'égalisation des zones
    Utilise la nouvelle stratégie d'égalisation globale basée sur les quantiles
    
    Args:
        zone_ortho_path: Chemin vers l'ortho de zone à égaliser
        logger: Logger pour les messages
    
    Returns:
        str: Chemin vers la zone égalisée
    """
    # Cette fonction est maintenant une interface qui appelle l'égalisation globale
    # Elle sera remplacée par l'appel direct dans le pipeline principal
    logger.warning("⚠️ Cette fonction est obsolète. Utilisez l'égalisation globale directement.")
    return zone_ortho_path
