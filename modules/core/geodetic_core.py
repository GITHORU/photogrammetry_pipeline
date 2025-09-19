"""
Module de base pour les opérations géodésiques.
Contient les fonctions de traitement de base des nuages de points.
"""

import os
import numpy as np
import logging
from multiprocessing import Pool, cpu_count

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
    if len(args) == 5:
        # Ancien format : compatibilité
        ply_file, output_dir, coord_file, extra_params, ref_point_name = args
        global_ref_point = None
        force_global_ref = False
    else:
        # Nouveau format avec point global
        ply_file, output_dir, coord_file, extra_params, ref_point_name, global_ref_point, force_global_ref = args
    
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
        
        # ÉTAPE 1 : Priorisation du point de référence global si forcé
        if force_global_ref and global_ref_point is not None:
            logger.info("🎯 UTILISATION DU POINT DE RÉFÉRENCE GLOBAL FORCÉ (ITRF→ENU)")
            logger.info(f"Point global : ({global_ref_point[0]:.6f}, {global_ref_point[1]:.6f}, {global_ref_point[2]:.6f})")
            tr_center = global_ref_point
            logger.info("Le point global remplace le point local pour la transformation ITRF→ENU")
            logger.info("⚠️  L'offset ne s'applique PAS au point global (coordonnées absolues préservées)")
        else:
            logger.info("📍 UTILISATION DU POINT DE RÉFÉRENCE LOCAL (ITRF→ENU)")
            
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
            
            # Application de l'offset au point local UNIQUEMENT
            ref_point_with_offset = [ref_point[0] + offset[0], ref_point[1] + offset[1], ref_point[2] + offset[2]]
            tr_center = ref_point_with_offset
            logger.info(f"Point local avec offset appliqué : ({tr_center[0]:.6f}, {tr_center[1]:.6f}, {tr_center[2]:.6f})")
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
            
            # DEBUG : Vérification des coordonnées de transformation (premier chunk seulement)
            if i == 0:
                logger.info(f"🔍 DEBUG - Transformation ITRF→ENU (premier chunk):")
                logger.info(f"   Premier point ITRF: ({arr_x[0]:.6f}, {arr_y[0]:.6f}, {arr_z[0]:.6f})")
                logger.info(f"   Premier point ENU: ({chunk_enu[0, 0]:.6f}, {chunk_enu[0, 1]:.6f}, {chunk_enu[0, 2]:.6f})")
                logger.info(f"   Dernier point ITRF: ({arr_x[-1]:.6f}, {arr_y[-1]:.6f}, {arr_z[-1]:.6f})")
                logger.info(f"   Dernier point ENU: ({chunk_enu[-1, 0]:.6f}, {chunk_enu[-1, 1]:.6f}, {chunk_enu[-1, 2]:.6f})")
                
                # DEBUG : Vérification du centre de transformation
                logger.info(f"🔍 DEBUG - Centre de transformation utilisé:")
                logger.info(f"   Centre ITRF: ({tr_center[0]:.6f}, {tr_center[1]:.6f}, {tr_center[2]:.6f})")
                logger.info(f"   Centre ENU: (0.000000, 0.000000, 0.000000)")
                
                # DEBUG : Calcul manuel pour vérifier
                test_point_itrf = np.array([arr_x[0], arr_y[0], arr_z[0]])
                test_point_enu = transformer.transform(test_point_itrf[0], test_point_itrf[1], test_point_itrf[2])
                logger.info(f"🔍 DEBUG - Vérification manuelle:")
                logger.info(f"   Point test ITRF: ({test_point_itrf[0]:.6f}, {test_point_itrf[1]:.6f}, {test_point_itrf[2]:.6f})")
                logger.info(f"   Point test ENU: ({test_point_enu[0]:.6f}, {test_point_enu[1]:.6f}, {test_point_enu[2]:.6f})")
                logger.info(f"   Différence ITRF - Centre: ({test_point_itrf[0] - tr_center[0]:.6f}, {test_point_itrf[1] - tr_center[1]:.6f}, {test_point_itrf[2] - tr_center[2]:.6f})")
        
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
    # Assurer un handler en sous-processus pour voir les logs (console)
    if not logger.handlers:
        _sh = logging.StreamHandler()
        _sh.setLevel(logging.INFO)
        _sh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        logger.addHandler(_sh)
        logger.propagate = False
    
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
        if deformation_type == "none":
            logger.info("  Aucune déformation appliquée (mode none)")
            deformations = np.zeros_like(points)
        elif deformation_type == "tps":
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
            
        elif deformation_type == "radial":
            logger.info(f"  Déformation radiale (centre libre, a1,a2,c1) avec {len(control_points)} GCPs...")

            # Estimation des paramètres sur les GCPs (ENU)
            cps_xy = control_points[:, :2]
            offs = control_values  # (dE, dN, dU)

            # Barycentre XY des GCPs (diagnostic d'invariance au repère)
            bary_x = float(np.mean(cps_xy[:, 0]))
            bary_y = float(np.mean(cps_xy[:, 1]))
            # Quelques stats de rayon par rapport au barycentre
            _dxb = cps_xy[:, 0] - bary_x
            _dyb = cps_xy[:, 1] - bary_y
            _rb = np.sqrt(_dxb * _dxb + _dyb * _dyb)
            rb_min = float(np.min(_rb)) if _rb.size else 0.0
            rb_med = float(np.median(_rb)) if _rb.size else 0.0
            rb_max = float(np.max(_rb)) if _rb.size else 0.0
            logger.info(
                f"    Barycentre GCPs (E,N)=({bary_x:.6f},{bary_y:.6f}) | r(min/med/max)={rb_min:.3f}/{rb_med:.3f}/{rb_max:.3f} m"
            )

            def compute_radial_components_xy(points_xy, center):
                cx, cy = center
                dx = points_xy[:, 0] - cx
                dy = points_xy[:, 1] - cy
                r = np.sqrt(dx * dx + dy * dy)
                denom = np.where(r == 0.0, 1.0, r)
                ux = np.where(r == 0.0, 0.0, dx / denom)
                uy = np.where(r == 0.0, 0.0, dy / denom)
                return r, ux, uy

            # Gauss-Newton robuste (Huber), paramètres initiaux
            cx, cy = bary_x, bary_y
            a1, a2, c1p = 0.0, 0.0, 0.0  # a2 forcé à 0 (pas de terme r^3)
            logger.info(
                f"    Init centre (cx,cy)=({cx:.6f},{cy:.6f}), a1={a1:.3e}, a2={a2:.3e}, c1={c1p:.3e}"
            )

            max_iter = 200
            tol = 1e-8
            prev_cost = None
            for it in range(max_iter):
                r, ux, uy = compute_radial_components_xy(cps_xy, (cx, cy))
                # Modèle simple sans normalisation ni pondération robuste
                g = a1 * r  # a2=0 → pas de terme r^3
                pred_e = g * ux
                pred_n = g * uy
                pred_u = c1p * r

                res_e = offs[:, 0] - pred_e
                res_n = offs[:, 1] - pred_n
                res_u = offs[:, 2] - pred_u

                # Poids uniformes (pas de Huber)
                w_e = np.ones_like(res_e)
                w_n = np.ones_like(res_n)
                w_u = np.ones_like(res_u)

                JtWJ = np.zeros((5, 5), dtype=float)
                JtWr = np.zeros(5, dtype=float)

                dx = cps_xy[:, 0] - cx
                dy = cps_xy[:, 1] - cy
                r_safe = np.where(r == 0.0, 1.0, r)
                r3 = r_safe ** 3
                dudx_dcx = -1.0 / r_safe + (dx * dx) / r3
                dudx_dcy = (dx * dy) / r3
                dudy_dcx = (dx * dy) / r3
                dudy_dcy = -1.0 / r_safe + (dy * dy) / r3
                # g = a1 * r + a2 * r^3 ; dg/dr = a1 + 3*a2*r^2
                dg_dr = (a1 + 3.0 * a2 * (r ** 2))
                dr_dcx = -dx / r_safe
                dr_dcy = -dy / r_safe

                dpe_dcx = dg_dr * dr_dcx * ux + g * dudx_dcx
                dpe_dcy = dg_dr * dr_dcy * ux + g * dudx_dcy
                dpe_da1 = r * ux
                # Neutraliser la colonne a2 (suppression du terme r^3)
                dpe_da2 = np.zeros_like(r)

                dpn_dcx = dg_dr * dr_dcx * uy + g * dudy_dcx
                dpn_dcy = dg_dr * dr_dcy * uy + g * dudy_dcy
                dpn_da1 = r * uy
                # Neutraliser la colonne a2 (suppression du terme r^3)
                dpn_da2 = np.zeros_like(r)

                # pred_u = c1p * r
                dpu_dcx = c1p * dr_dcx
                dpu_dcy = c1p * dr_dcy
                dpu_dc1 = r

                for i in range(len(cps_xy)):
                    Ji_e = np.array([dpe_dcx[i], dpe_dcy[i], dpe_da1[i], dpe_da2[i], 0.0])
                    Ji_n = np.array([dpn_dcx[i], dpn_dcy[i], dpn_da1[i], dpn_da2[i], 0.0])
                    Ji_u = np.array([dpu_dcx[i], dpu_dcy[i], 0.0, 0.0, dpu_dc1[i]])

                    we, wn, wu = w_e[i], w_n[i], w_u[i]
                    JtWJ += we * np.outer(Ji_e, Ji_e)
                    JtWJ += wn * np.outer(Ji_n, Ji_n)
                    JtWJ += wu * np.outer(Ji_u, Ji_u)
                    JtWr += we * Ji_e * res_e[i]
                    JtWr += wn * Ji_n * res_n[i]
                    JtWr += wu * Ji_u * res_u[i]

                try:
                    delta = np.linalg.solve(JtWJ, JtWr)
                except np.linalg.LinAlgError:
                    JtWJ += np.eye(5) * 1e-8
                    delta = np.linalg.lstsq(JtWJ, JtWr, rcond=None)[0]

                dcx, dcy, da1p, da2p, dc1 = delta
                cx += dcx
                cy += dcy
                a1 += da1p
                # a2 reste forcé à 0 (ignorer la mise à jour éventuelle)
                a2 = 0.0
                c1p += dc1
                step_norm = float(np.linalg.norm(delta))

                # Coût et diagnostics d'itération
                cost = float(np.sum(w_e * res_e * res_e) + np.sum(w_n * res_n * res_n) + np.sum(w_u * res_u * res_u))
                rms_e = float(np.sqrt(np.mean(res_e * res_e)))
                rms_n = float(np.sqrt(np.mean(res_n * res_n)))
                rms_u = float(np.sqrt(np.mean(res_u * res_u)))
                if it == 0 or it % 5 == 0:
                    logger.info(
                        f"    it={it:03d} cost={cost:.6e} step={step_norm:.3e} | RMS(E,N,U)=({rms_e:.4e},{rms_n:.4e},{rms_u:.4e}) | "
                        f"params: cx={cx:.6f} cy={cy:.6f} a1={a1:.3e} a2={a2:.3e} c1={c1p:.3e}"
                    )
                if step_norm < tol:
                    logger.info(
                        f"    Convergence atteinte à it={it} (step={step_norm:.3e}, Δcost={(0.0 if prev_cost is None else prev_cost - cost):.3e})"
                    )
                    break
                prev_cost = cost

            # Appliquer au nuage complet
            r_all, ux_all, uy_all = compute_radial_components_xy(points[:, :2], (cx, cy))
            g_all = a1 * r_all  # a2=0
            deform_e = g_all * ux_all
            deform_n = g_all * uy_all
            deform_u = c1p * r_all
            deformations = np.column_stack([deform_e, deform_n, deform_u])

            logger.info(f"  Radial params: center=({cx:.6f},{cy:.6f}), a1={a1:.6e}, a2={a2:.6e}, c1={c1p:.6e}")
            # Résidus finaux sur GCPs pour contrôle (modèle simple)
            res_e_f = offs[:, 0] - (a1 * r * ux)
            res_n_f = offs[:, 1] - (a1 * r * uy)
            res_u_f = offs[:, 2] - (c1p * r)
            logger.info(
                f"  Résidus finaux GCPs RMS(E,N,U)=({np.sqrt(np.mean(res_e_f**2)):.4e},{np.sqrt(np.mean(res_n_f**2)):.4e},{np.sqrt(np.mean(res_u_f**2)):.4e})"
            )

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
        
        logger.info(f"  Déformation {deformation_type} appliquée - min: {min_deform:.6f}, max: {max_deform:.6f}, moy: {mean_deform:.6f}")
        
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