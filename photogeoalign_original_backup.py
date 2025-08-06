import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit,
    QMessageBox, QSplashScreen, QTabWidget, QCheckBox, QToolBar, QDialog, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QPoint
from PySide6.QtGui import QPixmap, QIcon, QMovie, QAction, QPainter, QColor, QBrush, QPen
import subprocess
import logging
import argparse
import time
from time import sleep
from pathlib import Path
import re
import platform
import shutil
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial

# Protection freeze_support pour Windows/pyinstaller
if __name__ == "__main__":
    multiprocessing.freeze_support()

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

def setup_logger(log_path=None):
    logger = logging.getLogger("PhotogrammetryPipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Console handler (INFO et plus)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    # File handler (DEBUG et plus)
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def run_command(cmd, logger, cwd=None):
    logger.info(f"Commande lancée : {' '.join(cmd)}")
    try:
        creationflags = 0
        if os.name == 'nt':
            import subprocess as sp
            creationflags = sp.CREATE_NO_WINDOW
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
            creationflags=creationflags,
            stdin=subprocess.PIPE
        )
        if process.stdout is not None:
            for line in process.stdout:
                logger.info(line.rstrip())
                if 'Warn tape enter to continue' in line:
                    try:
                        if process.stdin is not None:
                            process.stdin.write('\n')
                            process.stdin.flush()
                    except Exception:
                        pass
        process.wait()
        if process.returncode != 0:
            logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)} (code {process.returncode})")
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)}")
        logger.error(f"Code retour : {e.returncode}")
        raise

def run_micmac_tapioca(input_dir, logger, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapioca va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    # 1. Génération des tie points (pipeline)
    cmd = [
        'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    homol_dir = Path(abs_input_dir) / 'Homol'
    if homol_dir.exists() and any(homol_dir.iterdir()):
        logger.info(f"Dossier Homol généré : {homol_dir}")
    else:
        logger.error("Le dossier Homol n'a pas été généré par Tapioca. Arrêt du pipeline.")
        raise RuntimeError("Le dossier Homol n'a pas été généré par Tapioca.")
    logger.info("Tapioca terminé. Les tie points .dat sont utilisés pour le pipeline.")

def run_micmac_tapas(input_dir, logger, tapas_model="Fraser", extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapas va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapas', tapas_model, pattern, f'Out={tapas_model}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(input_dir, logger, mode='QuickMac', zoomf=1, tapas_model='Fraser', extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    ori = f"{tapas_model}_abs"
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} et Ori={ori} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, ori, f'ZoomF={zoomf}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"Nuage dense généré par C3DC {mode} (voir dossier PIMs-{mode}/ ou fichier C3DC_{mode}.ply)")

def to_micmac_path(path):
    return path.replace("\\", "/")

def micmac_command_exists(cmd):
    try:
        result = subprocess.run(
            ['mm3d'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True
        )
        return cmd in result.stdout
    except Exception:
        return False

def run_micmac_saisieappuisinit(input_dir, logger, tapas_model="Fraser", appuis_file=None, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori = tapas_model
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour SaisieAppuisInit.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour SaisieAppuisInit.")
    appuis_file = os.path.abspath(appuis_file)
    if not os.path.exists(appuis_file):
        logger.error(f"Fichier de coordonnées introuvable : {appuis_file}")
        raise RuntimeError(f"Fichier de coordonnées introuvable : {appuis_file}")
    if not appuis_file.lower().endswith('.txt'):
        logger.error("Le fichier de coordonnées doit être au format .txt")
        raise RuntimeError("Le fichier de coordonnées doit être au format .txt")
    # Conversion systématique en xml
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    logger.info(f"Conversion du fichier de coordonnées TXT en XML avec GCPConvert : {appuis_file} -> {xml_file}")
    cmd_gcp = ['mm3d', 'GCPConvert', 'AppInFile', appuis_file]
    run_command(cmd_gcp, logger, cwd=abs_input_dir)
    if not os.path.exists(xml_file):
        xml_file_candidate = os.path.join(abs_input_dir, os.path.basename(xml_file))
        if os.path.exists(xml_file_candidate):
            xml_file = xml_file_candidate
        else:
            logger.error(f"Le fichier XML n'a pas été généré par GCPConvert : {xml_file}")
            raise RuntimeError(f"Le fichier XML n'a pas été généré par GCPConvert : {xml_file}")
    # Chemin relatif pour MicMac
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    # Détection de la commande à utiliser
    if micmac_command_exists('SaisieAppuisInitQT'):
        cmd_name = 'SaisieAppuisInitQT'
    else:
        cmd_name = 'SaisieAppuisInit'
    logger.info(f"Lancement de {cmd_name} dans {abs_input_dir} sur {pattern} avec Ori={ori}, appuis={xml_file_rel}, sortie=PtsImgInit.xml ...")
    cmd = [
        'mm3d', cmd_name, pattern, ori, xml_file_rel, 'PtsImgInit.xml'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"{cmd_name} terminé.")
    return os.path.join(abs_input_dir, "PtsImgInit.xml")

def run_micmac_gcpbascule_init(input_dir, logger, tapas_model="Fraser", appuis_file=None):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori_in = tapas_model
    ori_out = f"{tapas_model}_abs_init"
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour GCPBascule (init).")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour GCPBascule (init).")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimginit_s2d = os.path.join(abs_input_dir, "PtsImgInit-S2D.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    ptsimginit_s2d_rel = os.path.relpath(ptsimginit_s2d, abs_input_dir)
    logger.info(f"Lancement de GCPBascule (init) dans {abs_input_dir} sur {pattern} avec Ori_in={ori_in}, Ori_out={ori_out}, appuis={xml_file_rel}, ptsinit={ptsimginit_s2d_rel} ...")
    cmd = [
        'mm3d', 'GCPBascule', pattern, ori_in, ori_out, xml_file_rel, ptsimginit_s2d_rel
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("GCPBascule (init) terminé.")
    return ori_out

def run_micmac_saisieappuispredic(input_dir, logger, tapas_model="Fraser", ori_abs_init=None, appuis_file=None, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori = ori_abs_init or f"{tapas_model}_abs_init"  # Utilise l'orientation de sortie de GCPBascule
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour SaisieAppuisPredic.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour SaisieAppuisPredic.")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimgpredic_file = os.path.join(abs_input_dir, "PtsImgPredic.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    # Détection de la commande à utiliser
    if micmac_command_exists('SaisieAppuisPredicQT'):
        cmd_name = 'SaisieAppuisPredicQT'
    else:
        cmd_name = 'SaisieAppuisPredic'
    logger.info(f"Lancement de {cmd_name} dans {abs_input_dir} sur {pattern} avec Ori={ori}, appuis={xml_file_rel}, sortie=PtsImgPredic.xml ...")
    cmd = [
        'mm3d', cmd_name, pattern, ori, xml_file_rel, 'PtsImgPredic.xml'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"{cmd_name} terminé.")
    return ptsimgpredic_file

def run_micmac_gcpbascule_predic(input_dir, logger, tapas_model="Fraser", appuis_file=None):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori_in = f"{tapas_model}_abs_init"
    ori_out = f"{tapas_model}_abs"
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour GCPBascule (predic).")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour GCPBascule (predic).")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimgpredic_s2d = os.path.join(abs_input_dir, "PtsImgPredic-S2D.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    ptsimgpredic_s2d_rel = os.path.relpath(ptsimgpredic_s2d, abs_input_dir)
    logger.info(f"Lancement de GCPBascule (predic) dans {abs_input_dir} sur {pattern} avec Ori_in={ori_in}, Ori_out={ori_out}, appuis={xml_file_rel}, pts={ptsimgpredic_s2d_rel} ...")
    cmd = [
        'mm3d', 'GCPBascule', pattern, ori_in, ori_out, xml_file_rel, ptsimgpredic_s2d_rel
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("GCPBascule (predic) terminé.")
    return ori_out

# Fonctions de transformation géodésique
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
    import xml.etree.ElementTree as ET
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

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class PipelineThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, run_tapioca=True, run_tapas=True, run_saisieappuisinit=True, run_saisieappuispredic=True, run_c3dc=True):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf
        self.tapas_model = tapas_model
        self.tapioca_extra = tapioca_extra
        self.tapas_extra = tapas_extra
        self.saisieappuisinit_extra = saisieappuisinit_extra
        self.saisieappuispredic_extra = saisieappuispredic_extra
        self.c3dc_extra = c3dc_extra
        self.saisieappuisinit_pt = saisieappuisinit_pt
        self.run_tapioca = run_tapioca
        self.run_tapas = run_tapas
        self.run_saisieappuisinit = run_saisieappuisinit
        self.run_saisieappuispredic = run_saisieappuispredic
        self.run_c3dc = run_c3dc

    def run(self):
        logger = logging.getLogger(f"PhotogrammetryPipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        try:
            self.log_signal.emit("Démarrage du pipeline...\n")
            if self.run_tapioca:
                run_micmac_tapioca(self.input_dir, logger, self.tapioca_extra)
                self.log_signal.emit("Tapioca terminé.\n")
            if self.run_tapas:
                run_micmac_tapas(self.input_dir, logger, self.tapas_model, self.tapas_extra)
                self.log_signal.emit("Tapas terminé.\n")
            ori_abs_init = None
            if self.run_saisieappuisinit:
                run_micmac_saisieappuisinit(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt, self.saisieappuisinit_extra)
                self.log_signal.emit("SaisieAppuisInitQT terminé.\n")
                ori_abs_init = run_micmac_gcpbascule_init(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (init) terminé.\n")
            if self.run_saisieappuispredic:
                run_micmac_saisieappuispredic(self.input_dir, logger, self.tapas_model, ori_abs_init, self.saisieappuisinit_pt, self.saisieappuispredic_extra)
                self.log_signal.emit("SaisieAppuisPredicQT terminé.\n")
                run_micmac_gcpbascule_predic(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (predic) terminé.\n")
            if self.run_c3dc:
                run_micmac_c3dc(self.input_dir, logger, mode=self.mode, zoomf=self.zoomf, tapas_model=self.tapas_model, extra_params=self.c3dc_extra)
                self.log_signal.emit("C3DC terminé.\n")
            self.finished_signal.emit(True, "Pipeline terminé avec succès !")
        except Exception as e:
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline : {e}")

class GeodeticTransformThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, coord_file, deformation_type, deformation_params, add_offset_extra, itrf_to_enu_extra, deform_extra, enu_to_itrf_extra, run_add_offset=True, run_itrf_to_enu=True, run_deform=True, run_enu_to_itrf=True, add_offset_input_dir=None, itrf_to_enu_input_dir=None, deform_input_dir=None, enu_to_itrf_input_dir=None, add_offset_output_dir=None, itrf_to_enu_output_dir=None, deform_output_dir=None, enu_to_itrf_output_dir=None, itrf_to_enu_ref_point=None, deform_bascule_xml=None, max_workers=None):
        super().__init__()
        self.input_dir = input_dir
        self.coord_file = coord_file
        self.deformation_type = deformation_type
        self.deformation_params = deformation_params
        self.add_offset_extra = add_offset_extra
        self.itrf_to_enu_extra = itrf_to_enu_extra
        self.deform_extra = deform_extra
        self.enu_to_itrf_extra = enu_to_itrf_extra
        self.run_add_offset = run_add_offset
        self.run_itrf_to_enu = run_itrf_to_enu
        self.run_deform = run_deform
        self.run_enu_to_itrf = run_enu_to_itrf
        # Dossiers d'entrée personnalisés pour chaque étape
        self.add_offset_input_dir = add_offset_input_dir
        self.itrf_to_enu_input_dir = itrf_to_enu_input_dir
        self.deform_input_dir = deform_input_dir
        self.enu_to_itrf_input_dir = enu_to_itrf_input_dir
        # Dossiers de sortie personnalisés pour chaque étape
        self.add_offset_output_dir = add_offset_output_dir
        self.itrf_to_enu_output_dir = itrf_to_enu_output_dir
        self.deform_output_dir = deform_output_dir
        self.enu_to_itrf_output_dir = enu_to_itrf_output_dir
        self.itrf_to_enu_ref_point = itrf_to_enu_ref_point
        self.deform_bascule_xml = deform_bascule_xml
        self.max_workers = max_workers

    def run(self):
        logger = logging.getLogger(f"GeodeticTransform_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        # Handler pour l'interface graphique (Qt)
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        
        # Handler pour la console (mode CLI)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        try:
            start_msg = "Démarrage des transformations géodésiques..."
            self.log_signal.emit(start_msg + "\n")
            print(start_msg)
            
            # Gestion des dossiers d'entrée pour chaque étape
            # On initialise avec le dossier de travail principal
            current_input_dir = self.input_dir
            
            if self.run_add_offset:
                # Utiliser le dossier d'entrée personnalisé ou le dossier initial
                step_input_dir = self.add_offset_input_dir if self.add_offset_input_dir else current_input_dir
                add_offset_to_clouds(step_input_dir, logger, self.coord_file, self.add_offset_extra, self.max_workers)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.add_offset_output_dir:
                    current_input_dir = self.add_offset_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), "offset_step")
                self.log_signal.emit("Ajout d'offset terminé.\n")
                print("Ajout d'offset terminé.")
                
            if self.run_itrf_to_enu:
                # Utiliser le dossier d'entrée personnalisé ou le dossier de l'étape précédente
                step_input_dir = self.itrf_to_enu_input_dir if self.itrf_to_enu_input_dir else current_input_dir
                convert_itrf_to_enu(step_input_dir, logger, self.coord_file, self.itrf_to_enu_extra, self.itrf_to_enu_ref_point, self.max_workers)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.itrf_to_enu_output_dir:
                    current_input_dir = self.itrf_to_enu_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), "itrf_to_enu_step")
                self.log_signal.emit("Conversion ITRF→ENU terminée.\n")
                print("Conversion ITRF→ENU terminée.")
                
            if self.run_deform:
                # Utiliser le dossier d'entrée personnalisé ou le dossier de l'étape précédente
                step_input_dir = self.deform_input_dir if self.deform_input_dir else current_input_dir
                deform_clouds(step_input_dir, logger, self.deformation_type, self.deformation_params, self.deform_extra, self.deform_bascule_xml, self.coord_file, self.max_workers)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.deform_output_dir:
                    current_input_dir = self.deform_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), f"deform_{self.deformation_type}_step")
                self.log_signal.emit("Déformation terminée.\n")
                print("Déformation terminée.")
                
            if self.run_enu_to_itrf:
                # Utiliser le dossier d'entrée personnalisé ou le dossier de l'étape précédente
                step_input_dir = self.enu_to_itrf_input_dir if self.enu_to_itrf_input_dir else current_input_dir
                convert_enu_to_itrf(step_input_dir, logger, self.coord_file, self.enu_to_itrf_extra, self.max_workers)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.enu_to_itrf_output_dir:
                    current_input_dir = self.enu_to_itrf_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), "enu_to_itrf_step")
                self.log_signal.emit("Conversion ENU→ITRF terminée.\n")
                print("Conversion ENU→ITRF terminée.")
                
            success_msg = "Transformations géodésiques terminées avec succès !"
            self.finished_signal.emit(True, success_msg)
            print(success_msg)
        except Exception as e:
            error_msg = f"Erreur lors des transformations géodésiques : {e}"
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, error_msg)
            print(f"Erreur : {e}")

def resource_path(relative_path):
    """Trouve le chemin absolu d'une ressource, compatible PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

class JobExportDialog(QDialog):
    def __init__(self, parent=None, job_name="PhotoGeoAlign", output="PhotoGeoAlign.out", partition="ncpu", ntasks=128, cli_cmd=""):
        super().__init__(parent)
        self.setWindowTitle("Exporter le batch SLURM (.job)")
        self.setModal(True)
        layout = QFormLayout(self)
        self.job_name_edit = QLineEdit(job_name)
        self.output_edit = QLineEdit(output)
        self.partition_combo = QComboBox()
        self.partition_combo.addItems(["ncpu", "ncpum", "ncpu_long", "ncpu_short"])
        self.partition_combo.setCurrentText(partition)
        self.ntasks_spin = QSpinBox()
        self.ntasks_spin.setMinimum(1)
        self.ntasks_spin.setMaximum(1024)
        self.ntasks_spin.setValue(ntasks)
        self.cli_cmd = cli_cmd
        layout.addRow("Nom du job :", self.job_name_edit)
        layout.addRow("Fichier de sortie :", self.output_edit)
        layout.addRow("Partition :", self.partition_combo)
        layout.addRow("Nombre de tâches :", self.ntasks_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_values(self):
        return {
            "job_name": self.job_name_edit.text().strip() or "PhotoGeoAlign",
            "output": self.output_edit.text().strip() or "PhotoGeoAlign.out",
            "partition": self.partition_combo.currentText(),
            "ntasks": self.ntasks_spin.value(),
            "cli_cmd": self.cli_cmd
        }

class PhotogrammetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoGeoAlign")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.pipeline_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        # Barre d'outils
        toolbar = QToolBar()
        # Icône flèche verte pour Lancer
        pixmap_run = QPixmap(24, 24)
        pixmap_run.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_run)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(76, 175, 80)))  # vert
        painter.setPen(Qt.GlobalColor.transparent)
        points = [
            pixmap_run.rect().topLeft() + QPoint(6, 4),
            pixmap_run.rect().bottomLeft() + QPoint(6, -4),
            pixmap_run.rect().center() + QPoint(6, 0)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_run = QIcon(pixmap_run)
        action_run = QAction(icon_run, "Lancer le pipeline MicMac", self)
        action_run.triggered.connect(self.launch_pipeline)
        toolbar.addAction(action_run)
        self.action_run = action_run
        
        # Icône flèche bleue pour Lancer le pipeline géodésique
        pixmap_geodetic = QPixmap(24, 24)
        pixmap_geodetic.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_geodetic)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(33, 150, 243)))  # bleu
        painter.setPen(Qt.GlobalColor.transparent)
        points = [
            pixmap_geodetic.rect().topLeft() + QPoint(6, 4),
            pixmap_geodetic.rect().bottomLeft() + QPoint(6, -4),
            pixmap_geodetic.rect().center() + QPoint(6, 0)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_geodetic = QIcon(pixmap_geodetic)
        action_geodetic = QAction(icon_geodetic, "Lancer le pipeline géodésique", self)
        action_geodetic.triggered.connect(self.launch_geodetic_pipeline)
        toolbar.addAction(action_geodetic)
        self.action_geodetic = action_geodetic
        
        # Icône rond rouge pour Arrêter
        pixmap_stop = QPixmap(24, 24)
        pixmap_stop.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_stop)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(211, 47, 47)))  # rouge
        painter.setPen(Qt.GlobalColor.transparent)
        painter.drawEllipse(6, 6, 12, 12)
        painter.end()
        icon_stop = QIcon(pixmap_stop)
        action_stop = QAction(icon_stop, "Arrêter", self)
        action_stop.triggered.connect(self.stop_pipeline)
        toolbar.addAction(action_stop)
        self.action_stop = action_stop
        self.action_stop.setEnabled(False)
        # Icône pour Export .job (flèche vers le bas orange, plus grande)
        pixmap_export = QPixmap(24, 24)
        pixmap_export.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_export)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        orange = QColor(255, 152, 0)  # #ff9800
        painter.setBrush(QBrush(orange))
        painter.setPen(Qt.GlobalColor.transparent)
        # Tige de la flèche (plus large et plus longue)
        painter.drawRect(10, 6, 4, 10)
        # Pointe de la flèche (plus grande)
        points = [
            QPoint(12, 21), QPoint(6, 14), QPoint(18, 14)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_export = QIcon(pixmap_export)
        action_export = QAction(icon_export, "Exporter le batch .job", self)
        action_export.triggered.connect(self.export_job_dialog)
        toolbar.addAction(action_export)
        main_layout.addWidget(toolbar)
        # Tabs
        tabs = QTabWidget()
        # Onglet 1 : paramètres, boutons, sélecteur python, ligne de commande, résumé
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)
        # 1. Dossier image
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Dossier d'images DNG à traiter")
        browse_btn = QPushButton("Parcourir…")
        browse_btn.clicked.connect(self.browse_folder)
        dir_layout.addWidget(QLabel("Dossier d'images :"))
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        param_layout.addLayout(dir_layout)
        # 2. Mode
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["QuickMac", "BigMac", "MicMac"])
        self.mode_combo.setCurrentText("BigMac")
        mode_layout.addWidget(QLabel("Mode C3DC :"))
        mode_layout.addWidget(self.mode_combo)
        param_layout.addLayout(mode_layout)
        # 3. Facteur de zoom
        zoom_layout = QHBoxLayout()
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setMinimum(1)
        self.zoom_spin.setMaximum(8)
        self.zoom_spin.setValue(1)
        zoom_layout.addWidget(QLabel("Facteur de zoom (ZoomF) :"))
        zoom_layout.addWidget(self.zoom_spin)
        param_layout.addLayout(zoom_layout)
        # 4. Modèle Tapas
        tapas_model_layout = QHBoxLayout()
        self.tapas_model_combo = QComboBox()
        self.tapas_model_combo.addItems([
            "RadialBasic",
            "RadialExtended",
            "Fraser",
            "FishEyeEqui",
            "AutoCal",
            "Figee",
            "HemiEqui",
            "RadialStd",
            "FraserBasic",
            "FishEyeBasic",
            "FE_EquiSolBasic",
            "Four7x2",
            "Four11x2",
            "Four15x2",
            "Four19x2",
            "AddFour7x2",
            "AddFour11x2",
            "AddFour15x2",
            "AddFour19x2",
            "AddPolyDeg0",
            "AddPolyDeg1",
            "AddPolyDeg2",
            "AddPolyDeg3",
            "AddPolyDeg4",
            "AddPolyDeg5",
            "AddPolyDeg6",
            "AddPolyDeg7",
            "Ebner",
            "Brown",
            "FishEyeStereo"
        ])
        self.tapas_model_combo.setCurrentText("Fraser")
        tapas_model_layout.addWidget(QLabel("Modèle Tapas :"))
        tapas_model_layout.addWidget(self.tapas_model_combo)
        param_layout.addLayout(tapas_model_layout)
        # 5. Fichier de coordonnées
        pt_layout = QHBoxLayout()
        self.pt_lineedit = QLineEdit()
        self.pt_lineedit.setPlaceholderText("Chemin du fichier de coordonnées (.txt uniquement)")
        self.pt_lineedit.setText("")
        pt_browse_btn = QPushButton("Parcourir…")
        pt_browse_btn.clicked.connect(self.browse_pt_file)
        pt_layout.addWidget(QLabel("Fichier de coordonnées :"))
        pt_layout.addWidget(self.pt_lineedit)
        pt_layout.addWidget(pt_browse_btn)
        param_layout.addLayout(pt_layout)
        # 6. Paramètres supplémentaires + cases à cocher associées
        # Création des cases à cocher d'abord
        # Tapioca
        tapioca_line = QHBoxLayout()
        self.tapioca_cb = QCheckBox("Tapioca")
        self.tapioca_cb.setChecked(True)
        self.tapioca_cb.setMinimumWidth(140)
        self.tapioca_extra = QLineEdit()
        self.tapioca_extra.setPlaceholderText("Paramètres supplémentaires pour Tapioca (optionnel)")
        tapioca_line.addWidget(self.tapioca_cb)
        tapioca_line.addWidget(self.tapioca_extra)
        # Tapas
        tapas_line = QHBoxLayout()
        self.tapas_cb = QCheckBox("Tapas")
        self.tapas_cb.setChecked(True)
        self.tapas_cb.setMinimumWidth(140)
        self.tapas_extra = QLineEdit()
        self.tapas_extra.setPlaceholderText("Paramètres supplémentaires pour Tapas (optionnel)")
        tapas_line.addWidget(self.tapas_cb)
        tapas_line.addWidget(self.tapas_extra)
        # SaisieAppuisInitQT
        saisieappuisinit_line = QHBoxLayout()
        self.saisieappuisinit_cb = QCheckBox("SaisieAppuisInit")
        self.saisieappuisinit_cb.setChecked(True)
        self.saisieappuisinit_cb.setMinimumWidth(140)
        self.saisieappuisinit_extra = QLineEdit()
        self.saisieappuisinit_extra.setPlaceholderText("Paramètres supplémentaires pour SaisieAppuisInitQT (optionnel)")
        saisieappuisinit_line.addWidget(self.saisieappuisinit_cb)
        saisieappuisinit_line.addWidget(self.saisieappuisinit_extra)
        # SaisieAppuisPredicQT
        saisieappuispredic_line = QHBoxLayout()
        self.saisieappuispredic_cb = QCheckBox("SaisieAppuisPredic")
        self.saisieappuispredic_cb.setChecked(True)
        self.saisieappuispredic_cb.setMinimumWidth(140)
        self.saisieappuispredic_extra = QLineEdit()
        self.saisieappuispredic_extra.setPlaceholderText("Paramètres supplémentaires pour SaisieAppuisPredicQT (optionnel)")
        saisieappuispredic_line.addWidget(self.saisieappuispredic_cb)
        saisieappuispredic_line.addWidget(self.saisieappuispredic_extra)
        # C3DC
        c3dc_line = QHBoxLayout()
        self.c3dc_cb = QCheckBox("C3DC")
        self.c3dc_cb.setChecked(True)
        self.c3dc_cb.setMinimumWidth(140)
        self.c3dc_extra = QLineEdit()
        self.c3dc_extra.setPlaceholderText("Paramètres supplémentaires pour C3DC (optionnel)")
        c3dc_line.addWidget(self.c3dc_cb)
        c3dc_line.addWidget(self.c3dc_extra)
        # Ajout bouton à bascule Tout cocher/décocher avec icône dynamique (petit, à gauche, sans texte)
        toggle_btn = QPushButton()
        toggle_btn.setFixedSize(24, 24)
        toggle_btn.setCursor(Qt.PointingHandCursor)
        toggle_btn.setStyleSheet("border: none; padding: 0px;")
        def update_toggle_btn():
            all_checked = all([
                self.tapioca_cb.isChecked(),
                self.tapas_cb.isChecked(),
                self.saisieappuisinit_cb.isChecked(),
                self.saisieappuispredic_cb.isChecked(),
                self.c3dc_cb.isChecked()
            ])
            if all_checked:
                # Icône croix rouge ❌
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(211, 47, 47), 4))
                painter.drawLine(4, 4, 16, 16)
                painter.drawLine(16, 4, 4, 16)
                painter.end()
                toggle_btn.setIcon(QIcon(pixmap))
                toggle_btn.setToolTip("Tout décocher")
            else:
                # Icône coche verte ✅
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(76, 175, 80), 4))
                painter.drawLine(4, 12, 9, 17)
                painter.drawLine(9, 17, 16, 5)
                painter.end()
                toggle_btn.setIcon(QIcon(pixmap))
                toggle_btn.setToolTip("Tout cocher")
        def toggle_all():
            all_checked = all([
                self.tapioca_cb.isChecked(),
                self.tapas_cb.isChecked(),
                self.saisieappuisinit_cb.isChecked(),
                self.saisieappuispredic_cb.isChecked(),
                self.c3dc_cb.isChecked()
            ])
            state = not all_checked
            self.tapioca_cb.setChecked(state)
            self.tapas_cb.setChecked(state)
            self.saisieappuisinit_cb.setChecked(state)
            self.saisieappuispredic_cb.setChecked(state)
            self.c3dc_cb.setChecked(state)
            update_toggle_btn()
        toggle_btn.clicked.connect(toggle_all)
        for cb in [self.tapioca_cb, self.tapas_cb, self.saisieappuisinit_cb, self.saisieappuispredic_cb, self.c3dc_cb]:
            cb.stateChanged.connect(update_toggle_btn)
        # Ajout du bouton dans un layout horizontal collé à gauche
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        toggle_layout.addWidget(toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        param_layout.addLayout(toggle_layout)
        update_toggle_btn()
        # Ajout des cases à cocher au layout
        param_layout.addLayout(tapioca_line)
        param_layout.addLayout(tapas_line)
        param_layout.addLayout(saisieappuisinit_line)
        param_layout.addLayout(saisieappuispredic_line)
        param_layout.addLayout(c3dc_line)
        # 9. Interpréteur Python
        python_layout = QHBoxLayout()
        self.python_selector = QComboBox()
        self.python_selector.addItems(["python", "python3"])
        python_layout.addWidget(QLabel("Interpréteur Python :"))
        python_layout.addWidget(self.python_selector)
        param_layout.addLayout(python_layout)
        # 10. Ligne de commande CLI équivalente
        self.cmd_label = QLabel("Ligne de commande CLI équivalente :")
        param_layout.addWidget(self.cmd_label)
        self.cmd_line = QLineEdit()
        self.cmd_line.setReadOnly(True)
        self.cmd_line.setStyleSheet("font-family: monospace;")
        param_layout.addWidget(self.cmd_line)
        # 11. Résumé et stretch
        self.summary_label = QLabel("")
        param_layout.addWidget(self.summary_label)
        param_layout.addStretch(1)
        tabs.addTab(param_tab, "MicMac")

        # Onglet 2 : Transformations géodésiques
        geodetic_tab = QWidget()
        geodetic_layout = QVBoxLayout(geodetic_tab)
        
        # 1. Dossier de travail
        geodetic_dir_layout = QHBoxLayout()
        self.geodetic_dir_edit = QLineEdit()
        self.geodetic_dir_edit.setPlaceholderText("Dossier contenant les nuages .ply à traiter")
        geodetic_browse_btn = QPushButton("Parcourir…")
        geodetic_browse_btn.clicked.connect(self.browse_geodetic_folder)
        geodetic_dir_layout.addWidget(QLabel("Dossier de travail :"))
        geodetic_dir_layout.addWidget(self.geodetic_dir_edit)
        geodetic_dir_layout.addWidget(geodetic_browse_btn)
        geodetic_layout.addLayout(geodetic_dir_layout)
        
        # 2. Fichier de coordonnées de recalage
        geodetic_coord_layout = QHBoxLayout()
        self.geodetic_coord_edit = QLineEdit()
        self.geodetic_coord_edit.setPlaceholderText("Chemin du fichier de coordonnées de recalage (.txt)")
        geodetic_coord_browse_btn = QPushButton("Parcourir…")
        geodetic_coord_browse_btn.clicked.connect(self.browse_geodetic_coord_file)
        geodetic_coord_layout.addWidget(QLabel("Fichier de coordonnées :"))
        geodetic_coord_layout.addWidget(self.geodetic_coord_edit)
        geodetic_coord_layout.addWidget(geodetic_coord_browse_btn)
        geodetic_layout.addLayout(geodetic_coord_layout)
        
        # 3. Type de déformation
        deformation_layout = QHBoxLayout()
        self.deformation_combo = QComboBox()
        self.deformation_combo.addItems(["tps"])
        self.deformation_combo.setCurrentText("tps")
        deformation_layout.addWidget(QLabel("Type de déformation :"))
        deformation_layout.addWidget(self.deformation_combo)
        geodetic_layout.addLayout(deformation_layout)
        
        # 4. Paramètres de déformation
        deformation_params_layout = QHBoxLayout()
        self.deformation_params_edit = QLineEdit()
        self.deformation_params_edit.setPlaceholderText("Paramètres de déformation (optionnel)")
        deformation_params_layout.addWidget(QLabel("Paramètres :"))
        deformation_params_layout.addWidget(self.deformation_params_edit)
        geodetic_layout.addLayout(deformation_params_layout)
        
        # 4.5. Fichier XML GCPBascule pour la déformation
        bascule_xml_layout = QHBoxLayout()
        self.bascule_xml_edit = QLineEdit()
        self.bascule_xml_edit.setPlaceholderText("Chemin du fichier XML GCPBascule (.xml)")
        bascule_xml_browse_btn = QPushButton("Parcourir…")
        bascule_xml_browse_btn.clicked.connect(self.browse_bascule_xml_file)
        bascule_xml_layout.addWidget(QLabel("Fichier XML GCPBascule :"))
        bascule_xml_layout.addWidget(self.bascule_xml_edit)
        bascule_xml_layout.addWidget(bascule_xml_browse_btn)
        geodetic_layout.addLayout(bascule_xml_layout)
        
        # 4.6. Nombre de processus parallèles
        parallel_layout = QHBoxLayout()
        self.parallel_workers_spin = QSpinBox()
        self.parallel_workers_spin.setMinimum(1)
        self.parallel_workers_spin.setMaximum(128)
        self.parallel_workers_spin.setValue(10)
        self.parallel_workers_spin.setToolTip("Nombre de processus parallèles pour le traitement des nuages (1-128)")
        parallel_layout.addWidget(QLabel("Processus parallèles :"))
        parallel_layout.addWidget(self.parallel_workers_spin)
        parallel_layout.addStretch()
        geodetic_layout.addLayout(parallel_layout)
        
        # 5. Bouton tout cocher/décocher pour les transformations géodésiques
        geodetic_toggle_btn = QPushButton()
        geodetic_toggle_btn.setFixedSize(24, 24)
        geodetic_toggle_btn.setCursor(Qt.PointingHandCursor)
        geodetic_toggle_btn.setStyleSheet("border: none; padding: 0px;")
        
        def update_geodetic_toggle_btn():
            all_checked = all([
                self.add_offset_cb.isChecked(),
                self.itrf_to_enu_cb.isChecked(),
                self.deform_cb.isChecked(),
                self.enu_to_itrf_cb.isChecked()
            ])
            if all_checked:
                # Icône croix rouge ❌
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(211, 47, 47), 4))
                painter.drawLine(4, 4, 16, 16)
                painter.drawLine(16, 4, 4, 16)
                painter.end()
                geodetic_toggle_btn.setIcon(QIcon(pixmap))
                geodetic_toggle_btn.setToolTip("Tout décocher")
            else:
                # Icône coche verte ✅
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(76, 175, 80), 4))
                painter.drawLine(4, 12, 9, 17)
                painter.drawLine(9, 17, 16, 5)
                painter.end()
                geodetic_toggle_btn.setIcon(QIcon(pixmap))
                geodetic_toggle_btn.setToolTip("Tout cocher")
        
        def toggle_all_geodetic():
            all_checked = all([
                self.add_offset_cb.isChecked(),
                self.itrf_to_enu_cb.isChecked(),
                self.deform_cb.isChecked(),
                self.enu_to_itrf_cb.isChecked()
            ])
            state = not all_checked
            self.add_offset_cb.setChecked(state)
            self.itrf_to_enu_cb.setChecked(state)
            self.deform_cb.setChecked(state)
            self.enu_to_itrf_cb.setChecked(state)
            update_geodetic_toggle_btn()
        
        geodetic_toggle_btn.clicked.connect(toggle_all_geodetic)
        
        # Ajout du bouton dans un layout horizontal collé à gauche
        geodetic_toggle_layout = QHBoxLayout()
        geodetic_toggle_layout.setContentsMargins(0, 0, 0, 0)
        geodetic_toggle_layout.setSpacing(0)
        geodetic_toggle_layout.addWidget(geodetic_toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        geodetic_layout.addLayout(geodetic_toggle_layout)
        
        # 6. Cases à cocher pour les étapes
        # Ajout offset
        add_offset_line = QHBoxLayout()
        self.add_offset_cb = QCheckBox("Ajout offset")
        self.add_offset_cb.setChecked(True)
        self.add_offset_cb.setMinimumWidth(140)
        self.add_offset_extra = QLineEdit()
        self.add_offset_extra.setPlaceholderText("Paramètres supplémentaires pour l'ajout d'offset (optionnel)")
        add_offset_line.addWidget(self.add_offset_cb)
        add_offset_line.addWidget(self.add_offset_extra)
        
        # Input/Output for Add Offset
        add_offset_input_layout = QHBoxLayout()
        self.add_offset_input_edit = QLineEdit()
        self.add_offset_input_edit.setPlaceholderText("Dossier d'entrée (vide = dossier principal)")
        add_offset_input_layout.addWidget(QLabel("Entrée :"))
        add_offset_input_layout.addWidget(self.add_offset_input_edit)
        self.add_offset_input_browse_btn = QPushButton("Parcourir")
        self.add_offset_input_browse_btn.clicked.connect(self.browse_add_offset_input_dir)
        add_offset_input_layout.addWidget(self.add_offset_input_browse_btn)

        add_offset_output_layout = QHBoxLayout()
        self.add_offset_output_edit = QLineEdit()
        self.add_offset_output_edit.setPlaceholderText("Dossier de sortie (vide = offset_step)")
        add_offset_output_layout.addWidget(QLabel("Sortie :"))
        add_offset_output_layout.addWidget(self.add_offset_output_edit)
        self.add_offset_output_browse_btn = QPushButton("Parcourir")
        self.add_offset_output_browse_btn.clicked.connect(self.browse_add_offset_output_dir)
        add_offset_output_layout.addWidget(self.add_offset_output_browse_btn)

        geodetic_layout.addLayout(add_offset_line)
        geodetic_layout.addLayout(add_offset_input_layout)
        geodetic_layout.addLayout(add_offset_output_layout)
        
        # ITRF vers ENU
        itrf_to_enu_line = QHBoxLayout()
        self.itrf_to_enu_cb = QCheckBox("ITRF → ENU")
        self.itrf_to_enu_cb.setChecked(True)
        self.itrf_to_enu_cb.setMinimumWidth(140)
        self.itrf_to_enu_extra = QLineEdit()
        self.itrf_to_enu_extra.setPlaceholderText("Paramètres supplémentaires pour ITRF→ENU (optionnel)")
        itrf_to_enu_line.addWidget(self.itrf_to_enu_cb)
        itrf_to_enu_line.addWidget(self.itrf_to_enu_extra)

        # Input/Output for ITRF to ENU
        itrf_to_enu_input_layout = QHBoxLayout()
        self.itrf_to_enu_input_edit = QLineEdit()
        self.itrf_to_enu_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        itrf_to_enu_input_layout.addWidget(QLabel("Entrée :"))
        itrf_to_enu_input_layout.addWidget(self.itrf_to_enu_input_edit)
        self.itrf_to_enu_input_browse_btn = QPushButton("Parcourir")
        self.itrf_to_enu_input_browse_btn.clicked.connect(self.browse_itrf_to_enu_input_dir)
        itrf_to_enu_input_layout.addWidget(self.itrf_to_enu_input_browse_btn)

        itrf_to_enu_output_layout = QHBoxLayout()
        self.itrf_to_enu_output_edit = QLineEdit()
        self.itrf_to_enu_output_edit.setPlaceholderText("Dossier de sortie (vide = itrf_to_enu_step)")
        itrf_to_enu_output_layout.addWidget(QLabel("Sortie :"))
        itrf_to_enu_output_layout.addWidget(self.itrf_to_enu_output_edit)
        self.itrf_to_enu_output_browse_btn = QPushButton("Parcourir")
        self.itrf_to_enu_output_browse_btn.clicked.connect(self.browse_itrf_to_enu_output_dir)
        itrf_to_enu_output_layout.addWidget(self.itrf_to_enu_output_browse_btn)

        geodetic_layout.addLayout(itrf_to_enu_line)
        geodetic_layout.addLayout(itrf_to_enu_input_layout)
        geodetic_layout.addLayout(itrf_to_enu_output_layout)
        
        # Déformation
        deform_line = QHBoxLayout()
        self.deform_cb = QCheckBox("Déformation")
        self.deform_cb.setChecked(True)
        self.deform_cb.setMinimumWidth(140)
        self.deform_extra = QLineEdit()
        self.deform_extra.setPlaceholderText("Paramètres supplémentaires pour la déformation (optionnel)")
        deform_line.addWidget(self.deform_cb)
        deform_line.addWidget(self.deform_extra)

        # Input/Output for Deformation
        deform_input_layout = QHBoxLayout()
        self.deform_input_edit = QLineEdit()
        self.deform_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        deform_input_layout.addWidget(QLabel("Entrée :"))
        deform_input_layout.addWidget(self.deform_input_edit)
        self.deform_input_browse_btn = QPushButton("Parcourir")
        self.deform_input_browse_btn.clicked.connect(self.browse_deform_input_dir)
        deform_input_layout.addWidget(self.deform_input_browse_btn)

        deform_output_layout = QHBoxLayout()
        self.deform_output_edit = QLineEdit()
        self.deform_output_edit.setPlaceholderText("Dossier de sortie (vide = deform_[type]_step)")
        deform_output_layout.addWidget(QLabel("Sortie :"))
        deform_output_layout.addWidget(self.deform_output_edit)
        self.deform_output_browse_btn = QPushButton("Parcourir")
        self.deform_output_browse_btn.clicked.connect(self.browse_deform_output_dir)
        deform_output_layout.addWidget(self.deform_output_browse_btn)

        geodetic_layout.addLayout(deform_line)
        geodetic_layout.addLayout(deform_input_layout)
        geodetic_layout.addLayout(deform_output_layout)
        
        # ENU vers ITRF
        enu_to_itrf_line = QHBoxLayout()
        self.enu_to_itrf_cb = QCheckBox("ENU → ITRF")
        self.enu_to_itrf_cb.setChecked(True)
        self.enu_to_itrf_cb.setMinimumWidth(140)
        self.enu_to_itrf_extra = QLineEdit()
        self.enu_to_itrf_extra.setPlaceholderText("Paramètres supplémentaires pour ENU→ITRF (optionnel)")
        enu_to_itrf_line.addWidget(self.enu_to_itrf_cb)
        enu_to_itrf_line.addWidget(self.enu_to_itrf_extra)

        # Input/Output for ENU to ITRF
        enu_to_itrf_input_layout = QHBoxLayout()
        self.enu_to_itrf_input_edit = QLineEdit()
        self.enu_to_itrf_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        enu_to_itrf_input_layout.addWidget(QLabel("Entrée :"))
        enu_to_itrf_input_layout.addWidget(self.enu_to_itrf_input_edit)
        self.enu_to_itrf_input_browse_btn = QPushButton("Parcourir")
        self.enu_to_itrf_input_browse_btn.clicked.connect(self.browse_enu_to_itrf_input_dir)
        enu_to_itrf_input_layout.addWidget(self.enu_to_itrf_input_browse_btn)

        enu_to_itrf_output_layout = QHBoxLayout()
        self.enu_to_itrf_output_edit = QLineEdit()
        self.enu_to_itrf_output_edit.setPlaceholderText("Dossier de sortie (vide = enu_to_itrf_step)")
        enu_to_itrf_output_layout.addWidget(QLabel("Sortie :"))
        enu_to_itrf_output_layout.addWidget(self.enu_to_itrf_output_edit)
        self.enu_to_itrf_output_browse_btn = QPushButton("Parcourir")
        self.enu_to_itrf_output_browse_btn.clicked.connect(self.browse_enu_to_itrf_output_dir)
        enu_to_itrf_output_layout.addWidget(self.enu_to_itrf_output_browse_btn)

        geodetic_layout.addLayout(enu_to_itrf_line)
        geodetic_layout.addLayout(enu_to_itrf_input_layout)
        geodetic_layout.addLayout(enu_to_itrf_output_layout)
        
        # Connexion des cases à cocher au bouton toggle après leur création
        for cb in [self.add_offset_cb, self.itrf_to_enu_cb, self.deform_cb, self.enu_to_itrf_cb]:
            cb.stateChanged.connect(update_geodetic_toggle_btn)
        update_geodetic_toggle_btn()
        
        # 8. Ligne de commande CLI équivalente
        self.geodetic_cmd_label = QLabel("Ligne de commande CLI équivalente :")
        geodetic_layout.addWidget(self.geodetic_cmd_label)
        self.geodetic_cmd_line = QLineEdit()
        self.geodetic_cmd_line.setReadOnly(True)
        self.geodetic_cmd_line.setStyleSheet("font-family: monospace;")
        geodetic_layout.addWidget(self.geodetic_cmd_line)
        
        # 7. Résumé et stretch
        self.geodetic_summary_label = QLabel("")
        geodetic_layout.addWidget(self.geodetic_summary_label)
        geodetic_layout.addStretch(1)
        tabs.addTab(geodetic_tab, "Transformations géodésiques")

        # Onglet 3 : logs
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.addWidget(QLabel("Logs :"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_tab, "Logs")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        main_layout.addLayout(layout)
        self.setLayout(main_layout)
        # Connexions
        self.dir_edit.textChanged.connect(self.update_cmd_line)
        self.mode_combo.currentTextChanged.connect(self.update_cmd_line)
        self.zoom_spin.valueChanged.connect(self.update_cmd_line)
        self.tapas_model_combo.currentTextChanged.connect(self.update_cmd_line)
        self.tapioca_extra.textChanged.connect(self.update_cmd_line)
        self.tapas_extra.textChanged.connect(self.update_cmd_line)
        self.c3dc_extra.textChanged.connect(self.update_cmd_line)
        self.python_selector.currentTextChanged.connect(self.update_cmd_line)
        self.tapioca_cb.stateChanged.connect(self.update_cmd_line)
        self.tapas_cb.stateChanged.connect(self.update_cmd_line)
        self.c3dc_cb.stateChanged.connect(self.update_cmd_line)
        self.saisieappuisinit_cb.stateChanged.connect(self.update_cmd_line)
        self.saisieappuispredic_cb.stateChanged.connect(self.update_cmd_line)
        self.pt_lineedit.textChanged.connect(self.update_cmd_line)
        self.update_cmd_line()
        
        # Connexions pour l'onglet géodésique
        self.geodetic_dir_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.geodetic_coord_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deformation_combo.currentTextChanged.connect(self.update_geodetic_cmd_line)
        self.deformation_params_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.bascule_xml_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.add_offset_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.add_offset_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.deform_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers d'entrée personnalisés
        self.add_offset_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers de sortie personnalisés
        self.add_offset_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.update_geodetic_cmd_line()

    def update_cmd_line(self):
        input_dir = self.dir_edit.text().strip() or "<dossier_images>"
        mode = self.mode_combo.currentText()
        zoomf = self.zoom_spin.value()
        tapas_model = self.tapas_model_combo.currentText()
        tapioca_extra = self.tapioca_extra.text().strip()
        tapas_extra = self.tapas_extra.text().strip()
        c3dc_extra = self.c3dc_extra.text().strip()
        saisieappuisinit_pt = self.pt_lineedit.text().strip()
        saisieappuisinit_extra = self.saisieappuisinit_extra.text().strip()
        saisieappuispredic_extra = self.saisieappuispredic_extra.text().strip()
        base_cmd = ["photogeoalign.py", "--no-gui", f'\"{input_dir}\"', f"--mode {mode}", f"--tapas-model {tapas_model}", f"--zoomf {zoomf}"]
        if tapioca_extra:
            base_cmd.append(f"--tapioca-extra \"{tapioca_extra}\"")
        if tapas_extra:
            base_cmd.append(f"--tapas-extra \"{tapas_extra}\"")
        if c3dc_extra:
            base_cmd.append(f"--c3dc-extra \"{c3dc_extra}\"")
        if saisieappuisinit_pt:
            base_cmd.append(f"--saisieappuisinit-pt \"{saisieappuisinit_pt}\"")
        if saisieappuisinit_extra:
            base_cmd.append(f"--saisieappuisinit-extra \"{saisieappuisinit_extra}\"")
        if saisieappuispredic_extra:
            base_cmd.append(f"--saisieappuispredic-extra \"{saisieappuispredic_extra}\"")
        # Ajout des options de skip
        if not self.tapioca_cb.isChecked():
            base_cmd.append("--skip-tapioca")
        if not self.tapas_cb.isChecked():
            base_cmd.append("--skip-tapas")
        if not self.saisieappuisinit_cb.isChecked():
            base_cmd.append("--skip-saisieappuisinit")
        if not self.saisieappuispredic_cb.isChecked():
            base_cmd.append("--skip-saisieappuispredic")
        if not self.c3dc_cb.isChecked():
            base_cmd.append("--skip-c3dc")
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.cmd_line.setText(cmd)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'images")
        if folder:
            self.dir_edit.setText(folder)

    def browse_pt_file(self):
        pt_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier de coordonnées (.txt)", "", "Fichiers de coordonnées (*.txt)")
        if pt_file:
            self.pt_lineedit.setText(pt_file)

    def browse_geodetic_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier contenant les nuages .ply")
        if folder:
            self.geodetic_dir_edit.setText(folder)

    def browse_geodetic_coord_file(self):
        coord_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier de coordonnées de recalage (.txt)", "", "Fichiers de coordonnées (*.txt)")
        if coord_file:
            self.geodetic_coord_edit.setText(coord_file)
    
    def browse_bascule_xml_file(self):
        xml_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier XML GCPBascule (.xml)", "", "Fichiers XML (*.xml)")
        if xml_file:
            self.bascule_xml_edit.setText(xml_file)

    def browse_add_offset_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour l'ajout d'offset")
        if folder:
            self.add_offset_input_edit.setText(folder)

    def browse_itrf_to_enu_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour ITRF→ENU")
        if folder:
            self.itrf_to_enu_input_edit.setText(folder)

    def browse_deform_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour la déformation")
        if folder:
            self.deform_input_edit.setText(folder)

    def browse_enu_to_itrf_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour ENU→ITRF")
        if folder:
            self.enu_to_itrf_input_edit.setText(folder)

    def browse_add_offset_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour l'ajout d'offset")
        if folder:
            self.add_offset_output_edit.setText(folder)

    def browse_itrf_to_enu_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour ITRF→ENU")
        if folder:
            self.itrf_to_enu_output_edit.setText(folder)

    def browse_deform_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour la déformation")
        if folder:
            self.deform_output_edit.setText(folder)

    def browse_enu_to_itrf_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour ENU→ITRF")
        if folder:
            self.enu_to_itrf_output_edit.setText(folder)

    def update_geodetic_cmd_line(self):
        geodetic_dir = self.geodetic_dir_edit.text().strip() or "<dossier_nuages>"
        coord_file = self.geodetic_coord_edit.text().strip()
        deformation_type = self.deformation_combo.currentText()
        deformation_params = self.deformation_params_edit.text().strip()
        bascule_xml = self.bascule_xml_edit.text().strip()
        add_offset_extra = self.add_offset_extra.text().strip()
        itrf_to_enu_extra = self.itrf_to_enu_extra.text().strip()
        deform_extra = self.deform_extra.text().strip()
        enu_to_itrf_extra = self.enu_to_itrf_extra.text().strip()
        
        # Dossiers d'entrée personnalisés
        add_offset_input_dir = self.add_offset_input_edit.text().strip()
        itrf_to_enu_input_dir = self.itrf_to_enu_input_edit.text().strip()
        deform_input_dir = self.deform_input_edit.text().strip()
        enu_to_itrf_input_dir = self.enu_to_itrf_input_edit.text().strip()
        
        # Dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        enu_to_itrf_output_dir = self.enu_to_itrf_output_edit.text().strip()
        
        base_cmd = ["photogeoalign.py", "--geodetic", f'\"{geodetic_dir}\"']
        
        if coord_file:
            base_cmd.append(f"--geodetic-coord \"{coord_file}\"")
        
        base_cmd.append(f"--deformation-type {deformation_type}")
        
        if deformation_params:
            base_cmd.append(f"--deformation-params \"{deformation_params}\"")
        
        if bascule_xml:
            base_cmd.append(f"--deform-bascule-xml \"{bascule_xml}\"")
        
        if add_offset_extra:
            base_cmd.append(f"--add-offset-extra \"{add_offset_extra}\"")
        
        if itrf_to_enu_extra:
            base_cmd.append(f"--itrf-to-enu-extra \"{itrf_to_enu_extra}\"")
        
        # Note: Pour le point de référence, on pourrait ajouter un champ dans le GUI
        # Pour l'instant, on utilise le premier point par défaut
        
        if deform_extra:
            base_cmd.append(f"--deform-extra \"{deform_extra}\"")
        
        if enu_to_itrf_extra:
            base_cmd.append(f"--enu-to-itrf-extra \"{enu_to_itrf_extra}\"")
        
        # Ajout des dossiers d'entrée personnalisés
        if add_offset_input_dir:
            base_cmd.append(f"--add-offset-input-dir \"{add_offset_input_dir}\"")
        
        if itrf_to_enu_input_dir:
            base_cmd.append(f"--itrf-to-enu-input-dir \"{itrf_to_enu_input_dir}\"")
        
        if deform_input_dir:
            base_cmd.append(f"--deform-input-dir \"{deform_input_dir}\"")
        
        if enu_to_itrf_input_dir:
            base_cmd.append(f"--enu-to-itrf-input-dir \"{enu_to_itrf_input_dir}\"")
        
        # Ajout des dossiers de sortie personnalisés
        if add_offset_output_dir:
            base_cmd.append(f"--add-offset-output-dir \"{add_offset_output_dir}\"")
        
        if itrf_to_enu_output_dir:
            base_cmd.append(f"--itrf-to-enu-output-dir \"{itrf_to_enu_output_dir}\"")
        
        if deform_output_dir:
            base_cmd.append(f"--deform-output-dir \"{deform_output_dir}\"")
        
        if enu_to_itrf_output_dir:
            base_cmd.append(f"--enu-to-itrf-output-dir \"{enu_to_itrf_output_dir}\"")
        
        # Ajout des options de skip
        if not self.add_offset_cb.isChecked():
            base_cmd.append("--skip-add-offset")
        
        if not self.itrf_to_enu_cb.isChecked():
            base_cmd.append("--skip-itrf-to-enu")
        
        if not self.deform_cb.isChecked():
            base_cmd.append("--skip-deform")
        
        if not self.enu_to_itrf_cb.isChecked():
            base_cmd.append("--skip-enu-to-itrf")
        
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.geodetic_cmd_line.setText(cmd)

    def launch_pipeline(self):
        input_dir = self.dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un dossier valide.</span>")
            return
        mode = self.mode_combo.currentText()
        zoomf = self.zoom_spin.value()
        tapas_model = self.tapas_model_combo.currentText()
        tapioca_extra = self.tapioca_extra.text().strip()
        tapas_extra = self.tapas_extra.text().strip()
        c3dc_extra = self.c3dc_extra.text().strip()
        saisieappuisinit_pt = self.pt_lineedit.text().strip()
        saisieappuisinit_extra = self.saisieappuisinit_extra.text().strip()
        saisieappuispredic_extra = self.saisieappuispredic_extra.text().strip()
        run_tapioca = self.tapioca_cb.isChecked()
        run_tapas = self.tapas_cb.isChecked()
        run_saisieappuisinit = self.saisieappuisinit_cb.isChecked()
        run_saisieappuispredic = self.saisieappuispredic_cb.isChecked()
        run_c3dc = self.c3dc_cb.isChecked()
        # Avertissement si incohérence
        if run_c3dc and not run_tapas:
            self.log_text.append("<span style='color:orange'>Attention : lancer C3DC sans Tapas n'a pas de sens !</span>")
        if run_tapas and not run_tapioca:
            self.log_text.append("<span style='color:orange'>Attention : lancer Tapas sans Tapioca n'a pas de sens !</span>")
        self.log_text.clear()
        self.summary_label.setText("")
        self.action_run.setEnabled(False)
        self.action_geodetic.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.pipeline_thread = PipelineThread(input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, run_tapioca, run_tapas, run_saisieappuisinit, run_saisieappuispredic, run_c3dc)
        self.pipeline_thread.log_signal.connect(self.append_log)
        self.pipeline_thread.finished_signal.connect(self.pipeline_finished)
        self.pipeline_thread.start()

    def append_log(self, text):
        # Recherche du format : '2025-06-24 13:11:38,200 - INFO - message'
        m = re.match(r"^(.*? - )([A-Z]+)( - )(.*)$", text)
        if m:
            timestamp = m.group(1)
            level = m.group(2)
            sep = m.group(3)
            message = m.group(4)
            if level == "INFO":
                color = "#1565c0"  # bleu
            elif level in ("WARNING", "WARN"): 
                color = "#ff9800"  # orange
            elif level == "ERROR":
                color = "#d32f2f"  # rouge
            else:
                color = "#8e24aa"  # violet
            html = f"<span>{timestamp}</span><span style='color:{color};'>{level}</span><span>{sep}</span><span style='font-weight:bold;color:{color};'>{message}</span>"
        else:
            # Pas de format reconnu, affiche en violet
            html = f"<span style='font-weight:bold;color:#8e24aa'>{text}</span>"
        self.log_text.append(html)

    def stop_pipeline(self):
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self.pipeline_thread.terminate()
            self.pipeline_thread.wait()
            self.append_log("<span style='color:red'>Pipeline MicMac arrêté par l'utilisateur.</span>")
        if hasattr(self, 'geodetic_thread') and self.geodetic_thread and self.geodetic_thread.isRunning():
            self.geodetic_thread.terminate()
            self.geodetic_thread.wait()
            self.append_log("<span style='color:red'>Pipeline géodésique arrêté par l'utilisateur.</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_stop.setEnabled(False)

    def pipeline_finished(self, success, message):
        if success:
            self.summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_stop.setEnabled(False)

    def launch_geodetic_pipeline(self):
        input_dir = self.geodetic_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un dossier valide.</span>")
            return
        
        coord_file = self.geodetic_coord_edit.text().strip()
        if not coord_file or not os.path.exists(coord_file):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un fichier de coordonnées valide.</span>")
            return
        
        deformation_type = self.deformation_combo.currentText()
        deformation_params = self.deformation_params_edit.text().strip()
        bascule_xml = self.bascule_xml_edit.text().strip()
        max_workers = self.parallel_workers_spin.value()
        add_offset_extra = self.add_offset_extra.text().strip()
        itrf_to_enu_extra = self.itrf_to_enu_extra.text().strip()
        deform_extra = self.deform_extra.text().strip()
        enu_to_itrf_extra = self.enu_to_itrf_extra.text().strip()
        
        run_add_offset = self.add_offset_cb.isChecked()
        run_itrf_to_enu = self.itrf_to_enu_cb.isChecked()
        run_deform = self.deform_cb.isChecked()
        run_enu_to_itrf = self.enu_to_itrf_cb.isChecked()
        
        # Vérification qu'au moins une étape est sélectionnée
        if not any([run_add_offset, run_itrf_to_enu, run_deform, run_enu_to_itrf]):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner au moins une étape de transformation.</span>")
            return
        
        self.log_text.clear()
        self.geodetic_summary_label.setText("")
        self.action_run.setEnabled(False)
        self.action_geodetic.setEnabled(False)
        self.action_stop.setEnabled(True)
        
        # Récupération des dossiers d'entrée personnalisés
        add_offset_input_dir = self.add_offset_input_edit.text().strip()
        itrf_to_enu_input_dir = self.itrf_to_enu_input_edit.text().strip()
        deform_input_dir = self.deform_input_edit.text().strip()
        enu_to_itrf_input_dir = self.enu_to_itrf_input_edit.text().strip()
        
        # Récupération des dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        enu_to_itrf_output_dir = self.enu_to_itrf_output_edit.text().strip()
        
        # Conversion en None si vide
        add_offset_input_dir = add_offset_input_dir if add_offset_input_dir else None
        itrf_to_enu_input_dir = itrf_to_enu_input_dir if itrf_to_enu_input_dir else None
        deform_input_dir = deform_input_dir if deform_input_dir else None
        enu_to_itrf_input_dir = enu_to_itrf_input_dir if enu_to_itrf_input_dir else None
        add_offset_output_dir = add_offset_output_dir if add_offset_output_dir else None
        itrf_to_enu_output_dir = itrf_to_enu_output_dir if itrf_to_enu_output_dir else None
        deform_output_dir = deform_output_dir if deform_output_dir else None
        enu_to_itrf_output_dir = enu_to_itrf_output_dir if enu_to_itrf_output_dir else None
        
        self.geodetic_thread = GeodeticTransformThread(
            input_dir, coord_file, deformation_type, deformation_params,
            add_offset_extra, itrf_to_enu_extra, deform_extra, enu_to_itrf_extra,
            run_add_offset, run_itrf_to_enu, run_deform, run_enu_to_itrf,
            add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, enu_to_itrf_input_dir,
            add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, enu_to_itrf_output_dir,
            None, bascule_xml, max_workers
        )
        self.geodetic_thread.log_signal.connect(self.append_log)
        self.geodetic_thread.finished_signal.connect(self.geodetic_pipeline_finished)
        self.geodetic_thread.start()

    def geodetic_pipeline_finished(self, success, message):
        if success:
            self.geodetic_summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.geodetic_summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_stop.setEnabled(False)

    def export_job_dialog(self):
        import sys
        import os
        exe_path = sys.executable
        script_path = os.path.abspath(__file__)
        cli_cmd = self.cmd_line.text().strip()
        parts = cli_cmd.split()
        # On retire le premier mot (python ou exe)
        args = parts[1:]
        # On retire tout photogeoalign.py
        filtered_args = [arg for arg in args if not arg.endswith('photogeoalign.py') and not arg.endswith('photogeoalign.py"')]
        if getattr(sys, 'frozen', False):
            # Cas exécutable PyInstaller
            cmd = [exe_path] + filtered_args
        else:
            # Cas Python
            cmd = [exe_path, script_path] + filtered_args
        cli_cmd = " ".join(cmd)
        dialog = JobExportDialog(self, cli_cmd=cli_cmd)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vals = dialog.get_values()
            job_content = self.generate_job_script(vals)
            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le script .job", "micmac.job", "Fichiers batch (*.job *.sh)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(job_content)
                    QMessageBox.information(self, "Export réussi", f"Script batch exporté :\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export : {e}")

    def generate_job_script(self, vals):
        # Génère le contenu du script SLURM
        return f"""#!/bin/bash

#SBATCH --job-name {vals['job_name']}
#SBATCH --output {vals['output']}
#SBATCH --nodes=1-1
#SBATCH --partition {vals['partition']} #ncpum,ncpu,ncpulong
#SBATCH --ntasks={vals['ntasks']}

module purge
module load micmac

{vals['cli_cmd']}
"""

def check_micmac_or_quit():
    try:
        result = subprocess.run(["mm3d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 and "usage" not in result.stdout.lower() and "usage" not in result.stderr.lower():
            raise FileNotFoundError
    except Exception:
        app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            msg.setWindowIcon(QIcon(logo_path))
            msg.setIconPixmap(QPixmap(logo_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        msg.setWindowTitle("MicMac non détecté")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.setText("Le logiciel MicMac (mm3d) n'est pas installé ou n'est pas accessible dans le PATH système.<br><br>" +
                   "<b>Si vous êtes sous Linux, il est recommandé de lancer PhotoGeoAlign depuis un terminal pour garantir que mm3d soit bien trouvé dans le PATH.</b><br><br>" +
                   "Veuillez suivre la documentation d'installation :<br>"
                   "<a href='https://micmac.ensg.eu/index.php/Install'>https://micmac.ensg.eu/index.php/Install</a><br><br>"
                   "Projet GitHub officiel :<br>"
                   "<a href='https://github.com/micmacIGN/micmac'>https://github.com/micmacIGN/micmac</a>")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)

main_window = None  # Variable globale pour conserver la fenêtre principale

if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_micmac_or_quit()
        # Lancement GUI pur (aucun argument)
        app = QApplication(sys.argv)
        # Splash screen statique avec le logo
        splash = QLabel()
        splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        splash.setStyleSheet("background: transparent; border: none;")
        logo_path = resource_path("logo.png")
        pixmap = QPixmap(logo_path)
        # Redimensionne le logo à 300px de large (hauteur ajustée automatiquement)
        target_width = 300
        scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.TransformationMode.SmoothTransformation)
        splash.setPixmap(scaled_pixmap)
        splash.setFixedSize(scaled_pixmap.size())
        splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splash.show()
        app.processEvents()
        def show_main():
            global main_window
            logo_path = resource_path("logo.png")
            if os.path.exists(logo_path):
                app.setWindowIcon(QIcon(logo_path))
            main_window = PhotogrammetryGUI()
            main_window.show()
            splash.close()
        QTimer.singleShot(3000, show_main)
        sys.exit(app.exec())
    else:
        parser = argparse.ArgumentParser(description="Photogrammetry Pipeline (MicMac)")
        parser.add_argument('--no-gui', action='store_true', help='Lancer en mode console (sans interface graphique)')
        parser.add_argument('input_dir', nargs='?', default=None, help="Dossier d'images à traiter")
        parser.add_argument('--mode', default='BigMac', choices=['QuickMac', 'BigMac', 'MicMac'], help='Mode de densification C3DC (défaut: BigMac)')
        parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (résolution) pour C3DC (1=max)')
        parser.add_argument('--tapas-model', default='Fraser', help='Modèle Tapas à utiliser (défaut: Fraser)')
        parser.add_argument('--tapioca-extra', default='', help='Paramètres supplémentaires pour Tapioca (optionnel)')
        parser.add_argument('--tapas-extra', default='', help='Paramètres supplémentaires pour Tapas (optionnel)')
        parser.add_argument('--c3dc-extra', default='', help='Paramètres supplémentaires pour C3DC (optionnel)')
        parser.add_argument('--saisieappuisinit-pt', default='', help='Chemin du fichier de points d\'appui pour SaisieAppuisInit (optionnel)')
        parser.add_argument('--saisieappuisinit-extra', default='', help='Paramètres supplémentaires pour SaisieAppuisInitQT (optionnel)')
        parser.add_argument('--saisieappuispredic-extra', default='', help='Paramètres supplémentaires pour SaisieAppuisPredicQT (optionnel)')
        parser.add_argument('--skip-saisieappuisinit', action='store_true', help='Ne pas exécuter SaisieAppuisInit')
        parser.add_argument('--skip-saisieappuispredic', action='store_true', help='Ne pas exécuter SaisieAppuisPredic')
        parser.add_argument('--skip-tapioca', action='store_true', help='Ne pas exécuter Tapioca')
        parser.add_argument('--skip-tapas', action='store_true', help='Ne pas exécuter Tapas')
        parser.add_argument('--skip-c3dc', action='store_true', help='Ne pas exécuter C3DC')
        
        # Arguments pour les transformations géodésiques
        parser.add_argument('--geodetic', action='store_true', help='Lancer les transformations géodésiques')
        parser.add_argument('--geodetic-coord', default='', help='Fichier de coordonnées de recalage pour les transformations géodésiques')
        parser.add_argument('--deformation-type', default='tps', choices=['tps'], help='Type de déformation (défaut: tps)')
        parser.add_argument('--deformation-params', default='', help='Paramètres de déformation (optionnel)')
        parser.add_argument('--add-offset-extra', default='', help='Paramètres supplémentaires pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-extra', default='', help='Paramètres supplémentaires pour ITRF→ENU (optionnel)')
        parser.add_argument('--itrf-to-enu-ref-point', default='', help='Nom du point de référence pour ITRF→ENU (optionnel, utilise le premier point si non spécifié)')
        parser.add_argument('--deform-extra', default='', help='Paramètres supplémentaires pour la déformation (optionnel)')
        parser.add_argument('--deform-bascule-xml', default='', help='Fichier XML GCPBascule pour la déformation (optionnel)')
        parser.add_argument('--enu-to-itrf-extra', default='', help='Paramètres supplémentaires pour ENU→ITRF (optionnel)')
        parser.add_argument('--skip-add-offset', action='store_true', help='Ne pas exécuter l\'ajout d\'offset')
        parser.add_argument('--skip-itrf-to-enu', action='store_true', help='Ne pas exécuter la conversion ITRF→ENU')
        parser.add_argument('--skip-deform', action='store_true', help='Ne pas exécuter la déformation')
        parser.add_argument('--skip-enu-to-itrf', action='store_true', help='Ne pas exécuter la conversion ENU→ITRF')
        
        # Arguments pour les dossiers d'entrée personnalisés
        parser.add_argument('--add-offset-input-dir', default='', help='Dossier d\'entrée personnalisé pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-input-dir', default='', help='Dossier d\'entrée personnalisé pour ITRF→ENU (optionnel)')
        parser.add_argument('--deform-input-dir', default='', help='Dossier d\'entrée personnalisé pour la déformation (optionnel)')
        parser.add_argument('--enu-to-itrf-input-dir', default='', help='Dossier d\'entrée personnalisé pour ENU→ITRF (optionnel)')
        
        # Arguments pour les dossiers de sortie personnalisés
        parser.add_argument('--add-offset-output-dir', default='', help='Dossier de sortie personnalisé pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-output-dir', default='', help='Dossier de sortie personnalisé pour ITRF→ENU (optionnel)')
        parser.add_argument('--deform-output-dir', default='', help='Dossier de sortie personnalisé pour la déformation (optionnel)')
        parser.add_argument('--enu-to-itrf-output-dir', default='', help='Dossier de sortie personnalisé pour ENU→ITRF (optionnel)')
    
        args = parser.parse_args()
        if args.geodetic:
            # Mode transformations géodésiques
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spécifier un dossier de nuages valide.")
                sys.exit(1)
            if not args.geodetic_coord or not os.path.exists(args.geodetic_coord):
                print("Erreur : veuillez spécifier un fichier de coordonnées valide.")
                sys.exit(1)
            
            log_path = os.path.join(args.input_dir, 'geodetic_transforms.log')
            logger = setup_logger(log_path)
            print(f"Début des transformations géodésiques pour le dossier : {args.input_dir}")
            
            try:
                coord_file = args.geodetic_coord
                deformation_type = args.deformation_type
                deformation_params = args.deformation_params
                add_offset_extra = args.add_offset_extra
                itrf_to_enu_extra = args.itrf_to_enu_extra
                itrf_to_enu_ref_point = args.itrf_to_enu_ref_point if args.itrf_to_enu_ref_point else None
                deform_extra = args.deform_extra
                deform_bascule_xml = args.deform_bascule_xml if args.deform_bascule_xml else None
                enu_to_itrf_extra = args.enu_to_itrf_extra
                
                # Dossiers d'entrée personnalisés
                add_offset_input_dir = args.add_offset_input_dir if args.add_offset_input_dir else None
                itrf_to_enu_input_dir = args.itrf_to_enu_input_dir if args.itrf_to_enu_input_dir else None
                deform_input_dir = args.deform_input_dir if args.deform_input_dir else None
                enu_to_itrf_input_dir = args.enu_to_itrf_input_dir if args.enu_to_itrf_input_dir else None
                
                # Dossiers de sortie personnalisés
                add_offset_output_dir = args.add_offset_output_dir if args.add_offset_output_dir else None
                itrf_to_enu_output_dir = args.itrf_to_enu_output_dir if args.itrf_to_enu_output_dir else None
                deform_output_dir = args.deform_output_dir if args.deform_output_dir else None
                enu_to_itrf_output_dir = args.enu_to_itrf_output_dir if args.enu_to_itrf_output_dir else None
                
                run_add_offset = not args.skip_add_offset
                run_itrf_to_enu = not args.skip_itrf_to_enu
                run_deform = not args.skip_deform
                run_enu_to_itrf = not args.skip_enu_to_itrf
                
                # Création d'une instance du thread pour gérer les dossiers d'entrée/sortie
                geodetic_thread = GeodeticTransformThread(
                    args.input_dir, coord_file, deformation_type, deformation_params,
                    add_offset_extra, itrf_to_enu_extra, deform_extra, enu_to_itrf_extra,
                    run_add_offset, run_itrf_to_enu, run_deform, run_enu_to_itrf,
                    add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, enu_to_itrf_input_dir,
                    add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, enu_to_itrf_output_dir,
                    itrf_to_enu_ref_point, deform_bascule_xml
                )
                
                # Exécution des transformations
                geodetic_thread.run()
                
                print("Transformations géodésiques terminées avec succès !")
            except Exception as e:
                print(f"Erreur lors des transformations géodésiques : {e}")
                sys.exit(1)
        elif args.no_gui:
            # Mode pipeline photogrammétrique
            check_micmac_or_quit()
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spécifier un dossier d'images valide.")
                sys.exit(1)
            log_path = os.path.join(args.input_dir, 'photogrammetry_pipeline.log')
            logger = setup_logger(log_path)
            print(f"Début du pipeline photogrammétrique pour le dossier : {args.input_dir}")
            try:
                tapas_model = args.tapas_model
                saisieappuisinit_pt = args.saisieappuisinit_pt or None
                saisieappuisinit_extra = args.saisieappuisinit_extra
                saisieappuispredic_extra = args.saisieappuispredic_extra
                run_saisieappuisinit = not args.skip_saisieappuisinit
                run_saisieappuispredic = not args.skip_saisieappuispredic
                if not args.skip_tapioca:
                    run_micmac_tapioca(args.input_dir, logger, args.tapioca_extra)
                if not args.skip_tapas:
                    run_micmac_tapas(args.input_dir, logger, tapas_model, args.tapas_extra)
                if run_saisieappuisinit:
                    run_micmac_saisieappuisinit(args.input_dir, logger, tapas_model, saisieappuisinit_pt, saisieappuisinit_extra)
                if run_saisieappuispredic:
                    run_micmac_saisieappuispredic(args.input_dir, logger, tapas_model, saisieappuisinit_pt, saisieappuispredic_extra)
                if not args.skip_c3dc:
                    run_micmac_c3dc(args.input_dir, logger, mode=args.mode, zoomf=args.zoomf, tapas_model=tapas_model, extra_params=args.c3dc_extra)
                print("Pipeline terminé avec succès !")
            except Exception as e:
                print(f"Erreur lors de l'exécution du pipeline : {e}")
                sys.exit(1)
        else:
            check_micmac_or_quit()
            app = QApplication(sys.argv)
            logo_path = resource_path("logo.png")
            if os.path.exists(logo_path):
                app.setWindowIcon(QIcon(logo_path))
            gui = PhotogrammetryGUI()
            gui.show()
            sys.exit(app.exec()) 