import os
import subprocess
from pathlib import Path
from .utils import run_command, micmac_command_exists

def run_micmac_xifgps2xml(input_dir, logger, extra_params=""):
    """Extrait les coordonnées GPS depuis les métadonnées EXIF des images TIFF"""
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    logger.info(f"XifGps2Xml va extraire les coordonnées GPS depuis les EXIF des TIFF dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'XifGps2Xml', pattern, 'RAWGNSS'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    rawgnss_dir = Path(abs_input_dir) / 'Ori-RAWGNSS'
    if rawgnss_dir.exists() and rawgnss_dir.is_dir() and any(rawgnss_dir.iterdir()):
        logger.info(f"Dossier d'orientation RAWGNSS généré : {rawgnss_dir}")
    else:
        logger.warning("Le dossier d'orientation Ori-RAWGNSS n'a pas été généré par XifGps2Xml.")
    logger.info("XifGps2Xml terminé.")
    return 'RAWGNSS'  # Retourne le nom de l'orientation (sans préfixe Ori-)

def run_micmac_oriconvert(input_dir, logger, positions_file, tapas_model="Fraser", extra_params=""):
    """Convertit les coordonnées GPS en orientations initiales"""
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    if not positions_file:
        logger.error("Aucun fichier de positions RTK fourni pour OriConvert.")
        raise RuntimeError("Aucun fichier de positions RTK fourni pour OriConvert.")
    positions_file = os.path.abspath(positions_file)
    if not os.path.exists(positions_file):
        logger.error(f"Fichier de positions RTK introuvable : {positions_file}")
        raise RuntimeError(f"Fichier de positions RTK introuvable : {positions_file}")
    
    positions_file_rel = os.path.relpath(positions_file, abs_input_dir)
    rawgnss_n = 'RAWGNSS_N'
    chsys_file = f'ChSys=DegreeWGS84@RTLFromExif.xml'
    
    logger.info(f"OriConvert va convertir les coordonnées GPS en orientations initiales dans {abs_input_dir} ...")
    logger.info(f"  Fichier de positions : {positions_file_rel}")
    logger.info(f"  Système de coordonnées : {chsys_file}")
    
    cmd = [
        'mm3d', 'OriConvert', '#F=N X Y Z', positions_file_rel, rawgnss_n,
        chsys_file, 'MTD1=1', 'NameCple=FileImagesNeighbour.xml', 'OkNoIm=1'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    
    rawgnss_n_file = Path(abs_input_dir) / rawgnss_n
    neighbor_file = Path(abs_input_dir) / 'FileImagesNeighbour.xml'
    if rawgnss_n_file.exists():
        logger.info(f"Orientation RAWGNSS_N générée : {rawgnss_n_file}")
    if neighbor_file.exists():
        logger.info(f"Fichier FileImagesNeighbour.xml généré : {neighbor_file}")
    logger.info("OriConvert terminé.")
    return rawgnss_n, neighbor_file

def run_micmac_tapioca(input_dir, logger, use_neighbor_file=False, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    
    if use_neighbor_file:
        neighbor_file = Path(abs_input_dir) / 'FileImagesNeighbour.xml'
        if not neighbor_file.exists():
            logger.error(f"Le fichier FileImagesNeighbour.xml est introuvable : {neighbor_file}")
            raise RuntimeError(f"Le fichier FileImagesNeighbour.xml est introuvable : {neighbor_file}")
        neighbor_file_rel = os.path.relpath(neighbor_file, abs_input_dir)
        logger.info(f"Tapioca va utiliser le fichier FileImagesNeighbour.xml dans {abs_input_dir} ...")
        cmd = [
            'mm3d', 'Tapioca', 'File', neighbor_file_rel, '2700'
        ]
    else:
        logger.info(f"Tapioca va utiliser les TIFF dans {abs_input_dir} avec le motif {pattern} ...")
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

def run_micmac_tapas(input_dir, logger, tapas_model="Fraser", use_arbitrary=False, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    if use_arbitrary:
        out_ori = 'Arbitrary'
        logger.info(f"Tapas va utiliser les TIFF dans {abs_input_dir} avec le motif {pattern} et générer l'orientation Arbitrary ...")
    else:
        out_ori = tapas_model
        logger.info(f"Tapas va utiliser les TIFF dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapas', tapas_model, pattern, f'Out={out_ori}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")
    return out_ori

def run_micmac_centerbascule(input_dir, logger, ori_in="Arbitrary", rawgnss_n="RAWGNSS_N", tapas_model="Fraser", extra_params=""):
    """Recalage initial utilisant les positions GPS RTK"""
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    ori_out = f"{tapas_model}_Init_RTL"
    
    # MicMac utilise le nom d'orientation sans préfixe Ori-, il ajoute automatiquement le préfixe
    # Le dossier réel s'appelle Ori-RAWGNSS_N mais on passe juste RAWGNSS_N à MicMac
    rawgnss_n_name = rawgnss_n  # Nom de l'orientation (sans préfixe Ori-)
    
    logger.info(f"CenterBascule va recaler l'orientation {ori_in} avec les GPS RTK dans {abs_input_dir} ...")
    logger.info(f"  Orientation d'entrée : {ori_in}")
    logger.info(f"  Orientation de sortie : {ori_out}")
    logger.info(f"  Orientation GPS : {rawgnss_n_name} (MicMac utilisera Ori-{rawgnss_n_name})")
    
    cmd = [
        'mm3d', 'CenterBascule', pattern, ori_in, rawgnss_n_name, ori_out
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("CenterBascule terminé.")
    return ori_out

def run_micmac_campari(input_dir, logger, ori_in, tapas_model="Fraser", rawgnss_n="RAWGNSS_N", 
                       gps_weights=(0.05, 0.07), extra_params=""):
    """Ajustement bundle avec contraintes GPS RTK
    
    Args:
        gps_weights: Tuple (sigma_plani, sigma_alti) - poids GPS pour planimétrie et altitude
                     Par défaut: (0.05, 0.07) - 0.05m en planimétrie, 0.07m en altitude
    """
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    ori_out = tapas_model  # Normalisation : toujours utiliser le nom du modèle
    
    # MicMac utilise le nom d'orientation sans préfixe Ori-, il ajoute automatiquement le préfixe
    # Le dossier réel s'appelle Ori-RAWGNSS_N mais on passe juste RAWGNSS_N à MicMac
    rawgnss_n_name = rawgnss_n  # Nom de l'orientation (sans préfixe Ori-)
    # Campari utilise 2 poids : sigma_plani (pour X et Y) et sigma_alti (pour Z)
    gps_weight_str = f"[{rawgnss_n_name},{gps_weights[0]},{gps_weights[1]}]"
    
    logger.info(f"Campari va ajuster l'orientation {ori_in} avec contraintes GPS RTK dans {abs_input_dir} ...")
    logger.info(f"  Orientation d'entrée : {ori_in}")
    logger.info(f"  Orientation de sortie : {ori_out} (normalisée)")
    logger.info(f"  Contraintes GPS : {gps_weight_str} (sigma_plani={gps_weights[0]}m, sigma_alti={gps_weights[1]}m)")
    logger.info(f"  Orientation GPS : {rawgnss_n_name} (MicMac utilisera Ori-{rawgnss_n_name})")
    
    cmd = [
        'mm3d', 'Campari', pattern, ori_in, ori_out, f'EmGPS={gps_weight_str}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Campari terminé.")
    return ori_out

def run_micmac_c3dc(input_dir, logger, mode='QuickMac', zoomf=1, tapas_model='Fraser', extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
    ori = f"{tapas_model}_abs"
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} et Ori={ori} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, ori, f'ZoomF={zoomf}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"Nuage dense généré par C3DC {mode} (voir dossier PIMs-{mode}/ ou fichier C3DC_{mode}.ply)")

def run_micmac_saisieappuisinit(input_dir, logger, tapas_model="Fraser", appuis_file=None, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.tiff'
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
    pattern = '.*.tiff'
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
    pattern = '.*.tiff'
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
    pattern = '.*.tiff'
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