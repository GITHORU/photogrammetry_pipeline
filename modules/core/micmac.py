import os
import subprocess
from pathlib import Path
from .utils import run_command, micmac_command_exists

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