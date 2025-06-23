import os
import subprocess
import sys
from pathlib import Path
import logging

# Vérification des dépendances pour la conversion DNG -> TIF (SUPPRIMÉE)

def setup_logger(log_path):
    logger = logging.getLogger("PhotogrammetryPipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler (INFO et plus)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (DEBUG et plus)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def run_command(cmd, logger, cwd=None):
    logger.info(f"Commande lancée : {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd, text=True, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)}")
        logger.error(f"Code retour : {e.returncode}")
        logger.error(f"Sortie standard : voir terminal ci-dessus")
        logger.error(f"Sortie erreur : voir terminal ci-dessus")
        raise

def run_micmac_tapioca(input_dir, logger):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapioca va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700', 'ExpTxt=1'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    homol_dir = Path(abs_input_dir) / 'Homol'
    if homol_dir.exists() and any(homol_dir.iterdir()):
        logger.info(f"Dossier Homol généré : {homol_dir}")
    else:
        logger.error("Le dossier Homol n'a pas été généré par Tapioca. Arrêt du pipeline.")
        raise RuntimeError("Le dossier Homol n'a pas été généré par Tapioca.")
    logger.info("Tapioca terminé.")

def run_micmac_tapas(input_dir, logger):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapas va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapas', 'Fraser', pattern, 'Out=Fraser', 'ExpTxt=1'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(input_dir, logger):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Lancement de C3DC (densification du nuage de points) dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'C3DC', 'QuickMac', pattern, 'Fraser', 'ExpTxt=1', 'ZoomF=1'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    mec_dir = Path(abs_input_dir) / 'MEC-QuickMac'
    if mec_dir.exists():
        logger.info(f"Nuage dense généré dans : {mec_dir}")
    else:
        mec_dir_alt = Path(abs_input_dir) / 'MEC-QuickMac'
        if mec_dir_alt.exists():
            logger.info(f"Nuage dense généré dans : {mec_dir_alt}")
        else:
            logger.error("Le dossier du nuage dense n'a pas été trouvé après C3DC.")

def main():
    if len(sys.argv) == 2:
        input_dir = sys.argv[1]
    else:
        input_dir = 'short_dataset'
        print("Aucun dossier d'images spécifié, utilisation du dossier par défaut : short_dataset")
    log_path = os.path.join(input_dir, 'photogrammetry_pipeline.log')
    logger = setup_logger(log_path)
    logger.info(f"Début du pipeline photogrammétrique pour le dossier : {input_dir}")

    try:
        # 1. Points homologues sur DNG
        run_micmac_tapioca(input_dir, logger)

        # 2. Orientation sur DNG
        run_micmac_tapas(input_dir, logger)

        # 3. Densification sur DNG
        run_micmac_c3dc(input_dir, logger)

        logger.info("Traitement terminé ! Le nuage dense est dans le dossier 'MEC-QuickMac'.")
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        logger.info("Arrêt du pipeline suite à une erreur.")

if __name__ == "__main__":
    main() 