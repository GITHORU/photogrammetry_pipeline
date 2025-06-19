import os
import subprocess
import sys
from pathlib import Path
import logging

# Vérification des dépendances pour la conversion DNG -> TIF
try:
    import rawpy
    import imageio
except ImportError:
    print("Erreur : les modules rawpy et imageio sont nécessaires pour la conversion DNG -> TIF.\nInstallez-les avec : pip install rawpy imageio")
    sys.exit(1)

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
        # Affiche la sortie en temps réel dans le terminal
        result = subprocess.run(cmd, check=True, cwd=cwd, text=True, stdout=sys.stdout, stderr=sys.stderr)
        # On ne logge plus la sortie ici car elle est déjà affichée
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)}")
        logger.error(f"Code retour : {e.returncode}")
        logger.error(f"Sortie standard : voir terminal ci-dessus")
        logger.error(f"Sortie erreur : voir terminal ci-dessus")
        raise

def convert_dng_to_tif(input_dir, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    for dng_file in Path(input_dir).glob('*.DNG'):
        tif_file = Path(output_dir) / (dng_file.stem + '.tif')
        logger.info(f"Conversion {dng_file} -> {tif_file} avec rawpy")
        try:
            with rawpy.imread(str(dng_file)) as raw:
                rgb = raw.postprocess()
            imageio.imwrite(str(tif_file), rgb)
            if tif_file.exists():
                logger.info(f"Fichier généré : {tif_file}")
            else:
                logger.error(f"La conversion a échoué pour {dng_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de {dng_file} : {e}")

def run_micmac_tapioca(image_dir, logger, dng_dir=None):
    # Si un dossier DNG est fourni et contient des DNG, utiliser ces images pour Tapioca
    if dng_dir is not None and any(Path(dng_dir).glob('*.DNG')):
        abs_dng_dir = os.path.abspath(dng_dir)
        pattern = '.*.DNG'
        logger.info(f"Tapioca va utiliser les DNG originaux dans {abs_dng_dir} ...")
        cmd = [
            'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700'
        ]
        run_command(cmd, logger, cwd=abs_dng_dir)
        homol_dir = Path(abs_dng_dir) / 'Homol'
    else:
        abs_image_dir = os.path.abspath(image_dir)
        pattern = '.*.tif'
        logger.info(f"Tapioca va utiliser les TIFF dans {abs_image_dir} ...")
        cmd = [
            'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700'
        ]
        run_command(cmd, logger, cwd=abs_image_dir)
        homol_dir = Path(abs_image_dir) / 'Homol'
    if homol_dir.exists() and any(homol_dir.iterdir()):
        logger.info(f"Dossier Homol généré : {homol_dir}")
    else:
        logger.error("Le dossier Homol n'a pas été généré par Tapioca. Arrêt du pipeline.")
        raise RuntimeError("Le dossier Homol n'a pas été généré par Tapioca.")
    logger.info("Tapioca terminé.")

def run_micmac_tapas(image_dir, logger, dng_dir=None):
    abs_image_dir = os.path.abspath(image_dir)
    # Si un dossier DNG est fourni et contient des DNG, utiliser ces images pour Tapas
    if dng_dir is not None and any(Path(dng_dir).glob('*.DNG')):
        abs_dng_dir = os.path.abspath(dng_dir)
        pattern = '.*.DNG'
        logger.info(f"Tapas va utiliser les DNG originaux dans {abs_dng_dir} ...")
        cmd = [
            'mm3d', 'Tapas', 'Fraser', pattern, 'Out=Fraser', 'ExpTxt=1'
        ]
        run_command(cmd, logger, cwd=abs_dng_dir)
    else:
        # Sinon, fallback sur les TIFF
        pattern = '.*.tif'
        logger.info(f"Tapas va utiliser les TIFF dans {abs_image_dir} ...")
        cmd = [
            'mm3d', 'Tapas', 'Fraser', pattern, 'Out=Fraser', 'ExpTxt=1'
        ]
        run_command(cmd, logger, cwd=abs_image_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(image_dir, logger):
    abs_image_dir = os.path.abspath(image_dir)
    cmd = [
        'mm3d', 'C3DC', 'QuickMac', '.*.tif', 'Fraser', 'ExpTxt=1'
    ]
    logger.info(f"Lancement de C3DC (densification du nuage de points) dans {abs_image_dir} ...")
    run_command(cmd, logger, cwd=abs_image_dir)
    mec_dir = Path(abs_image_dir) / 'MEC-QuickMac'
    if mec_dir.exists():
        logger.info(f"Nuage dense généré dans : {mec_dir}")
    else:
        logger.error("Le dossier du nuage dense n'a pas été trouvé après C3DC.")

def main():
    if len(sys.argv) == 2:
        input_dir = sys.argv[1]
    else:
        input_dir = 'short_dataset'
        print("Aucun dossier d'images spécifié, utilisation du dossier par défaut : short_dataset")
    tif_dir = os.path.join(input_dir, 'tif')
    log_path = os.path.join(input_dir, 'photogrammetry_pipeline.log')
    logger = setup_logger(log_path)
    logger.info(f"Début du pipeline photogrammétrique pour le dossier : {input_dir}")

    try:
        # 1. Conversion DNG -> TIF (désactivée temporairement pour accélérer les tests)
        # convert_dng_to_tif(input_dir, tif_dir, logger)

        # 2. Points homologues sur DNG si possible, sinon TIFF
        run_micmac_tapioca(tif_dir, logger, dng_dir=input_dir)

        # 3. Orientation sur DNG si possible, sinon TIFF
        run_micmac_tapas(tif_dir, logger, dng_dir=input_dir)

        # 4. Densification sur TIFF
        run_micmac_c3dc(tif_dir, logger)

        logger.info("Traitement terminé ! Le nuage dense est dans le dossier 'tif/MEC-QuickMac'.")
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        logger.info("Arrêt du pipeline suite à une erreur.")

if __name__ == "__main__":
    main() 