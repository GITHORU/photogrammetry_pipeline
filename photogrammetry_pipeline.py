import os
import subprocess
import sys
from pathlib import Path
import logging
import argparse
import time

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

def run_micmac_tapioca(input_dir, logger, nb_proc=8):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapioca va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700', f'NbProc={nb_proc}'
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
        'mm3d', 'Tapas', 'Fraser', pattern, 'Out=Fraser' #, 'ExpTxt=1'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(input_dir, logger, mode='QuickMac', zoomf=1, nb_proc=8):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, 'Fraser', f'ZoomF={zoomf}', f'NbProc={nb_proc}'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"Nuage dense généré par C3DC {mode} (voir dossier PIMs-{mode}/ ou fichier C3DC_{mode}.ply)")

def setup_summary_logger(summary_log_path):
    summary_logger = logging.getLogger("PhotogrammetryPipelineSummary")
    summary_logger.setLevel(logging.INFO)
    # Évite les handlers multiples si la fonction est appelée plusieurs fois
    if not summary_logger.handlers:
        sh = logging.FileHandler(summary_log_path)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        summary_logger.addHandler(sh)
    return summary_logger

def main():
    parser = argparse.ArgumentParser(description="Pipeline photogrammétrique MicMac (C3DC)")
    parser.add_argument('input_dir', nargs='?', default='short_dataset', help='Dossier d\'images à traiter')
    parser.add_argument('--mode', default='QuickMac', choices=['QuickMac', 'BigMac'], help='Mode de densification C3DC')
    parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (résolution) pour C3DC (1=max)')
    parser.add_argument('--nb-proc', type=int, default=8, help='Nombre de processeurs à utiliser pour Tapioca et C3DC (défaut: 8)')
    args = parser.parse_args()

    input_dir = args.input_dir
    mode = args.mode
    zoomf = args.zoomf
    nb_proc = args.nb_proc

    os.makedirs(input_dir, exist_ok=True)  # S'assurer que le dossier existe
    log_path = os.path.join(input_dir, 'photogrammetry_pipeline.log')
    summary_log_path = os.path.join(input_dir, 'photogrammetry_pipeline_summary.log')
    logger = setup_logger(log_path)
    summary_logger = setup_summary_logger(summary_log_path)
    logger.info(f"Début du pipeline photogrammétrique pour le dossier : {input_dir}")
    summary_logger.info(f"--- Résumé pipeline pour {input_dir} ---")
    global_start = time.time()

    try:
        # 1. Points homologues sur DNG
        t0 = time.time()
        try:
            run_micmac_tapioca(input_dir, logger, nb_proc=nb_proc)
            summary_logger.info(f"Tapioca : SUCCÈS ({time.time()-t0:.1f}s)")
        except Exception as e:
            summary_logger.info(f"Tapioca : ÉCHEC ({time.time()-t0:.1f}s)")
            raise

        # 2. Orientation sur DNG
        t1 = time.time()
        try:
            run_micmac_tapas(input_dir, logger)
            summary_logger.info(f"Tapas   : SUCCÈS ({time.time()-t1:.1f}s)")
        except Exception as e:
            summary_logger.info(f"Tapas   : ÉCHEC ({time.time()-t1:.1f}s)")
            raise

        # 3. Densification sur DNG avec C3DC
        t2 = time.time()
        try:
            run_micmac_c3dc(input_dir, logger, mode=mode, zoomf=zoomf, nb_proc=nb_proc)
            summary_logger.info(f"C3DC-{mode:8} : SUCCÈS ({time.time()-t2:.1f}s)")
        except Exception as e:
            summary_logger.info(f"C3DC-{mode:8} : ÉCHEC ({time.time()-t2:.1f}s)")
            raise

        total = time.time() - global_start
        summary_logger.info(f"--- Pipeline terminé avec SUCCÈS en {total:.1f}s ---\n")
        logger.info(f"Traitement terminé ! Le nuage dense est dans le dossier PIMs-{mode}/ ou C3DC_{mode}.ply")
    except Exception as e:
        total = time.time() - global_start
        summary_logger.info(f"--- Pipeline TERMINÉ AVEC ÉCHEC en {total:.1f}s ---\n")
        logger.error(f"Erreur critique : {e}")
        logger.info("Arrêt du pipeline suite à une erreur.")

if __name__ == "__main__":
    main() 