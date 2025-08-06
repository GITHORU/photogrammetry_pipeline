import os
import sys
import logging
import subprocess
from pathlib import Path

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

def resource_path(relative_path):
    """Trouve le chemin absolu d'une ressource, compatible PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # Remonte de 2 niveaux depuis modules/core/ pour atteindre le répertoire racine
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Si le fichier n'existe pas dans le répertoire racine, cherche dans resources/
    if not os.path.exists(os.path.join(root_dir, relative_path)):
        return os.path.join(root_dir, "resources", relative_path)
    return os.path.join(root_dir, relative_path) 