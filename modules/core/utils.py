import os
import sys
import logging
import subprocess
from pathlib import Path

def setup_logger(log_path=None):
    logger = logging.getLogger("PhotogrammetryPipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Console handler (INFO et plus) - écrire vers stdout pour que SLURM capture les logs dans le fichier .out
    ch = logging.StreamHandler(sys.stdout)
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
    # Forcer le flush immédiat des logs
    for handler in logger.handlers:
        handler.flush()
    
    try:
        creationflags = 0
        if os.name == 'nt':
            import subprocess as sp
            creationflags = sp.CREATE_NO_WINDOW
        
        # Préparer l'environnement avec unbuffering forcé
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        
        # Sur Linux/Unix, essayer d'utiliser stdbuf pour forcer le unbuffering des processus externes
        # (notamment pour MicMac qui peut bufferiser ses sorties)
        original_cmd = cmd
        if os.name != 'nt':  # Pas sur Windows
            # Vérifier si stdbuf est disponible
            try:
                subprocess.run(['stdbuf', '--version'], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, 
                             timeout=1)
                # stdbuf disponible, l'utiliser pour forcer unbuffering
                cmd = ['stdbuf', '-o0', '-e0'] + list(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                # stdbuf non disponible, continuer sans
                pass
        
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=0, universal_newlines=True,  # bufsize=0 = unbuffered
            creationflags=creationflags,
            stdin=subprocess.PIPE,
            env=env
        )
        if process.stdout is not None:
            for line in process.stdout:
                logger.info(line.rstrip())
                # Forcer le flush immédiat après chaque ligne pour garantir l'écriture
                for handler in logger.handlers:
                    handler.flush()
                if 'Warn tape enter to continue' in line:
                    try:
                        if process.stdin is not None:
                            process.stdin.write('\n')
                            process.stdin.flush()
                    except Exception:
                        pass
        process.wait()
        # Flush final pour s'assurer que tous les logs sont écrits
        for handler in logger.handlers:
            handler.flush()
        if process.returncode != 0:
            logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(original_cmd)} (code {process.returncode})")
            raise subprocess.CalledProcessError(process.returncode, original_cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(original_cmd)}")
        logger.error(f"Code retour : {e.returncode}")
        # Flush final en cas d'erreur
        for handler in logger.handlers:
            handler.flush()
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