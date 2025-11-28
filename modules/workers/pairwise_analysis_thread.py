#!/usr/bin/env python3
"""
Thread de travail pour le pipeline d'analyse paire par paire
"""

import os
import logging
from PySide6.QtCore import QThread, Signal
from ..core.analysis import run_pairwise_analysis_pipeline
from .utils import QtLogHandler

class PairwiseAnalysisThread(QThread):
    """
    Thread pour exécuter le pipeline d'analyse paire par paire
    """
    
    # Signaux pour communiquer avec l'interface
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    progress_signal = Signal(int, int)  # (current, total)
    
    def __init__(self, model_paths: list, analysis_type: str, resolution: float,
                 output_dir: str, farneback_params: dict = None, mnt_paths: list = None,
                 parallel: bool = True, max_workers: int = None):
        super().__init__()
        
        self.model_paths = model_paths
        self.analysis_type = analysis_type
        self.resolution = resolution
        self.output_dir = output_dir
        self.farneback_params = farneback_params or {}
        self.mnt_paths = mnt_paths
        self.parallel = parallel
        self.max_workers = max_workers
        
        # Configuration du logger
        self.logger = logging.getLogger(f"PairwiseAnalysisPipeline_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Handler pour envoyer les logs vers l'interface
        self.qt_handler = QtLogHandler(self.log_signal)
        self.qt_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.qt_handler)
        
        # Création du dossier de sortie avant de créer le fichier de log
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Handler pour les logs dans un fichier
        log_path = os.path.join(output_dir, 'pairwise_analysis_pipeline.log')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.results = None
    
    def run(self):
        """
        Exécution du pipeline d'analyse paire par paire
        """
        try:
            self.logger.info("Démarrage du pipeline d'analyse paire par paire")
            self.logger.info(f"Nombre de modèles: {len(self.model_paths)}")
            self.logger.info(f"Type d'analyse: {self.analysis_type}")
            self.logger.info(f"Résolution: {self.resolution} m")
            self.logger.info(f"Dossier de sortie: {self.output_dir}")
            
            # Exécution du pipeline d'analyse paire par paire
            self.results = run_pairwise_analysis_pipeline(
                self.model_paths,
                self.analysis_type,
                self.resolution,
                self.output_dir,
                self.farneback_params,
                self.mnt_paths,
                self.parallel,
                self.max_workers
            )
            
            if self.results:
                self.logger.info("Pipeline d'analyse paire par paire terminé avec succès")
                self.logger.info(f"Nombre de comparaisons: {self.results.get('n_comparisons', 'N/A')}")
                
                self.finished_signal.emit(True, "Pipeline d'analyse paire par paire terminé avec succès")
            else:
                self.logger.warning("Aucun résultat obtenu")
                self.finished_signal.emit(False, "Aucun résultat obtenu")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du pipeline d'analyse paire par paire: {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
            self.finished_signal.emit(False, f"Erreur: {str(e)}")
    
    def get_results(self):
        """
        Retourne les résultats de l'analyse
        """
        return self.results

