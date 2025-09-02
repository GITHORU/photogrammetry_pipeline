#!/usr/bin/env python3
"""
Thread de travail pour le pipeline d'analyse
"""

import os
import logging
from PySide6.QtCore import QThread, Signal
from ..core.analysis import run_analysis_pipeline
from .utils import QtLogHandler

class AnalysisThread(QThread):
    """
    Thread pour exécuter le pipeline d'analyse
    """
    
    # Signaux pour communiquer avec l'interface
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    
    def __init__(self, image1_path: str, image2_path: str, analysis_type: str, 
                 resolution: float, output_dir: str, farneback_params: dict = None):
        super().__init__()
        
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.analysis_type = analysis_type
        self.resolution = resolution
        self.output_dir = output_dir
        self.farneback_params = farneback_params or {}
        
        # Configuration du logger
        self.logger = logging.getLogger(f"AnalysisPipeline_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Handler pour envoyer les logs vers l'interface
        self.qt_handler = QtLogHandler(self.log_signal)
        self.qt_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.qt_handler)
        
        # Création du dossier de sortie avant de créer le fichier de log
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Handler pour les logs dans un fichier
        log_path = os.path.join(output_dir, 'analysis_pipeline.log')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.results = None
    
    def run(self):
        """
        Exécution du pipeline d'analyse
        """
        try:
            self.logger.info("Démarrage du pipeline d'analyse")
            self.logger.info(f"Image 1: {str(self.image1_path)}")
            self.logger.info(f"Image 2: {str(self.image2_path)}")
            self.logger.info(f"Type d'analyse: {str(self.analysis_type)}")
            self.logger.info(f"Résolution: {str(self.resolution)} m")
            self.logger.info(f"Dossier de sortie: {str(self.output_dir)}")
            
            # Exécution du pipeline d'analyse
            self.results = run_analysis_pipeline(
                self.image1_path,
                self.image2_path,
                self.analysis_type,
                self.resolution,
                self.output_dir,
                self.farneback_params
            )
            
            if self.results:
                self.logger.info("Pipeline d'analyse terminé avec succès")
                self.logger.info(f"RMSE: {str(self.results.get('rmse', 'N/A'))}")
                self.logger.info(f"Corrélation Pearson: {str(self.results.get('correlation_pearson', 'N/A'))}")
                self.logger.info(f"Nombre de points: {str(self.results.get('n_points', 'N/A'))}")
                
                if 'report_path' in self.results:
                    self.logger.info(f"Rapport généré: {str(self.results['report_path'])}")
                
                self.finished_signal.emit(True, "Pipeline d'analyse terminé avec succès")
            else:
                self.logger.warning("Aucun résultat obtenu")
                self.finished_signal.emit(False, "Aucun résultat obtenu")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du pipeline d'analyse: {str(e)} (fichier: {__file__}, ligne: {e.__traceback__.tb_lineno})")
            self.finished_signal.emit(False, f"Erreur: {str(e)}")
    
    def get_results(self):
        """
        Retourne les résultats de l'analyse
        """
        return self.results
