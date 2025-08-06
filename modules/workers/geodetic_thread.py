import logging
import os
from PySide6.QtCore import QThread, Signal
from ..core.geodetic import (
    add_offset_to_clouds, convert_itrf_to_enu, deform_clouds, convert_enu_to_itrf
)
from .utils import QtLogHandler

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