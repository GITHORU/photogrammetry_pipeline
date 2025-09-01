import logging
import os
from PySide6.QtCore import QThread, Signal
# Import des fonctions depuis les nouveaux modules refactorisés
from ..core.geodetic_processing import (
    add_offset_to_clouds, convert_itrf_to_enu, deform_clouds
)
from ..core.geodetic_orthoimage_basic import (
    create_orthoimage_from_pointcloud, merge_orthoimages_and_dtm
)
from ..core.geodetic_orthoimage_fusion import (
    unified_ortho_mnt_fusion
)
from .utils import QtLogHandler

class GeodeticTransformThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, coord_file, deformation_type, deformation_params, add_offset_extra, itrf_to_enu_extra, deform_extra, run_add_offset=True, run_itrf_to_enu=True, run_deform=True, run_orthoimage=True, run_unified_orthoimage=True, add_offset_input_dir=None, itrf_to_enu_input_dir=None, deform_input_dir=None, orthoimage_input_dir=None, unified_orthoimage_input_dir=None, add_offset_output_dir=None, itrf_to_enu_output_dir=None, deform_output_dir=None, orthoimage_output_dir=None, unified_orthoimage_output_dir=None, itrf_to_enu_ref_point=None, deform_bascule_xml=None, orthoimage_resolution=0.1, orthoimage_height_field="z", orthoimage_color_field="rgb", unified_orthoimage_resolution=0.1, max_workers=None, color_fusion_method="average", zone_size_meters=5.0, global_ref_point=None, force_global_ref=False):
        super().__init__()
        self.input_dir = input_dir
        self.coord_file = coord_file
        self.deformation_type = deformation_type
        self.deformation_params = deformation_params
        self.add_offset_extra = add_offset_extra
        self.itrf_to_enu_extra = itrf_to_enu_extra
        self.deform_extra = deform_extra

        self.run_add_offset = run_add_offset
        self.run_itrf_to_enu = run_itrf_to_enu
        self.run_deform = run_deform
        self.run_orthoimage = run_orthoimage
        self.run_unified_orthoimage = run_unified_orthoimage

        # Dossiers d'entrée personnalisés pour chaque étape
        self.add_offset_input_dir = add_offset_input_dir
        self.itrf_to_enu_input_dir = itrf_to_enu_input_dir
        self.deform_input_dir = deform_input_dir
        self.orthoimage_input_dir = orthoimage_input_dir
        self.unified_orthoimage_input_dir = unified_orthoimage_input_dir

        # Dossiers de sortie personnalisés pour chaque étape
        self.add_offset_output_dir = add_offset_output_dir
        self.itrf_to_enu_output_dir = itrf_to_enu_output_dir
        self.deform_output_dir = deform_output_dir
        self.orthoimage_output_dir = orthoimage_output_dir
        self.unified_orthoimage_output_dir = unified_orthoimage_output_dir

        self.itrf_to_enu_ref_point = itrf_to_enu_ref_point
        self.deform_bascule_xml = deform_bascule_xml
        self.orthoimage_resolution = orthoimage_resolution
        self.orthoimage_height_field = orthoimage_height_field
        self.orthoimage_color_field = orthoimage_color_field
        self.unified_orthoimage_resolution = unified_orthoimage_resolution
        self.max_workers = max_workers
        self.color_fusion_method = color_fusion_method
        
        # Paramètre de taille des zones
        self.zone_size_meters = zone_size_meters
        
        # Paramètres du point de référence global
        self.global_ref_point = global_ref_point  # [x, y, z] ou None
        self.force_global_ref = force_global_ref  # bool

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
                
                # DEBUG : Vérification des paramètres du point de référence global
                logger.info(f"🔍 DEBUG - Paramètres transmis à convert_itrf_to_enu:")
                logger.info(f"   global_ref_point: {self.global_ref_point}")
                logger.info(f"   force_global_ref: {self.force_global_ref}")
                logger.info(f"   itrf_to_enu_ref_point: {self.itrf_to_enu_ref_point}")
                
                convert_itrf_to_enu(step_input_dir, logger, self.coord_file, self.itrf_to_enu_extra, self.itrf_to_enu_ref_point, self.max_workers, self.global_ref_point, self.force_global_ref)
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
                
                # DEBUG : Vérification des paramètres du point de référence global pour la déformation
                logger.info(f"🔍 DEBUG - Paramètres transmis à deform_clouds:")
                logger.info(f"   global_ref_point: {self.global_ref_point}")
                logger.info(f"   force_global_ref: {self.force_global_ref}")
                
                deform_clouds(step_input_dir, logger, self.deformation_type, self.deformation_params, self.deform_extra, self.deform_bascule_xml, self.coord_file, self.max_workers, self.global_ref_point, self.force_global_ref)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.deform_output_dir:
                    current_input_dir = self.deform_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), f"deform_{self.deformation_type}_step")
                self.log_signal.emit("Déformation terminée.\n")
                print("Déformation terminée.")
                
            if self.run_orthoimage:
                # Utiliser le dossier d'entrée personnalisé ou le dossier de l'étape précédente
                step_input_dir = self.orthoimage_input_dir if self.orthoimage_input_dir else current_input_dir
                create_orthoimage_from_pointcloud(step_input_dir, logger, self.orthoimage_output_dir, self.orthoimage_resolution, self.orthoimage_height_field, self.orthoimage_color_field, self.max_workers)
                # Utiliser le dossier de sortie personnalisé ou le dossier par défaut
                if self.orthoimage_output_dir:
                    current_input_dir = self.orthoimage_output_dir
                else:
                    current_input_dir = os.path.join(os.path.dirname(step_input_dir), "orthoimage_step")
                self.log_signal.emit("Création d'orthoimage terminée.\n")
                print("Création d'orthoimage terminée.")

            # Étape 4 : Fusion des orthoimages et MNT unifiés
            if self.run_unified_orthoimage:
                # Pour l'orthoimage unifiée, on utilise les orthoimages .tif déjà générées
                # Si un dossier d'entrée personnalisé est spécifié, on l'utilise
                # Sinon, on utilise le dossier de l'étape précédente (qui contient les .tif)
                if self.unified_orthoimage_input_dir:
                    step_input_dir = self.unified_orthoimage_input_dir
                else:
                    # Utiliser le dossier de l'étape précédente qui contient les orthoimages .tif
                    step_input_dir = current_input_dir
                
                # 🎯 FUSION FINALE : Assemblage des orthoimages et MNT unifiés
                # Déterminer le dossier de sortie pour la fusion finale
                if self.unified_orthoimage_output_dir:
                    fusion_output_dir = self.unified_orthoimage_output_dir
                else:
                    fusion_output_dir = os.path.join(os.path.dirname(step_input_dir), "ortho_mnt_unified")
                
                # Appeler la fonction de fusion finale
                from ..core.geodetic_orthoimage_fusion import unified_ortho_mnt_fusion
                # Fusion avec grille automatique et zones paramétrables
                unified_ortho_mnt_fusion(
                    step_input_dir, 
                    logger, 
                    fusion_output_dir, 
                    self.unified_orthoimage_resolution,
                    zone_size_meters=self.zone_size_meters,   # Depuis l'interface
                    max_workers=self.max_workers              # Nombre de workers parallèles
                )
                current_input_dir = fusion_output_dir
                self.log_signal.emit("Fusion des orthoimages et MNT unifiés terminée.\n")
                print("Fusion des orthoimages et MNT unifiés terminée.")

                
            success_msg = "Transformations géodésiques terminées avec succès !"
            self.finished_signal.emit(True, success_msg)
            print(success_msg)
        except Exception as e:
            error_msg = f"Erreur lors des transformations géodésiques : {e}"
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, error_msg)
            print(f"Erreur : {e}") 