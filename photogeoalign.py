#!/usr/bin/env python3
"""
PhotoGeoAlign - Pipeline de photogrammétrie avec MicMac
Version refactorisée avec structure modulaire
"""

import sys
import os
import argparse
import subprocess
import multiprocessing
from PySide6.QtWidgets import QApplication, QMessageBox, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon, QAction

# Protection freeze_support pour Windows/pyinstaller
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Import rasterio AVANT le patch
try:
    import rasterio
except ImportError:
    rasterio = None

# PATCH RASTERIO UNIVERSEL - Exécuté APRÈS l'import de rasterio
def patch_rasterio_essentials():
    """Patch universel pour les modules rasterio essentiels"""
    if rasterio is None:
        return  # Pas de rasterio, pas de patch
    
    import types
    import logging
    
    # Modules vraiment utilisés dans le code
    essential_modules = [
        'rasterio.sample',    # Utilisé dans process_single_cloud_orthoimage
        'rasterio.vrt',       # Utilisé dans process_single_cloud_orthoimage  
        'rasterio._features', # Erreur actuelle
        'rasterio.coords',    # Utilisé pour BoundingBox
    ]
    
    for module_name in essential_modules:
        try:
            __import__(module_name)
        except ImportError:
            # Créer un module minimal avec seulement ce qui est nécessaire
            module = types.ModuleType(module_name)
            
            # Cas spéciaux pour certains modules
            if module_name == 'rasterio.coords':
                class BoundingBox:
                    def __init__(self, left, bottom, right, top):
                        self.left = left
                        self.bottom = bottom
                        self.right = right
                        self.top = top
                module.BoundingBox = BoundingBox
                logging.getLogger(__name__).warning(f"PATCH: {module_name}.BoundingBox créé")
            
            # Injecter le module dans rasterio
            module_parts = module_name.split('.')
            if len(module_parts) == 2:
                parent_name, child_name = module_parts
                if parent_name in globals():
                    parent = globals()[parent_name]
                    setattr(parent, child_name, module)
            
            logging.getLogger(__name__).warning(f"PATCH: {module_name} créé (module minimal)")

# Appliquer le patch APRÈS l'import de rasterio
patch_rasterio_essentials()

# Import des modules refactorisés
from modules.core.utils import setup_logger, resource_path
from modules.core.micmac import (
    run_micmac_tapioca, run_micmac_tapas, run_micmac_c3dc,
    run_micmac_saisieappuisinit, run_micmac_saisieappuispredic
)
from modules.workers import GeodeticTransformThread
from modules.gui.main_window import PhotogrammetryGUI

def check_micmac_or_quit():
    try:
        result = subprocess.run(["mm3d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 and "usage" not in result.stdout.lower() and "usage" not in result.stderr.lower():
            raise FileNotFoundError
    except Exception:
        app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            msg.setWindowIcon(QIcon(logo_path))
            msg.setIconPixmap(QPixmap(logo_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        msg.setWindowTitle("MicMac non détecté")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.setText("Le logiciel MicMac (mm3d) n'est pas installé ou n'est pas accessible dans le PATH système.<br><br>" +
                   "<b>Si vous êtes sous Linux, il est recommandé de lancer PhotoGeoAlign depuis un terminal pour garantir que mm3d soit bien trouvé dans le PATH.</b><br><br>" +
                   "Veuillez suivre la documentation d'installation :<br>"
                   "<a href='https://micmac.ensg.eu/index.php/Install'>https://micmac.ensg.eu/index.php/Install</a><br><br>"
                   "Projet GitHub officiel :<br>"
                   "<a href='https://github.com/micmacIGN/micmac'>https://github.com/micmacIGN/micmac</a>")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)

main_window = None  # Variable globale pour conserver la fenêtre principale

if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_micmac_or_quit()
        # Lancement GUI pur (aucun argument)
        app = QApplication(sys.argv)
        # Splash screen statique avec le logo
        splash = QLabel()
        splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        splash.setStyleSheet("background: transparent; border: none;")
        logo_path = resource_path("logo.png")
        pixmap = QPixmap(logo_path)
        # Redimensionne le logo à 300px de large (hauteur ajustée automatiquement)
        target_width = 300
        scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.TransformationMode.SmoothTransformation)
        splash.setPixmap(scaled_pixmap)
        splash.setFixedSize(scaled_pixmap.size())
        splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splash.show()
        app.processEvents()
        def show_main():
            global main_window
            logo_path = resource_path("logo.png")
            if os.path.exists(logo_path):
                app.setWindowIcon(QIcon(logo_path))
            main_window = PhotogrammetryGUI()
            main_window.show()
            splash.close()
        QTimer.singleShot(3000, show_main)
        sys.exit(app.exec())
    else:
        parser = argparse.ArgumentParser(description="Photogrammetry Pipeline (MicMac)")
        parser.add_argument('--no-gui', action='store_true', help='Lancer en mode console (sans interface graphique)')
        parser.add_argument('input_dir', nargs='?', default=None, help="Dossier d'images à traiter")
        parser.add_argument('--mode', default='BigMac', choices=['QuickMac', 'BigMac', 'MicMac'], help='Mode de densification C3DC (défaut: BigMac)')
        parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (résolution) pour C3DC (1=max)')
        parser.add_argument('--tapas-model', default='Fraser', help='Modèle Tapas à utiliser (défaut: Fraser)')
        parser.add_argument('--tapioca-extra', default='', help='Paramètres supplémentaires pour Tapioca (optionnel)')
        parser.add_argument('--tapas-extra', default='', help='Paramètres supplémentaires pour Tapas (optionnel)')
        parser.add_argument('--c3dc-extra', default='', help='Paramètres supplémentaires pour C3DC (optionnel)')
        parser.add_argument('--saisieappuisinit-pt', default='', help='Chemin du fichier de points d\'appui pour SaisieAppuisInit (optionnel)')
        parser.add_argument('--saisieappuisinit-extra', default='', help='Paramètres supplémentaires pour SaisieAppuisInitQT (optionnel)')
        parser.add_argument('--saisieappuispredic-extra', default='', help='Paramètres supplémentaires pour SaisieAppuisPredicQT (optionnel)')
        parser.add_argument('--skip-saisieappuisinit', action='store_true', help='Ne pas exécuter SaisieAppuisInit')
        parser.add_argument('--skip-saisieappuispredic', action='store_true', help='Ne pas exécuter SaisieAppuisPredic')
        parser.add_argument('--skip-tapioca', action='store_true', help='Ne pas exécuter Tapioca')
        parser.add_argument('--skip-tapas', action='store_true', help='Ne pas exécuter Tapas')
        parser.add_argument('--skip-c3dc', action='store_true', help='Ne pas exécuter C3DC')
        
        # Arguments pour les transformations géodésiques
        parser.add_argument('--geodetic', action='store_true', help='Lancer les transformations géodésiques')
        parser.add_argument('--geodetic-coord', default='', help='Fichier de coordonnées de recalage pour les transformations géodésiques')
        parser.add_argument('--deformation-type', default='tps', choices=['tps'], help='Type de déformation (défaut: tps)')
        parser.add_argument('--deformation-params', default='', help='Paramètres de déformation (optionnel)')
        parser.add_argument('--add-offset-extra', default='', help='Paramètres supplémentaires pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-extra', default='', help='Paramètres supplémentaires pour ITRF vers ENU (optionnel)')
        parser.add_argument('--itrf-to-enu-ref-point', default='', help='Nom du point de référence pour ITRF vers ENU (optionnel, utilise le premier point si non spécifié)')
        parser.add_argument('--deform-extra', default='', help='Paramètres supplémentaires pour la déformation (optionnel)')
        parser.add_argument('--deform-bascule-xml', default='', help='Fichier XML GCPBascule pour la déformation (optionnel)')

        parser.add_argument('--skip-add-offset', action='store_true', help='Ne pas exécuter l\'ajout d\'offset')
        parser.add_argument('--skip-itrf-to-enu', action='store_true', help='Ne pas exécuter la conversion ITRF vers ENU')
        parser.add_argument('--skip-deform', action='store_true', help='Ne pas exécuter la déformation')
        parser.add_argument('--skip-orthoimage', action='store_true', help='Ne pas exécuter la création d\'orthoimage')
        parser.add_argument('--skip-unified-orthoimage', action='store_true', help='Ne pas exécuter la création d\'orthoimage et MNT unifiés')

        
        # Arguments pour les dossiers d'entrée personnalisés
        parser.add_argument('--add-offset-input-dir', default='', help='Dossier d\'entrée personnalisé pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-input-dir', default='', help='Dossier d\'entrée personnalisé pour ITRF vers ENU (optionnel)')
        parser.add_argument('--deform-input-dir', default='', help='Dossier d\'entrée personnalisé pour la déformation (optionnel)')
        parser.add_argument('--orthoimage-input-dir', default='', help='Dossier d\'entrée personnalisé pour l\'orthoimage (optionnel)')
        parser.add_argument('--unified-orthoimage-input-dir', default='', help='Dossier d\'entrée personnalisé pour l\'orthoimage unifiée (optionnel)')

        
        # Arguments pour les dossiers de sortie personnalisés
        parser.add_argument('--add-offset-output-dir', default='', help='Dossier de sortie personnalisé pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-output-dir', default='', help='Dossier de sortie personnalisé pour ITRF vers ENU (optionnel)')
        parser.add_argument('--deform-output-dir', default='', help='Dossier de sortie personnalisé pour la déformation (optionnel)')
        parser.add_argument('--orthoimage-output-dir', default='', help='Dossier de sortie personnalisé pour l\'orthoimage (optionnel)')
        parser.add_argument('--unified-orthoimage-output-dir', default='', help='Dossier de sortie personnalisé pour l\'orthoimage unifiée (optionnel)')

        
        # Arguments pour les paramètres d'orthoimage
        parser.add_argument('--orthoimage-resolution', type=float, default=0.1, help='Résolution de l\'orthoimage en mètres (défaut: 0.1)')
        parser.add_argument('--unified-orthoimage-resolution', type=float, default=0.1, help='Résolution de l\'orthoimage unifiée en mètres (défaut: 0.1)')
        
        # Arguments pour la méthode de fusion des couleurs
        parser.add_argument('--color-fusion-median', action='store_true', help='Utiliser la méthode de médiane pour la fusion des couleurs')

        parser.add_argument('--max-workers', type=int, default=4, help='Nombre maximum de processus parallèles (défaut: 4)')
    
        args = parser.parse_args()
        if args.geodetic:
            # Mode transformations géodésiques
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spécifier un dossier de nuages valide.")
                sys.exit(1)
            if not args.geodetic_coord or not os.path.exists(args.geodetic_coord):
                print("Erreur : veuillez spécifier un fichier de coordonnées valide.")
                sys.exit(1)
            
            log_path = os.path.join(args.input_dir, 'geodetic_transforms.log')
            logger = setup_logger(log_path)
            print(f"Début des transformations géodésiques pour le dossier : {args.input_dir}")
            
            try:
                coord_file = args.geodetic_coord
                deformation_type = args.deformation_type
                deformation_params = args.deformation_params
                add_offset_extra = args.add_offset_extra
                itrf_to_enu_extra = args.itrf_to_enu_extra
                itrf_to_enu_ref_point = args.itrf_to_enu_ref_point if args.itrf_to_enu_ref_point else None
                deform_extra = args.deform_extra
                deform_bascule_xml = args.deform_bascule_xml if args.deform_bascule_xml else None

                
                # Dossiers d'entrée personnalisés
                add_offset_input_dir = args.add_offset_input_dir if args.add_offset_input_dir else None
                itrf_to_enu_input_dir = args.itrf_to_enu_input_dir if args.itrf_to_enu_input_dir else None
                deform_input_dir = args.deform_input_dir if args.deform_input_dir else None
                orthoimage_input_dir = args.orthoimage_input_dir if args.orthoimage_input_dir else None
                unified_orthoimage_input_dir = args.unified_orthoimage_input_dir if args.unified_orthoimage_input_dir else None

                
                # Dossiers de sortie personnalisés
                add_offset_output_dir = args.add_offset_output_dir if args.add_offset_output_dir else None
                itrf_to_enu_output_dir = args.itrf_to_enu_output_dir if args.itrf_to_enu_output_dir else None
                deform_output_dir = args.deform_output_dir if args.deform_output_dir else None
                orthoimage_output_dir = args.orthoimage_output_dir if args.orthoimage_output_dir else None
                unified_orthoimage_output_dir = args.unified_orthoimage_output_dir if args.unified_orthoimage_output_dir else None

                
                run_add_offset = not args.skip_add_offset
                run_itrf_to_enu = not args.skip_itrf_to_enu
                run_deform = not args.skip_deform
                run_orthoimage = not args.skip_orthoimage
                run_unified_orthoimage = not args.skip_unified_orthoimage

                
                # Méthode de fusion des couleurs
                color_fusion_method = "median" if args.color_fusion_median else "average"
                
                # Création d'une instance du thread pour gérer les dossiers d'entrée/sortie
                geodetic_thread = GeodeticTransformThread(
                    args.input_dir, coord_file, deformation_type, deformation_params,
                    add_offset_extra, itrf_to_enu_extra, deform_extra,
                    run_add_offset, run_itrf_to_enu, run_deform, run_orthoimage, run_unified_orthoimage,
                    add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, orthoimage_input_dir, unified_orthoimage_input_dir,
                    add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, orthoimage_output_dir, unified_orthoimage_output_dir,
                    itrf_to_enu_ref_point, deform_bascule_xml, args.orthoimage_resolution, "z", "rgb", args.unified_orthoimage_resolution, args.max_workers, color_fusion_method
                )
                
                # Exécution des transformations
                geodetic_thread.run()
                
                print("Transformations géodésiques terminées avec succès !")
            except Exception as e:
                print(f"Erreur lors des transformations géodésiques : {e}")
                sys.exit(1)
        elif args.no_gui:
            # Mode pipeline photogrammétrique
            check_micmac_or_quit()
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spécifier un dossier d'images valide.")
                sys.exit(1)
            log_path = os.path.join(args.input_dir, 'photogrammetry_pipeline.log')
            logger = setup_logger(log_path)
            print(f"Début du pipeline photogrammétrique pour le dossier : {args.input_dir}")
            try:
                tapas_model = args.tapas_model
                saisieappuisinit_pt = args.saisieappuisinit_pt or None
                saisieappuisinit_extra = args.saisieappuisinit_extra
                saisieappuispredic_extra = args.saisieappuispredic_extra
                run_saisieappuisinit = not args.skip_saisieappuisinit
                run_saisieappuispredic = not args.skip_saisieappuispredic
                if not args.skip_tapioca:
                    run_micmac_tapioca(args.input_dir, logger, args.tapioca_extra)
                if not args.skip_tapas:
                    run_micmac_tapas(args.input_dir, logger, tapas_model, args.tapas_extra)
                if run_saisieappuisinit:
                    run_micmac_saisieappuisinit(args.input_dir, logger, tapas_model, saisieappuisinit_pt, saisieappuisinit_extra)
                if run_saisieappuispredic:
                    run_micmac_saisieappuispredic(args.input_dir, logger, tapas_model, saisieappuisinit_pt, saisieappuispredic_extra)
                if not args.skip_c3dc:
                    run_micmac_c3dc(args.input_dir, logger, mode=args.mode, zoomf=args.zoomf, tapas_model=tapas_model, extra_params=args.c3dc_extra)
                print("Pipeline terminé avec succès !")
            except Exception as e:
                print(f"Erreur lors de l'exécution du pipeline : {e}")
                sys.exit(1)
        else:
            check_micmac_or_quit()
            app = QApplication(sys.argv)
            logo_path = resource_path("logo.png")
            if os.path.exists(logo_path):
                app.setWindowIcon(QIcon(logo_path))
            gui = PhotogrammetryGUI()
            gui.show()
            sys.exit(app.exec()) 