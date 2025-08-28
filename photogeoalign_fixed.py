ï»¿#!/usr/bin/env python3
"""
PhotoGeoAlign - Pipeline de photogrammÃÂ©trie avec MicMac
Version refactorisÃÂ©e avec structure modulaire
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

# Import des modules refactorisÃÂ©s
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
        msg.setWindowTitle("MicMac non dÃÂ©tectÃÂ©")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.setText("Le logiciel MicMac (mm3d) n'est pas installÃÂ© ou n'est pas accessible dans le PATH systÃÂ¨me.<br><br>" +
                   "<b>Si vous ÃÂªtes sous Linux, il est recommandÃÂ© de lancer PhotoGeoAlign depuis un terminal pour garantir que mm3d soit bien trouvÃÂ© dans le PATH.</b><br><br>" +
                   "Veuillez suivre la documentation d'installation :<br>"
                   "<a href='https://micmac.ensg.eu/index.php/Install'>https://micmac.ensg.eu/index.php/Install</a><br><br>"
                   "Projet GitHub officiel :<br>"
                   "<a href='https://github.com/micmacIGN/micmac'>https://github.com/micmacIGN/micmac</a>")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)

main_window = None  # Variable globale pour conserver la fenÃÂªtre principale

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
        # Redimensionne le logo ÃÂ  300px de large (hauteur ajustÃÂ©e automatiquement)
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
        parser.add_argument('input_dir', nargs='?', default=None, help="Dossier d'images ÃÂ  traiter")
        parser.add_argument('--mode', default='BigMac', choices=['QuickMac', 'BigMac', 'MicMac'], help='Mode de densification C3DC (dÃÂ©faut: BigMac)')
        parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (rÃÂ©solution) pour C3DC (1=max)')
        parser.add_argument('--tapas-model', default='Fraser', help='ModÃÂ¨le Tapas ÃÂ  utiliser (dÃÂ©faut: Fraser)')
        parser.add_argument('--tapioca-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour Tapioca (optionnel)')
        parser.add_argument('--tapas-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour Tapas (optionnel)')
        parser.add_argument('--c3dc-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour C3DC (optionnel)')
        parser.add_argument('--saisieappuisinit-pt', default='', help='Chemin du fichier de points d\'appui pour SaisieAppuisInit (optionnel)')
        parser.add_argument('--saisieappuisinit-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour SaisieAppuisInitQT (optionnel)')
        parser.add_argument('--saisieappuispredic-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour SaisieAppuisPredicQT (optionnel)')
        parser.add_argument('--skip-saisieappuisinit', action='store_true', help='Ne pas exÃÂ©cuter SaisieAppuisInit')
        parser.add_argument('--skip-saisieappuispredic', action='store_true', help='Ne pas exÃÂ©cuter SaisieAppuisPredic')
        parser.add_argument('--skip-tapioca', action='store_true', help='Ne pas exÃÂ©cuter Tapioca')
        parser.add_argument('--skip-tapas', action='store_true', help='Ne pas exÃÂ©cuter Tapas')
        parser.add_argument('--skip-c3dc', action='store_true', help='Ne pas exÃÂ©cuter C3DC')
        
        # Arguments pour les transformations gÃÂ©odÃÂ©siques
        parser.add_argument('--geodetic', action='store_true', help='Lancer les transformations gÃÂ©odÃÂ©siques')
        parser.add_argument('--geodetic-coord', default='', help='Fichier de coordonnÃÂ©es de recalage pour les transformations gÃÂ©odÃÂ©siques')
        parser.add_argument('--deformation-type', default='tps', choices=['tps'], help='Type de dÃÂ©formation (dÃÂ©faut: tps)')
        parser.add_argument('--deformation-params', default='', help='ParamÃÂ¨tres de dÃÂ©formation (optionnel)')
        parser.add_argument('--add-offset-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour ITRF vers ENU (optionnel)')
        parser.add_argument('--itrf-to-enu-ref-point', default='', help='Nom du point de rÃÂ©fÃÂ©rence pour ITRF vers ENU (optionnel, utilise le premier point si non spÃÂ©cifiÃÂ©)')
        parser.add_argument('--global-ref-point', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Point de rÃÂ©fÃÂ©rence global X Y Z en mÃÂ¨tres (ITRF) pour unifier le repÃÂ¨re ENU')
        parser.add_argument('--force-global-ref', action='store_true', help='Forcer l\'utilisation du point de rÃÂ©fÃÂ©rence global au lieu du point local')
        parser.add_argument('--deform-extra', default='', help='ParamÃÂ¨tres supplÃÂ©mentaires pour la dÃÂ©formation (optionnel)')
        parser.add_argument('--deform-bascule-xml', default='', help='Fichier XML GCPBascule pour la dÃÂ©formation (optionnel)')

        parser.add_argument('--skip-add-offset', action='store_true', help='Ne pas exÃÂ©cuter l\'ajout d\'offset')
        parser.add_argument('--skip-itrf-to-enu', action='store_true', help='Ne pas exÃÂ©cuter la conversion ITRF vers ENU')
        parser.add_argument('--skip-deform', action='store_true', help='Ne pas exÃÂ©cuter la dÃÂ©formation')
        parser.add_argument('--skip-orthoimage', action='store_true', help='Ne pas exÃÂ©cuter la crÃÂ©ation d\'orthoimage')
        parser.add_argument('--skip-unified-orthoimage', action='store_true', help='Ne pas exÃÂ©cuter la crÃÂ©ation d\'orthoimage et MNT unifiÃÂ©s')

        
        # Arguments pour les dossiers d'entrÃÂ©e personnalisÃÂ©s
        parser.add_argument('--add-offset-input-dir', default='', help='Dossier d\'entrÃÂ©e personnalisÃÂ© pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-input-dir', default='', help='Dossier d\'entrÃÂ©e personnalisÃÂ© pour ITRF vers ENU (optionnel)')
        parser.add_argument('--deform-input-dir', default='', help='Dossier d\'entrÃÂ©e personnalisÃÂ© pour la dÃÂ©formation (optionnel)')
        parser.add_argument('--orthoimage-input-dir', default='', help='Dossier d\'entrÃÂ©e personnalisÃÂ© pour l\'orthoimage (optionnel)')
        parser.add_argument('--unified-orthoimage-input-dir', default='', help='Dossier d\'entrÃÂ©e personnalisÃÂ© pour l\'orthoimage unifiÃÂ©e (optionnel)')

        
        # Arguments pour les dossiers de sortie personnalisÃÂ©s
        parser.add_argument('--add-offset-output-dir', default='', help='Dossier de sortie personnalisÃÂ© pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-output-dir', default='', help='Dossier de sortie personnalisÃÂ© pour ITRF vers ENU (optionnel)')
        parser.add_argument('--deform-output-dir', default='', help='Dossier de sortie personnalisÃÂ© pour la dÃÂ©formation (optionnel)')
        parser.add_argument('--orthoimage-output-dir', default='', help='Dossier de sortie personnalisÃÂ© pour l\'orthoimage (optionnel)')
        parser.add_argument('--unified-orthoimage-output-dir', default='', help='Dossier de sortie personnalisÃÂ© pour l\'orthoimage unifiÃÂ©e (optionnel)')

        
        # Arguments pour les paramÃÂ¨tres d'orthoimage
        parser.add_argument('--orthoimage-resolution', type=float, default=0.1, help='RÃÂ©solution de l\'orthoimage en mÃÂ¨tres (dÃÂ©faut: 0.1)')
        parser.add_argument('--unified-orthoimage-resolution', type=float, default=0.1, help='RÃÂ©solution de l\'orthoimage unifiÃÂ©e en mÃÂ¨tres (dÃÂ©faut: 0.1)')
        
        # Arguments pour les paramÃÂ¨tres de taille de grille et de zones

        parser.add_argument('--zone-size', type=float, default=5.0, help='Taille de chaque zone en mÃÂ¨tres (dÃÂ©faut: 5.0)')
        
        # Arguments pour la mÃÂ©thode de fusion des couleurs
        parser.add_argument('--color-fusion-median', action='store_true', help='Utiliser la mÃÂ©thode de mÃÂ©diane pour la fusion des couleurs')

        parser.add_argument('--max-workers', type=int, default=4, help='Nombre maximum de processus parallÃÂ¨les (dÃÂ©faut: 4)')
    
        args = parser.parse_args()
        if args.geodetic:
            # Mode transformations gÃÂ©odÃÂ©siques
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spÃÂ©cifier un dossier de nuages valide.")
                sys.exit(1)
            if not args.geodetic_coord or not os.path.exists(args.geodetic_coord):
                print("Erreur : veuillez spÃÂ©cifier un fichier de coordonnÃÂ©es valide.")
                sys.exit(1)
            
            log_path = os.path.join(args.input_dir, 'geodetic_transforms.log')
            logger = setup_logger(log_path)
            print(f"DÃÂ©but des transformations gÃÂ©odÃÂ©siques pour le dossier : {args.input_dir}")
            
            try:
                coord_file = args.geodetic_coord
                deformation_type = args.deformation_type
                deformation_params = args.deformation_params
                add_offset_extra = args.add_offset_extra
                itrf_to_enu_extra = args.itrf_to_enu_extra
                itrf_to_enu_ref_point = args.itrf_to_enu_ref_point if args.itrf_to_enu_ref_point else None
                
                # Point de rÃ©fÃ©rence global
                global_ref_point = None
                force_global_ref = False
                if args.global_ref_point and args.force_global_ref:
                    global_ref_point = list(args.global_ref_point)  # Convertir tuple en list
                    force_global_ref = True
                    print(f"Point de rÃ©fÃ©rence global forcÃ© : X={global_ref_point[0]:.3f}, Y={global_ref_point[1]:.3f}, Z={global_ref_point[2]:.3f}")
                
                deform_extra = args.deform_extra
                deform_bascule_xml = args.deform_bascule_xml if args.deform_bascule_xml else None

                
                # Dossiers d'entrÃÂ©e personnalisÃÂ©s
                add_offset_input_dir = args.add_offset_input_dir if args.add_offset_input_dir else None
                itrf_to_enu_input_dir = args.itrf_to_enu_input_dir if args.itrf_to_enu_input_dir else None
                deform_input_dir = args.deform_input_dir if args.deform_input_dir else None
                orthoimage_input_dir = args.orthoimage_input_dir if args.orthoimage_input_dir else None
                unified_orthoimage_input_dir = args.unified_orthoimage_input_dir if args.unified_orthoimage_input_dir else None

                
                # Dossiers de sortie personnalisÃÂ©s
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

                
                # MÃÂ©thode de fusion des couleurs
                color_fusion_method = "median" if args.color_fusion_median else "average"
                
                # ParamÃ¨tre de taille des zones
                zone_size = args.zone_size
                
                # CrÃÂ©ation d'une instance du thread pour gÃÂ©rer les dossiers d'entrÃÂ©e/sortie
                geodetic_thread = GeodeticTransformThread(
                    args.input_dir, coord_file, deformation_type, deformation_params,
                    add_offset_extra, itrf_to_enu_extra, deform_extra,
                    run_add_offset, run_itrf_to_enu, run_deform, run_orthoimage, run_unified_orthoimage,
                    add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, orthoimage_input_dir, unified_orthoimage_input_dir,
                    add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, orthoimage_output_dir, unified_orthoimage_output_dir,
                    itrf_to_enu_ref_point, deform_bascule_xml, args.orthoimage_resolution, "z", "rgb", args.unified_orthoimage_resolution, args.max_workers, color_fusion_method,
                    zone_size, global_ref_point, force_global_ref
                )
                
                # ExÃÂ©cution des transformations
                geodetic_thread.run()
                
                print("Transformations gÃÂ©odÃÂ©siques terminÃÂ©es avec succÃÂ¨s !")
            except Exception as e:
                print(f"Erreur lors des transformations gÃÂ©odÃÂ©siques : {e}")
                sys.exit(1)
        elif args.no_gui:
            # Mode pipeline photogrammÃÂ©trique
            check_micmac_or_quit()
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spÃÂ©cifier un dossier d'images valide.")
                sys.exit(1)
            log_path = os.path.join(args.input_dir, 'photogrammetry_pipeline.log')
            logger = setup_logger(log_path)
            print(f"DÃÂ©but du pipeline photogrammÃÂ©trique pour le dossier : {args.input_dir}")
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
                print("Pipeline terminÃÂ© avec succÃÂ¨s !")
            except Exception as e:
                print(f"Erreur lors de l'exÃÂ©cution du pipeline : {e}")
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

# Cache invalidation - 2025-08-27 15:00:00
