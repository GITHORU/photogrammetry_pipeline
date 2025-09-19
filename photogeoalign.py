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
import traceback

# Protection contre les conflits PROJ/GDAL système sur cluster
if getattr(sys, 'frozen', False):
    # Si on est dans un exécutable PyInstaller, nettoyer les variables d'environnement système
    # qui pourraient interférer avec les bibliothèques embarquées
    # Peut être désactivé avec PHOTOGEOSKIP_PROJ_PROTECTION=1
    if os.environ.get('PHOTOGEOSKIP_PROJ_PROTECTION', '').lower() not in ['1', 'true', 'yes']:
        env_vars_to_clean = ['PROJ_LIB', 'GDAL_DATA', 'GDAL_DRIVER_PATH']
        for var in env_vars_to_clean:
            if var in os.environ:
                # Garder une copie de la valeur système pour debug si nécessaire
                sys_val = os.environ[var]
                print(f"[INFO] Protection PROJ: suppression de {var}={sys_val}")
                # Nettoyer pour forcer l'utilisation des données de l'exécutable
                del os.environ[var]
    else:
        print("[INFO] Protection PROJ désactivée par PHOTOGEOSKIP_PROJ_PROTECTION")

from PySide6.QtWidgets import QApplication, QMessageBox, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon, QAction

# Protection freeze_support pour Windows/pyinstaller
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Gestionnaire d'exception global pour capturer les erreurs de formatage
def global_exception_handler(exctype, value, traceback_obj):
    if exctype == ValueError and "Unknown format code 'f' for object of type 'str'" in str(value):
        print(f"❌ ERREUR DE FORMATAGE DÉTECTÉE:")
        print(f"   Type: {exctype}")
        print(f"   Message: {value}")
        print(f"   Fichier: {traceback_obj.tb_frame.f_code.co_filename}")
        print(f"   Ligne: {traceback_obj.tb_lineno}")
        print(f"   Fonction: {traceback_obj.tb_frame.f_code.co_name}")
        print(f"   Traceback complet:")
        traceback.print_tb(traceback_obj)
        sys.exit(1)
    else:
        # Comportement par défaut pour les autres exceptions
        sys.__excepthook__(exctype, value, traceback_obj)

# Installation du gestionnaire d'exception global
sys.excepthook = global_exception_handler

# Import des modules refactorisés
from modules.core.utils import setup_logger, resource_path

# Initialisation GDAL/PROJ pour environnements packagés (exécutables)
def _init_gdal_env():
    """Initialise l'environnement GDAL/PROJ pour éviter les conflits de versions"""
    try:
        import rasterio
        
        # FORCER l'utilisation des données de l'exécutable (priorité sur système)
        gdal_data = getattr(rasterio, 'gdal_data', None)
        if gdal_data and os.path.exists(gdal_data):
            os.environ['GDAL_DATA'] = gdal_data
        
        # FORCER l'utilisation des données PROJ de l'exécutable
        try:
            import pyproj
            proj_data = pyproj.datadir.get_data_dir()
            if proj_data and os.path.exists(proj_data):
                os.environ['PROJ_LIB'] = proj_data
        except Exception:
            pass
            
        # Désactiver les chemins système qui pourraient interférer
        if 'PROJ_LIB' in os.environ and getattr(sys, 'frozen', False):
            # Si on est dans un exécutable, s'assurer que PROJ_LIB pointe vers les données de l'exécutable
            current_proj_lib = os.environ['PROJ_LIB']
            if not current_proj_lib.startswith(sys._MEIPASS):
                # Récupérer le chemin depuis pyproj
                try:
                    import pyproj
                    proj_data = pyproj.datadir.get_data_dir()
                    if proj_data and os.path.exists(proj_data):
                        os.environ['PROJ_LIB'] = proj_data
                except Exception:
                    pass
                    
    except Exception:
        pass

_init_gdal_env()
from modules.core.micmac import (
    run_micmac_tapioca, run_micmac_tapas, run_micmac_c3dc,
    run_micmac_saisieappuisinit, run_micmac_saisieappuispredic
)
from modules.core.analysis import run_analysis_pipeline
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
        # Redimensionne le logo à  300px de large (hauteur ajustée automatiquement)
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
        parser.add_argument('input_dir', nargs='?', default=None, help="Dossier d'images à  traiter")
        parser.add_argument('--mode', default='BigMac', choices=['QuickMac', 'BigMac', 'MicMac'], help='Mode de densification C3DC (défaut: BigMac)')
        parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (résolution) pour C3DC (1=max)')
        parser.add_argument('--tapas-model', default='Fraser', help='Modèle Tapas à  utiliser (défaut: Fraser)')
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
        parser.add_argument('--deformation-type', default='none', choices=['none', 'tps', 'radial'], help='Type de déformation (none, tps, radial)')
        parser.add_argument('--deformation-params', default='', help='Paramètres de déformation (optionnel)')
        parser.add_argument('--add-offset-extra', default='', help='Paramètres supplémentaires pour l\'ajout d\'offset (optionnel)')
        parser.add_argument('--itrf-to-enu-extra', default='', help='Paramètres supplémentaires pour ITRF vers ENU (optionnel)')
        parser.add_argument('--itrf-to-enu-ref-point', default='', help='Nom du point de référence pour ITRF vers ENU (optionnel, utilise le premier point si non spécifié)')
        parser.add_argument('--global-ref-point', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Point de référence global X Y Z en mètres (ITRF) pour unifier le repère ENU')
        parser.add_argument('--force-global-ref', action='store_true', help='Forcer l\'utilisation du point de référence global au lieu du point local')
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
        
        # Arguments pour les paramètres de taille de grille et de zones

        parser.add_argument('--zone-size', type=float, default=5.0, help='Taille de chaque zone en mètres (défaut: 5.0)')
        
        # Arguments pour la méthode de fusion des couleurs
        parser.add_argument('--color-fusion-median', action='store_true', help='Utiliser la méthode de médiane pour la fusion des couleurs')

        parser.add_argument('--max-workers', type=int, default=4, help='Nombre maximum de processus parallèles (défaut: 4)')
        
        # Arguments pour le pipeline d'analyse
        parser.add_argument('--analysis', action='store_true', help='Lancer le pipeline d\'analyse')
        parser.add_argument('--type', choices=['mnt', 'ortho', 'mnt_ortho'], default='mnt', help='Type d\'analyse : mnt, ortho ou mnt_ortho (défaut: mnt)')
        parser.add_argument('--image1', default='', help='Chemin vers l\'image 1 pour l\'analyse')
        parser.add_argument('--image2', default='', help='Chemin vers l\'image 2 pour l\'analyse')
        parser.add_argument('--mnt1', default='', help='Chemin vers le MNT 1 pour l\'analyse mnt_ortho')
        parser.add_argument('--mnt2', default='', help='Chemin vers le MNT 2 pour l\'analyse mnt_ortho')
        parser.add_argument('--resolution', type=float, default=10.0, help='Résolution d\'analyse en mètres (défaut: 10.0)')
        
        # Arguments pour les paramètres Farneback (configuration optimisée par défaut)
        parser.add_argument('--pyr-scale', type=float, default=0.8, help='Facteur d\'échelle de la pyramide (défaut: 0.8)')
        parser.add_argument('--levels', type=int, default=5, help='Nombre de niveaux de la pyramide (défaut: 5)')
        parser.add_argument('--winsize', type=int, default=101, help='Taille de la fenêtre de recherche (défaut: 101, adapté automatiquement)')
        parser.add_argument('--iterations', type=int, default=10, help='Nombre d\'itérations (défaut: 10)')
        parser.add_argument('--poly-n', type=int, default=7, help='Taille du filtre polynomial (défaut: 7)')
        parser.add_argument('--poly-sigma', type=float, default=1.2, help='Écart-type du filtre polynomial (défaut: 1.2)')
        
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
                
                # Point de référence global
                global_ref_point = None
                force_global_ref = False
                if args.global_ref_point and args.force_global_ref:
                    global_ref_point = list(args.global_ref_point)  # Convertir tuple en list
                    force_global_ref = True
                    print(f"Point de référence global forcé : X={global_ref_point[0]:.3f}, Y={global_ref_point[1]:.3f}, Z={global_ref_point[2]:.3f}")
                
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
                
                # Paramètre de taille des zones
                zone_size = args.zone_size
                
                # Création d'une instance du thread pour gérer les dossiers d'entrée/sortie
                geodetic_thread = GeodeticTransformThread(
                    args.input_dir, coord_file, deformation_type, deformation_params,
                    add_offset_extra, itrf_to_enu_extra, deform_extra,
                    run_add_offset, run_itrf_to_enu, run_deform, run_orthoimage, run_unified_orthoimage,
                    add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, orthoimage_input_dir, unified_orthoimage_input_dir,
                    add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, orthoimage_output_dir, unified_orthoimage_output_dir,
                    itrf_to_enu_ref_point, deform_bascule_xml, args.orthoimage_resolution, "z", "rgb", args.unified_orthoimage_resolution, args.max_workers, color_fusion_method,
                    zone_size, global_ref_point, force_global_ref
                )
                
                # Exécution des transformations
                geodetic_thread.run()
                
                print("Transformations géodésiques terminées avec succès !")
            except Exception as e:
                print(f"Erreur lors des transformations géodésiques : {str(e)}")
                sys.exit(1)
        elif args.analysis:
            # Mode pipeline d'analyse
            if not args.image1 or not os.path.exists(args.image1):
                print("Erreur : veuillez spécifier un fichier image1 valide.")
                sys.exit(1)
            if not args.image2 or not os.path.exists(args.image2):
                print("Erreur : veuillez spécifier un fichier image2 valide.")
                sys.exit(1)
            
            # Validation spécifique pour le mode mnt_ortho
            if args.type == 'mnt_ortho':
                if not args.mnt1 or not os.path.exists(args.mnt1):
                    print("Erreur : veuillez spécifier un fichier mnt1 valide pour le mode mnt_ortho.")
                    sys.exit(1)
                if not args.mnt2 or not os.path.exists(args.mnt2):
                    print("Erreur : veuillez spécifier un fichier mnt2 valide pour le mode mnt_ortho.")
                    sys.exit(1)
            
            log_path = os.path.join(os.path.dirname(args.image1), 'analysis_pipeline.log')
            logger = setup_logger(log_path)
            if args.type == 'mnt_ortho':
                print(f"Début du pipeline d'analyse 3D pour :")
                print(f"  - Images : {args.image1} et {args.image2}")
                print(f"  - MNTs : {args.mnt1} et {args.mnt2}")
            else:
                print(f"Début du pipeline d'analyse pour les images : {args.image1} et {args.image2}")
            
            try:
                analysis_type = args.type
                image1_path = args.image1
                image2_path = args.image2
                mnt1_path = args.mnt1 if args.type == 'mnt_ortho' else None
                mnt2_path = args.mnt2 if args.type == 'mnt_ortho' else None
                resolution = args.resolution
                
                # Paramètres Farneback (seulement si spécifiés explicitement)
                farneback_params = None
                if (args.pyr_scale != 0.8 or args.levels != 5 or args.winsize != 101 or 
                    args.iterations != 10 or args.poly_n != 7 or args.poly_sigma != 1.2):
                    # Paramètres personnalisés fournis
                    farneback_params = {
                        'pyr_scale': args.pyr_scale,
                        'levels': args.levels,
                        'winsize': args.winsize,  # Sera adapté automatiquement
                        'iterations': args.iterations,
                        'poly_n': args.poly_n,
                        'poly_sigma': args.poly_sigma
                    }
                
                print(f"Type d'analyse : {analysis_type}")
                print(f"Image 1 : {image1_path}")
                print(f"Image 2 : {image2_path}")
                print(f"Résolution : {resolution} m")
                if farneback_params:
                    print(f"Paramètres Farneback personnalisés : {farneback_params}")
                else:
                    print("Paramètres Farneback : Configuration optimale automatique")
                    # Afficher les paramètres optimisés qui seront utilisés
                    base_config = {
                        'pyr_scale': 0.8, 'levels': 5, 'winsize': 101,
                        'iterations': 10, 'poly_n': 7, 'poly_sigma': 1.2
                    }
                    ratio = 0.01 / resolution
                    adapted_winsize = max(3, int(101 * ratio))
                    if adapted_winsize % 2 == 0:
                        adapted_winsize += 1
                    print(f"  - pyr_scale: {base_config['pyr_scale']} (constant)")
                    print(f"  - levels: {base_config['levels']} (constant)")
                    print(f"  - winsize: {adapted_winsize} (adapté: 101 * {ratio:.2f} = {101 * ratio:.0f})")
                    print(f"  - iterations: {base_config['iterations']} (constant)")
                    print(f"  - poly_n: {base_config['poly_n']} (constant)")
                    print(f"  - poly_sigma: {base_config['poly_sigma']} (constant)")
                
                # Exécution du pipeline d'analyse
                output_dir = os.path.join(os.path.dirname(image1_path), 'analysis_results')
                results = run_analysis_pipeline(image1_path, image2_path, analysis_type, resolution, output_dir, farneback_params, mnt1_path, mnt2_path)
                
                if results:
                    print(f"RMSE: {results.get('rmse', 'N/A')}")
                    print(f"Corrélation Pearson: {results.get('correlation_pearson', 'N/A')}")
                    print(f"Nombre de points: {results.get('n_points', 'N/A')}")
                    if 'report_path' in results:
                        print(f"Rapport généré: {results['report_path']}")
                
                print("Pipeline d'analyse terminé avec succès !")
            except Exception as e:
                print(f"Erreur lors de l'exécution du pipeline d'analyse : {str(e)}")
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
                print(f"Erreur lors de l'exécution du pipeline : {str(e)}")
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
