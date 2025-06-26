import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit,
    QMessageBox, QSplashScreen
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon, QMovie
import subprocess
import logging
import argparse
import time
from time import sleep
from pathlib import Path
import re

# Protection freeze_support pour Windows/pyinstaller
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

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

def run_micmac_tapioca(input_dir, logger, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapioca va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    homol_dir = Path(abs_input_dir) / 'Homol'
    if homol_dir.exists() and any(homol_dir.iterdir()):
        logger.info(f"Dossier Homol généré : {homol_dir}")
    else:
        logger.error("Le dossier Homol n'a pas été généré par Tapioca. Arrêt du pipeline.")
        raise RuntimeError("Le dossier Homol n'a pas été généré par Tapioca.")
    logger.info("Tapioca terminé.")

def run_micmac_tapas(input_dir, logger, tapas_model="Fraser", extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapas va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapas', tapas_model, pattern, f'Out={tapas_model}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(input_dir, logger, mode='QuickMac', zoomf=1, tapas_model='Fraser', extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, tapas_model, f'ZoomF={zoomf}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"Nuage dense généré par C3DC {mode} (voir dossier PIMs-{mode}/ ou fichier C3DC_{mode}.ply)")

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class PipelineThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, c3dc_extra):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf
        self.tapas_model = tapas_model
        self.tapioca_extra = tapioca_extra
        self.tapas_extra = tapas_extra
        self.c3dc_extra = c3dc_extra

    def run(self):
        # Logger pour la GUI : uniquement dans la zone de logs
        logger = logging.getLogger(f"PhotogrammetryPipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        # Supprime les handlers existants (évite les doublons)
        logger.handlers = []
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        try:
            self.log_signal.emit("Démarrage du pipeline...\n")
            run_micmac_tapioca(self.input_dir, logger, self.tapioca_extra)
            self.log_signal.emit("Tapioca terminé.\n")
            run_micmac_tapas(self.input_dir, logger, self.tapas_model, self.tapas_extra)
            self.log_signal.emit("Tapas terminé.\n")
            run_micmac_c3dc(self.input_dir, logger, mode=self.mode, zoomf=self.zoomf, tapas_model=self.tapas_model, extra_params=self.c3dc_extra)
            self.log_signal.emit("C3DC terminé.\n")
            self.finished_signal.emit(True, "Pipeline terminé avec succès !")
        except Exception as e:
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline : {e}")

def resource_path(relative_path):
    """Trouve le chemin absolu d'une ressource, compatible PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

class PhotogrammetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoGeoAlign")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.init_ui()
        self.pipeline_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Sélection dossier
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Dossier d'images DNG à traiter")
        browse_btn = QPushButton("Parcourir…")
        browse_btn.clicked.connect(self.browse_folder)
        dir_layout.addWidget(QLabel("Dossier d'images :"))
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        # Mode
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["QuickMac", "BigMac", "MicMac"])
        self.mode_combo.setCurrentText("BigMac")
        mode_layout.addWidget(QLabel("Mode C3DC :"))
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # ZoomF
        zoom_layout = QHBoxLayout()
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setMinimum(1)
        self.zoom_spin.setMaximum(8)
        self.zoom_spin.setValue(1)
        zoom_layout.addWidget(QLabel("Facteur de zoom (ZoomF) :"))
        zoom_layout.addWidget(self.zoom_spin)
        layout.addLayout(zoom_layout)

        # Modèle Tapas
        tapas_model_layout = QHBoxLayout()
        self.tapas_model_combo = QComboBox()
        self.tapas_model_combo.addItems([
            "RadialBasic",
            "RadialExtended",
            "Fraser",
            "FishEyeEqui",
            "AutoCal",
            "Figee",
            "HemiEqui",
            "RadialStd",
            "FraserBasic",
            "FishEyeBasic",
            "FE_EquiSolBasic",
            "Four7x2",
            "Four11x2",
            "Four15x2",
            "Four19x2",
            "AddFour7x2",
            "AddFour11x2",
            "AddFour15x2",
            "AddFour19x2",
            "AddPolyDeg0",
            "AddPolyDeg1",
            "AddPolyDeg2",
            "AddPolyDeg3",
            "AddPolyDeg4",
            "AddPolyDeg5",
            "AddPolyDeg6",
            "AddPolyDeg7",
            "Ebner",
            "Brown",
            "FishEyeStereo"
        ])
        self.tapas_model_combo.setCurrentText("Fraser")
        tapas_model_layout.addWidget(QLabel("Modèle Tapas :"))
        tapas_model_layout.addWidget(self.tapas_model_combo)
        layout.addLayout(tapas_model_layout)

        # Paramètres supplémentaires MicMac
        extra_layout = QVBoxLayout()
        self.tapioca_extra = QLineEdit()
        self.tapioca_extra.setPlaceholderText("Paramètres supplémentaires pour Tapioca (optionnel)")
        extra_layout.addWidget(self.tapioca_extra)
        self.tapas_extra = QLineEdit()
        self.tapas_extra.setPlaceholderText("Paramètres supplémentaires pour Tapas (optionnel)")
        extra_layout.addWidget(self.tapas_extra)
        self.c3dc_extra = QLineEdit()
        self.c3dc_extra.setPlaceholderText("Paramètres supplémentaires pour C3DC (optionnel)")
        extra_layout.addWidget(self.c3dc_extra)
        layout.addLayout(extra_layout)

        # Bouton lancer
        self.run_btn = QPushButton("Lancer le pipeline")
        self.run_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:enabled:hover {
                background-color: #388E3C;
            }
        """)
        self.run_btn.clicked.connect(self.launch_pipeline)
        layout.addWidget(self.run_btn)

        # Bouton arrêter
        self.stop_btn = QPushButton("Arrêter")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #f44336;
                color: white;
                font-weight: bold;
            }
            QPushButton:enabled:hover {
                background-color: #b71c1c;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        layout.addWidget(self.stop_btn)

        # Logs
        layout.addWidget(QLabel("Logs :"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Résumé
        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)

        # Ligne de commande équivalente (non éditable)
        self.cmd_label = QLabel("Ligne de commande CLI équivalente :")
        layout.addWidget(self.cmd_label)
        self.cmd_line = QLineEdit()
        self.cmd_line.setReadOnly(True)
        self.cmd_line.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.cmd_line)

        self.setLayout(layout)

        # Connecte les changements de paramètres à la mise à jour de la ligne de commande
        self.dir_edit.textChanged.connect(self.update_cmd_line)
        self.mode_combo.currentTextChanged.connect(self.update_cmd_line)
        self.zoom_spin.valueChanged.connect(self.update_cmd_line)
        self.tapas_model_combo.currentTextChanged.connect(self.update_cmd_line)
        self.tapioca_extra.textChanged.connect(self.update_cmd_line)
        self.tapas_extra.textChanged.connect(self.update_cmd_line)
        self.c3dc_extra.textChanged.connect(self.update_cmd_line)
        self.update_cmd_line()

    def update_cmd_line(self):
        input_dir = self.dir_edit.text().strip() or "<dossier_images>"
        mode = self.mode_combo.currentText()
        zoomf = self.zoom_spin.value()
        tapas_model = self.tapas_model_combo.currentText()
        tapioca_extra = self.tapioca_extra.text().strip()
        tapas_extra = self.tapas_extra.text().strip()
        c3dc_extra = self.c3dc_extra.text().strip()
        cmd = ["python photogeoalign.py", "--no-gui", f'"{input_dir}"', f"--mode {mode}", f"--tapas-model {tapas_model}", f"--zoomf {zoomf}"]
        if tapioca_extra:
            cmd.append(f"--tapioca-extra \"{tapioca_extra}\"")
        if tapas_extra:
            cmd.append(f"--tapas-extra \"{tapas_extra}\"")
        if c3dc_extra:
            cmd.append(f"--c3dc-extra \"{c3dc_extra}\"")
        self.cmd_line.setText(" ".join(cmd))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'images")
        if folder:
            self.dir_edit.setText(folder)

    def launch_pipeline(self):
        input_dir = self.dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un dossier valide.</span>")
            return
        mode = self.mode_combo.currentText()
        zoomf = self.zoom_spin.value()
        tapas_model = self.tapas_model_combo.currentText()
        tapioca_extra = self.tapioca_extra.text().strip()
        tapas_extra = self.tapas_extra.text().strip()
        c3dc_extra = self.c3dc_extra.text().strip()
        self.log_text.clear()
        self.summary_label.setText("")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pipeline_thread = PipelineThread(input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, c3dc_extra)
        self.pipeline_thread.log_signal.connect(self.append_log)
        self.pipeline_thread.finished_signal.connect(self.pipeline_finished)
        self.pipeline_thread.start()

    def append_log(self, text):
        # Recherche du format : '2025-06-24 13:11:38,200 - INFO - message'
        m = re.match(r"^(.*? - )([A-Z]+)( - )(.*)$", text)
        if m:
            timestamp = m.group(1)
            level = m.group(2)
            sep = m.group(3)
            message = m.group(4)
            if level == "INFO":
                color = "#1565c0"  # bleu
            elif level in ("WARNING", "WARN"): 
                color = "#ff9800"  # orange
            elif level == "ERROR":
                color = "#d32f2f"  # rouge
            else:
                color = "#8e24aa"  # violet
            html = f"<span>{timestamp}</span><span style='color:{color};'>{level}</span><span>{sep}</span><span style='font-weight:bold;color:{color};'>{message}</span>"
        else:
            # Pas de format reconnu, affiche en violet
            html = f"<span style='font-weight:bold;color:#8e24aa'>{text}</span>"
        self.log_text.append(html)

    def stop_pipeline(self):
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self.pipeline_thread.terminate()
            self.pipeline_thread.wait()
            self.append_log("<span style='color:red'>Pipeline arrêté par l'utilisateur.</span>")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def pipeline_finished(self, success, message):
        if success:
            self.summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

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
        args = parser.parse_args()
        if args.no_gui:
            check_micmac_or_quit()
            if not args.input_dir or not os.path.isdir(args.input_dir):
                print("Erreur : veuillez spécifier un dossier d'images valide.")
                sys.exit(1)
            log_path = os.path.join(args.input_dir, 'photogrammetry_pipeline.log')
            logger = setup_logger(log_path)
            print(f"Début du pipeline photogrammétrique pour le dossier : {args.input_dir}")
            try:
                tapas_model = args.tapas_model
                run_micmac_tapioca(args.input_dir, logger, args.tapioca_extra)
                run_micmac_tapas(args.input_dir, logger, tapas_model, args.tapas_extra)
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