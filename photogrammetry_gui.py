import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
import subprocess
import logging
import argparse
import time
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
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        if process.stdout is not None:
            for line in process.stdout:
                logger.info(line.rstrip())
        process.wait()
        if process.returncode != 0:
            logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)} (code {process.returncode})")
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande : {' '.join(cmd)}")
        logger.error(f"Code retour : {e.returncode}")
        raise

def run_micmac_tapioca(input_dir, logger):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapioca va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapioca', 'MulScale', pattern, '500', '2700'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    homol_dir = Path(abs_input_dir) / 'Homol'
    if homol_dir.exists() and any(homol_dir.iterdir()):
        logger.info(f"Dossier Homol généré : {homol_dir}")
    else:
        logger.error("Le dossier Homol n'a pas été généré par Tapioca. Arrêt du pipeline.")
        raise RuntimeError("Le dossier Homol n'a pas été généré par Tapioca.")
    logger.info("Tapioca terminé.")

def run_micmac_tapas(input_dir, logger):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Tapas va utiliser les DNG dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'Tapas', 'Fraser', pattern, 'Out=Fraser'
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("Tapas terminé.")

def run_micmac_c3dc(input_dir, logger, mode='QuickMac', zoomf=1):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*.DNG'
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, 'Fraser', f'ZoomF={zoomf}'
    ]
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

    def __init__(self, input_dir, mode, zoomf):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf

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
            run_micmac_tapioca(self.input_dir, logger)
            self.log_signal.emit("Tapioca terminé.\n")
            run_micmac_tapas(self.input_dir, logger)
            self.log_signal.emit("Tapas terminé.\n")
            run_micmac_c3dc(self.input_dir, logger, mode=self.mode, zoomf=self.zoomf)
            self.log_signal.emit("C3DC terminé.\n")
            self.finished_signal.emit(True, "Pipeline terminé avec succès !")
        except Exception as e:
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline : {e}")

class PhotogrammetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photogrammetry Pipeline (MicMac)")
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.init_ui()
        self.pipeline_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Logo en haut, centré
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(logo_label)

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
        self.mode_combo.addItems(["QuickMac", "BigMac"])
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

        # Bouton lancer
        self.run_btn = QPushButton("Lancer le pipeline")
        self.run_btn.clicked.connect(self.launch_pipeline)
        layout.addWidget(self.run_btn)

        # Logs
        layout.addWidget(QLabel("Logs :"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Résumé
        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)

        self.setLayout(layout)

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
        self.log_text.clear()
        self.summary_label.setText("")
        self.run_btn.setEnabled(False)
        self.pipeline_thread = PipelineThread(input_dir, mode, zoomf)
        self.pipeline_thread.log_signal.connect(self.append_log)
        self.pipeline_thread.finished_signal.connect(self.pipeline_finished)
        self.pipeline_thread.start()

    def append_log(self, text):
        self.log_text.append(text)

    def pipeline_finished(self, success, message):
        if success:
            self.summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.run_btn.setEnabled(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photogrammetry Pipeline (MicMac)")
    parser.add_argument('--no-gui', action='store_true', help='Lancer en mode console (sans interface graphique)')
    parser.add_argument('input_dir', nargs='?', default=None, help="Dossier d'images à traiter")
    parser.add_argument('--mode', default='QuickMac', choices=['QuickMac', 'BigMac'], help='Mode de densification C3DC')
    parser.add_argument('--zoomf', type=int, default=1, help='Facteur de zoom (résolution) pour C3DC (1=max)')
    args = parser.parse_args()
    if args.no_gui:
        if not args.input_dir or not os.path.isdir(args.input_dir):
            print("Erreur : veuillez spécifier un dossier d'images valide.")
            sys.exit(1)
        log_path = os.path.join(args.input_dir, 'photogrammetry_pipeline.log')
        logger = setup_logger(log_path)
        print(f"Début du pipeline photogrammétrique pour le dossier : {args.input_dir}")
        try:
            run_micmac_tapioca(args.input_dir, logger)
            run_micmac_tapas(args.input_dir, logger)
            run_micmac_c3dc(args.input_dir, logger, mode=args.mode, zoomf=args.zoomf)
            print("Pipeline terminé avec succès !")
        except Exception as e:
            print(f"Erreur lors de l'exécution du pipeline : {e}")
            sys.exit(1)
    else:
        app = QApplication(sys.argv)
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            app.setWindowIcon(QIcon(logo_path))
        gui = PhotogrammetryGUI()
        gui.show()
        sys.exit(app.exec()) 