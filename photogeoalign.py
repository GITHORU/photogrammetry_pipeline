import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit,
    QMessageBox, QSplashScreen, QTabWidget, QCheckBox, QToolBar, QDialog, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QPoint
from PySide6.QtGui import QPixmap, QIcon, QMovie, QAction, QPainter, QColor, QBrush, QPen
import subprocess
import logging
import argparse
import time
from time import sleep
from pathlib import Path
import re
import platform

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
    # 1. Génération des tie points (pipeline)
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
    logger.info("Tapioca terminé. Les tie points .dat sont utilisés pour le pipeline.")

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
    ori = f"{tapas_model}_abs"
    logger.info(f"Lancement de C3DC ({mode}) dans {abs_input_dir} avec le motif {pattern} et Ori={ori} ...")
    cmd = [
        'mm3d', 'C3DC', mode, pattern, ori, f'ZoomF={zoomf}'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info(f"Nuage dense généré par C3DC {mode} (voir dossier PIMs-{mode}/ ou fichier C3DC_{mode}.ply)")

def to_micmac_path(path):
    return path.replace("\\", "/")

def run_micmac_saisieappuisinit(input_dir, logger, tapas_model="Fraser", appuis_file=None, extra_params=""):
    import shutil
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori = tapas_model
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour SaisieAppuisInitQT.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour SaisieAppuisInitQT.")
    appuis_file = os.path.abspath(appuis_file)
    if not os.path.exists(appuis_file):
        logger.error(f"Fichier de coordonnées introuvable : {appuis_file}")
        raise RuntimeError(f"Fichier de coordonnées introuvable : {appuis_file}")
    if not appuis_file.lower().endswith('.txt'):
        logger.error("Le fichier de coordonnées doit être au format .txt")
        raise RuntimeError("Le fichier de coordonnées doit être au format .txt")
    # Conversion systématique en xml
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    logger.info(f"Conversion du fichier de coordonnées TXT en XML avec GCPConvert : {appuis_file} -> {xml_file}")
    cmd_gcp = ['mm3d', 'GCPConvert', 'AppInFile', appuis_file]
    run_command(cmd_gcp, logger, cwd=abs_input_dir)
    if not os.path.exists(xml_file):
        xml_file_candidate = os.path.join(abs_input_dir, os.path.basename(xml_file))
        if os.path.exists(xml_file_candidate):
            xml_file = xml_file_candidate
        else:
            logger.error(f"Le fichier XML n'a pas été généré par GCPConvert : {xml_file}")
            raise RuntimeError(f"Le fichier XML n'a pas été généré par GCPConvert : {xml_file}")
    # Chemin relatif pour MicMac
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    logger.info(f"Lancement de SaisieAppuisInitQT dans {abs_input_dir} sur {pattern} avec Ori={ori}, appuis={xml_file_rel}, sortie=PtsImgInit.xml ...")
    cmd = [
        'mm3d', 'SaisieAppuisInitQT', pattern, ori, xml_file_rel, 'PtsImgInit.xml'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("SaisieAppuisInitQT terminé.")
    return os.path.join(abs_input_dir, "PtsImgInit.xml")

def run_micmac_gcpbascule_init(input_dir, logger, tapas_model="Fraser", appuis_file=None):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori_in = tapas_model
    ori_out = f"{tapas_model}_abs_init"
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour GCPBascule (init).")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour GCPBascule (init).")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimginit_s2d = os.path.join(abs_input_dir, "PtsImgInit-S2D.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    ptsimginit_s2d_rel = os.path.relpath(ptsimginit_s2d, abs_input_dir)
    logger.info(f"Lancement de GCPBascule (init) dans {abs_input_dir} sur {pattern} avec Ori_in={ori_in}, Ori_out={ori_out}, appuis={xml_file_rel}, ptsinit={ptsimginit_s2d_rel} ...")
    cmd = [
        'mm3d', 'GCPBascule', pattern, ori_in, ori_out, xml_file_rel, ptsimginit_s2d_rel
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("GCPBascule (init) terminé.")
    return ori_out

def run_micmac_saisieappuispredic(input_dir, logger, tapas_model="Fraser", ori_abs_init=None, appuis_file=None, extra_params=""):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori = ori_abs_init or f"{tapas_model}_abs_init"  # Utilise l'orientation de sortie de GCPBascule
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour SaisieAppuisPredicQT.")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour SaisieAppuisPredicQT.")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimgpredic_file = os.path.join(abs_input_dir, "PtsImgPredic.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    logger.info(f"Lancement de SaisieAppuisPredicQT dans {abs_input_dir} sur {pattern} avec Ori={ori}, appuis={xml_file_rel}, sortie=PtsImgPredic.xml ...")
    cmd = [
        'mm3d', 'SaisieAppuisPredicQT', pattern, ori, xml_file_rel, 'PtsImgPredic.xml'
    ]
    if extra_params:
        cmd += extra_params.split()
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("SaisieAppuisPredicQT terminé.")
    return ptsimgpredic_file

def run_micmac_gcpbascule_predic(input_dir, logger, tapas_model="Fraser", appuis_file=None):
    abs_input_dir = os.path.abspath(input_dir)
    pattern = '.*DNG'
    ori_in = f"{tapas_model}_abs_init"
    ori_out = f"{tapas_model}_abs"
    if not appuis_file:
        logger.error("Aucun fichier de coordonnées fourni pour GCPBascule (predic).")
        raise RuntimeError("Aucun fichier de coordonnées fourni pour GCPBascule (predic).")
    appuis_file = os.path.abspath(appuis_file)
    xml_file = os.path.splitext(appuis_file)[0] + '.xml'
    ptsimgpredic_s2d = os.path.join(abs_input_dir, "PtsImgPredic-S2D.xml")
    xml_file_rel = os.path.relpath(xml_file, abs_input_dir)
    ptsimgpredic_s2d_rel = os.path.relpath(ptsimgpredic_s2d, abs_input_dir)
    logger.info(f"Lancement de GCPBascule (predic) dans {abs_input_dir} sur {pattern} avec Ori_in={ori_in}, Ori_out={ori_out}, appuis={xml_file_rel}, pts={ptsimgpredic_s2d_rel} ...")
    cmd = [
        'mm3d', 'GCPBascule', pattern, ori_in, ori_out, xml_file_rel, ptsimgpredic_s2d_rel
    ]
    run_command(cmd, logger, cwd=abs_input_dir)
    logger.info("GCPBascule (predic) terminé.")
    return ori_out

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

    def __init__(self, input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, run_tapioca=True, run_tapas=True, run_saisieappuisinit=True, run_saisieappuispredic=True, run_c3dc=True):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf
        self.tapas_model = tapas_model
        self.tapioca_extra = tapioca_extra
        self.tapas_extra = tapas_extra
        self.saisieappuisinit_extra = saisieappuisinit_extra
        self.saisieappuispredic_extra = saisieappuispredic_extra
        self.c3dc_extra = c3dc_extra
        self.saisieappuisinit_pt = saisieappuisinit_pt
        self.run_tapioca = run_tapioca
        self.run_tapas = run_tapas
        self.run_saisieappuisinit = run_saisieappuisinit
        self.run_saisieappuispredic = run_saisieappuispredic
        self.run_c3dc = run_c3dc

    def run(self):
        logger = logging.getLogger(f"PhotogrammetryPipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        try:
            self.log_signal.emit("Démarrage du pipeline...\n")
            if self.run_tapioca:
                run_micmac_tapioca(self.input_dir, logger, self.tapioca_extra)
                self.log_signal.emit("Tapioca terminé.\n")
            if self.run_tapas:
                run_micmac_tapas(self.input_dir, logger, self.tapas_model, self.tapas_extra)
                self.log_signal.emit("Tapas terminé.\n")
            ori_abs_init = None
            if self.run_saisieappuisinit:
                run_micmac_saisieappuisinit(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt, self.saisieappuisinit_extra)
                self.log_signal.emit("SaisieAppuisInitQT terminé.\n")
                ori_abs_init = run_micmac_gcpbascule_init(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (init) terminé.\n")
            if self.run_saisieappuispredic:
                run_micmac_saisieappuispredic(self.input_dir, logger, self.tapas_model, ori_abs_init, self.saisieappuisinit_pt, self.saisieappuispredic_extra)
                self.log_signal.emit("SaisieAppuisPredicQT terminé.\n")
                run_micmac_gcpbascule_predic(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (predic) terminé.\n")
            if self.run_c3dc:
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

class JobExportDialog(QDialog):
    def __init__(self, parent=None, job_name="PhotoGeoAlign", output="PhotoGeoAlign.out", partition="ncpu", ntasks=128, cli_cmd=""):
        super().__init__(parent)
        self.setWindowTitle("Exporter le batch SLURM (.job)")
        self.setModal(True)
        layout = QFormLayout(self)
        self.job_name_edit = QLineEdit(job_name)
        self.output_edit = QLineEdit(output)
        self.partition_combo = QComboBox()
        self.partition_combo.addItems(["ncpu", "ncpum", "ncpu_long", "ncpu_short"])
        self.partition_combo.setCurrentText(partition)
        self.ntasks_spin = QSpinBox()
        self.ntasks_spin.setMinimum(1)
        self.ntasks_spin.setMaximum(1024)
        self.ntasks_spin.setValue(ntasks)
        self.cli_cmd = cli_cmd
        layout.addRow("Nom du job :", self.job_name_edit)
        layout.addRow("Fichier de sortie :", self.output_edit)
        layout.addRow("Partition :", self.partition_combo)
        layout.addRow("Nombre de tâches :", self.ntasks_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_values(self):
        return {
            "job_name": self.job_name_edit.text().strip() or "PhotoGeoAlign",
            "output": self.output_edit.text().strip() or "PhotoGeoAlign.out",
            "partition": self.partition_combo.currentText(),
            "ntasks": self.ntasks_spin.value(),
            "cli_cmd": self.cli_cmd
        }

class PhotogrammetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoGeoAlign")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.pipeline_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        # Barre d'outils
        toolbar = QToolBar()
        # Icône flèche verte pour Lancer
        pixmap_run = QPixmap(24, 24)
        pixmap_run.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_run)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(76, 175, 80)))  # vert
        painter.setPen(Qt.GlobalColor.transparent)
        points = [
            pixmap_run.rect().topLeft() + QPoint(6, 4),
            pixmap_run.rect().bottomLeft() + QPoint(6, -4),
            pixmap_run.rect().center() + QPoint(6, 0)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_run = QIcon(pixmap_run)
        action_run = QAction(icon_run, "Lancer le pipeline", self)
        action_run.triggered.connect(self.launch_pipeline)
        toolbar.addAction(action_run)
        self.action_run = action_run
        # Icône rond rouge pour Arrêter
        pixmap_stop = QPixmap(24, 24)
        pixmap_stop.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_stop)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(211, 47, 47)))  # rouge
        painter.setPen(Qt.GlobalColor.transparent)
        painter.drawEllipse(6, 6, 12, 12)
        painter.end()
        icon_stop = QIcon(pixmap_stop)
        action_stop = QAction(icon_stop, "Arrêter", self)
        action_stop.triggered.connect(self.stop_pipeline)
        toolbar.addAction(action_stop)
        self.action_stop = action_stop
        self.action_stop.setEnabled(False)
        # Icône pour Export .job (flèche vers le bas orange, plus grande)
        pixmap_export = QPixmap(24, 24)
        pixmap_export.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_export)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        orange = QColor(255, 152, 0)  # #ff9800
        painter.setBrush(QBrush(orange))
        painter.setPen(Qt.GlobalColor.transparent)
        # Tige de la flèche (plus large et plus longue)
        painter.drawRect(10, 6, 4, 10)
        # Pointe de la flèche (plus grande)
        points = [
            QPoint(12, 21), QPoint(6, 14), QPoint(18, 14)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_export = QIcon(pixmap_export)
        action_export = QAction(icon_export, "Exporter le batch .job", self)
        action_export.triggered.connect(self.export_job_dialog)
        toolbar.addAction(action_export)
        main_layout.addWidget(toolbar)
        # Tabs
        tabs = QTabWidget()
        # Onglet 1 : paramètres, boutons, sélecteur python, ligne de commande, résumé
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)
        # 1. Dossier image
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Dossier d'images DNG à traiter")
        browse_btn = QPushButton("Parcourir…")
        browse_btn.clicked.connect(self.browse_folder)
        dir_layout.addWidget(QLabel("Dossier d'images :"))
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        param_layout.addLayout(dir_layout)
        # 2. Mode
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["QuickMac", "BigMac", "MicMac"])
        self.mode_combo.setCurrentText("BigMac")
        mode_layout.addWidget(QLabel("Mode C3DC :"))
        mode_layout.addWidget(self.mode_combo)
        param_layout.addLayout(mode_layout)
        # 3. Facteur de zoom
        zoom_layout = QHBoxLayout()
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setMinimum(1)
        self.zoom_spin.setMaximum(8)
        self.zoom_spin.setValue(1)
        zoom_layout.addWidget(QLabel("Facteur de zoom (ZoomF) :"))
        zoom_layout.addWidget(self.zoom_spin)
        param_layout.addLayout(zoom_layout)
        # 4. Modèle Tapas
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
        param_layout.addLayout(tapas_model_layout)
        # 5. Fichier de coordonnées
        pt_layout = QHBoxLayout()
        self.pt_lineedit = QLineEdit()
        self.pt_lineedit.setPlaceholderText("Chemin du fichier de coordonnées (.txt uniquement)")
        self.pt_lineedit.setText("")
        pt_browse_btn = QPushButton("Parcourir…")
        pt_browse_btn.clicked.connect(self.browse_pt_file)
        pt_layout.addWidget(QLabel("Fichier de coordonnées :"))
        pt_layout.addWidget(self.pt_lineedit)
        pt_layout.addWidget(pt_browse_btn)
        param_layout.addLayout(pt_layout)
        # 6. Paramètres supplémentaires + cases à cocher associées
        # Création des cases à cocher d'abord
        # Tapioca
        tapioca_line = QHBoxLayout()
        self.tapioca_cb = QCheckBox("Tapioca")
        self.tapioca_cb.setChecked(True)
        self.tapioca_cb.setMinimumWidth(140)
        self.tapioca_extra = QLineEdit()
        self.tapioca_extra.setPlaceholderText("Paramètres supplémentaires pour Tapioca (optionnel)")
        tapioca_line.addWidget(self.tapioca_cb)
        tapioca_line.addWidget(self.tapioca_extra)
        # Tapas
        tapas_line = QHBoxLayout()
        self.tapas_cb = QCheckBox("Tapas")
        self.tapas_cb.setChecked(True)
        self.tapas_cb.setMinimumWidth(140)
        self.tapas_extra = QLineEdit()
        self.tapas_extra.setPlaceholderText("Paramètres supplémentaires pour Tapas (optionnel)")
        tapas_line.addWidget(self.tapas_cb)
        tapas_line.addWidget(self.tapas_extra)
        # SaisieAppuisInitQT
        saisieappuisinit_line = QHBoxLayout()
        self.saisieappuisinit_cb = QCheckBox("SaisieAppuisInit")
        self.saisieappuisinit_cb.setChecked(True)
        self.saisieappuisinit_cb.setMinimumWidth(140)
        self.saisieappuisinit_extra = QLineEdit()
        self.saisieappuisinit_extra.setPlaceholderText("Paramètres supplémentaires pour SaisieAppuisInitQT (optionnel)")
        saisieappuisinit_line.addWidget(self.saisieappuisinit_cb)
        saisieappuisinit_line.addWidget(self.saisieappuisinit_extra)
        # SaisieAppuisPredicQT
        saisieappuispredic_line = QHBoxLayout()
        self.saisieappuispredic_cb = QCheckBox("SaisieAppuisPredic")
        self.saisieappuispredic_cb.setChecked(True)
        self.saisieappuispredic_cb.setMinimumWidth(140)
        self.saisieappuispredic_extra = QLineEdit()
        self.saisieappuispredic_extra.setPlaceholderText("Paramètres supplémentaires pour SaisieAppuisPredicQT (optionnel)")
        saisieappuispredic_line.addWidget(self.saisieappuispredic_cb)
        saisieappuispredic_line.addWidget(self.saisieappuispredic_extra)
        # C3DC
        c3dc_line = QHBoxLayout()
        self.c3dc_cb = QCheckBox("C3DC")
        self.c3dc_cb.setChecked(True)
        self.c3dc_cb.setMinimumWidth(140)
        self.c3dc_extra = QLineEdit()
        self.c3dc_extra.setPlaceholderText("Paramètres supplémentaires pour C3DC (optionnel)")
        c3dc_line.addWidget(self.c3dc_cb)
        c3dc_line.addWidget(self.c3dc_extra)
        # Ajout bouton à bascule Tout cocher/décocher avec icône dynamique (petit, à gauche, sans texte)
        toggle_btn = QPushButton()
        toggle_btn.setFixedSize(24, 24)
        toggle_btn.setCursor(Qt.PointingHandCursor)
        toggle_btn.setStyleSheet("border: none; padding: 0px;")
        def update_toggle_btn():
            all_checked = all([
                self.tapioca_cb.isChecked(),
                self.tapas_cb.isChecked(),
                self.saisieappuisinit_cb.isChecked(),
                self.saisieappuispredic_cb.isChecked(),
                self.c3dc_cb.isChecked()
            ])
            if all_checked:
                # Icône croix rouge ❌
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(211, 47, 47), 4))
                painter.drawLine(4, 4, 16, 16)
                painter.drawLine(16, 4, 4, 16)
                painter.end()
                toggle_btn.setIcon(QIcon(pixmap))
                toggle_btn.setToolTip("Tout décocher")
            else:
                # Icône coche verte ✅
                pixmap = QPixmap(20, 20)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QPen(QColor(76, 175, 80), 4))
                painter.drawLine(4, 12, 9, 17)
                painter.drawLine(9, 17, 16, 5)
                painter.end()
                toggle_btn.setIcon(QIcon(pixmap))
                toggle_btn.setToolTip("Tout cocher")
        def toggle_all():
            all_checked = all([
                self.tapioca_cb.isChecked(),
                self.tapas_cb.isChecked(),
                self.saisieappuisinit_cb.isChecked(),
                self.saisieappuispredic_cb.isChecked(),
                self.c3dc_cb.isChecked()
            ])
            state = not all_checked
            self.tapioca_cb.setChecked(state)
            self.tapas_cb.setChecked(state)
            self.saisieappuisinit_cb.setChecked(state)
            self.saisieappuispredic_cb.setChecked(state)
            self.c3dc_cb.setChecked(state)
            update_toggle_btn()
        toggle_btn.clicked.connect(toggle_all)
        for cb in [self.tapioca_cb, self.tapas_cb, self.saisieappuisinit_cb, self.saisieappuispredic_cb, self.c3dc_cb]:
            cb.stateChanged.connect(update_toggle_btn)
        # Ajout du bouton dans un layout horizontal collé à gauche
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        toggle_layout.addWidget(toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        param_layout.addLayout(toggle_layout)
        update_toggle_btn()
        # Ajout des cases à cocher au layout
        param_layout.addLayout(tapioca_line)
        param_layout.addLayout(tapas_line)
        param_layout.addLayout(saisieappuisinit_line)
        param_layout.addLayout(saisieappuispredic_line)
        param_layout.addLayout(c3dc_line)
        # 9. Interpréteur Python
        python_layout = QHBoxLayout()
        self.python_selector = QComboBox()
        self.python_selector.addItems(["python", "python3"])
        python_layout.addWidget(QLabel("Interpréteur Python :"))
        python_layout.addWidget(self.python_selector)
        param_layout.addLayout(python_layout)
        # 10. Ligne de commande CLI équivalente
        self.cmd_label = QLabel("Ligne de commande CLI équivalente :")
        param_layout.addWidget(self.cmd_label)
        self.cmd_line = QLineEdit()
        self.cmd_line.setReadOnly(True)
        self.cmd_line.setStyleSheet("font-family: monospace;")
        param_layout.addWidget(self.cmd_line)
        # 11. Résumé et stretch
        self.summary_label = QLabel("")
        param_layout.addWidget(self.summary_label)
        param_layout.addStretch(1)
        tabs.addTab(param_tab, "Paramètres")

        # Onglet 2 : logs
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.addWidget(QLabel("Logs :"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_tab, "Logs")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        main_layout.addLayout(layout)
        self.setLayout(main_layout)
        # Connexions
        self.dir_edit.textChanged.connect(self.update_cmd_line)
        self.mode_combo.currentTextChanged.connect(self.update_cmd_line)
        self.zoom_spin.valueChanged.connect(self.update_cmd_line)
        self.tapas_model_combo.currentTextChanged.connect(self.update_cmd_line)
        self.tapioca_extra.textChanged.connect(self.update_cmd_line)
        self.tapas_extra.textChanged.connect(self.update_cmd_line)
        self.c3dc_extra.textChanged.connect(self.update_cmd_line)
        self.python_selector.currentTextChanged.connect(self.update_cmd_line)
        self.tapioca_cb.stateChanged.connect(self.update_cmd_line)
        self.tapas_cb.stateChanged.connect(self.update_cmd_line)
        self.c3dc_cb.stateChanged.connect(self.update_cmd_line)
        self.saisieappuisinit_cb.stateChanged.connect(self.update_cmd_line)
        self.saisieappuispredic_cb.stateChanged.connect(self.update_cmd_line)
        self.pt_lineedit.textChanged.connect(self.update_cmd_line)
        self.update_cmd_line()

    def update_cmd_line(self):
        input_dir = self.dir_edit.text().strip() or "<dossier_images>"
        mode = self.mode_combo.currentText()
        zoomf = self.zoom_spin.value()
        tapas_model = self.tapas_model_combo.currentText()
        tapioca_extra = self.tapioca_extra.text().strip()
        tapas_extra = self.tapas_extra.text().strip()
        c3dc_extra = self.c3dc_extra.text().strip()
        saisieappuisinit_pt = self.pt_lineedit.text().strip()
        saisieappuisinit_extra = self.saisieappuisinit_extra.text().strip()
        saisieappuispredic_extra = self.saisieappuispredic_extra.text().strip()
        base_cmd = ["photogeoalign.py", "--no-gui", f'\"{input_dir}\"', f"--mode {mode}", f"--tapas-model {tapas_model}", f"--zoomf {zoomf}"]
        if tapioca_extra:
            base_cmd.append(f"--tapioca-extra \"{tapioca_extra}\"")
        if tapas_extra:
            base_cmd.append(f"--tapas-extra \"{tapas_extra}\"")
        if c3dc_extra:
            base_cmd.append(f"--c3dc-extra \"{c3dc_extra}\"")
        if saisieappuisinit_pt:
            base_cmd.append(f"--saisieappuisinit-pt \"{saisieappuisinit_pt}\"")
        if saisieappuisinit_extra:
            base_cmd.append(f"--saisieappuisinit-extra \"{saisieappuisinit_extra}\"")
        if saisieappuispredic_extra:
            base_cmd.append(f"--saisieappuispredic-extra \"{saisieappuispredic_extra}\"")
        # Ajout des options de skip
        if not self.tapioca_cb.isChecked():
            base_cmd.append("--skip-tapioca")
        if not self.tapas_cb.isChecked():
            base_cmd.append("--skip-tapas")
        if not self.saisieappuisinit_cb.isChecked():
            base_cmd.append("--skip-saisieappuisinit")
        if not self.saisieappuispredic_cb.isChecked():
            base_cmd.append("--skip-saisieappuispredic")
        if not self.c3dc_cb.isChecked():
            base_cmd.append("--skip-c3dc")
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.cmd_line.setText(cmd)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'images")
        if folder:
            self.dir_edit.setText(folder)

    def browse_pt_file(self):
        pt_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier de coordonnées (.txt)", "", "Fichiers de coordonnées (*.txt)")
        if pt_file:
            self.pt_lineedit.setText(pt_file)

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
        saisieappuisinit_pt = self.pt_lineedit.text().strip()
        saisieappuisinit_extra = self.saisieappuisinit_extra.text().strip()
        saisieappuispredic_extra = self.saisieappuispredic_extra.text().strip()
        run_tapioca = self.tapioca_cb.isChecked()
        run_tapas = self.tapas_cb.isChecked()
        run_saisieappuisinit = self.saisieappuisinit_cb.isChecked()
        run_saisieappuispredic = self.saisieappuispredic_cb.isChecked()
        run_c3dc = self.c3dc_cb.isChecked()
        # Avertissement si incohérence
        if run_c3dc and not run_tapas:
            self.log_text.append("<span style='color:orange'>Attention : lancer C3DC sans Tapas n'a pas de sens !</span>")
        if run_tapas and not run_tapioca:
            self.log_text.append("<span style='color:orange'>Attention : lancer Tapas sans Tapioca n'a pas de sens !</span>")
        self.log_text.clear()
        self.summary_label.setText("")
        self.action_run.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.pipeline_thread = PipelineThread(input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, run_tapioca, run_tapas, run_saisieappuisinit, run_saisieappuispredic, run_c3dc)
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
        self.action_run.setEnabled(True)
        self.action_stop.setEnabled(False)

    def pipeline_finished(self, success, message):
        if success:
            self.summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.action_run.setEnabled(True)
        self.action_stop.setEnabled(False)

    def export_job_dialog(self):
        import sys
        import os
        exe_path = sys.executable
        script_path = os.path.abspath(__file__)
        cli_cmd = self.cmd_line.text().strip()
        parts = cli_cmd.split()
        # On retire le premier mot (python ou exe)
        args = parts[1:]
        # On retire tout photogeoalign.py
        filtered_args = [arg for arg in args if not arg.endswith('photogeoalign.py') and not arg.endswith('photogeoalign.py"')]
        if getattr(sys, 'frozen', False):
            # Cas exécutable PyInstaller
            cmd = [exe_path] + filtered_args
        else:
            # Cas Python
            cmd = [exe_path, script_path] + filtered_args
        cli_cmd = " ".join(cmd)
        dialog = JobExportDialog(self, cli_cmd=cli_cmd)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vals = dialog.get_values()
            job_content = self.generate_job_script(vals)
            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le script .job", "micmac.job", "Fichiers batch (*.job *.sh)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(job_content)
                    QMessageBox.information(self, "Export réussi", f"Script batch exporté :\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export : {e}")

    def generate_job_script(self, vals):
        # Génère le contenu du script SLURM
        return f"""#!/bin/bash

#SBATCH --job-name {vals['job_name']}
#SBATCH --output {vals['output']}
#SBATCH --nodes=1-1
#SBATCH --partition {vals['partition']} #ncpum,ncpu,ncpulong
#SBATCH --ntasks={vals['ntasks']}

module purge
module load micmac

{vals['cli_cmd']}
"""

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