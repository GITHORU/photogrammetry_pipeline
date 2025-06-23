import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
import subprocess

class PipelineThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, mode, zoomf):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf

    def run(self):
        cmd = [
            sys.executable, 'photogrammetry_pipeline.py',
            self.input_dir,
            '--mode', self.mode,
            '--zoomf', str(self.zoomf)
        ]
        self.log_signal.emit(f"Commande lancée : {' '.join(cmd)}\n")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            if process.stdout is not None:
                for line in process.stdout:
                    self.log_signal.emit(line)
            process.wait()
            if process.returncode == 0:
                self.finished_signal.emit(True, "Pipeline terminé avec succès !")
            else:
                self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline (code {process.returncode})")
        except Exception as e:
            self.finished_signal.emit(False, f"Exception : {e}")

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
    app = QApplication(sys.argv)
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))
    gui = PhotogrammetryGUI()
    gui.show()
    sys.exit(app.exec()) 