#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhotoGeoAlign - Pipeline photogrammétrique automatisé
Interface graphique pour le traitement d'images DNG avec MicMac
"""

import sys
import os
import argparse
import multiprocessing
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QPushButton, QLabel, QFileDialog, QTextEdit, 
                               QProgressBar, QGroupBox, QGridLayout, QMessageBox,
                               QSpinBox, QCheckBox, QLineEdit)
from PySide6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PySide6.QtGui import QPixmap, QIcon, QFont
import subprocess
import threading
import time

# Support pour PyInstaller
if getattr(sys, 'frozen', False):
    multiprocessing.freeze_support()

class PipelineThread(QThread):
    """Thread pour exécuter le pipeline photogrammétrique"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, input_dir, output_dir, nb_proc, use_gpu):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nb_proc = nb_proc
        self.use_gpu = use_gpu
        self.process = None
        
    def run(self):
        try:
            self.progress_signal.emit("Démarrage du pipeline PhotoGeoAlign...")
            
            # Vérification des chemins
            if not os.path.exists(self.input_dir):
                raise Exception(f"Le dossier d'entrée n'existe pas: {self.input_dir}")
            
            # Création du dossier de sortie
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Changement vers le dossier de sortie
            os.chdir(self.output_dir)
            
            # Pipeline MicMac
            self.progress_signal.emit("Étape 1/3: Tapioca - Détection des points d'intérêt...")
            tapioca_cmd = [
                "mm3d", "Tapioca", "MulScale", 
                self.input_dir, 
                "800", "1200", 
                "-1", "Exe", "1"
            ]
            if self.use_gpu:
                tapioca_cmd.extend(["-1", "GPU", "1"])
            
            self.process = subprocess.Popen(
                tapioca_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.progress_signal.emit(line.strip())
            
            self.process.wait()
            if self.process.returncode != 0:
                raise Exception("Erreur lors de l'exécution de Tapioca")
            
            self.progress_signal.emit("Étape 2/3: Tapas - Calcul de l'orientation...")
            tapas_cmd = ["mm3d", "Tapas", "Fraser", "Basic", ".*JPG", "Out=Calib"]
            self.process = subprocess.Popen(
                tapas_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.progress_signal.emit(line.strip())
            
            self.process.wait()
            if self.process.returncode != 0:
                raise Exception("Erreur lors de l'exécution de Tapas")
            
            self.progress_signal.emit("Étape 3/3: C3DC - Génération du nuage de points dense...")
            c3dc_cmd = ["mm3d", "C3DC", ".*JPG", "Calib", "Out=Nuage"]
            if self.use_gpu:
                c3dc_cmd.extend(["-1", "GPU", "1"])
            
            self.process = subprocess.Popen(
                c3dc_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.progress_signal.emit(line.strip())
            
            self.process.wait()
            if self.process.returncode != 0:
                raise Exception("Erreur lors de l'exécution de C3DC")
            
            self.progress_signal.emit("Pipeline PhotoGeoAlign terminé avec succès!")
            self.finished_signal.emit(True, "Traitement terminé avec succès")
            
        except Exception as e:
            self.progress_signal.emit(f"Erreur: {str(e)}")
            self.finished_signal.emit(False, str(e))
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

class PhotoGeoAlignGUI(QMainWindow):
    """Interface graphique principale de PhotoGeoAlign"""
    
    def __init__(self):
        super().__init__()
        self.pipeline_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialisation de l'interface utilisateur"""
        self.setWindowTitle("PhotoGeoAlign - Pipeline Photogrammétrique")
        self.setGeometry(100, 100, 800, 600)
        
        # Icône de l'application
        if os.path.exists("logo.png"):
            self.setWindowIcon(QIcon("logo.png"))
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Logo en haut
        if os.path.exists("logo.png"):
            logo_label = QLabel()
            logo_pixmap = QPixmap("logo.png")
            logo_pixmap = logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(logo_label)
        
        # Titre
        title_label = QLabel("PhotoGeoAlign")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Pipeline photogrammétrique automatisé")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout(config_group)
        
        # Dossier d'entrée
        self.input_label = QLabel("Dossier d'images DNG:")
        self.input_path = QLineEdit()
        self.input_button = QPushButton("Parcourir...")
        self.input_button.clicked.connect(self.select_input_folder)
        config_layout.addWidget(self.input_label, 0, 0)
        config_layout.addWidget(self.input_path, 0, 1)
        config_layout.addWidget(self.input_button, 0, 2)
        
        # Dossier de sortie
        self.output_label = QLabel("Dossier de sortie:")
        self.output_path = QLineEdit()
        self.output_button = QPushButton("Parcourir...")
        self.output_button.clicked.connect(self.select_output_folder)
        config_layout.addWidget(self.output_label, 1, 0)
        config_layout.addWidget(self.output_path, 1, 1)
        config_layout.addWidget(self.output_button, 1, 2)
        
        # Nombre de processus
        self.nb_proc_label = QLabel("Nombre de processus:")
        self.nb_proc_spin = QSpinBox()
        self.nb_proc_spin.setRange(1, 16)
        self.nb_proc_spin.setValue(4)
        config_layout.addWidget(self.nb_proc_label, 2, 0)
        config_layout.addWidget(self.nb_proc_spin, 2, 1)
        
        # Utilisation GPU
        self.gpu_checkbox = QCheckBox("Utiliser GPU (si disponible)")
        self.gpu_checkbox.setChecked(True)
        config_layout.addWidget(self.gpu_checkbox, 2, 2)
        
        main_layout.addWidget(config_group)
        
        # Boutons de contrôle
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Démarrer le pipeline")
        self.start_button.clicked.connect(self.start_pipeline)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; }")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Zone de logs
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
        # Timer pour la barre de progression
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        
    def select_input_folder(self):
        """Sélection du dossier d'images d'entrée"""
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier d'images DNG")
        if folder:
            self.input_path.setText(folder)
            
    def select_output_folder(self):
        """Sélection du dossier de sortie"""
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if folder:
            self.output_path.setText(folder)
            
    def start_pipeline(self):
        """Démarrage du pipeline photogrammétrique"""
        input_dir = self.input_path.text().strip()
        output_dir = self.output_path.text().strip()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner les dossiers d'entrée et de sortie")
            return
            
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "Erreur", "Le dossier d'entrée n'existe pas")
            return
            
        # Vérification de MicMac
        try:
            subprocess.run(["mm3d", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.critical(self, "Erreur", "MicMac n'est pas installé ou n'est pas dans le PATH")
            return
        
        # Désactivation des boutons
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Mode indéterminé
        self.progress_value = 0
        
        # Nettoyage des logs
        self.log_text.clear()
        
        # Création et démarrage du thread
        self.pipeline_thread = PipelineThread(
            input_dir,
            output_dir,
            self.nb_proc_spin.value(),
            self.gpu_checkbox.isChecked()
        )
        self.pipeline_thread.progress_signal.connect(self.update_log)
        self.pipeline_thread.finished_signal.connect(self.pipeline_finished)
        self.pipeline_thread.start()
        
        # Démarrage du timer de progression
        self.progress_timer.start(100)
        
    def stop_pipeline(self):
        """Arrêt du pipeline"""
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self.pipeline_thread.stop()
            self.pipeline_thread.wait()
            self.update_log("Pipeline arrêté par l'utilisateur")
            
    def update_log(self, message):
        """Mise à jour des logs"""
        self.log_text.append(message)
        # Auto-scroll vers le bas
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def update_progress(self):
        """Mise à jour de la barre de progression"""
        self.progress_value += 1
        if self.progress_value > 100:
            self.progress_value = 0
            
    def pipeline_finished(self, success, message):
        """Callback de fin du pipeline"""
        self.progress_timer.stop()
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "Succès", "Pipeline PhotoGeoAlign terminé avec succès!")
        else:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement: {message}")

def run_pipeline_console(input_dir, output_dir, nb_proc, use_gpu):
    """Exécution du pipeline en mode console"""
    print("PhotoGeoAlign - Pipeline photogrammétrique")
    print("=" * 50)
    
    try:
        # Vérification des chemins
        if not os.path.exists(input_dir):
            raise Exception(f"Le dossier d'entrée n'existe pas: {input_dir}")
        
        # Création du dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Changement vers le dossier de sortie
        os.chdir(output_dir)
        
        print("Étape 1/3: Tapioca - Détection des points d'intérêt...")
        tapioca_cmd = [
            "mm3d", "Tapioca", "MulScale", 
            input_dir, 
            "800", "1200", 
            "-1", "Exe", "1"
        ]
        if use_gpu:
            tapioca_cmd.extend(["-1", "GPU", "1"])
        
        result = subprocess.run(tapioca_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Erreur Tapioca: {result.stderr}")
        print("✓ Tapioca terminé")
        
        print("Étape 2/3: Tapas - Calcul de l'orientation...")
        tapas_cmd = ["mm3d", "Tapas", "Fraser", "Basic", ".*JPG", "Out=Calib"]
        result = subprocess.run(tapas_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Erreur Tapas: {result.stderr}")
        print("✓ Tapas terminé")
        
        print("Étape 3/3: C3DC - Génération du nuage de points dense...")
        c3dc_cmd = ["mm3d", "C3DC", ".*JPG", "Calib", "Out=Nuage"]
        if use_gpu:
            c3dc_cmd.extend(["-1", "GPU", "1"])
        
        result = subprocess.run(c3dc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Erreur C3DC: {result.stderr}")
        print("✓ C3DC terminé")
        
        print("\nPipeline PhotoGeoAlign terminé avec succès!")
        print(f"Résultats disponibles dans: {output_dir}")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        sys.exit(1)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="PhotoGeoAlign - Pipeline photogrammétrique automatisé")
    parser.add_argument("--no-gui", action="store_true", help="Mode console sans interface graphique")
    parser.add_argument("--input", help="Dossier d'images DNG (mode console)")
    parser.add_argument("--output", help="Dossier de sortie (mode console)")
    parser.add_argument("--nb-proc", type=int, default=4, help="Nombre de processus (mode console)")
    parser.add_argument("--gpu", action="store_true", help="Utiliser GPU (mode console)")
    
    args = parser.parse_args()
    
    if args.no_gui:
        if not args.input or not args.output:
            print("Erreur: --input et --output sont requis en mode console")
            sys.exit(1)
        run_pipeline_console(args.input, args.output, args.nb_proc, args.gpu)
    else:
        # Mode interface graphique
        app = QApplication(sys.argv)
        app.setApplicationName("PhotoGeoAlign")
        
        # Icône globale
        if os.path.exists("logo.png"):
            app.setWindowIcon(QIcon("logo.png"))
        
        window = PhotoGeoAlignGUI()
        window.show()
        
        sys.exit(app.exec())

if __name__ == "__main__":
    main() 