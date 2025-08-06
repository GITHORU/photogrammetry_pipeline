import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QLineEdit,
    QMessageBox, QTabWidget, QCheckBox, QToolBar
)
from PySide6.QtGui import QPixmap, QIcon, QPainter, QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt, QTimer, QPoint
from ..core.utils import resource_path
from ..workers import PipelineThread, GeodeticTransformThread
from .dialogs import JobExportDialog

class PhotogrammetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoGeoAlign")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.pipeline_thread = None
        self.geodetic_thread = None
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
        action_run = QAction(icon_run, "Lancer le pipeline MicMac", self)
        action_run.triggered.connect(self.launch_pipeline)
        toolbar.addAction(action_run)
        self.action_run = action_run
        
        # Icône flèche bleue pour Lancer le pipeline géodésique
        pixmap_geodetic = QPixmap(24, 24)
        pixmap_geodetic.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_geodetic)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(33, 150, 243)))  # bleu
        painter.setPen(Qt.GlobalColor.transparent)
        points = [
            pixmap_geodetic.rect().topLeft() + QPoint(6, 4),
            pixmap_geodetic.rect().bottomLeft() + QPoint(6, -4),
            pixmap_geodetic.rect().center() + QPoint(6, 0)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_geodetic = QIcon(pixmap_geodetic)
        action_geodetic = QAction(icon_geodetic, "Lancer le pipeline géodésique", self)
        action_geodetic.triggered.connect(self.launch_geodetic_pipeline)
        toolbar.addAction(action_geodetic)
        self.action_geodetic = action_geodetic
        
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
        tabs.addTab(param_tab, "MicMac")

        # Onglet 2 : Transformations géodésiques
        geodetic_tab = QWidget()
        geodetic_layout = QVBoxLayout(geodetic_tab)
        
        # 1. Dossier de travail
        geodetic_dir_layout = QHBoxLayout()
        self.geodetic_dir_edit = QLineEdit()
        self.geodetic_dir_edit.setPlaceholderText("Dossier contenant les nuages .ply à traiter")
        geodetic_browse_btn = QPushButton("Parcourir…")
        geodetic_browse_btn.clicked.connect(self.browse_geodetic_folder)
        geodetic_dir_layout.addWidget(QLabel("Dossier de travail :"))
        geodetic_dir_layout.addWidget(self.geodetic_dir_edit)
        geodetic_dir_layout.addWidget(geodetic_browse_btn)
        geodetic_layout.addLayout(geodetic_dir_layout)
        
        # 2. Fichier de coordonnées de recalage
        geodetic_coord_layout = QHBoxLayout()
        self.geodetic_coord_edit = QLineEdit()
        self.geodetic_coord_edit.setPlaceholderText("Chemin du fichier de coordonnées de recalage (.txt)")
        geodetic_coord_browse_btn = QPushButton("Parcourir…")
        geodetic_coord_browse_btn.clicked.connect(self.browse_geodetic_coord_file)
        geodetic_coord_layout.addWidget(QLabel("Fichier de coordonnées :"))
        geodetic_coord_layout.addWidget(self.geodetic_coord_edit)
        geodetic_coord_layout.addWidget(geodetic_coord_browse_btn)
        geodetic_layout.addLayout(geodetic_coord_layout)
        
        # 3. Type de déformation
        deformation_layout = QHBoxLayout()
        self.deformation_combo = QComboBox()
        self.deformation_combo.addItems(["tps"])
        self.deformation_combo.setCurrentText("tps")
        deformation_layout.addWidget(QLabel("Type de déformation :"))
        deformation_layout.addWidget(self.deformation_combo)
        geodetic_layout.addLayout(deformation_layout)
        
        # 4. Paramètres de déformation
        deformation_params_layout = QHBoxLayout()
        self.deformation_params_edit = QLineEdit()
        self.deformation_params_edit.setPlaceholderText("Paramètres de déformation (optionnel)")
        deformation_params_layout.addWidget(QLabel("Paramètres :"))
        deformation_params_layout.addWidget(self.deformation_params_edit)
        geodetic_layout.addLayout(deformation_params_layout)
        
        # 4.5. Fichier XML GCPBascule pour la déformation
        bascule_xml_layout = QHBoxLayout()
        self.bascule_xml_edit = QLineEdit()
        self.bascule_xml_edit.setPlaceholderText("Chemin du fichier XML GCPBascule (.xml)")
        bascule_xml_browse_btn = QPushButton("Parcourir…")
        bascule_xml_browse_btn.clicked.connect(self.browse_bascule_xml_file)
        bascule_xml_layout.addWidget(QLabel("Fichier XML GCPBascule :"))
        bascule_xml_layout.addWidget(self.bascule_xml_edit)
        bascule_xml_layout.addWidget(bascule_xml_browse_btn)
        geodetic_layout.addLayout(bascule_xml_layout)
        
        # 4.6. Nombre de processus parallèles
        parallel_layout = QHBoxLayout()
        self.parallel_workers_spin = QSpinBox()
        self.parallel_workers_spin.setMinimum(1)
        self.parallel_workers_spin.setMaximum(128)
        self.parallel_workers_spin.setValue(10)
        self.parallel_workers_spin.setToolTip("Nombre de processus parallèles pour le traitement des nuages (1-128)")
        parallel_layout.addWidget(QLabel("Processus parallèles :"))
        parallel_layout.addWidget(self.parallel_workers_spin)
        parallel_layout.addStretch()
        geodetic_layout.addLayout(parallel_layout)
        
        # 5. Bouton tout cocher/décocher pour les transformations géodésiques
        geodetic_toggle_btn = QPushButton()
        geodetic_toggle_btn.setFixedSize(24, 24)
        geodetic_toggle_btn.setCursor(Qt.PointingHandCursor)
        geodetic_toggle_btn.setStyleSheet("border: none; padding: 0px;")
        
        def update_geodetic_toggle_btn():
            all_checked = all([
                self.add_offset_cb.isChecked(),
                self.itrf_to_enu_cb.isChecked(),
                self.deform_cb.isChecked(),
                self.enu_to_itrf_cb.isChecked()
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
                geodetic_toggle_btn.setIcon(QIcon(pixmap))
                geodetic_toggle_btn.setToolTip("Tout décocher")
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
                geodetic_toggle_btn.setIcon(QIcon(pixmap))
                geodetic_toggle_btn.setToolTip("Tout cocher")
        
        def toggle_all_geodetic():
            all_checked = all([
                self.add_offset_cb.isChecked(),
                self.itrf_to_enu_cb.isChecked(),
                self.deform_cb.isChecked(),
                self.enu_to_itrf_cb.isChecked()
            ])
            state = not all_checked
            self.add_offset_cb.setChecked(state)
            self.itrf_to_enu_cb.setChecked(state)
            self.deform_cb.setChecked(state)
            self.enu_to_itrf_cb.setChecked(state)
            update_geodetic_toggle_btn()
        
        geodetic_toggle_btn.clicked.connect(toggle_all_geodetic)
        
        # Ajout du bouton dans un layout horizontal collé à gauche
        geodetic_toggle_layout = QHBoxLayout()
        geodetic_toggle_layout.setContentsMargins(0, 0, 0, 0)
        geodetic_toggle_layout.setSpacing(0)
        geodetic_toggle_layout.addWidget(geodetic_toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        geodetic_layout.addLayout(geodetic_toggle_layout)
        
        # 6. Cases à cocher pour les étapes
        # Ajout offset
        add_offset_line = QHBoxLayout()
        self.add_offset_cb = QCheckBox("Ajout offset")
        self.add_offset_cb.setChecked(True)
        self.add_offset_cb.setMinimumWidth(140)
        self.add_offset_extra = QLineEdit()
        self.add_offset_extra.setPlaceholderText("Paramètres supplémentaires pour l'ajout d'offset (optionnel)")
        add_offset_line.addWidget(self.add_offset_cb)
        add_offset_line.addWidget(self.add_offset_extra)
        
        # Input/Output for Add Offset
        add_offset_input_layout = QHBoxLayout()
        self.add_offset_input_edit = QLineEdit()
        self.add_offset_input_edit.setPlaceholderText("Dossier d'entrée (vide = dossier principal)")
        add_offset_input_layout.addWidget(QLabel("Entrée :"))
        add_offset_input_layout.addWidget(self.add_offset_input_edit)
        self.add_offset_input_browse_btn = QPushButton("Parcourir")
        self.add_offset_input_browse_btn.clicked.connect(self.browse_add_offset_input_dir)
        add_offset_input_layout.addWidget(self.add_offset_input_browse_btn)

        add_offset_output_layout = QHBoxLayout()
        self.add_offset_output_edit = QLineEdit()
        self.add_offset_output_edit.setPlaceholderText("Dossier de sortie (vide = offset_step)")
        add_offset_output_layout.addWidget(QLabel("Sortie :"))
        add_offset_output_layout.addWidget(self.add_offset_output_edit)
        self.add_offset_output_browse_btn = QPushButton("Parcourir")
        self.add_offset_output_browse_btn.clicked.connect(self.browse_add_offset_output_dir)
        add_offset_output_layout.addWidget(self.add_offset_output_browse_btn)

        geodetic_layout.addLayout(add_offset_line)
        geodetic_layout.addLayout(add_offset_input_layout)
        geodetic_layout.addLayout(add_offset_output_layout)
        
        # ITRF vers ENU
        itrf_to_enu_line = QHBoxLayout()
        self.itrf_to_enu_cb = QCheckBox("ITRF → ENU")
        self.itrf_to_enu_cb.setChecked(True)
        self.itrf_to_enu_cb.setMinimumWidth(140)
        self.itrf_to_enu_extra = QLineEdit()
        self.itrf_to_enu_extra.setPlaceholderText("Paramètres supplémentaires pour ITRF→ENU (optionnel)")
        itrf_to_enu_line.addWidget(self.itrf_to_enu_cb)
        itrf_to_enu_line.addWidget(self.itrf_to_enu_extra)

        # Input/Output for ITRF to ENU
        itrf_to_enu_input_layout = QHBoxLayout()
        self.itrf_to_enu_input_edit = QLineEdit()
        self.itrf_to_enu_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        itrf_to_enu_input_layout.addWidget(QLabel("Entrée :"))
        itrf_to_enu_input_layout.addWidget(self.itrf_to_enu_input_edit)
        self.itrf_to_enu_input_browse_btn = QPushButton("Parcourir")
        self.itrf_to_enu_input_browse_btn.clicked.connect(self.browse_itrf_to_enu_input_dir)
        itrf_to_enu_input_layout.addWidget(self.itrf_to_enu_input_browse_btn)

        itrf_to_enu_output_layout = QHBoxLayout()
        self.itrf_to_enu_output_edit = QLineEdit()
        self.itrf_to_enu_output_edit.setPlaceholderText("Dossier de sortie (vide = itrf_to_enu_step)")
        itrf_to_enu_output_layout.addWidget(QLabel("Sortie :"))
        itrf_to_enu_output_layout.addWidget(self.itrf_to_enu_output_edit)
        self.itrf_to_enu_output_browse_btn = QPushButton("Parcourir")
        self.itrf_to_enu_output_browse_btn.clicked.connect(self.browse_itrf_to_enu_output_dir)
        itrf_to_enu_output_layout.addWidget(self.itrf_to_enu_output_browse_btn)

        geodetic_layout.addLayout(itrf_to_enu_line)
        geodetic_layout.addLayout(itrf_to_enu_input_layout)
        geodetic_layout.addLayout(itrf_to_enu_output_layout)
        
        # Déformation
        deform_line = QHBoxLayout()
        self.deform_cb = QCheckBox("Déformation")
        self.deform_cb.setChecked(True)
        self.deform_cb.setMinimumWidth(140)
        self.deform_extra = QLineEdit()
        self.deform_extra.setPlaceholderText("Paramètres supplémentaires pour la déformation (optionnel)")
        deform_line.addWidget(self.deform_cb)
        deform_line.addWidget(self.deform_extra)

        # Input/Output for Deformation
        deform_input_layout = QHBoxLayout()
        self.deform_input_edit = QLineEdit()
        self.deform_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        deform_input_layout.addWidget(QLabel("Entrée :"))
        deform_input_layout.addWidget(self.deform_input_edit)
        self.deform_input_browse_btn = QPushButton("Parcourir")
        self.deform_input_browse_btn.clicked.connect(self.browse_deform_input_dir)
        deform_input_layout.addWidget(self.deform_input_browse_btn)

        deform_output_layout = QHBoxLayout()
        self.deform_output_edit = QLineEdit()
        self.deform_output_edit.setPlaceholderText("Dossier de sortie (vide = deform_[type]_step)")
        deform_output_layout.addWidget(QLabel("Sortie :"))
        deform_output_layout.addWidget(self.deform_output_edit)
        self.deform_output_browse_btn = QPushButton("Parcourir")
        self.deform_output_browse_btn.clicked.connect(self.browse_deform_output_dir)
        deform_output_layout.addWidget(self.deform_output_browse_btn)

        geodetic_layout.addLayout(deform_line)
        geodetic_layout.addLayout(deform_input_layout)
        geodetic_layout.addLayout(deform_output_layout)
        
        # ENU vers ITRF
        enu_to_itrf_line = QHBoxLayout()
        self.enu_to_itrf_cb = QCheckBox("ENU → ITRF")
        self.enu_to_itrf_cb.setChecked(True)
        self.enu_to_itrf_cb.setMinimumWidth(140)
        self.enu_to_itrf_extra = QLineEdit()
        self.enu_to_itrf_extra.setPlaceholderText("Paramètres supplémentaires pour ENU→ITRF (optionnel)")
        enu_to_itrf_line.addWidget(self.enu_to_itrf_cb)
        enu_to_itrf_line.addWidget(self.enu_to_itrf_extra)

        # Input/Output for ENU to ITRF
        enu_to_itrf_input_layout = QHBoxLayout()
        self.enu_to_itrf_input_edit = QLineEdit()
        self.enu_to_itrf_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        enu_to_itrf_input_layout.addWidget(QLabel("Entrée :"))
        enu_to_itrf_input_layout.addWidget(self.enu_to_itrf_input_edit)
        self.enu_to_itrf_input_browse_btn = QPushButton("Parcourir")
        self.enu_to_itrf_input_browse_btn.clicked.connect(self.browse_enu_to_itrf_input_dir)
        enu_to_itrf_input_layout.addWidget(self.enu_to_itrf_input_browse_btn)

        enu_to_itrf_output_layout = QHBoxLayout()
        self.enu_to_itrf_output_edit = QLineEdit()
        self.enu_to_itrf_output_edit.setPlaceholderText("Dossier de sortie (vide = enu_to_itrf_step)")
        enu_to_itrf_output_layout.addWidget(QLabel("Sortie :"))
        enu_to_itrf_output_layout.addWidget(self.enu_to_itrf_output_edit)
        self.enu_to_itrf_output_browse_btn = QPushButton("Parcourir")
        self.enu_to_itrf_output_browse_btn.clicked.connect(self.browse_enu_to_itrf_output_dir)
        enu_to_itrf_output_layout.addWidget(self.enu_to_itrf_output_browse_btn)

        geodetic_layout.addLayout(enu_to_itrf_line)
        geodetic_layout.addLayout(enu_to_itrf_input_layout)
        geodetic_layout.addLayout(enu_to_itrf_output_layout)
        
        # Connexion des cases à cocher au bouton toggle après leur création
        for cb in [self.add_offset_cb, self.itrf_to_enu_cb, self.deform_cb, self.enu_to_itrf_cb]:
            cb.stateChanged.connect(update_geodetic_toggle_btn)
        update_geodetic_toggle_btn()
        
        # 8. Ligne de commande CLI équivalente
        self.geodetic_cmd_label = QLabel("Ligne de commande CLI équivalente :")
        geodetic_layout.addWidget(self.geodetic_cmd_label)
        self.geodetic_cmd_line = QLineEdit()
        self.geodetic_cmd_line.setReadOnly(True)
        self.geodetic_cmd_line.setStyleSheet("font-family: monospace;")
        geodetic_layout.addWidget(self.geodetic_cmd_line)
        
        # 7. Résumé et stretch
        self.geodetic_summary_label = QLabel("")
        geodetic_layout.addWidget(self.geodetic_summary_label)
        geodetic_layout.addStretch(1)
        tabs.addTab(geodetic_tab, "Transformations géodésiques")

        # Onglet 3 : logs
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
        
        # Connexions pour l'onglet géodésique
        self.geodetic_dir_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.geodetic_coord_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deformation_combo.currentTextChanged.connect(self.update_geodetic_cmd_line)
        self.deformation_params_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.bascule_xml_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.add_offset_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_extra.textChanged.connect(self.update_geodetic_cmd_line)
        self.add_offset_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.deform_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers d'entrée personnalisés
        self.add_offset_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers de sortie personnalisés
        self.add_offset_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.enu_to_itrf_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.update_geodetic_cmd_line() 

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

    def browse_geodetic_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier contenant les nuages .ply")
        if folder:
            self.geodetic_dir_edit.setText(folder)

    def browse_geodetic_coord_file(self):
        coord_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier de coordonnées de recalage (.txt)", "", "Fichiers de coordonnées (*.txt)")
        if coord_file:
            self.geodetic_coord_edit.setText(coord_file)
    
    def browse_bascule_xml_file(self):
        xml_file, _ = QFileDialog.getOpenFileName(self, "Choisir le fichier XML GCPBascule (.xml)", "", "Fichiers XML (*.xml)")
        if xml_file:
            self.bascule_xml_edit.setText(xml_file)

    def browse_add_offset_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour l'ajout d'offset")
        if folder:
            self.add_offset_input_edit.setText(folder)

    def browse_itrf_to_enu_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour ITRF→ENU")
        if folder:
            self.itrf_to_enu_input_edit.setText(folder)

    def browse_deform_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour la déformation")
        if folder:
            self.deform_input_edit.setText(folder)

    def browse_enu_to_itrf_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour ENU→ITRF")
        if folder:
            self.enu_to_itrf_input_edit.setText(folder)

    def browse_add_offset_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour l'ajout d'offset")
        if folder:
            self.add_offset_output_edit.setText(folder)

    def browse_itrf_to_enu_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour ITRF→ENU")
        if folder:
            self.itrf_to_enu_output_edit.setText(folder)

    def browse_deform_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour la déformation")
        if folder:
            self.deform_output_edit.setText(folder)

    def browse_enu_to_itrf_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour ENU→ITRF")
        if folder:
            self.enu_to_itrf_output_edit.setText(folder)

    def update_geodetic_cmd_line(self):
        geodetic_dir = self.geodetic_dir_edit.text().strip() or "<dossier_nuages>"
        coord_file = self.geodetic_coord_edit.text().strip()
        deformation_type = self.deformation_combo.currentText()
        deformation_params = self.deformation_params_edit.text().strip()
        bascule_xml = self.bascule_xml_edit.text().strip()
        add_offset_extra = self.add_offset_extra.text().strip()
        itrf_to_enu_extra = self.itrf_to_enu_extra.text().strip()
        deform_extra = self.deform_extra.text().strip()
        enu_to_itrf_extra = self.enu_to_itrf_extra.text().strip()
        
        # Dossiers d'entrée personnalisés
        add_offset_input_dir = self.add_offset_input_edit.text().strip()
        itrf_to_enu_input_dir = self.itrf_to_enu_input_edit.text().strip()
        deform_input_dir = self.deform_input_edit.text().strip()
        enu_to_itrf_input_dir = self.enu_to_itrf_input_edit.text().strip()
        
        # Dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        enu_to_itrf_output_dir = self.enu_to_itrf_output_edit.text().strip()
        
        base_cmd = ["photogeoalign.py", "--geodetic", f'\"{geodetic_dir}\"']
        
        if coord_file:
            base_cmd.append(f"--geodetic-coord \"{coord_file}\"")
        
        base_cmd.append(f"--deformation-type {deformation_type}")
        
        if deformation_params:
            base_cmd.append(f"--deformation-params \"{deformation_params}\"")
        
        if bascule_xml:
            base_cmd.append(f"--deform-bascule-xml \"{bascule_xml}\"")
        
        if add_offset_extra:
            base_cmd.append(f"--add-offset-extra \"{add_offset_extra}\"")
        
        if itrf_to_enu_extra:
            base_cmd.append(f"--itrf-to-enu-extra \"{itrf_to_enu_extra}\"")
        
        # Note: Pour le point de référence, on pourrait ajouter un champ dans le GUI
        # Pour l'instant, on utilise le premier point par défaut
        
        if deform_extra:
            base_cmd.append(f"--deform-extra \"{deform_extra}\"")
        
        if enu_to_itrf_extra:
            base_cmd.append(f"--enu-to-itrf-extra \"{enu_to_itrf_extra}\"")
        
        # Ajout des dossiers d'entrée personnalisés
        if add_offset_input_dir:
            base_cmd.append(f"--add-offset-input-dir \"{add_offset_input_dir}\"")
        
        if itrf_to_enu_input_dir:
            base_cmd.append(f"--itrf-to-enu-input-dir \"{itrf_to_enu_input_dir}\"")
        
        if deform_input_dir:
            base_cmd.append(f"--deform-input-dir \"{deform_input_dir}\"")
        
        if enu_to_itrf_input_dir:
            base_cmd.append(f"--enu-to-itrf-input-dir \"{enu_to_itrf_input_dir}\"")
        
        # Ajout des dossiers de sortie personnalisés
        if add_offset_output_dir:
            base_cmd.append(f"--add-offset-output-dir \"{add_offset_output_dir}\"")
        
        if itrf_to_enu_output_dir:
            base_cmd.append(f"--itrf-to-enu-output-dir \"{itrf_to_enu_output_dir}\"")
        
        if deform_output_dir:
            base_cmd.append(f"--deform-output-dir \"{deform_output_dir}\"")
        
        if enu_to_itrf_output_dir:
            base_cmd.append(f"--enu-to-itrf-output-dir \"{enu_to_itrf_output_dir}\"")
        
        # Ajout des options de skip
        if not self.add_offset_cb.isChecked():
            base_cmd.append("--skip-add-offset")
        
        if not self.itrf_to_enu_cb.isChecked():
            base_cmd.append("--skip-itrf-to-enu")
        
        if not self.deform_cb.isChecked():
            base_cmd.append("--skip-deform")
        
        if not self.enu_to_itrf_cb.isChecked():
            base_cmd.append("--skip-enu-to-itrf")
        
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.geodetic_cmd_line.setText(cmd)

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
        self.action_geodetic.setEnabled(False)
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
            self.append_log("<span style='color:red'>Pipeline MicMac arrêté par l'utilisateur.</span>")
        if hasattr(self, 'geodetic_thread') and self.geodetic_thread and self.geodetic_thread.isRunning():
            self.geodetic_thread.terminate()
            self.geodetic_thread.wait()
            self.append_log("<span style='color:red'>Pipeline géodésique arrêté par l'utilisateur.</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_stop.setEnabled(False)

    def pipeline_finished(self, success, message):
        if success:
            self.summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_stop.setEnabled(False)

    def launch_geodetic_pipeline(self):
        input_dir = self.geodetic_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un dossier valide.</span>")
            return
        
        coord_file = self.geodetic_coord_edit.text().strip()
        if not coord_file or not os.path.exists(coord_file):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner un fichier de coordonnées valide.</span>")
            return
        
        deformation_type = self.deformation_combo.currentText()
        deformation_params = self.deformation_params_edit.text().strip()
        bascule_xml = self.bascule_xml_edit.text().strip()
        max_workers = self.parallel_workers_spin.value()
        add_offset_extra = self.add_offset_extra.text().strip()
        itrf_to_enu_extra = self.itrf_to_enu_extra.text().strip()
        deform_extra = self.deform_extra.text().strip()
        enu_to_itrf_extra = self.enu_to_itrf_extra.text().strip()
        
        run_add_offset = self.add_offset_cb.isChecked()
        run_itrf_to_enu = self.itrf_to_enu_cb.isChecked()
        run_deform = self.deform_cb.isChecked()
        run_enu_to_itrf = self.enu_to_itrf_cb.isChecked()
        
        # Vérification qu'au moins une étape est sélectionnée
        if not any([run_add_offset, run_itrf_to_enu, run_deform, run_enu_to_itrf]):
            self.log_text.append("<span style='color:red'>Veuillez sélectionner au moins une étape de transformation.</span>")
            return
        
        self.log_text.clear()
        self.geodetic_summary_label.setText("")
        self.action_run.setEnabled(False)
        self.action_geodetic.setEnabled(False)
        self.action_stop.setEnabled(True)
        
        # Récupération des dossiers d'entrée personnalisés
        add_offset_input_dir = self.add_offset_input_edit.text().strip()
        itrf_to_enu_input_dir = self.itrf_to_enu_input_edit.text().strip()
        deform_input_dir = self.deform_input_edit.text().strip()
        enu_to_itrf_input_dir = self.enu_to_itrf_input_edit.text().strip()
        
        # Récupération des dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        enu_to_itrf_output_dir = self.enu_to_itrf_output_edit.text().strip()
        
        # Conversion en None si vide
        add_offset_input_dir = add_offset_input_dir if add_offset_input_dir else None
        itrf_to_enu_input_dir = itrf_to_enu_input_dir if itrf_to_enu_input_dir else None
        deform_input_dir = deform_input_dir if deform_input_dir else None
        enu_to_itrf_input_dir = enu_to_itrf_input_dir if enu_to_itrf_input_dir else None
        add_offset_output_dir = add_offset_output_dir if add_offset_output_dir else None
        itrf_to_enu_output_dir = itrf_to_enu_output_dir if itrf_to_enu_output_dir else None
        deform_output_dir = deform_output_dir if deform_output_dir else None
        enu_to_itrf_output_dir = enu_to_itrf_output_dir if enu_to_itrf_output_dir else None
        
        self.geodetic_thread = GeodeticTransformThread(
            input_dir, coord_file, deformation_type, deformation_params,
            add_offset_extra, itrf_to_enu_extra, deform_extra, enu_to_itrf_extra,
            run_add_offset, run_itrf_to_enu, run_deform, run_enu_to_itrf,
            add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, enu_to_itrf_input_dir,
            add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, enu_to_itrf_output_dir,
            None, bascule_xml, max_workers
        )
        self.geodetic_thread.log_signal.connect(self.append_log)
        self.geodetic_thread.finished_signal.connect(self.geodetic_pipeline_finished)
        self.geodetic_thread.start()

    def geodetic_pipeline_finished(self, success, message):
        if success:
            self.geodetic_summary_label.setText(f"<span style='color:green'>{message}</span>")
        else:
            self.geodetic_summary_label.setText(f"<span style='color:red'>{message}</span>")
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
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
        if dialog.exec() == QMessageBox.DialogCode.Accepted:
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