import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QLineEdit,
    QMessageBox, QTabWidget, QCheckBox, QToolBar, QDialog, QRadioButton, QGroupBox,
    QButtonGroup
)
from PySide6.QtGui import QPixmap, QIcon, QPainter, QColor, QBrush, QPen, QAction
from PySide6.QtCore import Qt, QTimer, QPoint
from ..core.utils import resource_path
from ..workers import PipelineThread, GeodeticTransformThread, AnalysisThread
from .dialogs import JobExportDialog

class PhotogrammetryGUI(QWidget):
    def create_folder_icon(self):
        """Crée une icône de dossier standard"""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Couleur du dossier
        folder_color = QColor(255, 193, 7)  # Jaune standard
        
        # Dessiner le dossier
        painter.setBrush(QBrush(folder_color))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        # Base du dossier
        painter.drawRect(2, 6, 12, 8)
        # Partie supérieure du dossier
        painter.drawRect(2, 4, 8, 4)
        
        painter.end()
        return QIcon(pixmap)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoGeoAlign")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setMinimumWidth(600)
        self.pipeline_thread = None
        self.geodetic_thread = None
        self.analysis_thread = None
        self.init_ui()
    


    def init_ui(self):
        main_layout = QVBoxLayout()
        # Barre d'outils
        toolbar = QToolBar()
        
        # Icône rond rouge pour Arrêter (tout à gauche)
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
        
        # Icône flèche orange pour Lancer le pipeline géodésique
        pixmap_geodetic = QPixmap(24, 24)
        pixmap_geodetic.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_geodetic)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 152, 0)))  # orange
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
        
        # Icône flèche rouge pour Lancer le nouvel onglet
        pixmap_new = QPixmap(24, 24)
        pixmap_new.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_new)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(244, 67, 54)))  # rouge
        painter.setPen(Qt.GlobalColor.transparent)
        points = [
            pixmap_new.rect().topLeft() + QPoint(6, 4),
            pixmap_new.rect().bottomLeft() + QPoint(6, -4),
            pixmap_new.rect().center() + QPoint(6, 0)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_new = QIcon(pixmap_new)
        action_new = QAction(icon_new, "Lancer l'analyse", self)
        action_new.triggered.connect(self.launch_new_pipeline)
        toolbar.addAction(action_new)
        self.action_new = action_new
        
        # Icône pour Export .job (flèche vers le bas verte - même couleur que lancement MicMac)
        pixmap_export = QPixmap(24, 24)
        pixmap_export.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_export)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        green = QColor(76, 175, 80)  # même vert que lancement MicMac
        painter.setBrush(QBrush(green))
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
        action_export = QAction(icon_export, "Exporter le batch .job (MicMac)", self)
        action_export.triggered.connect(self.export_job_dialog)
        toolbar.addAction(action_export)
        
        # Icône pour Export .job géodésique (flèche vers le bas orange - même couleur que lancement géodésique)
        pixmap_export_geodetic = QPixmap(24, 24)
        pixmap_export_geodetic.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_export_geodetic)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        orange = QColor(255, 152, 0)  # même orange que lancement géodésique
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
        icon_export_geodetic = QIcon(pixmap_export_geodetic)
        action_export_geodetic = QAction(icon_export_geodetic, "Exporter le batch .job (Géodésique)", self)
        action_export_geodetic.triggered.connect(self.export_geodetic_job_dialog)
        toolbar.addAction(action_export_geodetic)
        
        # Icône pour Export .job du nouvel onglet (flèche vers le bas rouge - même couleur que lancement)
        pixmap_export_new = QPixmap(24, 24)
        pixmap_export_new.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap_export_new)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        red = QColor(244, 67, 54)  # même rouge que lancement
        painter.setBrush(QBrush(red))
        painter.setPen(Qt.GlobalColor.transparent)
        # Tige de la flèche (plus large et plus longue)
        painter.drawRect(10, 6, 4, 10)
        # Pointe de la flèche (plus grande)
        points = [
            QPoint(12, 21), QPoint(6, 14), QPoint(18, 14)
        ]
        painter.drawPolygon(points)
        painter.end()
        icon_export_new = QIcon(pixmap_export_new)
        action_export_new = QAction(icon_export_new, "Exporter le batch .job (Analyse)", self)
        action_export_new.triggered.connect(self.export_new_job_dialog)
        toolbar.addAction(action_export_new)
        
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
        browse_btn = QPushButton()
        browse_btn.setIcon(self.create_folder_icon())
        browse_btn.setToolTip("Parcourir")
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
        pt_browse_btn = QPushButton()
        pt_browse_btn.setIcon(self.create_folder_icon())
        pt_browse_btn.setToolTip("Parcourir")
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
        geodetic_browse_btn = QPushButton()
        geodetic_browse_btn.setIcon(self.create_folder_icon())
        geodetic_browse_btn.setToolTip("Parcourir")
        geodetic_browse_btn.clicked.connect(self.browse_geodetic_folder)
        geodetic_dir_layout.addWidget(QLabel("Dossier de travail :"))
        geodetic_dir_layout.addWidget(self.geodetic_dir_edit)
        geodetic_dir_layout.addWidget(geodetic_browse_btn)
        geodetic_layout.addLayout(geodetic_dir_layout)
        
        # 2. Fichier de coordonnées de recalage
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Fichier de coordonnées :"))
        self.geodetic_coord_edit = QLineEdit()
        self.geodetic_coord_edit.setPlaceholderText("Sélectionner un fichier de coordonnées (.txt)")
        coord_layout.addWidget(self.geodetic_coord_edit)
        self.geodetic_coord_browse_btn = QPushButton()
        self.geodetic_coord_browse_btn.setIcon(self.create_folder_icon())
        self.geodetic_coord_browse_btn.setToolTip("Parcourir")
        self.geodetic_coord_browse_btn.clicked.connect(self.browse_geodetic_coord_file)
        coord_layout.addWidget(self.geodetic_coord_browse_btn)
        geodetic_layout.addLayout(coord_layout)
        
        # Choix du type de point de référence (exclusif)
        ref_point_choice_layout = QHBoxLayout()
        ref_point_choice_layout.addWidget(QLabel("Type de point de référence :"))
        
        # Boutons radio pour le choix exclusif
        self.local_ref_radio = QRadioButton("Point local (depuis le fichier)")
        self.local_ref_radio.setToolTip("Utilise un point de référence lu depuis le fichier de coordonnées")
        self.local_ref_radio.setChecked(True)  # Par défaut, point local
        
        self.global_ref_radio = QRadioButton("Point global (coordonnées fixes)")
        self.global_ref_radio.setToolTip("Utilise un point de référence global fixe pour unifier le repère ENU")
        
        ref_point_choice_layout.addWidget(self.local_ref_radio)
        ref_point_choice_layout.addWidget(self.global_ref_radio)
        ref_point_choice_layout.addStretch(1)
        geodetic_layout.addLayout(ref_point_choice_layout)
        
        # Point de référence local (depuis le fichier de coordonnées)
        ref_point_layout = QHBoxLayout()
        ref_point_layout.addWidget(QLabel("Point de référence local :"))
        self.ref_point_combo = QComboBox()
        self.ref_point_combo.setPlaceholderText("Sélectionner un point de référence")
        self.ref_point_combo.setMinimumWidth(200)
        ref_point_layout.addWidget(self.ref_point_combo)
        ref_point_layout.addStretch(1)
        geodetic_layout.addLayout(ref_point_layout)
        
        # Point de référence global (pour unifier le repère ENU)
        global_ref_point_layout = QHBoxLayout()
        global_ref_point_layout.addWidget(QLabel("Point de référence global :"))
        
        # Coordonnées X, Y, Z
        self.global_ref_x_spin = QDoubleSpinBox()
        self.global_ref_x_spin.setRange(-999999999, 999999999)  # Aucune limite pratique
        self.global_ref_x_spin.setDecimals(3)
        self.global_ref_x_spin.setSuffix(" m")
        self.global_ref_x_spin.setToolTip("Coordonnée X du point de référence global (ITRF)")
        global_ref_point_layout.addWidget(QLabel("X:"))
        global_ref_point_layout.addWidget(self.global_ref_x_spin)
        
        self.global_ref_y_spin = QDoubleSpinBox()
        self.global_ref_y_spin.setRange(-999999999, 999999999)  # Aucune limite pratique
        self.global_ref_y_spin.setDecimals(3)
        self.global_ref_y_spin.setSuffix(" m")
        self.global_ref_y_spin.setToolTip("Coordonnée Y du point de référence global (ITRF)")
        global_ref_point_layout.addWidget(QLabel("Y:"))
        global_ref_point_layout.addWidget(self.global_ref_y_spin)
        
        self.global_ref_z_spin = QDoubleSpinBox()
        self.global_ref_z_spin.setRange(-999999999, 999999999)  # Aucune limite pratique
        self.global_ref_z_spin.setDecimals(3)
        self.global_ref_z_spin.setSuffix(" m")
        self.global_ref_z_spin.setToolTip("Coordonnée Z du point de référence global (ITRF)")
        global_ref_point_layout.addWidget(QLabel("Z:"))
        global_ref_point_layout.addWidget(self.global_ref_z_spin)
        
        global_ref_point_layout.addStretch(1)
        geodetic_layout.addLayout(global_ref_point_layout)
        
        # 3. Type de déformation
        deformation_layout = QHBoxLayout()
        self.deformation_combo = QComboBox()
        self.deformation_combo.addItems(["none", "tps", "radial"])  # radial ajouté
        self.deformation_combo.setCurrentText("none")
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
        bascule_xml_browse_btn = QPushButton()
        bascule_xml_browse_btn.setIcon(self.create_folder_icon())
        bascule_xml_browse_btn.setToolTip("Parcourir")
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
                self.orthoimage_cb.isChecked(),
                self.unified_orthoimage_cb.isChecked()
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
                self.orthoimage_cb.isChecked(),
                self.unified_orthoimage_cb.isChecked()
            ])
            state = not all_checked
            self.add_offset_cb.setChecked(state)
            self.itrf_to_enu_cb.setChecked(state)
            self.deform_cb.setChecked(state)
            self.orthoimage_cb.setChecked(state)
            self.unified_orthoimage_cb.setChecked(state)
            update_geodetic_toggle_btn()
        
        geodetic_toggle_btn.clicked.connect(toggle_all_geodetic)
        
        # Ajout du bouton dans un layout horizontal collé à gauche
        geodetic_toggle_layout = QHBoxLayout()
        geodetic_toggle_layout.setContentsMargins(0, 0, 0, 0)
        geodetic_toggle_layout.setSpacing(0)
        geodetic_toggle_layout.addWidget(geodetic_toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        geodetic_layout.addLayout(geodetic_toggle_layout)
        
        # 6. Cases à cocher pour les étapes avec champs de recherche
        # Ajout offset
        add_offset_line = QHBoxLayout()
        self.add_offset_cb = QCheckBox("Ajout offset")
        self.add_offset_cb.setChecked(True)
        self.add_offset_cb.setMinimumWidth(140)
        add_offset_line.addWidget(self.add_offset_cb)
        add_offset_line.addWidget(QLabel("Entrée :"))
        self.add_offset_input_edit = QLineEdit()
        self.add_offset_input_edit.setPlaceholderText("Dossier d'entrée (vide = dossier principal)")
        add_offset_line.addWidget(self.add_offset_input_edit)
        self.add_offset_input_browse_btn = QPushButton()
        self.add_offset_input_browse_btn.setIcon(self.create_folder_icon())
        self.add_offset_input_browse_btn.setToolTip("Parcourir")
        self.add_offset_input_browse_btn.clicked.connect(self.browse_add_offset_input_dir)
        add_offset_line.addWidget(self.add_offset_input_browse_btn)
        add_offset_line.addWidget(QLabel("Sortie :"))
        self.add_offset_output_edit = QLineEdit()
        self.add_offset_output_edit.setPlaceholderText("Dossier de sortie (vide = offset_step)")
        add_offset_line.addWidget(self.add_offset_output_edit)
        self.add_offset_output_browse_btn = QPushButton()
        self.add_offset_output_browse_btn.setIcon(self.create_folder_icon())
        self.add_offset_output_browse_btn.setToolTip("Parcourir")
        self.add_offset_output_browse_btn.clicked.connect(self.browse_add_offset_output_dir)
        add_offset_line.addWidget(self.add_offset_output_browse_btn)
        geodetic_layout.addLayout(add_offset_line)
        
        # ITRF vers ENU
        itrf_to_enu_line = QHBoxLayout()
        self.itrf_to_enu_cb = QCheckBox("ITRF → ENU")
        self.itrf_to_enu_cb.setChecked(True)
        self.itrf_to_enu_cb.setMinimumWidth(140)
        itrf_to_enu_line.addWidget(self.itrf_to_enu_cb)
        itrf_to_enu_line.addWidget(QLabel("Entrée :"))
        self.itrf_to_enu_input_edit = QLineEdit()
        self.itrf_to_enu_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        itrf_to_enu_line.addWidget(self.itrf_to_enu_input_edit)
        self.itrf_to_enu_input_browse_btn = QPushButton()
        self.itrf_to_enu_input_browse_btn.setIcon(self.create_folder_icon())
        self.itrf_to_enu_input_browse_btn.setToolTip("Parcourir")
        self.itrf_to_enu_input_browse_btn.clicked.connect(self.browse_itrf_to_enu_input_dir)
        itrf_to_enu_line.addWidget(self.itrf_to_enu_input_browse_btn)
        itrf_to_enu_line.addWidget(QLabel("Sortie :"))
        self.itrf_to_enu_output_edit = QLineEdit()
        self.itrf_to_enu_output_edit.setPlaceholderText("Dossier de sortie (vide = itrf_to_enu_step)")
        itrf_to_enu_line.addWidget(self.itrf_to_enu_output_edit)
        self.itrf_to_enu_output_browse_btn = QPushButton()
        self.itrf_to_enu_output_browse_btn.setIcon(self.create_folder_icon())
        self.itrf_to_enu_output_browse_btn.setToolTip("Parcourir")
        self.itrf_to_enu_output_browse_btn.clicked.connect(self.browse_itrf_to_enu_output_dir)
        itrf_to_enu_line.addWidget(self.itrf_to_enu_output_browse_btn)
        geodetic_layout.addLayout(itrf_to_enu_line)
        
        # Déformation
        deform_line = QHBoxLayout()
        self.deform_cb = QCheckBox("Déformation")
        self.deform_cb.setChecked(True)
        self.deform_cb.setMinimumWidth(140)
        deform_line.addWidget(self.deform_cb)
        deform_line.addWidget(QLabel("Entrée :"))
        self.deform_input_edit = QLineEdit()
        self.deform_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        deform_line.addWidget(self.deform_input_edit)
        self.deform_input_browse_btn = QPushButton()
        self.deform_input_browse_btn.setIcon(self.create_folder_icon())
        self.deform_input_browse_btn.setToolTip("Parcourir")
        self.deform_input_browse_btn.clicked.connect(self.browse_deform_input_dir)
        deform_line.addWidget(self.deform_input_browse_btn)
        deform_line.addWidget(QLabel("Sortie :"))
        self.deform_output_edit = QLineEdit()
        self.deform_output_edit.setPlaceholderText("Dossier de sortie (vide = deform_[type]_step)")
        deform_line.addWidget(self.deform_output_edit)
        self.deform_output_browse_btn = QPushButton()
        self.deform_output_browse_btn.setIcon(self.create_folder_icon())
        self.deform_output_browse_btn.setToolTip("Parcourir")
        self.deform_output_browse_btn.clicked.connect(self.browse_deform_output_dir)
        deform_line.addWidget(self.deform_output_browse_btn)
        geodetic_layout.addLayout(deform_line)
        
        # Orthoimage
        orthoimage_line = QHBoxLayout()
        self.orthoimage_cb = QCheckBox("Orthoimage")
        self.orthoimage_cb.setChecked(True)
        self.orthoimage_cb.setMinimumWidth(140)
        orthoimage_line.addWidget(self.orthoimage_cb)
        orthoimage_line.addWidget(QLabel("Entrée :"))
        self.orthoimage_input_edit = QLineEdit()
        self.orthoimage_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        orthoimage_line.addWidget(self.orthoimage_input_edit)
        self.orthoimage_input_browse_btn = QPushButton()
        self.orthoimage_input_browse_btn.setIcon(self.create_folder_icon())
        self.orthoimage_input_browse_btn.setToolTip("Parcourir")
        self.orthoimage_input_browse_btn.clicked.connect(self.browse_orthoimage_input_dir)
        orthoimage_line.addWidget(self.orthoimage_input_browse_btn)
        orthoimage_line.addWidget(QLabel("Sortie :"))
        self.orthoimage_output_edit = QLineEdit()
        self.orthoimage_output_edit.setPlaceholderText("Dossier de sortie (vide = orthoimage_step)")
        orthoimage_line.addWidget(self.orthoimage_output_edit)
        self.orthoimage_output_browse_btn = QPushButton()
        self.orthoimage_output_browse_btn.setIcon(self.create_folder_icon())
        self.orthoimage_output_browse_btn.setToolTip("Parcourir")
        self.orthoimage_output_browse_btn.clicked.connect(self.browse_orthoimage_output_dir)
        orthoimage_line.addWidget(self.orthoimage_output_browse_btn)
        geodetic_layout.addLayout(orthoimage_line)
        
        # Paramètres d'orthoimage
        orthoimage_params_line = QHBoxLayout()
        orthoimage_params_line.addWidget(QLabel("Résolution :"))
        self.orthoimage_resolution_spin = QDoubleSpinBox()
        self.orthoimage_resolution_spin.setRange(0.1, 20000.0)
        self.orthoimage_resolution_spin.setValue(100.0)  # 100 mm par défaut
        self.orthoimage_resolution_spin.setSuffix(" mm")
        self.orthoimage_resolution_spin.setDecimals(1)
        orthoimage_params_line.addWidget(self.orthoimage_resolution_spin)
        orthoimage_params_line.addStretch(1)
        geodetic_layout.addLayout(orthoimage_params_line)
        
        # Méthode de fusion des couleurs
        geodetic_layout.addWidget(QLabel(""))  # Ligne vide pour séparer
        color_fusion_line = QHBoxLayout()
        color_fusion_line.addWidget(QLabel("Méthode de fusion des couleurs :"))
        self.color_fusion_combo = QComboBox()
        self.color_fusion_combo.addItems(["Moyenne", "Médiane"])
        self.color_fusion_combo.setCurrentText("Moyenne")
        color_fusion_line.addWidget(self.color_fusion_combo)
        color_fusion_line.addStretch(1)
        geodetic_layout.addLayout(color_fusion_line)
        
        # Orthoimage unifiée
        unified_orthoimage_line = QHBoxLayout()
        self.unified_orthoimage_cb = QCheckBox("Orthoimage unifiée")
        self.unified_orthoimage_cb.setChecked(True)
        self.unified_orthoimage_cb.setMinimumWidth(140)
        unified_orthoimage_line.addWidget(self.unified_orthoimage_cb)
        unified_orthoimage_line.addWidget(QLabel("Entrée :"))
        self.unified_orthoimage_input_edit = QLineEdit()
        self.unified_orthoimage_input_edit.setPlaceholderText("Dossier d'entrée (vide = sortie précédente)")
        unified_orthoimage_line.addWidget(self.unified_orthoimage_input_edit)
        self.unified_orthoimage_input_browse_btn = QPushButton()
        self.unified_orthoimage_input_browse_btn.setIcon(self.create_folder_icon())
        self.unified_orthoimage_input_browse_btn.setToolTip("Parcourir")
        self.unified_orthoimage_input_browse_btn.clicked.connect(self.browse_unified_orthoimage_input_dir)
        unified_orthoimage_line.addWidget(self.unified_orthoimage_input_browse_btn)
        unified_orthoimage_line.addWidget(QLabel("Sortie :"))
        self.unified_orthoimage_output_edit = QLineEdit()
        self.unified_orthoimage_output_edit.setPlaceholderText("Dossier de sortie (vide = ortho_mnt_unified)")
        unified_orthoimage_line.addWidget(self.unified_orthoimage_output_edit)
        self.unified_orthoimage_output_browse_btn = QPushButton()
        self.unified_orthoimage_output_browse_btn.setIcon(self.create_folder_icon())
        self.unified_orthoimage_output_browse_btn.setToolTip("Parcourir")
        self.unified_orthoimage_output_browse_btn.clicked.connect(self.browse_unified_orthoimage_output_dir)
        unified_orthoimage_line.addWidget(self.unified_orthoimage_output_browse_btn)
        geodetic_layout.addLayout(unified_orthoimage_line)
        
        # Paramètres d'orthoimage unifiée
        unified_orthoimage_params_line = QHBoxLayout()
        
        # Résolution
        unified_orthoimage_params_line.addWidget(QLabel("Résolution :"))
        self.unified_orthoimage_resolution_spin = QDoubleSpinBox()
        self.unified_orthoimage_resolution_spin.setRange(0.1, 20000.0)
        self.unified_orthoimage_resolution_spin.setValue(100.0)  # 100 mm par défaut
        self.unified_orthoimage_resolution_spin.setSuffix(" mm")
        self.unified_orthoimage_resolution_spin.setDecimals(1)
        unified_orthoimage_params_line.addWidget(self.unified_orthoimage_resolution_spin)
        

        
        # Taille des zones
        unified_orthoimage_params_line.addWidget(QLabel("Taille zones :"))
        self.zone_size_spin = QDoubleSpinBox()
        self.zone_size_spin.setRange(0.5, 100.0)
        self.zone_size_spin.setValue(5.0)  # 5m par défaut
        self.zone_size_spin.setSuffix(" m")
        self.zone_size_spin.setDecimals(1)
        unified_orthoimage_params_line.addWidget(self.zone_size_spin)
        
        unified_orthoimage_params_line.addStretch(1)
        geodetic_layout.addLayout(unified_orthoimage_params_line)
        
        # Connexion des cases à cocher au bouton toggle après leur création
        for cb in [self.add_offset_cb, self.itrf_to_enu_cb, self.deform_cb, self.orthoimage_cb, self.unified_orthoimage_cb]:
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

        # Onglet 3 : Analyse
        new_tab = QWidget()
        new_layout = QVBoxLayout(new_tab)
        
        # Titre de l'onglet
        new_layout.addWidget(QLabel("Analyse"))
        
        # Type d'analyse (Radio buttons)
        analysis_type_group = QGroupBox("Type d'analyse")
        analysis_type_layout = QVBoxLayout(analysis_type_group)
        
        # Groupe de boutons mutuellement exclusifs
        self.analysis_radio_group = QButtonGroup()
        
        self.mnt_radio = QRadioButton("MNT")
        self.mnt_radio.setChecked(True)
        self.ortho_radio = QRadioButton("Ortho")
        self.mnt_ortho_radio = QRadioButton("MNT et Ortho")
        
        # Ajouter les boutons au groupe
        self.analysis_radio_group.addButton(self.mnt_radio, 0)
        self.analysis_radio_group.addButton(self.ortho_radio, 1)
        self.analysis_radio_group.addButton(self.mnt_ortho_radio, 2)
        
        analysis_type_layout.addWidget(self.mnt_radio)
        analysis_type_layout.addWidget(self.ortho_radio)
        analysis_type_layout.addWidget(self.mnt_ortho_radio)
        new_layout.addWidget(analysis_type_group)
        
        # Image 1
        image1_layout = QHBoxLayout()
        self.image1_edit = QLineEdit()
        self.image1_edit.setPlaceholderText("Chemin vers l'image 1")
        image1_browse_btn = QPushButton()
        image1_browse_btn.setIcon(self.create_folder_icon())
        image1_browse_btn.setToolTip("Parcourir")
        image1_browse_btn.clicked.connect(lambda: self.browse_analysis_file(self.image1_edit, "image1"))
        
        self.image1_label = QLabel("Image 1 :")
        image1_layout.addWidget(self.image1_label)
        image1_layout.addWidget(self.image1_edit)
        image1_layout.addWidget(image1_browse_btn)
        new_layout.addLayout(image1_layout)
        
        # Image 2
        image2_layout = QHBoxLayout()
        self.image2_edit = QLineEdit()
        self.image2_edit.setPlaceholderText("Chemin vers l'image 2")
        image2_browse_btn = QPushButton()
        image2_browse_btn.setIcon(self.create_folder_icon())
        image2_browse_btn.setToolTip("Parcourir")
        image2_browse_btn.clicked.connect(lambda: self.browse_analysis_file(self.image2_edit, "image2"))
        
        self.image2_label = QLabel("Image 2 :")
        image2_layout.addWidget(self.image2_label)
        image2_layout.addWidget(self.image2_edit)
        image2_layout.addWidget(image2_browse_btn)
        new_layout.addLayout(image2_layout)
        
        # MNT 1 (pour MNT et Ortho)
        mnt1_layout = QHBoxLayout()
        self.mnt1_edit = QLineEdit()
        self.mnt1_edit.setPlaceholderText("Chemin du premier MNT")
        mnt1_browse_btn = QPushButton()
        mnt1_browse_btn.setIcon(self.create_folder_icon())
        mnt1_browse_btn.setToolTip("Parcourir")
        mnt1_browse_btn.clicked.connect(lambda: self.browse_file(self.mnt1_edit, "MNT (*.tif *.tiff)"))
        
        mnt1_layout.addWidget(QLabel("MNT 1 :"))
        mnt1_layout.addWidget(self.mnt1_edit)
        mnt1_layout.addWidget(mnt1_browse_btn)
        new_layout.addLayout(mnt1_layout)
        
        # MNT 2 (pour MNT et Ortho)
        mnt2_layout = QHBoxLayout()
        self.mnt2_edit = QLineEdit()
        self.mnt2_edit.setPlaceholderText("Chemin du deuxième MNT")
        mnt2_browse_btn = QPushButton()
        mnt2_browse_btn.setIcon(self.create_folder_icon())
        mnt2_browse_btn.setToolTip("Parcourir")
        mnt2_browse_btn.clicked.connect(lambda: self.browse_file(self.mnt2_edit, "MNT (*.tif *.tiff)"))
        
        mnt2_layout.addWidget(QLabel("MNT 2 :"))
        mnt2_layout.addWidget(self.mnt2_edit)
        mnt2_layout.addWidget(mnt2_browse_btn)
        new_layout.addLayout(mnt2_layout)
        
        # Résolution
        resolution_layout = QHBoxLayout()
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.1, 20000.0)
        self.resolution_spin.setValue(100.0)  # 100 mm par défaut
        self.resolution_spin.setSuffix(" mm")
        self.resolution_spin.setDecimals(1)
        
        resolution_layout.addWidget(QLabel("Résolution :"))
        resolution_layout.addWidget(self.resolution_spin)
        resolution_layout.addStretch(1)
        new_layout.addLayout(resolution_layout)
        
        # Dossier de sortie
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Dossier de sortie (laisser vide pour analysis_results dans le dossier de l'image 1)")
        self.output_dir_browse_btn = QPushButton()
        self.output_dir_browse_btn.setIcon(self.create_folder_icon())
        self.output_dir_browse_btn.setToolTip("Parcourir")
        self.output_dir_browse_btn.clicked.connect(self.browse_analysis_output_dir)
        
        output_dir_layout.addWidget(QLabel("Dossier de sortie :"))
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_browse_btn)
        new_layout.addLayout(output_dir_layout)
        
        # Paramètres Farneback
        self.farneback_group = QGroupBox("Paramètres Farneback")
        farneback_layout = QVBoxLayout(self.farneback_group)
        
        # Pyr_scale
        pyr_scale_layout = QHBoxLayout()
        self.pyr_scale_spin = QDoubleSpinBox()
        self.pyr_scale_spin.setRange(0.1, 0.9)
        self.pyr_scale_spin.setValue(0.8)  # Configuration optimisée
        self.pyr_scale_spin.setDecimals(1)
        self.pyr_scale_spin.setSingleStep(0.1)
        pyr_scale_layout.addWidget(QLabel("Pyr_scale :"))
        pyr_scale_layout.addWidget(self.pyr_scale_spin)
        pyr_scale_layout.addStretch(1)
        farneback_layout.addLayout(pyr_scale_layout)
        
        # Levels
        levels_layout = QHBoxLayout()
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(1, 10)
        self.levels_spin.setValue(5)  # Configuration optimisée
        levels_layout.addWidget(QLabel("Levels :"))
        levels_layout.addWidget(self.levels_spin)
        levels_layout.addStretch(1)
        farneback_layout.addLayout(levels_layout)
        
        # Winsize (calculé automatiquement selon la résolution)
        winsize_layout = QHBoxLayout()
        self.winsize_spin = QSpinBox()
        self.winsize_spin.setRange(3, 1000)  # Plage étendue pour l'affichage
        self.winsize_spin.setValue(101)  # Valeur par défaut pour 0.01m
        self.winsize_spin.setSingleStep(2)
        self.winsize_spin.setReadOnly(True)  # Lecture seule
        self.winsize_spin.setToolTip("Winsize calculé automatiquement selon la résolution (référence: 101 pour 0.01m)")
        winsize_layout.addWidget(QLabel("Winsize (auto) :"))
        winsize_layout.addWidget(self.winsize_spin)
        winsize_layout.addStretch(1)
        farneback_layout.addLayout(winsize_layout)
        
        # Iterations
        iterations_layout = QHBoxLayout()
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 50)
        self.iterations_spin.setValue(10)  # Configuration optimisée
        iterations_layout.addWidget(QLabel("Iterations :"))
        iterations_layout.addWidget(self.iterations_spin)
        iterations_layout.addStretch(1)
        farneback_layout.addLayout(iterations_layout)
        
        # Poly_n
        poly_n_layout = QHBoxLayout()
        self.poly_n_spin = QSpinBox()
        self.poly_n_spin.setRange(5, 7)
        self.poly_n_spin.setValue(7)  # Configuration optimisée
        poly_n_layout.addWidget(QLabel("Poly_n :"))
        poly_n_layout.addWidget(self.poly_n_spin)
        poly_n_layout.addStretch(1)
        farneback_layout.addLayout(poly_n_layout)
        
        # Poly_sigma
        poly_sigma_layout = QHBoxLayout()
        self.poly_sigma_spin = QDoubleSpinBox()
        self.poly_sigma_spin.setRange(0.5, 2.0)
        self.poly_sigma_spin.setValue(1.2)  # Configuration optimisée
        self.poly_sigma_spin.setDecimals(1)
        self.poly_sigma_spin.setSingleStep(0.1)
        poly_sigma_layout.addWidget(QLabel("Poly_sigma :"))
        poly_sigma_layout.addWidget(self.poly_sigma_spin)
        poly_sigma_layout.addStretch(1)
        farneback_layout.addLayout(poly_sigma_layout)
        
        new_layout.addWidget(self.farneback_group)
        
        # Ligne de commande CLI équivalente
        self.new_cmd_label = QLabel("Ligne de commande CLI équivalente :")
        new_layout.addWidget(self.new_cmd_label)
        self.new_cmd_line = QLineEdit()
        self.new_cmd_line.setReadOnly(True)
        self.new_cmd_line.setStyleSheet("font-family: monospace;")
        new_layout.addWidget(self.new_cmd_line)
        
        # Résumé et stretch
        self.new_summary_label = QLabel("")
        new_layout.addWidget(self.new_summary_label)
        new_layout.addStretch(1)
        tabs.addTab(new_tab, "Analyse")

        # Onglet 4 : logs
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
        
        # Initialisation du winsize automatique et de l'interface d'analyse
        self.update_winsize_auto()
        self.update_analysis_ui()
        
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
        self.geodetic_coord_edit.textChanged.connect(lambda: self.update_ref_point_combo(self.geodetic_coord_edit.text()))
        self.ref_point_combo.currentTextChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour l'onglet d'analyse
        self.mnt_radio.toggled.connect(self.update_new_cmd_line)
        self.ortho_radio.toggled.connect(self.update_new_cmd_line)
        self.mnt_ortho_radio.toggled.connect(self.update_new_cmd_line)
        self.image1_edit.textChanged.connect(self.update_new_cmd_line)
        self.image2_edit.textChanged.connect(self.update_new_cmd_line)
        self.mnt1_edit.textChanged.connect(self.update_new_cmd_line)
        self.mnt2_edit.textChanged.connect(self.update_new_cmd_line)
        self.resolution_spin.valueChanged.connect(self.update_new_cmd_line)
        self.resolution_spin.valueChanged.connect(self.update_winsize_auto)
        
        # Connexions pour les radio buttons d'analyse
        self.mnt_radio.toggled.connect(self.update_analysis_ui)
        self.ortho_radio.toggled.connect(self.update_analysis_ui)
        self.mnt_ortho_radio.toggled.connect(self.update_analysis_ui)
        
        # Connexions pour les paramètres Farneback
        self.pyr_scale_spin.valueChanged.connect(self.update_new_cmd_line)
        self.levels_spin.valueChanged.connect(self.update_new_cmd_line)
        self.winsize_spin.valueChanged.connect(self.update_new_cmd_line)
        self.iterations_spin.valueChanged.connect(self.update_new_cmd_line)
        self.poly_n_spin.valueChanged.connect(self.update_new_cmd_line)
        self.poly_sigma_spin.valueChanged.connect(self.update_new_cmd_line)
        self.output_dir_edit.textChanged.connect(self.update_new_cmd_line)
        
        # Connexions pour le point de référence global
        self.global_ref_x_spin.valueChanged.connect(self.update_geodetic_cmd_line)
        self.global_ref_y_spin.valueChanged.connect(self.update_geodetic_cmd_line)
        self.global_ref_z_spin.valueChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les boutons radio
        self.local_ref_radio.toggled.connect(self.on_ref_point_type_changed)
        self.global_ref_radio.toggled.connect(self.on_ref_point_type_changed)
        
        self.deformation_combo.currentTextChanged.connect(self.update_geodetic_cmd_line)
        self.deformation_params_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.bascule_xml_edit.textChanged.connect(self.update_geodetic_cmd_line)

        self.add_offset_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.deform_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.orthoimage_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        self.unified_orthoimage_cb.stateChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers d'entrée personnalisés
        self.add_offset_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.orthoimage_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.unified_orthoimage_input_edit.textChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexions pour les dossiers de sortie personnalisés
        self.add_offset_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.itrf_to_enu_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.deform_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.orthoimage_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        self.unified_orthoimage_output_edit.textChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexion pour le nombre de workers
        self.parallel_workers_spin.valueChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexion pour les paramètres d'orthoimage
        self.orthoimage_resolution_spin.valueChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexion pour la méthode de fusion des couleurs
        self.color_fusion_combo.currentTextChanged.connect(self.update_geodetic_cmd_line)
        
        # Connexion pour les paramètres d'orthoimage unifiée
        self.unified_orthoimage_resolution_spin.valueChanged.connect(self.update_geodetic_cmd_line)

        self.zone_size_spin.valueChanged.connect(self.update_geodetic_cmd_line) 
        
        # Initialisation de la ligne de commande pour le nouvel onglet
        self.update_new_cmd_line()
        
        # Initialisation de l'interface d'analyse
        self.update_analysis_ui()
        
        # Initialisation de l'état des contrôles de point de référence
        self.on_ref_point_type_changed()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'images")
        if folder:
            self.dir_edit.setText(folder)

    def browse_file(self, line_edit, filter_str):
        """Ouvre un dialogue pour sélectionner un fichier"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier", "", filter_str)
        if file_path:
            line_edit.setText(file_path)

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
            self.update_ref_point_combo(coord_file)
    
    def update_ref_point_combo(self, coord_file):
        """Met à jour le menu déroulant des points de référence en lisant le fichier de coordonnées"""
        self.ref_point_combo.clear()
        self.ref_point_combo.addItem("Premier point (par défaut)", None)
        
        if not coord_file or not os.path.exists(coord_file):
            return
        
        try:
            with open(coord_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 4:  # Format: NOM X Y Z
                            try:
                                point_name = parts[0]
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                # Ajouter le point avec ses coordonnées dans le tooltip
                                tooltip = f"{point_name}: ({x:.3f}, {y:.3f}, {z:.3f})"
                                self.ref_point_combo.addItem(point_name, point_name)
                                # Définir le tooltip pour le dernier item ajouté
                                self.ref_point_combo.setItemData(self.ref_point_combo.count() - 1, tooltip, Qt.ToolTipRole)
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
    
    def get_selected_ref_point(self):
        """Retourne le point de référence sélectionné"""
        current_data = self.ref_point_combo.currentData()
        return current_data  # None pour "Premier point", sinon le nom du point
    
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

    def browse_orthoimage_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour l'orthoimage")
        if folder:
            self.orthoimage_input_edit.setText(folder)

    def browse_orthoimage_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour l'orthoimage")
        if folder:
            self.orthoimage_output_edit.setText(folder)

    def browse_unified_orthoimage_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'entrée pour l'orthoimage unifiée")
        if folder:
            self.unified_orthoimage_input_edit.setText(folder)

    def browse_unified_orthoimage_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour l'orthoimage unifiée")
        if folder:
            self.unified_orthoimage_output_edit.setText(folder)
    
    def browse_analysis_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie pour l'analyse")
        if folder:
            self.output_dir_edit.setText(folder)
    
    def browse_analysis_file(self, line_edit, field_type):
        """Ouvre un dialogue de sélection de fichier adapté au type d'analyse"""
        is_mnt = self.mnt_radio.isChecked()
        is_ortho = self.ortho_radio.isChecked()
        is_mnt_ortho = self.mnt_ortho_radio.isChecked()
        
        if is_mnt:
            # Mode MNT : sélectionner des fichiers MNT
            file_filter = "MNT (*.tif *.tiff)"
            title = f"Sélectionner le {field_type.replace('image', 'MNT')}"
        elif is_ortho or is_mnt_ortho:
            # Mode Ortho ou MNT+Ortho : sélectionner des orthoimages
            file_filter = "Images (*.tif *.tiff *.jpg *.jpeg *.png)"
            title = f"Sélectionner l'orthoimage {field_type.replace('image', '')}"
        
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def on_ref_point_type_changed(self):
        """Gère l'activation/désactivation des contrôles selon le type de point de référence choisi"""
        is_local = self.local_ref_radio.isChecked()
        is_global = self.global_ref_radio.isChecked()
        
        # Activation/désactivation des contrôles du point local
        self.ref_point_combo.setEnabled(is_local)
        
        # Activation/désactivation des contrôles du point global
        self.global_ref_x_spin.setEnabled(is_global)
        self.global_ref_y_spin.setEnabled(is_global)
        self.global_ref_z_spin.setEnabled(is_global)
        
        # Mise à jour de la ligne de commande
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
        run_tapioca = self.tapioca_cb.isChecked()
        run_tapas = self.tapas_cb.isChecked()
        run_saisieappuisinit = self.saisieappuisinit_cb.isChecked()
        run_saisieappuispredic = self.saisieappuispredic_cb.isChecked()
        run_c3dc = self.c3dc_cb.isChecked()
        
        # Construction de la ligne de commande
        base_cmd = ["photogeoalign.py", "--no-gui", f'\"{input_dir}\"']
        
        if mode != "MulScale":
            base_cmd.append(f"--mode {mode}")
        
        if zoomf != 500:
            base_cmd.append(f"--zoomf {zoomf}")
        
        if tapas_model != "Fraser":
            base_cmd.append(f"--tapas-model {tapas_model}")
        
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
        if not run_tapioca:
            base_cmd.append("--skip-tapioca")
        
        if not run_tapas:
            base_cmd.append("--skip-tapas")
        
        if not run_saisieappuisinit:
            base_cmd.append("--skip-saisieappuisinit")
        
        if not run_saisieappuispredic:
            base_cmd.append("--skip-saisieappuispredic")
        
        if not run_c3dc:
            base_cmd.append("--skip-c3dc")
        
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.cmd_line.setText(cmd)

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

    def update_geodetic_cmd_line(self):
        geodetic_dir = self.geodetic_dir_edit.text().strip() or "<dossier_nuages>"
        coord_file = self.geodetic_coord_edit.text().strip()
        deformation_type = self.deformation_combo.currentText()
        deformation_params = self.deformation_params_edit.text().strip()
        bascule_xml = self.bascule_xml_edit.text().strip()
        
        # Dossiers d'entrée personnalisés
        add_offset_input_dir = self.add_offset_input_edit.text().strip()
        itrf_to_enu_input_dir = self.itrf_to_enu_input_edit.text().strip()
        deform_input_dir = self.deform_input_edit.text().strip()
        orthoimage_input_dir = self.orthoimage_input_edit.text().strip()
        
        # Dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        orthoimage_output_dir = self.orthoimage_output_edit.text().strip()
        
        base_cmd = ["photogeoalign.py", "--geodetic", "--no-gui", f'\"{geodetic_dir}\"']
        
        if coord_file:
            base_cmd.append(f"--geodetic-coord \"{coord_file}\"")
        
        base_cmd.append(f"--deformation-type {deformation_type}")
        
        if deformation_params:
            base_cmd.append(f"--deformation-params \"{deformation_params}\"")
        
        if bascule_xml:
            base_cmd.append(f"--deform-bascule-xml \"{bascule_xml}\"")
        
        # Ajout du point de référence selon le type choisi
        if self.local_ref_radio.isChecked():
            # Point de référence local
            selected_ref_point = self.get_selected_ref_point()
            if selected_ref_point:
                base_cmd.append(f"--itrf-to-enu-ref-point \"{selected_ref_point}\"")
        elif self.global_ref_radio.isChecked():
            # Point de référence global
            global_x = self.global_ref_x_spin.value()
            global_y = self.global_ref_y_spin.value()
            global_z = self.global_ref_z_spin.value()
            base_cmd.append(f"--global-ref-point {global_x:.3f} {global_y:.3f} {global_z:.3f}")
            base_cmd.append("--force-global-ref")
        
        # Ajout des dossiers d'entrée personnalisés
        if add_offset_input_dir:
            base_cmd.append(f"--add-offset-input-dir \"{add_offset_input_dir}\"")
        
        if itrf_to_enu_input_dir:
            base_cmd.append(f"--itrf-to-enu-input-dir \"{itrf_to_enu_input_dir}\"")
        
        if deform_input_dir:
            base_cmd.append(f"--deform-input-dir \"{deform_input_dir}\"")
        
        if orthoimage_input_dir:
            base_cmd.append(f"--orthoimage-input-dir \"{orthoimage_input_dir}\"")
        

        
        # Ajout des dossiers de sortie personnalisés
        if add_offset_output_dir:
            base_cmd.append(f"--add-offset-output-dir \"{add_offset_output_dir}\"")
        
        if itrf_to_enu_output_dir:
            base_cmd.append(f"--itrf-to-enu-output-dir \"{itrf_to_enu_output_dir}\"")
        
        if deform_output_dir:
            base_cmd.append(f"--deform-output-dir \"{deform_output_dir}\"")
        
        if orthoimage_output_dir:
            base_cmd.append(f"--orthoimage-output-dir \"{orthoimage_output_dir}\"")
        
        # Ajout des dossiers d'entrée/sortie pour l'orthoimage unifiée
        unified_orthoimage_input_dir = self.unified_orthoimage_input_edit.text().strip()
        unified_orthoimage_output_dir = self.unified_orthoimage_output_edit.text().strip()
        
        if unified_orthoimage_input_dir:
            base_cmd.append(f"--unified-orthoimage-input-dir \"{unified_orthoimage_input_dir}\"")
        
        if unified_orthoimage_output_dir:
            base_cmd.append(f"--unified-orthoimage-output-dir \"{unified_orthoimage_output_dir}\"")
        
        # Ajout des paramètres d'orthoimage
        orthoimage_resolution = self.orthoimage_resolution_spin.value() / 1000.0  # Conversion mm vers m
        base_cmd.append(f"--orthoimage-resolution {orthoimage_resolution}")
        
        # Ajout des paramètres d'orthoimage unifiée
        unified_orthoimage_resolution = self.unified_orthoimage_resolution_spin.value() / 1000.0  # Conversion mm vers m
        base_cmd.append(f"--unified-orthoimage-resolution {unified_orthoimage_resolution}")
        
        # Ajout des paramètres de taille de grille et de zones
        zone_size = self.zone_size_spin.value()

        base_cmd.append(f"--zone-size {zone_size}")
        
        # Ajout de la méthode de fusion des couleurs
        color_fusion_method = self.color_fusion_combo.currentText()
        if color_fusion_method == "Médiane":
            base_cmd.append("--color-fusion-median")
        

        
        # Ajout du nombre de workers
        max_workers = self.parallel_workers_spin.value()
        base_cmd.append(f"--max-workers {max_workers}")
        
        # Ajout des options de skip
        if not self.add_offset_cb.isChecked():
            base_cmd.append("--skip-add-offset")
        
        if not self.itrf_to_enu_cb.isChecked():
            base_cmd.append("--skip-itrf-to-enu")
        
        if not self.deform_cb.isChecked():
            base_cmd.append("--skip-deform")
        
        if not self.orthoimage_cb.isChecked():
            base_cmd.append("--skip-orthoimage")
        
        if not self.unified_orthoimage_cb.isChecked():
            base_cmd.append("--skip-unified-orthoimage")
        
        python_cmd = self.python_selector.currentText()
        cmd = python_cmd + " " + " ".join(base_cmd)
        self.geodetic_cmd_line.setText(cmd)

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
        if hasattr(self, 'analysis_thread') and self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            self.append_log("<span style='color:red'>Pipeline d'analyse arrêté par l'utilisateur.</span>")
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
        add_offset_extra = ""  # Champ non implémenté dans la GUI
        itrf_to_enu_extra = ""  # Champ non implémenté dans la GUI
        deform_extra = ""  # Champ non implémenté dans la GUI

        
        run_add_offset = self.add_offset_cb.isChecked()
        run_itrf_to_enu = self.itrf_to_enu_cb.isChecked()
        run_deform = self.deform_cb.isChecked()
        run_orthoimage = self.orthoimage_cb.isChecked()
        run_unified_orthoimage = self.unified_orthoimage_cb.isChecked()

        
        # Vérification qu'au moins une étape est sélectionnée
        if not any([run_add_offset, run_itrf_to_enu, run_deform, run_orthoimage, run_unified_orthoimage]):
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
        orthoimage_input_dir = self.orthoimage_input_edit.text().strip()
        unified_orthoimage_input_dir = self.unified_orthoimage_input_edit.text().strip()
        
        # Récupération des dossiers de sortie personnalisés
        add_offset_output_dir = self.add_offset_output_edit.text().strip()
        itrf_to_enu_output_dir = self.itrf_to_enu_output_edit.text().strip()
        deform_output_dir = self.deform_output_edit.text().strip()
        orthoimage_output_dir = self.orthoimage_output_edit.text().strip()
        unified_orthoimage_output_dir = self.unified_orthoimage_output_edit.text().strip()
        
        # Conversion en None si vide
        add_offset_input_dir = add_offset_input_dir if add_offset_input_dir else None
        itrf_to_enu_input_dir = itrf_to_enu_input_dir if itrf_to_enu_input_dir else None
        deform_input_dir = deform_input_dir if deform_input_dir else None
        orthoimage_input_dir = orthoimage_input_dir if orthoimage_input_dir else None
        unified_orthoimage_input_dir = unified_orthoimage_input_dir if unified_orthoimage_input_dir else None
        add_offset_output_dir = add_offset_output_dir if add_offset_output_dir else None
        itrf_to_enu_output_dir = itrf_to_enu_output_dir if itrf_to_enu_output_dir else None
        deform_output_dir = deform_output_dir if deform_output_dir else None
        orthoimage_output_dir = orthoimage_output_dir if orthoimage_output_dir else None
        unified_orthoimage_output_dir = unified_orthoimage_output_dir if unified_orthoimage_output_dir else None
        
        # Paramètres d'orthoimage
        orthoimage_resolution = self.orthoimage_resolution_spin.value() / 1000.0  # Conversion mm vers m
        
        # Paramètres d'orthoimage unifiée
        unified_orthoimage_resolution = self.unified_orthoimage_resolution_spin.value() / 1000.0  # Conversion mm vers m
        
        # Paramètre de taille des zones
        zone_size = self.zone_size_spin.value()
        
        # Méthode de fusion des couleurs
        color_fusion_method = self.color_fusion_combo.currentText()
        
        # Préparation des paramètres du point de référence selon le type choisi
        global_ref_point = None
        force_global_ref = False
        if self.global_ref_radio.isChecked():
            global_ref_point = [
                self.global_ref_x_spin.value(),
                self.global_ref_y_spin.value(),
                self.global_ref_z_spin.value()
            ]
            force_global_ref = True
        
        self.geodetic_thread = GeodeticTransformThread(
            input_dir, coord_file, deformation_type, deformation_params,
            add_offset_extra, itrf_to_enu_extra, deform_extra,
            run_add_offset, run_itrf_to_enu, run_deform, run_orthoimage, run_unified_orthoimage,
            add_offset_input_dir, itrf_to_enu_input_dir, deform_input_dir, orthoimage_input_dir, unified_orthoimage_input_dir,
            add_offset_output_dir, itrf_to_enu_output_dir, deform_output_dir, orthoimage_output_dir, unified_orthoimage_output_dir,
            self.get_selected_ref_point(), bascule_xml, orthoimage_resolution, "z", "rgb", unified_orthoimage_resolution, max_workers, color_fusion_method,
            zone_size, global_ref_point, force_global_ref
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
        
        cli_cmd = self.cmd_line.text().strip()
        parts = cli_cmd.split()
        # On retire le premier mot (python ou exe)
        args = parts[1:]
        # On retire tout photogeoalign.py
        filtered_args = [arg for arg in args if not arg.endswith('photogeoalign.py') and not arg.endswith('photogeoalign.py"')]
        
        if getattr(sys, 'frozen', False):
            # Cas exécutable PyInstaller - utiliser sys.executable (chemin réel exe)
            exe_path = sys.executable
            cmd = [exe_path] + filtered_args
        else:
            # Cas Python - utiliser le script principal
            exe_path = sys.executable
            # Trouver photogeoalign.py dans le répertoire parent du projet
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            script_path = os.path.join(current_dir, 'photogeoalign.py')
            cmd = [exe_path, script_path] + filtered_args
        cli_cmd = " ".join(cmd)
        dialog = JobExportDialog(self, job_name="PhotoGeoAlign_MicMac", output="PhotoGeoAlign_MicMac.out", ntasks=self.parallel_workers_spin.value(), cli_cmd=cli_cmd)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vals = dialog.get_values()
            job_content = self.generate_job_script(vals, "micmac")
            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le script .job", "micmac.job", "Fichiers batch (*.out *.job *.sh)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(job_content)
                    QMessageBox.information(self, "Export réussi", f"Script batch exporté :\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export : {e}")

    def export_geodetic_job_dialog(self):
        import sys
        import os
        exe_path = sys.executable
        script_path = os.path.abspath(__file__)
        cli_cmd = self.geodetic_cmd_line.text().strip()
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
        dialog = JobExportDialog(self, job_name="PhotoGeoAlign_Geodetic", output="PhotoGeoAlign_Geodetic.out", ntasks=self.parallel_workers_spin.value(), cli_cmd=cli_cmd)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vals = dialog.get_values()
            job_content = self.generate_job_script(vals, "geodetic")
            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le script .job", "geodetic.job", "Fichiers batch (*.out *.job *.sh)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(job_content)
                    QMessageBox.information(self, "Export réussi", f"Script batch exporté :\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export : {e}")

    def launch_new_pipeline(self):
        """Lance le pipeline d'analyse"""
        self.log_text.clear()
        self.new_summary_label.setText("")
        self.action_run.setEnabled(False)
        self.action_geodetic.setEnabled(False)
        self.action_new.setEnabled(False)
        self.action_stop.setEnabled(True)
        
        # Récupération des paramètres
        if self.mnt_radio.isChecked():
            analysis_type = "mnt"
        elif self.ortho_radio.isChecked():
            analysis_type = "ortho"
        else:  # mnt_ortho_radio.isChecked()
            analysis_type = "mnt_ortho"
        image1 = self.image1_edit.text().strip()
        image2 = self.image2_edit.text().strip()
        resolution = self.resolution_spin.value() / 1000.0  # Conversion mm vers m
        
        # Paramètres Farneback
        farneback_params = {
            'pyr_scale': self.pyr_scale_spin.value(),
            'levels': self.levels_spin.value(),
            'winsize': self.winsize_spin.value(),
            'iterations': self.iterations_spin.value(),
            'poly_n': self.poly_n_spin.value(),
            'poly_sigma': self.poly_sigma_spin.value()
        }
        
        # Validation des paramètres
        if not image1 or not image2:
            self.append_log("<span style='color:red'>Erreur : Veuillez spécifier les deux images</span>")
            self.analysis_pipeline_finished(False, "Images manquantes")
            return
        
        if not os.path.exists(image1):
            self.append_log(f"<span style='color:red'>Erreur : Image 1 introuvable : {image1}</span>")
            self.analysis_pipeline_finished(False, "Image 1 introuvable")
            return
        
        if not os.path.exists(image2):
            self.append_log(f"<span style='color:red'>Erreur : Image 2 introuvable : {image2}</span>")
            self.analysis_pipeline_finished(False, "Image 2 introuvable")
            return
        
        # Validation des MNTs pour le mode mnt_ortho
        if analysis_type == "mnt_ortho":
            mnt1 = self.mnt1_edit.text().strip()
            mnt2 = self.mnt2_edit.text().strip()
            if not mnt1 or not mnt2:
                self.append_log("<span style='color:red'>Erreur : Mode MNT+Ortho nécessite les chemins des deux MNTs</span>")
                self.analysis_pipeline_finished(False, "MNTs manquants pour le mode mnt_ortho")
                return
            if not os.path.exists(mnt1):
                self.append_log(f"<span style='color:red'>Erreur : MNT 1 introuvable : {mnt1}</span>")
                self.analysis_pipeline_finished(False, "MNT 1 introuvable")
                return
            if not os.path.exists(mnt2):
                self.append_log(f"<span style='color:red'>Erreur : MNT 2 introuvable : {mnt2}</span>")
                self.analysis_pipeline_finished(False, "MNT 2 introuvable")
                return
        
        # Création du dossier de sortie
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(image1), "analysis_results")
        
        # Création et lancement du thread d'analyse
        self.analysis_thread = AnalysisThread(
            image1, image2, analysis_type, resolution, output_dir, farneback_params
        )
        self.analysis_thread.log_signal.connect(self.append_log)
        self.analysis_thread.finished_signal.connect(self.analysis_pipeline_finished)
        self.analysis_thread.start()

    def analysis_pipeline_finished(self, success, message):
        """Appelé quand le pipeline d'analyse se termine"""
        if success:
            self.new_summary_label.setText(f"<span style='color:green'>{message}</span>")
            
            # Affichage des résultats si disponibles
            if self.analysis_thread and self.analysis_thread.get_results():
                results = self.analysis_thread.get_results()
                self.append_log(f"<span style='color:green'>RMSE: {results.get('rmse', 'N/A'):.6f}</span>")
                self.append_log(f"<span style='color:green'>Corrélation Pearson: {results.get('correlation_pearson', 'N/A'):.6f}</span>")
                self.append_log(f"<span style='color:green'>Nombre de points: {results.get('n_points', 'N/A')}</span>")
                
                if 'report_path' in results:
                    self.append_log(f"<span style='color:blue'>Rapport généré: {results['report_path']}</span>")
        else:
            self.new_summary_label.setText(f"<span style='color:red'>{message}</span>")
        
        self.action_run.setEnabled(True)
        self.action_geodetic.setEnabled(True)
        self.action_new.setEnabled(True)
        self.action_stop.setEnabled(False)

    def update_analysis_ui(self, checked=None):
        """Met à jour l'interface selon le type d'analyse sélectionné"""
        print(f"[DEBUG] update_analysis_ui appelée, checked={checked}")
        is_mnt = self.mnt_radio.isChecked()
        is_ortho = self.ortho_radio.isChecked()
        is_mnt_ortho = self.mnt_ortho_radio.isChecked()
        print(f"[DEBUG] MNT={is_mnt}, Ortho={is_ortho}, MNT+Ortho={is_mnt_ortho}")
        
        # S'assurer que tous les éléments sont visibles au départ
        self.farneback_group.setVisible(True)
        self.mnt1_edit.parent().setVisible(True)
        self.mnt2_edit.parent().setVisible(True)
        
        # Mise à jour des placeholders, labels et états des champs
        if is_mnt:
            # Mode MNT : images = MNTs
            self.image1_label.setText("MNT 1 :")
            self.image2_label.setText("MNT 2 :")
            self.image1_edit.setPlaceholderText("Chemin vers le premier MNT")
            self.image2_edit.setPlaceholderText("Chemin vers le deuxième MNT")
            self.image1_edit.setEnabled(True)
            self.image2_edit.setEnabled(True)
            # Cacher les champs MNT séparés
            self.mnt1_edit.parent().setVisible(False)
            self.mnt2_edit.parent().setVisible(False)
            self.farneback_group.setVisible(False)  # Pas de Farneback pour MNT
        elif is_ortho:
            # Mode Ortho : images = orthoimages
            self.image1_label.setText("Ortho 1 :")
            self.image2_label.setText("Ortho 2 :")
            self.image1_edit.setPlaceholderText("Chemin vers la première orthoimage")
            self.image2_edit.setPlaceholderText("Chemin vers la deuxième orthoimage")
            self.image1_edit.setEnabled(True)
            self.image2_edit.setEnabled(True)
            # Cacher les champs MNT séparés
            self.mnt1_edit.parent().setVisible(False)
            self.mnt2_edit.parent().setVisible(False)
            self.farneback_group.setVisible(True)  # Farneback pour ortho
        elif is_mnt_ortho:
            # Mode MNT+Ortho : images = orthos, MNTs séparés
            self.image1_label.setText("Ortho 1 :")
            self.image2_label.setText("Ortho 2 :")
            self.image1_edit.setPlaceholderText("Chemin vers la première orthoimage")
            self.image2_edit.setPlaceholderText("Chemin vers la deuxième orthoimage")
            self.image1_edit.setEnabled(True)
            self.image2_edit.setEnabled(True)
            # Afficher les champs MNT séparés
            self.mnt1_edit.parent().setVisible(True)
            self.mnt2_edit.parent().setVisible(True)
            self.farneback_group.setVisible(True)  # Farneback pour ortho
        
        # Mise à jour de la ligne de commande
        self.update_new_cmd_line()

    def update_winsize_auto(self):
        """Met à jour automatiquement le winsize selon la résolution"""
        resolution_mm = self.resolution_spin.value()
        resolution_m = resolution_mm / 1000.0  # Conversion mm vers m
        
        # Configuration de référence optimisée pour 0.01m (10mm)
        base_winsize = 101
        base_resolution = 0.01  # 10mm
        
        # Calcul du winsize adapté
        ratio = base_resolution / resolution_m
        adapted_winsize = max(3, int(base_winsize * ratio))
        
        # S'assurer que winsize est impair (requis par OpenCV)
        if adapted_winsize % 2 == 0:
            adapted_winsize += 1
        
        # Mise à jour du spinbox (sans déclencher le signal valueChanged)
        self.winsize_spin.blockSignals(True)
        self.winsize_spin.setValue(adapted_winsize)
        self.winsize_spin.blockSignals(False)
        
        # Mise à jour du tooltip
        self.winsize_spin.setToolTip(f"Winsize calculé automatiquement: {adapted_winsize} (référence: {base_winsize} pour {base_resolution*1000:.0f}mm)")


    def update_new_cmd_line(self):
        """Met à jour la ligne de commande CLI pour l'analyse"""
        # Type d'analyse
        if self.mnt_radio.isChecked():
            analysis_type = "mnt"
        elif self.ortho_radio.isChecked():
            analysis_type = "ortho"
        else:  # mnt_ortho_radio.isChecked()
            analysis_type = "mnt_ortho"
        
        # Images
        image1 = self.image1_edit.text().strip()
        image2 = self.image2_edit.text().strip()
        
        # MNTs (pour le mode mnt_ortho)
        mnt1 = self.mnt1_edit.text().strip()
        mnt2 = self.mnt2_edit.text().strip()
        
        # Résolution (conversion mm vers m)
        resolution = self.resolution_spin.value() / 1000.0
        
        # Dossier de sortie
        output_dir = self.output_dir_edit.text().strip()
        
        # Paramètres Farneback
        pyr_scale = self.pyr_scale_spin.value()
        levels = self.levels_spin.value()
        winsize = self.winsize_spin.value()
        iterations = self.iterations_spin.value()
        poly_n = self.poly_n_spin.value()
        poly_sigma = self.poly_sigma_spin.value()
        
        # Construction de la commande
        cmd_parts = ["python", "photogeoalign.py", "--analysis", "--no-gui"]
        cmd_parts.append(f"--type={analysis_type}")
        
        if analysis_type == "mnt_ortho":
            # Mode MNT et Ortho : 4 fichiers
            if image1:
                cmd_parts.append(f"--image1={image1}")
            if image2:
                cmd_parts.append(f"--image2={image2}")
            if mnt1:
                cmd_parts.append(f"--mnt1={mnt1}")
            if mnt2:
                cmd_parts.append(f"--mnt2={mnt2}")
        else:
            # Mode MNT ou Ortho seul : 2 fichiers
            if image1:
                cmd_parts.append(f"--image1={image1}")
            if image2:
                cmd_parts.append(f"--image2={image2}")
        
        cmd_parts.append(f"--resolution={resolution}")
        
        # Ajout du dossier de sortie si spécifié
        if output_dir:
            cmd_parts.append(f"--output-dir \"{output_dir}\"")
        
        # Ajout des paramètres Farneback (winsize calculé automatiquement)
        cmd_parts.append(f"--pyr-scale={pyr_scale}")
        cmd_parts.append(f"--levels={levels}")
        # winsize est calculé automatiquement selon la résolution, pas besoin de l'inclure
        cmd_parts.append(f"--iterations={iterations}")
        cmd_parts.append(f"--poly-n={poly_n}")
        cmd_parts.append(f"--poly-sigma={poly_sigma}")
        
        cmd = " ".join(cmd_parts)
        self.new_cmd_line.setText(cmd)

    def export_new_job_dialog(self):
        """Exporte le job pour l'analyse"""
        import sys
        import os
        exe_path = sys.executable
        script_path = os.path.abspath(__file__)
        cli_cmd = self.new_cmd_line.text().strip()
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
        dialog = JobExportDialog(self, job_name="PhotoGeoAlign_Analysis", output="PhotoGeoAlign_Analysis.out", ntasks=self.parallel_workers_spin.value(), cli_cmd=cli_cmd)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            vals = dialog.get_values()
            job_content = self.generate_job_script(vals, "new")
            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le script .job", "analysis.job", "Fichiers batch (*.out *.job *.sh)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(job_content)
                    QMessageBox.information(self, "Export réussi", f"Script batch exporté :\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export : {e}")

    def generate_job_script(self, vals, pipeline_type="micmac"):
        # Génère le contenu du script SLURM
        # NOTE: L'exécutable Python est détecté automatiquement par le script :
        # - Si exécutable PyInstaller : utilise directement l'exécutable
        # - Si script Python : utilise sys.executable + chemin du script
        # Pas besoin de module load python/3.9 car on utilise un venv ou un exécutable
        
        if pipeline_type == "geodetic":
            # Pour le pipeline géodésique, utiliser l'exécutable détecté automatiquement
            modules = """module purge
# Charger MicMac si nécessaire
module load micmac

# L'exécutable Python est détecté automatiquement par le script
# Pas besoin de module load python/3.9 car on utilise un venv ou un exécutable"""
        elif pipeline_type == "new":
            # Pour le pipeline d'analyse
            modules = """module purge
# L'exécutable Python est détecté automatiquement par le script
# Pas besoin de module load python/3.9 car on utilise un venv ou un exécutable"""
        else:
            # Pour le pipeline MicMac
            modules = """module purge
module load micmac

# L'exécutable Python est détecté automatiquement par le script"""
        
        return f"""#!/bin/bash

#SBATCH --job-name {vals['job_name']}
#SBATCH --output {vals['output']}
#SBATCH --nodes=1-1
#SBATCH --partition {vals['partition']} #ncpum,ncpu,ncpulong
#SBATCH --ntasks={vals['ntasks']}

{modules}

{vals['cli_cmd']}
""" 