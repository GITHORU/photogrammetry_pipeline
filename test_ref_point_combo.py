#!/usr/bin/env python3
"""
Test pour vérifier la fonction update_ref_point_combo
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt

class TestRefPointCombo(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Label
        layout.addWidget(QLabel("Test du menu déroulant des points de référence"))
        
        # ComboBox
        self.ref_point_combo = QComboBox()
        self.ref_point_combo.setPlaceholderText("Sélectionner un point de référence")
        self.ref_point_combo.setMinimumWidth(200)
        layout.addWidget(self.ref_point_combo)
        
        # Bouton de test
        test_btn = QPushButton("Tester avec fichier E1")
        test_btn.clicked.connect(lambda: self.update_ref_point_combo("GNSS_STA_ITRF2020_e2024e9446_E1_mm3d_corr.txt"))
        layout.addWidget(test_btn)
        
        # Bouton de test 2
        test_btn2 = QPushButton("Tester avec fichier E2")
        test_btn2.clicked.connect(lambda: self.update_ref_point_combo("GNSS_STA_ITRF2020_e2024e9446_E2_mm3d_corr.txt"))
        layout.addWidget(test_btn2)
        
        # Bouton pour afficher la sélection
        show_btn = QPushButton("Afficher la sélection")
        show_btn.clicked.connect(self.show_selection)
        layout.addWidget(show_btn)
        
        self.setLayout(layout)
        self.setWindowTitle("Test Ref Point Combo")
        
    def update_ref_point_combo(self, coord_file):
        """Met à jour le menu déroulant des points de référence en lisant le fichier de coordonnées"""
        self.ref_point_combo.clear()
        self.ref_point_combo.addItem("Premier point (par défaut)", None)
        
        if not coord_file or not os.path.exists(coord_file):
            print(f"Fichier non trouvé : {coord_file}")
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
                                print(f"Ajouté : {point_name} - {tooltip}")
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de coordonnées : {e}")
    
    def get_selected_ref_point(self):
        """Retourne le point de référence sélectionné"""
        current_data = self.ref_point_combo.currentData()
        return current_data  # None pour "Premier point", sinon le nom du point
    
    def show_selection(self):
        """Affiche la sélection actuelle"""
        selected = self.get_selected_ref_point()
        current_text = self.ref_point_combo.currentText()
        print(f"Sélection actuelle : '{current_text}' -> {selected}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestRefPointCombo()
    window.show()
    sys.exit(app.exec())
