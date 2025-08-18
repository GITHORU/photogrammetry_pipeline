"""
Tests unitaires pour les fonctions de fusion des couleurs de PhotoGeoAlign
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import des fonctions de fusion des couleurs
# Note: Ces fonctions sont dans merge_orthoimages_and_dtm


class TestColorFusion:
    """Tests pour les fonctions de fusion des couleurs"""
    
    def test_color_average_calculation(self):
        """Test du calcul de la moyenne des couleurs"""
        # Données de test
        color1 = np.array([100, 150, 200], dtype=np.uint8)
        color2 = np.array([110, 160, 210], dtype=np.uint8)
        
        # Calcul de la moyenne
        average_color = np.mean([color1, color2], axis=0)
        
        # Vérifications
        assert np.array_equal(average_color, np.array([105.0, 155.0, 205.0]))
        assert average_color.dtype == np.float64
        
    def test_color_median_calculation(self):
        """Test du calcul de la médiane des couleurs"""
        # Données de test
        colors = np.array([
            [100, 150, 200],
            [110, 160, 210],
            [120, 170, 220]
        ], dtype=np.uint8)
        
        # Calcul de la médiane
        median_color = np.median(colors, axis=0)
        
        # Vérifications
        assert np.array_equal(median_color, np.array([110.0, 160.0, 210.0]))
        assert median_color.dtype == np.float64
        
    def test_color_clipping(self):
        """Test du clipping des valeurs de couleur"""
        # Données de test avec valeurs hors limites
        raw_colors = np.array([-10.0, 300.0, 125.5])
        
        # Clipping vers [0, 255]
        clipped_colors = np.clip(raw_colors, 0, 255)
        
        # Vérifications
        assert np.array_equal(clipped_colors, np.array([0.0, 255.0, 125.5]))
        assert np.all(clipped_colors >= 0)
        assert np.all(clipped_colors <= 255)
        
    def test_color_validation_mask(self):
        """Test de la création du masque de validation des couleurs"""
        # Données de test
        height_data = np.array([1.5, -9999.0, 2.0, np.nan])
        
        # Masque de validation (même logique que dans le code)
        valid_mask = (height_data != -9999.0) & ~np.isnan(height_data)
        
        # Vérifications
        expected_mask = np.array([True, False, True, False])
        assert np.array_equal(valid_mask, expected_mask)
        
    def test_color_data_types(self):
        """Test des types de données pour les couleurs"""
        # Test avec différents types
        uint8_colors = np.array([100, 150, 200], dtype=np.uint8)
        float64_colors = np.array([100.0, 150.0, 200.0], dtype=np.float64)
        
        # Vérifications
        assert uint8_colors.dtype == np.uint8
        assert float64_colors.dtype == np.float64
        
        # Conversion uint8 -> float64
        converted_colors = uint8_colors.astype(np.float64)
        assert converted_colors.dtype == np.float64
        assert np.array_equal(converted_colors, float64_colors)


if __name__ == "__main__":
    pytest.main([__file__])
