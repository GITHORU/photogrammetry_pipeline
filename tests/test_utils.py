"""
Tests unitaires pour les fonctions utilitaires de PhotoGeoAlign
"""
import pytest
import os
import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.utils import resource_path, setup_logger


class TestUtils:
    """Tests pour les fonctions utilitaires"""
    
    def test_resource_path_exists(self):
        """Test que resource_path retourne un chemin valide"""
        # Test avec un fichier qui existe
        logo_path = resource_path("resources/logo.png")
        assert isinstance(logo_path, str)
        assert "logo.png" in logo_path
        
    def test_resource_path_relative(self):
        """Test que resource_path gère les chemins relatifs"""
        relative_path = "test_file.txt"
        result = resource_path(relative_path)
        assert isinstance(result, str)
        assert result.endswith(relative_path)
        
    def test_setup_logger(self):
        """Test de la configuration du logger"""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "PhotogrammetryPipeline"  # Nom fixe dans l'implémentation
        assert logger.level == 10  # DEBUG level (fixe dans l'implémentation)
        
    def test_setup_logger_custom_level(self):
        """Test de la configuration du logger (niveau fixe DEBUG)"""
        logger = setup_logger("test_logger_debug")  # Pas de paramètre level
        assert logger is not None
        assert logger.name == "PhotogrammetryPipeline"
        assert logger.level == 10  # DEBUG level fixe


if __name__ == "__main__":
    pytest.main([__file__])
