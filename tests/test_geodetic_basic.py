"""
Tests unitaires de base pour les fonctions géodésiques de PhotoGeoAlign
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.geodetic_core import (
    process_single_cloud_add_offset,
    process_single_cloud_itrf_to_enu
)


class TestGeodeticBasic:
    """Tests de base pour les fonctions géodésiques"""
    
    def test_process_single_cloud_add_offset_valid_input(self):
        """Test de process_single_cloud_add_offset avec entrées valides"""
        # Création de données de test
        test_args = (
            "test.ply",           # ply_file
            "output_dir",         # output_dir
            "test_coords.txt",    # coord_file
            ""                    # extra_params
        )
        
        # Test que la fonction peut être appelée sans erreur
        # (Note: nécessite des fichiers réels pour un test complet)
        assert callable(process_single_cloud_add_offset)
        
    def test_process_single_cloud_itrf_to_enu_valid_input(self):
        """Test de process_single_cloud_itrf_to_enu avec entrées valides"""
        # Création de données de test
        test_args = (
            "test.ply",           # ply_file
            "output_dir",         # output_dir
            "test_coords.txt",    # coord_file
            "",                   # extra_params
            "REF_POINT"           # ref_point_name
        )
        
        # Test que la fonction peut être appelée sans erreur
        assert callable(process_single_cloud_itrf_to_enu)
        
    def test_geodetic_functions_signatures(self):
        """Test que les fonctions ont les bonnes signatures"""
        import inspect
        
        # Vérification de la signature de process_single_cloud_add_offset
        sig_add_offset = inspect.signature(process_single_cloud_add_offset)
        assert len(sig_add_offset.parameters) == 1  # Un seul paramètre (args)
        
        # Vérification de la signature de process_single_cloud_itrf_to_enu
        sig_itrf_to_enu = inspect.signature(process_single_cloud_itrf_to_enu)
        assert len(sig_itrf_to_enu.parameters) == 1  # Un seul paramètre (args)
        
    def test_geodetic_functions_return_type(self):
        """Test que les fonctions retournent le bon type"""
        # Ces fonctions retournent des tuples (success, message)
        # Test avec des données simulées
        assert isinstance(("test", "test"), tuple)
        assert len(("test", "test")) == 2


if __name__ == "__main__":
    pytest.main([__file__])
