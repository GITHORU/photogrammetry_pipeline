#!/usr/bin/env python3
"""
Test simple pour vérifier que convert_enu_to_itrf fonctionne avec ref_point_name
"""

import os
import sys
import logging
from modules.core.geodetic import convert_enu_to_itrf

def test_enu_to_itrf_with_ref_point():
    """Test de la fonction convert_enu_to_itrf avec ref_point_name"""
    
    # Configuration du logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('test_enu_to_itrf')
    
    # Paramètres de test
    test_input_dir = "test_complet"
    test_coord_file = "GNSS_STA_ITRF2020_e2024e9446_E2_mm3d_corr.txt"
    ref_point_name = None  # Utilise le premier point par défaut
    
    # Vérification de l'existence des fichiers
    if not os.path.exists(test_input_dir):
        print(f"❌ Dossier de test non trouvé : {test_input_dir}")
        return False
    
    if not os.path.exists(test_coord_file):
        print(f"❌ Fichier de coordonnées non trouvé : {test_coord_file}")
        return False
    
    print(f"✅ Test de la transformation ENU→ITRF avec ref_point_name={ref_point_name}")
    print(f"   Dossier d'entrée : {test_input_dir}")
    print(f"   Fichier de coordonnées : {test_coord_file}")
    
    try:
        # Appel de la fonction de test avec ref_point_name
        convert_enu_to_itrf(test_input_dir, logger, test_coord_file, ref_point_name=ref_point_name, max_workers=1)
        print("✅ Test réussi !")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        return False

if __name__ == "__main__":
    success = test_enu_to_itrf_with_ref_point()
    sys.exit(0 if success else 1)
