#!/usr/bin/env python3
"""
Script de test pour la nouvelle fonctionnalité d'orthoimage
"""

import os
import sys
import logging
from modules.core.geodetic import create_orthoimage_from_pointcloud

def test_orthoimage():
    """Test de la fonction de création d'orthoimage"""
    
    # Configuration du logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Dossier de test (à adapter selon votre environnement)
    test_input_dir = "test_data/pointclouds"  # Dossier contenant des fichiers .ply
    test_output_dir = "test_data/orthoimages"
    
    # Vérification de l'existence du dossier d'entrée
    if not os.path.exists(test_input_dir):
        logger.error(f"Le dossier de test {test_input_dir} n'existe pas.")
        logger.info("Veuillez créer ce dossier et y placer des fichiers .ply pour tester.")
        return False
    
    # Création du dossier de sortie
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # Test de la fonction de création d'orthoimage
        logger.info("Test de la création d'orthoimage...")
        create_orthoimage_from_pointcloud(
            input_dir=test_input_dir,
            logger=logger,
            output_dir=test_output_dir,
            resolution=0.1,  # 10 cm
            height_field="z",
            color_field="rgb",
            max_workers=2
        )
        
        logger.info("Test réussi ! Vérifiez les fichiers générés dans le dossier de sortie.")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du test : {e}")
        return False

if __name__ == "__main__":
    success = test_orthoimage()
    sys.exit(0 if success else 1)
