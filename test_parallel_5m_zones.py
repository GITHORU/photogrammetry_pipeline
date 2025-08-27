#!/usr/bin/env python3
"""
Test de la fusion parallélisée des orthoimages avec zones de 5m × 5m
"""

import os
import sys
import logging
from modules.core.geodetic import merge_orthoimages_and_dtm_parallel_5m_zones

def setup_logger():
    """Configure le logger pour les tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_parallel_5m_zones.log')
        ]
    )
    return logging.getLogger(__name__)

def test_parallel_5m_zones():
    """Test de la fonction parallélisée avec zones de 5m"""
    logger = setup_logger()
    
    # Dossier d'entrée (à adapter selon votre structure)
    input_dir = "test_orthos"  # Dossier contenant les orthos individuelles
    
    if not os.path.exists(input_dir):
        logger.error(f"Le dossier d'entrée {input_dir} n'existe pas")
        logger.info("Veuillez créer un dossier 'test_orthos' avec vos orthos individuelles")
        return False
    
    logger.info(f"Test de la fusion parallélisée avec zones de 5m")
    logger.info(f"Dossier d'entrée : {input_dir}")
    
    try:
        # Test avec différents paramètres
        logger.info("=== TEST 1 : Résolution par défaut ===")
        merge_orthoimages_and_dtm_parallel_5m_zones(
            input_dir=input_dir,
            logger=logger,
            max_workers=4,  # Limiter à 4 processus pour le test
            color_fusion_method="average"
        )
        
        logger.info("=== TEST 2 : Résolution personnalisée ===")
        merge_orthoimages_and_dtm_parallel_5m_zones(
            input_dir=input_dir,
            logger=logger,
            target_resolution=0.1,  # 10cm par pixel
            max_workers=4,
            color_fusion_method="median"
        )
        
        logger.info("✅ Tous les tests ont réussi !")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors des tests : {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_5m_zones()
    sys.exit(0 if success else 1)

