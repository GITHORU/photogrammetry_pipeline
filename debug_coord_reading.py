#!/usr/bin/env python3
"""
Script de debug pour tester la lecture du fichier de coordonnées
"""

import logging

# Configuration du logger avec debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_coord')

def test_coord_reading():
    """Test de lecture du fichier de coordonnées E2"""
    
    coord_file = "GNSS_STA_ITRF2020_e2024e9446_E2_mm3d_corr.txt"
    
    print(f"Test de lecture du fichier : {coord_file}")
    
    ref_point = None
    offset = None
    
    try:
        with open(coord_file, 'r', encoding='utf-8') as f:
            print("=== Lecture des points ===")
            for line in f:
                line = line.strip()
                print(f"Ligne : '{line}'")
                if line and not line.startswith('#'):
                    parts = line.split()
                    print(f"  Parts : {parts}")
                    if len(parts) >= 4:  # Format: NOM X Y Z
                        try:
                            point_name = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            print(f"  Point trouvé : {point_name} ({x}, {y}, {z})")
                            if ref_point is None:  # Prendre le premier point
                                ref_point = [x, y, z]
                                print(f"  Point de référence sélectionné : {point_name}")
                                break
                        except ValueError as e:
                            print(f"  Erreur de conversion pour la ligne : {line} - {e}")
                            continue
            
            print("\n=== Lecture de l'offset ===")
            f.seek(0)  # Retour au début du fichier
            for line in f:
                line = line.strip()
                print(f"Ligne : '{line}'")
                if line.startswith('#Offset to add :'):
                    parts = line.split(':')[1].strip().split()
                    print(f"  Parts offset : {parts}")
                    if len(parts) == 3:
                        offset = [float(parts[0]), float(parts[1]), float(parts[2])]
                        print(f"  Offset trouvé : {offset}")
                        break
            
            print(f"\n=== Résultat ===")
            print(f"ref_point : {ref_point}")
            print(f"offset : {offset}")
            
            if ref_point is None:
                print("❌ Aucun point de référence trouvé")
                return False
            
            if offset is None:
                print("❌ Aucun offset trouvé")
                return False
            
            print("✅ Lecture réussie")
            return True
                
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

if __name__ == "__main__":
    success = test_coord_reading()
    print(f"\nTest {'réussi' if success else 'échoué'}")
