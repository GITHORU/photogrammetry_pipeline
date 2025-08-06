#!/usr/bin/env python3
"""
Script de test pour vérifier que le build refactorisé fonctionne correctement
"""
import os
import sys
import subprocess
import time

def test_executable():
    """Teste l'exécutable généré"""
    exe_path = "dist/photogeoalign_windows.exe"
    
    if not os.path.exists(exe_path):
        print("❌ Erreur : L'exécutable n'existe pas")
        return False
    
    print(f"✅ Exécutable trouvé : {exe_path}")
    print(f"📁 Taille : {os.path.getsize(exe_path) / (1024*1024):.1f} MB")
    
    # Test avec --help
    try:
        print("\n🔍 Test de la commande --help...")
        result = subprocess.run([exe_path, "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Commande --help fonctionne")
            print(f"📝 Sortie (premiers 200 caractères) :")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"❌ Erreur avec --help : {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors du test --help")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test --help : {e}")
        return False
    
    # Test de lancement GUI (court)
    try:
        print("\n🖥️ Test de lancement GUI (3 secondes)...")
        process = subprocess.Popen([exe_path], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        time.sleep(3)  # Attendre 3 secondes
        
        if process.poll() is None:
            print("✅ GUI lancée avec succès")
            process.terminate()
            process.wait()
        else:
            print(f"❌ GUI s'est fermée prématurément (code: {process.returncode})")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Erreur stderr : {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test GUI : {e}")
        return False
    
    return True

def test_imports():
    """Teste que tous les modules peuvent être importés"""
    print("\n📦 Test des imports...")
    
    try:
        # Test des modules principaux
        from modules.core.utils import setup_logger, resource_path
        print("✅ modules.core.utils")
        
        from modules.core.micmac import run_micmac_tapioca
        print("✅ modules.core.micmac")
        
        from modules.core.geodetic import add_offset_to_clouds
        print("✅ modules.core.geodetic")
        
        from modules.gui.main_window import PhotogrammetryGUI
        print("✅ modules.gui.main_window")
        
        from modules.gui.dialogs import JobExportDialog
        print("✅ modules.gui.dialogs")
        
        from modules.workers.pipeline_thread import PipelineThread
        print("✅ modules.workers.pipeline_thread")
        
        from modules.workers.geodetic_thread import GeodeticTransformThread
        print("✅ modules.workers.geodetic_thread")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 Test du build refactorisé PhotoGeoAlign")
    print("=" * 50)
    
    # Test des imports
    if not test_imports():
        print("\n❌ Échec des tests d'import")
        return False
    
    # Test de l'exécutable
    if not test_executable():
        print("\n❌ Échec des tests de l'exécutable")
        return False
    
    print("\n🎉 Tous les tests sont passés avec succès !")
    print("✅ Le build refactorisé fonctionne correctement")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 