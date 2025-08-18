#!/usr/bin/env python3
"""
Script de téléchargement automatique des artifacts GitHub Actions
Usage: python3 download_github_artifact.py [--token YOUR_TOKEN]
"""

import requests
import sys
import os
import zipfile
import argparse
from pathlib import Path

REPO_OWNER = "GITHORU"
REPO_NAME = "photogrammetry_pipeline"
ARTIFACT_NAME = "photogeoalign-linux"

def download_latest_artifact(token=None):
    """Télécharge le dernier artifact depuis GitHub Actions"""
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PhotoGeoAlign-Downloader"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    # URL de l'API GitHub pour les workflow runs
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs"
    
    print("🔍 Recherche du dernier build réussi...")
    
    try:
        # Récupérer les derniers runs
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        runs = response.json()["workflow_runs"]
        
        # Trouver le dernier run réussi
        successful_run = None
        for run in runs:
            if run["status"] == "completed" and run["conclusion"] == "success":
                successful_run = run
                break
        
        if not successful_run:
            print("❌ Aucun build réussi trouvé")
            return False
        
        print(f"✅ Build trouvé: {successful_run['head_sha'][:8]} ({successful_run['created_at']})")
        
        # Récupérer les artifacts de ce run
        artifacts_url = successful_run["artifacts_url"]
        response = requests.get(artifacts_url, headers=headers)
        response.raise_for_status()
        
        artifacts = response.json()["artifacts"]
        
        # Trouver l'artifact Linux
        linux_artifact = None
        for artifact in artifacts:
            if artifact["name"] == ARTIFACT_NAME:
                linux_artifact = artifact
                break
        
        if not linux_artifact:
            print(f"❌ Artifact '{ARTIFACT_NAME}' non trouvé")
            print("Artifacts disponibles:", [a["name"] for a in artifacts])
            return False
        
        print(f"📥 Téléchargement de {ARTIFACT_NAME} ({linux_artifact['size_in_bytes']} bytes)...")
        
        # Télécharger l'artifact
        download_url = linux_artifact["archive_download_url"]
        response = requests.get(download_url, headers=headers)
        response.raise_for_status()
        
        # Créer le dossier de destination
        download_dir = Path("./executables")
        download_dir.mkdir(exist_ok=True)
        
        # Sauvegarder le zip
        zip_path = download_dir / f"{ARTIFACT_NAME}.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extraire le zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        # Supprimer le zip
        zip_path.unlink()
        
        # Rendre l'exécutable exécutable
        for exe_file in download_dir.glob("photogeoalign*"):
            if exe_file.is_file() and not exe_file.suffix:
                exe_file.chmod(0o755)
                print(f"🎯 Exécutable prêt: {exe_file}")
                break
        
        print("✅ Téléchargement terminé avec succès!")
        print(f"📁 Fichiers dans {download_dir}:")
        for file in download_dir.iterdir():
            print(f"  - {file.name}")
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ Erreur réseau: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Télécharge l'exécutable Linux depuis GitHub Actions")
    parser.add_argument("--token", help="Token GitHub (optionnel pour repos publics)")
    args = parser.parse_args()
    
    print("🚀 PhotoGeoAlign - Téléchargeur d'exécutable Linux")
    print("=" * 50)
    
    # Essayer de récupérer le token depuis les variables d'environnement
    token = args.token or os.environ.get("GITHUB_TOKEN")
    
    if not token:
        print("ℹ️  Aucun token GitHub fourni (OK pour repos publics)")
    
    success = download_latest_artifact(token)
    
    if success:
        print("\n🎉 Prêt à utiliser!")
        print("Commande de test: ./executables/photogeoalign --help")
    else:
        print("\n❌ Échec du téléchargement")
        print("\n📋 Solutions alternatives:")
        print("1. Vérifiez que le repo est public ou fournissez un token")
        print("2. Téléchargez manuellement depuis: https://github.com/GITHORU/photogrammetry_pipeline/actions")
        sys.exit(1)

if __name__ == "__main__":
    main()
