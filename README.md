# Photogrammetry Pipeline

Ce projet propose un pipeline photogrammétrique automatisé en Python, s'appuyant sur MicMac pour la reconstruction 3D dense à partir d'un jeu d'images DNG. Il est conçu pour les chercheurs, ingénieurs et étudiants en géodésie, topographie ou vision par ordinateur.

## Fonctionnalités principales
- Détection des points homologues (Tapioca)
- Calibration et orientation (Tapas)
- Densification du nuage de points (C3DC)
- Logs détaillés (console et fichier)
- Compatible cluster de calcul (environnement Python isolé)
- Parallélisation configurable (nombre de processeurs)

## Prérequis
- **Python 3.8+**
- **MicMac** installé et accessible via la commande `mm3d`

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/GITHORU/photogrammetry_pipeline.git
   cd photogrammetry_pipeline
   ```
2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   .\venv\Scripts\activate  # Windows
   ```
3. Installez les dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```
4. Vérifiez que la commande `mm3d` fonctionne dans votre terminal.

## Utilisation
1. Placez vos images DNG dans un dossier (ex : `short_dataset/`).
2. Lancez le pipeline avec la commande suivante :
   ```bash
   python photogrammetry_pipeline.py <dossier_images> [--mode QuickMac|BigMac] [--zoomf 1] [--nb-proc 8]
   ```
   - `<dossier_images>` : dossier contenant les images DNG à traiter (par défaut : `short_dataset`)
   - `--mode` : mode de densification C3DC (`QuickMac` ou `BigMac`, défaut : `QuickMac`)
   - `--zoomf` : facteur de zoom/résolution pour C3DC (défaut : 1)
   - `--nb-proc` : nombre de processeurs à utiliser pour Tapioca et C3DC (défaut : 8)

   **Exemple :**
   ```bash
   python photogrammetry_pipeline.py short_dataset --mode QuickMac --zoomf 1 --nb-proc 16
   ```
   - Les logs détaillés sont enregistrés dans `<dossier_images>/photogrammetry_pipeline.log`.
   - Un résumé synthétique est disponible dans `<dossier_images>/photogrammetry_pipeline_summary.log`.
   - Les résultats (nuage dense) sont générés dans `<dossier_images>/PIMs-QuickMac/` ou `<dossier_images>/C3DC_QuickMac.ply` (selon le mode).

## Structure du projet
```
photogrammetry_pipeline.py   # Script principal
requirements.txt             # Dépendances Python
.gitignore                  # Fichiers ignorés par git
README.md                   # Ce fichier
short_dataset/              # Exemple de dossier d'images (à créer)
```

## Conseils pour le cluster
- Clonez le dépôt et installez les dépendances comme ci-dessus.
- Vérifiez que MicMac est installé sur le cluster et accessible dans le PATH.
- Utilisez un environnement virtuel pour isoler les dépendances Python.
- Adaptez le paramètre `--nb-proc` selon les ressources disponibles.

## Contact
Pour toute question, suggestion ou contribution :
- Auteur : **Hugo R.**
- Dépôt : [github.com/GITHORU/photogrammetry_pipeline](https://github.com/GITHORU/photogrammetry_pipeline)

---
**Bon traitement photogrammétrique !** 