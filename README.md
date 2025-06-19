# Photogrammetry Pipeline

Ce projet propose un pipeline photogrammétrique automatisé en Python, s'appuyant sur MicMac pour la reconstruction 3D dense à partir d'un jeu d'images DNG. Il est conçu pour les chercheurs, ingénieurs et étudiants en géodésie, topographie ou vision par ordinateur.

## Fonctionnalités principales
- Conversion automatique des images DNG en TIFF (optionnel)
- Détection des points homologues (Tapioca)
- Calibration et orientation (Tapas)
- Densification du nuage de points (C3DC)
- Logs détaillés (console et fichier)
- Compatible cluster de calcul (environnement Python isolé)

## Prérequis
- **Python 3.8+**
- **MicMac** installé et accessible via la commande `mm3d`
- **rawpy** et **imageio** pour la conversion DNG → TIFF
- (Optionnel) [ImageMagick](https://imagemagick.org/) ou [exiftool](https://exiftool.org/) pour manipuler les métadonnées

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
2. Lancez le pipeline :
   ```bash
   python photogrammetry_pipeline.py short_dataset
   ```
   - Par défaut, le script traite le dossier `short_dataset`.
   - Les logs détaillés sont enregistrés dans `short_dataset/photogrammetry_pipeline.log`.
3. Les résultats (nuage dense) sont générés dans `short_dataset/tif/MEC-QuickMac/`.

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

## Contact
Pour toute question, suggestion ou contribution :
- Auteur : **Hugo R.**
- Dépôt : [github.com/GITHORU/photogrammetry_pipeline](https://github.com/GITHORU/photogrammetry_pipeline)

---
**Bon traitement photogrammétrique !** 