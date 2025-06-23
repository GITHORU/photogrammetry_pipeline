# PhotoGeoAlign

![PhotoGeoAlign Logo](logo.png)

**PhotoGeoAlign** est un pipeline photogrammétrique automatisé développé pour la recherche en géodésie. Il permet de traiter des images DNG pour générer des nuages de points denses avec une précision centimétrique.

## 🎯 Fonctionnalités

- **Interface graphique intuitive** avec PySide6
- **Pipeline automatisé** basé sur MicMac (Tapioca, Tapas, C3DC)
- **Support GPU** pour accélérer les calculs
- **Mode console** pour l'intégration dans des scripts
- **Logs en temps réel** dans l'interface
- **Packaging Windows** automatisé avec PyInstaller

## 🚀 Installation

### Prérequis

- Python 3.8+
- MicMac installé et configuré dans le PATH
- PySide6 pour l'interface graphique

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Interface graphique

```bash
python photogeoalign.py
```

### Mode console

```bash
python photogeoalign.py --no-gui --input /chemin/vers/images --output /chemin/vers/sortie --nb-proc 4 --gpu
```

### Options disponibles

- `--no-gui` : Mode console sans interface graphique
- `--input` : Dossier contenant les images DNG
- `--output` : Dossier de sortie pour les résultats
- `--nb-proc` : Nombre de processus (défaut: 4)
- `--gpu` : Utiliser le GPU si disponible

## 🔧 Pipeline photogrammétrique

Le pipeline PhotoGeoAlign exécute automatiquement les étapes suivantes :

1. **Tapioca** : Détection des points d'intérêt et mise en correspondance
2. **Tapas** : Calcul de l'orientation des caméras
3. **C3DC** : Génération du nuage de points dense

## 🏗️ Build Windows

Pour créer un exécutable Windows autonome :

```bash
build_exe.bat
```

L'exécutable sera généré dans le dossier `dist/` avec le logo intégré.

## 📁 Structure du projet

```
photogeoalign/
├── photogeoalign.py      # Application principale
├── build_exe.bat         # Script de build Windows
├── requirements.txt      # Dépendances Python
├── logo.png             # Logo de l'application
└── README.md            # Documentation
```

## 🎨 Interface utilisateur

L'interface PhotoGeoAlign propose :

- **Sélection des dossiers** d'entrée et de sortie
- **Configuration** du nombre de processus et GPU
- **Suivi en temps réel** de l'exécution
- **Logs détaillés** de chaque étape
- **Barre de progression** visuelle

## 🔍 Précision

PhotoGeoAlign est optimisé pour atteindre une **précision centimétrique** dans la génération de nuages de points, adapté aux besoins de la recherche en géodésie.

## 📝 Licence

Développé pour la recherche en géodésie - Tous droits réservés.

## 🤝 Contribution

Ce projet est développé dans le cadre de recherches en géodésie. Pour toute question ou contribution, veuillez contacter l'équipe de développement. 