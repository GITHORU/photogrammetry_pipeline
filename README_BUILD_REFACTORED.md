# Build PhotoGeoAlign Refactorisé

## 📋 Vue d'ensemble

Ce document décrit le processus de build pour la version refactorisée de PhotoGeoAlign, qui utilise une structure modulaire.

## 🏗️ Structure du projet refactorisé

```
photogrammetry_cursor/
├── photogeoalign_refactored.py     # Point d'entrée principal
├── modules/                        # Structure modulaire
│   ├── core/                      # Fonctions utilitaires et métier
│   │   ├── utils.py              # Fonctions utilitaires
│   │   ├── micmac.py             # Pipeline MicMac
│   │   └── geodetic.py           # Transformations géodésiques
│   ├── gui/                      # Interface graphique
│   │   ├── main_window.py        # Fenêtre principale
│   │   └── dialogs.py            # Boîtes de dialogue
│   └── workers/                  # Threads de traitement
│       ├── pipeline_thread.py    # Thread pipeline MicMac
│       ├── geodetic_thread.py    # Thread transformations géodésiques
│       └── utils.py              # Utilitaires pour les threads
├── resources/
│   └── logo.png                  # Logo de l'application
└── dist/
    └── photogeoalign_windows.exe # Exécutable généré
```

## 🔧 Scripts de build

### Windows
```batch
# Utiliser le script automatique
build_refactored.bat

# Ou utiliser PyInstaller directement
venv\Scripts\pyinstaller.exe photogeoalign_refactored.spec
```

### Linux
```bash
# Utiliser le script automatique
chmod +x build_refactored.sh
./build_refactored.sh

# Ou utiliser PyInstaller directement
pyinstaller photogeoalign_refactored.spec
```

## 📦 Configuration PyInstaller

Le fichier `photogeoalign_refactored.spec` contient :

- **Point d'entrée** : `photogeoalign_refactored.py`
- **Icône** : `resources/logo.png`
- **Données** : Logo inclus dans l'exécutable
- **Modules cachés** : Tous les modules de la structure modulaire
- **Exclusions** : `PySide6.QtNetwork` (non utilisé)

## 🧪 Tests

### Test des imports
```bash
venv\Scripts\python.exe test_build_refactored.py
```

### Test de l'exécutable
```bash
# Test CLI
dist\photogeoalign_windows.exe --help

# Test GUI
dist\photogeoalign_windows.exe
```

## 📊 Résultats du build

- **Taille** : ~161 MB (normal avec toutes les dépendances)
- **Dépendances incluses** : PySide6, NumPy, SciPy, Open3D, PyProj
- **Compatibilité** : Windows 64-bit

## 🔍 Dépannage

### Problèmes courants

1. **Erreur de chemin de logo** :
   - Vérifier que `resources/logo.png` existe
   - La fonction `resource_path` cherche dans `resources/` si pas trouvé dans le répertoire racine

2. **Erreur d'import de modules** :
   - Vérifier que tous les fichiers `__init__.py` existent
   - Vérifier les chemins d'import relatifs

3. **Exécutable trop volumineux** :
   - Utiliser `--exclude-module` pour exclure des modules non utilisés
   - Vérifier les `hiddenimports` dans le fichier `.spec`

### Logs de build

Les logs détaillés sont disponibles dans :
- `build/photogeoalign_refactored/warn-photogeoalign_refactored.txt`
- `build/photogeoalign_refactored/xref-photogeoalign_refactored.html`

## 🚀 Utilisation

### Mode GUI
```bash
dist\photogeoalign_windows.exe
```

### Mode CLI
```bash
dist\photogeoalign_windows.exe --help
dist\photogeoalign_windows.exe input_dir --mode BigMac
```

### Mode géodésique
```bash
dist\photogeoalign_windows.exe input_dir --geodetic --geodetic-coord coords.txt
```

## 📝 Notes importantes

- ✅ **Refactoring** : Aucune modification de la logique métier, uniquement réorganisation
- ✅ **Compatibilité** : Même fonctionnalités que la version originale
- ✅ **Structure modulaire** : Code organisé en packages logiques
- ✅ **Build fonctionnel** : Exécutable généré avec succès 