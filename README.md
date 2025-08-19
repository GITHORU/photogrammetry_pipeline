# PhotoGeoAlign

<p align="center">
  <img src="resources/logo.png" alt="Logo PhotoGeoAlign" height="300"/>
</p>

[![Tests](https://github.com/GITHORU/photogrammetry_pipeline/workflows/Tests%20PhotoGeoAlign/badge.svg)](https://github.com/GITHORU/photogrammetry_pipeline/actions)

PhotoGeoAlign est un pipeline photogrammétrique automatisé et personnalisable basé sur MicMac, utilisable en interface graphique (GUI) ou en ligne de commande (CLI). Il fonctionne sous Windows, Linux et macOS.

## Fonctionnalités principales
- Interface graphique moderne (PySide6)
- Pipeline automatisé (Tapioca, Tapas, C3DC)
- Choix du modèle Tapas parmi de nombreux modèles (Fraser, RadialBasic, etc.)
- Modes C3DC : QuickMac, BigMac, MicMac
- Paramètres personnalisés pour chaque étape (Tapioca, Tapas, C3DC)
- Logs détaillés et colorés dans l'interface
- Affichage dynamique de la ligne de commande CLI équivalente
- Arrêt du pipeline à tout moment
- Compatible Windows, Linux, macOS

## Utilisation

### Interface graphique (recommandée)
```bash
python photogeoalign.py
```
- Sélectionnez le dossier d'images DNG
- Choisissez le mode C3DC et le modèle Tapas
- (Optionnel) Ajoutez des paramètres personnalisés pour chaque étape
- La ligne de commande CLI équivalente s'affiche dynamiquement
- Cliquez sur "Lancer le pipeline"

### Ligne de commande (mode console)
Vous pouvez lancer le pipeline sans interface graphique :

```bash
python photogeoalign.py --no-gui <dossier_images> [options]
```

Ou, si vous avez généré un exécutable Windows (PyInstaller) :

```bash
photogeoalign.exe --no-gui <dossier_images> [options]
```

Ou, si vous avez généré un exécutable Linux (PyInstaller) :

```bash
./photogeoalign --no-gui <dossier_images> [options]
```

## Paramètres de la ligne de commande (CLI)

Voici la liste des paramètres disponibles pour lancer le pipeline en mode console :

### Paramètres principaux
- `input_dir`  
  **(positionnel)**  
  Dossier contenant les images à traiter.
- `--no-gui`  
  Lance le pipeline en mode console (sans interface graphique).
- `--mode`  
  Mode de densification C3DC (`QuickMac`, `BigMac`, `MicMac`).  
  **Défaut** : `BigMac`
- `--zoomf`  
  Facteur de zoom pour C3DC (1 = max résolution).  
  **Défaut** : `1`
- `--tapas-model`  
  Modèle Tapas à utiliser (ex : `Fraser`, `RadialBasic`, etc.).  
  **Défaut** : `Fraser`

### Paramètres supplémentaires pour chaque étape
- `--tapioca-extra`  
  Paramètres supplémentaires pour Tapioca (ex : `"NbMin=3"`)
- `--tapas-extra`  
  Paramètres supplémentaires pour Tapas (ex : `"ExpTxt=1"`)
- `--c3dc-extra`  
  Paramètres supplémentaires pour C3DC (ex : `"EZA=1"`)

### Contrôle des étapes du pipeline
- `--skip-tapioca`  
  Ne pas exécuter Tapioca
- `--skip-tapas`  
  Ne pas exécuter Tapas
- `--skip-c3dc`  
  Ne pas exécuter C3DC
- `--skip-saisieappuisinit`  
  Ne pas exécuter SaisieAppuisInitQT
- `--skip-saisieappuispredic`  
  Ne pas exécuter SaisieAppuisPredicQT

### Paramètres pour les points d'appui
- `--saisieappuisinit-pt <fichier.txt>`  
  Chemin du fichier de points d'appui (format TXT, obligatoire pour les étapes d'appuis et GCPBascule).

### Exemple de commande complète

```bash
python photogeoalign.py --no-gui "C:/chemin/vers/images" --mode BigMac --tapas-model RadialBasic --zoomf 2 --saisieappuisinit-pt "GNSS_STA_ITRF2020.txt" --tapioca-extra "NbMin=3" --tapas-extra "ExpTxt=1" --c3dc-extra "EZA=1"
```

**Remarques** :
- Les étapes GCPBascule (init et predic) sont automatiques si les étapes Init ou Predic sont activées.
- Les fichiers intermédiaires sont gérés automatiquement.
- Les logs détaillent chaque étape et les fichiers utilisés.

## Structure du projet
```
photogeoalign.py         # Script principal (GUI + CLI)
modules/                 # Structure modulaire
├── core/               # Fonctions utilitaires et métier
│   ├── utils.py       # Fonctions utilitaires
│   ├── micmac.py      # Pipeline MicMac
│   └── geodetic.py    # Transformations géodésiques et orthoimages
├── gui/                # Interface graphique
│   ├── main_window.py  # Fenêtre principale
│   └── dialogs.py      # Boîtes de dialogue et export SLURM
└── workers/            # Threads de traitement
    ├── pipeline_thread.py    # Thread pipeline MicMac
    ├── geodetic_thread.py    # Thread transformations géodésiques
    └── utils.py             # Utilitaires pour les threads
resources/
└── logo.png            # Logo de l'application
requirements.txt        # Dépendances Python
README.md               # Documentation principale
README_ORTHOIMAGE.md    # Documentation du pipeline géodésique
build_windows.bat       # Script de build Windows
build_linux.sh          # Script de build Linux
photogeoalign_windows.spec  # Configuration PyInstaller Windows
photogeoalign.spec    # Configuration PyInstaller Linux
```

## Prérequis
- Python 3.8+
- MicMac installé et accessible via la commande `mm3d`
- Dépendances Python : `pip install -r requirements.txt`

### Dépendances principales
- **Interface** : PySide6 (Qt6)
- **3D et géospatial** : Open3D, pyproj, rasterio
- **Images** : Pillow (PIL), numpy, scipy
- **Build** : PyInstaller

## Build et distribution

### Build automatisé (GitHub Actions)

PhotoGeoAlign utilise GitHub Actions pour construire automatiquement les exécutables Windows et Linux à chaque modification du code.

#### Télécharger les exécutables pré-compilés

**Windows :**
1. Allez sur [GitHub Actions](https://github.com/GITHORU/photogrammetry_pipeline/actions)
2. Cliquez sur le dernier build ✅
3. Téléchargez `photogeoalign-windows-docker`

**Linux (choisissez votre version) :**

**RHEL 8.2+ / CentOS 8+ / GLIBC 2.28+ (clusters anciens) :**
```bash
curl -L https://nightly.link/GITHORU/photogrammetry_pipeline/workflows/build/main/photogeoalign-centos8-docker.zip -o photogeoalign-linux.zip && \
unzip -o photogeoalign-linux.zip && rm photogeoalign-linux.zip && chmod +x photogeoalign* && ./photogeoalign --help
```

**Ubuntu 22.04+ / GLIBC 2.35+ (clusters modernes) :**
```bash
curl -L https://nightly.link/GITHORU/photogrammetry_pipeline/workflows/build/main/photogeoalign-ubuntu22-docker.zip -o photogeoalign-linux.zip && \
unzip -o photogeoalign-linux.zip && rm photogeoalign-linux.zip && chmod +x photogeoalign* && ./photogeoalign --help
```

### Build manuel avec PyInstaller
```bash
# Windows
build_windows.bat

# Linux
chmod +x build_linux.sh
./build_linux.sh

# Ou directement avec PyInstaller
pyinstaller photogeoalign_windows.spec    # Windows
pyinstaller photogeoalign.spec      # Linux
```

### Dépannage clusters Linux

**Erreur GLIBC** (`GLIBC_2.35' not found`) :
- L'exécutable est construit sur Ubuntu 22.04 (GLIBC 2.35)
- Compatible avec Ubuntu 22.04+, CentOS 9+, RHEL 9+
- **Pour CentOS 7/8** : buildez directement sur le cluster
- Si votre cluster est plus ancien, buildez directement dessus :
```bash
# Sur le cluster directement
git clone https://github.com/GITHORU/photogrammetry_pipeline.git
cd photogrammetry_pipeline
pip install -r requirements.txt
pip install pyinstaller
./build_linux.sh
```



## Contact
Pour toute question, suggestion ou contribution : reveneau@ipgp.fr

## Exécution de l'exécutable Windows

Après compilation ou téléchargement, l'exécutable Windows se nomme :

```
photogeoalign.exe
```

Lancez-le depuis l'explorateur ou un terminal :

```
photogeoalign.exe --no-gui <dossier_images> [options]
```

---

## Exécution de l'exécutable Linux

Après compilation ou téléchargement, l'exécutable Linux se nomme :

```
photogeoalign
```

Il est nécessaire de lui donner les droits d'exécution avant de pouvoir le lancer :

```bash
chmod +x photogeoalign
```

Ensuite, lancez-le avec :

```bash
./photogeoalign --no-gui <dossier_images> [options]
```

Ceci est une mesure de sécurité standard sous Linux : tout fichier téléchargé n'est pas exécutable par défaut. 

---

## Note de développement

Cette application a été développée avec l'aide d'une Intelligence Artificielle (IA) pour assister dans la conception, l'implémentation et l'optimisation du code. 

## Pipeline photogrammétrique complet

Le pipeline PhotoGeoAlign intègre désormais la gestion avancée des points d'appui et du recalage absolu :

1. **SaisieAppuisInitQT** (optionnelle)
   - Saisie manuelle initiale de quelques points d'appui.
   - Utilise l'orientation Tapas choisie (ex : `RadialBasic`).
   - Fichier d'appuis : fichier TXT (converti automatiquement en XML).
   - Fichier de sortie : `PtsImgInit.xml` (et `PtsImgInit-S2D.xml` généré par MicMac).

2. **GCPBascule (init)** (automatique si Init cochée)
   - Recalage absolu initial.
   - Entrée : orientation Tapas (ex : `RadialBasic`).
   - Sortie : orientation recalée (ex : `RadialBasic_abs_init`).
   - Fichier d'appuis : XML issu du TXT.
   - Fichier de points : `PtsImgInit-S2D.xml`.

3. **SaisieAppuisPredicQT** (optionnelle)
   - Saisie assistée de tous les points, sur la base du recalage initial.
   - Utilise l'orientation recalée (ex : `RadialBasic_abs_init`).
   - Fichier d'appuis : XML issu du TXT.
   - Fichier de sortie : `PtsImgPredic.xml` (et `PtsImgPredic-S2D.xml` généré par MicMac).

4. **GCPBascule (predic)** (automatique si Predic cochée)
   - Recalage absolu final sur tous les points.
   - Entrée : orientation recalée (ex : `RadialBasic_abs_init`).
   - Sortie : orientation finale (ex : `RadialBasic_abs`).
   - Fichier d'appuis : XML issu du TXT.
   - Fichier de points : `PtsImgPredic-S2D.xml`.

5. **C3DC**
   - Densification du nuage de points, utilisant l'orientation finale (ex : `RadialBasic_abs`).

### Exemple de séquence complète

1. Cochez "SaisieAppuisInitQT" et "SaisieAppuisPredicQT" dans l'interface (ou utilisez les options CLI correspondantes).
2. Sélectionnez le fichier TXT d'appuis GNSS.
3. Le pipeline exécutera automatiquement :
   - SaisieAppuisInitQT
   - GCPBascule (init)
   - SaisieAppuisPredicQT
   - GCPBascule (predic)
   - C3DC (si activé)

### Options CLI associées
- `--skip-saisieappuisinit` : ne pas exécuter SaisieAppuisInitQT
- `--skip-saisieappuispredic` : ne pas exécuter SaisieAppuisPredicQT
- `--saisieappuisinit-pt <fichier.txt>` : fichier de points d'appui TXT (obligatoire pour les étapes d'appuis)

### Remarques
- Les étapes GCPBascule sont automatiques et gérées par le pipeline.
- Les fichiers intermédiaires sont générés et utilisés automatiquement.
- Les logs détaillent chaque étape et les fichiers utilisés.

### 2. Pipeline géodésique avancé

PhotoGeoAlign intègre également un pipeline géodésique complet pour la transformation et fusion de nuages de points :

#### Étapes disponibles
- **Ajout d'offset** : Application d'un décalage géométrique
- **ITRF → ENU** : Conversion du système de coordonnées global vers local
- **Déformation TPS** : Transformation géométrique non-linéaire
- **Orthoimage unitaire** : Création d'orthoimages individuelles depuis chaque nuage
- **Orthoimage unifiée** : Fusion de toutes les orthoimages en une seule

#### Fonctionnalités
- **Gestion des résolutions** : Paramétrage de la résolution des orthoimages (0.1mm à 20m)
- **Fusion des couleurs** : Choix entre méthode "Moyenne" et "Médiane"
- **Gestion des nodata** : Traitement automatique des zones sans données
- **Export SLURM** : Génération de scripts de batch pour clusters

#### Utilisation
```bash
# Interface graphique
python photogeoalign.py

# Ligne de commande
python photogeoalign.py --geodetic [options]
```

Pour plus de détails, consultez `README_ORTHOIMAGE.md`. 