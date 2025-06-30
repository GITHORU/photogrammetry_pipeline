# PhotoGeoAlign

<p align="center">
  <img src="logo.png" alt="Logo PhotoGeoAlign" height="300"/>
</p>

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
photogeoalign_windows.exe --no-gui <dossier_images> [options]
```

Ou, si vous avez généré un exécutable Linux (PyInstaller) :

```bash
./photogeoalign_linux.sh --no-gui <dossier_images> [options]
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
logo.png                # Logo de l'application
requirements.txt        # Dépendances Python
README.md               # Documentation
```

## Prérequis
- Python 3.8+
- MicMac installé et accessible via la commande `mm3d`
- PySide6 (`pip install -r requirements.txt`)

## Contact
Pour toute question, suggestion ou contribution : reveneau@ipgp.fr

## Exécution de l'exécutable Windows

Après compilation ou téléchargement, l'exécutable Windows se nomme :

```
photogeoalign_windows.exe
```

Lancez-le depuis l'explorateur ou un terminal :

```
photogeoalign_windows.exe --no-gui <dossier_images> [options]
```

---

## Exécution de l'exécutable Linux

Après compilation ou téléchargement, l'exécutable Linux se nomme :

```
photogeoalign_linux.sh
```

Il est nécessaire de lui donner les droits d'exécution avant de pouvoir le lancer :

```bash
chmod +x photogeoalign_linux.sh
```

Ensuite, lancez-le avec :

```bash
./photogeoalign_linux.sh --no-gui <dossier_images> [options]
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