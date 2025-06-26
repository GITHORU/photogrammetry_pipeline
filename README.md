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

#### Options disponibles
- `--mode` : QuickMac, BigMac, MicMac (défaut : BigMac)
- `--tapas-model` : Modèle Tapas (défaut : Fraser)
- `--zoomf` : Facteur de zoom pour C3DC (défaut : 1)
- `--tapioca-extra` : Paramètres supplémentaires pour Tapioca (ex : "NbMin=3")
- `--tapas-extra` : Paramètres supplémentaires pour Tapas (ex : "ExpTxt=1")
- `--c3dc-extra` : Paramètres supplémentaires pour C3DC (ex : "EZA=1")

#### Exemple complet
```bash
photogeoalign_windows.exe --no-gui "C:\chemin\vers\images" --mode QuickMac --tapas-model RadialBasic --zoomf 2 --tapioca-extra "NbMin=3" --tapas-extra "ExpTxt=1" --c3dc-extra "EZA=1"
```

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
Pour toute question, suggestion ou contribution : hugor[at]protonmail.com 

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