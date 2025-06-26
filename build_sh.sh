#!/bin/bash
# Script d'automatisation pour créer l'exécutable Linux
# Nécessite pyinstaller installé dans l'environnement

# Nettoyage des anciens builds
rm -rf build dist photogeoalign.spec

SCRIPT=photogeoalign.py
ICON=logo.png

# Construction de l'exécutable principal
pyinstaller --noconfirm --onefile --windowed --icon=$ICON --add-data "logo.png:." $SCRIPT

echo
echo "Compilation terminée. L'exécutable se trouve dans le dossier dist/" 