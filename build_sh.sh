#!/bin/bash
# Script d'automatisation pour créer l'exécutable Linux
# Nécessite pyinstaller installé dans l'environnement

# Nettoyage des anciens builds
rm -rf build dist photogeoalign.spec

SCRIPT=photogeoalign.py
ICON=logo.png
NAME=photogeoalign_linux.sh

# Construction de l'exécutable principal
pyinstaller --noconfirm --onefile --windowed --icon=$ICON --add-data "logo.png:." --name $NAME $SCRIPT

echo
if [ -f dist/$NAME ]; then
    echo "Compilation terminée. L'exécutable se trouve dans le dossier dist/ sous le nom $NAME"
else
    echo "Erreur : l'exécutable n'a pas été généré."
fi 