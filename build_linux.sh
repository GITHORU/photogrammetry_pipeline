#!/bin/bash
# Script d'automatisation pour créer l'exécutable Linux (version refactorisée)
# Nécessite pyinstaller installé dans l'environnement

# Nettoyage des anciens builds
rm -rf build dist

# Construction de l'exécutable principal en utilisant le fichier .spec existant
pyinstaller --noconfirm photogeoalign_linux.spec

echo
if [ -f dist/photogeoalign_linux ]; then
    echo "Compilation terminée. L'exécutable se trouve dans le dossier dist/ sous le nom photogeoalign_linux"
else
    echo "Erreur : l'exécutable n'a pas été généré."
fi 