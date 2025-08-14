@echo off
REM Script d'automatisation pour créer l'exécutable Windows (version refactorisée)
REM Nécessite pyinstaller installé dans l'environnement

REM Nettoyage des anciens builds
rmdir /s /q build
rmdir /s /q dist

REM Construction de l'exécutable principal en utilisant le fichier .spec existant
pyinstaller --noconfirm photogeoalign_windows.spec

echo.
echo Compilation terminée. L'exécutable se trouve dans le dossier dist\ sous le nom photogeoalign_windows.exe 