@echo off
REM Script d'automatisation pour créer l'exécutable Windows
REM Nécessite pyinstaller installé dans l'environnement

REM Nettoyage des anciens builds
rmdir /s /q build
rmdir /s /q dist
del photogeoalign.spec

set SCRIPT=photogeoalign.py
set ICON=logo.png

REM Construction de l'exécutable principal
pyinstaller --noconfirm --onefile --windowed --icon=%ICON% --add-data "logo.png;." %SCRIPT%

echo.
echo Compilation terminée. L'exécutable se trouve dans le dossier dist\ 