#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Configuration de l'encodage pour Windows
import sys
import os
if sys.platform.startswith('win'):
    # Forcer l'encodage UTF-8 sur Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

print("Test d'encodage UTF-8")
print("Transformations géodésiques terminées avec succès !")
print("Démarrage des transformations géodésiques...")
print("Erreur lors des transformations géodésiques")
