#!/bin/bash

# Script de téléchargement automatique de l'exécutable Linux depuis GitHub Actions
# Usage: ./download_latest_executable.sh

REPO="GITHORU/photogrammetry_pipeline"
ARTIFACT_NAME="photogeoalign-linux"
DOWNLOAD_DIR="./executables"

echo "🚀 Téléchargement de l'exécutable Linux depuis GitHub Actions..."

# Créer le dossier de destination
mkdir -p "$DOWNLOAD_DIR"

# Méthode 1: Via GitHub CLI (si disponible)
if command -v gh &> /dev/null; then
    echo "📥 Utilisation de GitHub CLI..."
    
    # Télécharger le dernier artifact
    cd "$DOWNLOAD_DIR"
    gh run download --repo "$REPO" --name "$ARTIFACT_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ Téléchargement réussi via GitHub CLI!"
        
        # Rendre l'exécutable exécutable
        chmod +x photogeoalign 2>/dev/null || chmod +x photogeoalign_linux 2>/dev/null
        
        echo "🎯 Exécutable prêt dans: $DOWNLOAD_DIR"
        ls -la
        exit 0
    else
        echo "❌ Échec du téléchargement via GitHub CLI"
    fi
fi

# Méthode 2: Via curl/wget (fallback)
echo "📥 Utilisation de curl/wget..."
echo "⚠️  ATTENTION: Cette méthode nécessite un token GitHub pour les repos privés"
echo ""
echo "Pour télécharger manuellement:"
echo "1. Allez sur: https://github.com/$REPO/actions"
echo "2. Cliquez sur le dernier build vert"
echo "3. Téléchargez l'artifact '$ARTIFACT_NAME'"
echo "4. Transférez sur le cluster avec scp/rsync"
echo ""

# Afficher les instructions de transfert
cat << 'EOF'
📋 INSTRUCTIONS DE TRANSFERT MANUEL:

# Depuis votre machine locale vers le cluster:
scp photogeoalign-linux.tar.gz user@cluster:/path/to/destination/

# Sur le cluster, extraire:
tar -xzf photogeoalign-linux.tar.gz
chmod +x photogeoalign
./photogeoalign --help

EOF
