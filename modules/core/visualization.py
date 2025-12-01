#!/usr/bin/env python3
"""
Module de visualisation pour les résultats d'analyse photogrammétrique
"""

import os
import json
import numpy as np
import rasterio
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use('Qt5Agg')  # Backend compatible avec PySide6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

logger = logging.getLogger(__name__)

# Configuration du style seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_data_if_needed(data_or_path: Any) -> Optional[np.ndarray]:
    """
    Charge les données raster si un chemin est fourni, sinon retourne les données.
    
    Cette fonction permet un chargement à la demande des données raster pour économiser la mémoire.
    
    Args:
        data_or_path: Chemin de fichier (str) ou données numpy (np.ndarray)
        
    Returns:
        Array numpy ou None si erreur
    """
    if isinstance(data_or_path, str) and os.path.exists(data_or_path):
        try:
            with rasterio.open(data_or_path) as src:
                return src.read(1)
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de {data_or_path}: {e}")
            return None
    elif data_or_path is not None:
        # C'est déjà des données (compatibilité avec ancien code)
        return data_or_path
    return None


def load_metadata_if_needed(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Charge uniquement les métadonnées d'un fichier raster (sans charger les données).
    
    Args:
        filepath: Chemin vers le fichier raster
        
    Returns:
        Dictionnaire avec les métadonnées ou None si erreur
    """
    if not isinstance(filepath, str) or not os.path.exists(filepath):
        return None
    
    try:
        with rasterio.open(filepath) as src:
            # Sérialiser les métadonnées en format simple (pas d'objets rasterio)
            transform = src.transform
            return {
                'transform': [transform.a, transform.b, transform.c, 
                             transform.d, transform.e, transform.f],
                'crs': str(src.crs) if src.crs else None,
                'width': src.width,
                'height': src.height
            }
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des métadonnées de {filepath}: {e}")
        return None


def load_pairwise_results(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Charge les résultats d'analyse paire par paire depuis le dossier de sortie.
    
    Args:
        output_dir: Dossier contenant les résultats
        
    Returns:
        Dictionnaire contenant les résultats ou None si erreur
    """
    try:
        # Chercher le fichier de mapping
        mapping_file = os.path.join(output_dir, 'model_mapping.json')
        if not os.path.exists(mapping_file):
            logger.warning(f"Fichier de mapping introuvable: {mapping_file}")
            return None
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            model_mapping = json.load(f)
        
        # Charger les matrices CSV depuis aggregated_results/matrices
        matrices_dir = os.path.join(output_dir, 'aggregated_results', 'matrices')
        if not os.path.exists(matrices_dir):
            logger.warning(f"Dossier de matrices introuvable: {matrices_dir}")
            return None
        
        # Charger toutes les matrices disponibles
        matrices = {}
        import pandas as pd
        
        matrix_files = {
            'displacement_mean_3d': 'displacement_matrix_mean_3d.csv',
            'displacement_median_3d': 'displacement_matrix_median_3d.csv',
            'displacement_std_3d': 'displacement_matrix_std_3d.csv',
            'displacement_max_3d': 'displacement_matrix_max_3d.csv',
            'displacement_p95_3d': 'displacement_matrix_p95_3d.csv',
            'displacement_p99_3d': 'displacement_matrix_p99_3d.csv',
            'displacement_x_mean': 'displacement_x_matrix_mean.csv',
            'displacement_y_mean': 'displacement_y_matrix_mean.csv',
            'displacement_z_mean': 'displacement_z_matrix_mean.csv',
            'displacement_x_median': 'displacement_x_matrix_median.csv',
            'displacement_y_median': 'displacement_y_matrix_median.csv',
            'displacement_z_median': 'displacement_z_matrix_median.csv',
            'displacement_x_std': 'displacement_x_matrix_std.csv',
            'displacement_y_std': 'displacement_y_matrix_std.csv',
            'displacement_z_std': 'displacement_z_matrix_std.csv',
            'rmse': 'rmse_matrix.csv',
            'mae': 'mae_matrix.csv',
            'coverage': 'coverage_matrix.csv',
        }
        
        for key, filename in matrix_files.items():
            filepath = os.path.join(matrices_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0)
                    matrices[key] = df.values
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement de {filename}: {e}")
        
        # Charger les cartes de déplacement depuis les comparaisons individuelles
        comparisons_dir = os.path.join(output_dir, 'comparisons')
        comparisons = {}
        
        if os.path.exists(comparisons_dir):
            for pair_dir in os.listdir(comparisons_dir):
                pair_path = os.path.join(comparisons_dir, pair_dir)
                if os.path.isdir(pair_path):
                    pair_id = pair_dir  # Format: "0_vs_1"
                    comparisons[pair_id] = {
                        'displacement_x': None,
                        'displacement_y': None,
                        'displacement_z': None,
                        'displacement_magnitude': None,
                        'valid_mask': None,
                        'metadata': {}
                    }
                    
                    # Chercher les fichiers de déplacement
                    displacement_files = {
                        'displacement_x': ['displacement_x.tif'],
                        'displacement_y': ['displacement_y.tif'],
                        'displacement_z': ['displacement_z.tif'],
                        'displacement_magnitude': ['displacement_magnitude_3d.tif', 'displacement_magnitude.tif'],  # Fallback pour ortho seul
                        'valid_mask': ['valid_mask.tif']
                    }
                    
                    for key, filenames in displacement_files.items():
                        # Essayer chaque nom de fichier possible
                        # Stocker le chemin au lieu de charger les données (chargement à la demande)
                        data_loaded = False
                        for filename in filenames:
                            filepath = os.path.join(pair_path, filename)
                            if os.path.exists(filepath):
                                # Stocker le chemin au lieu de charger les données
                                comparisons[pair_id][key] = filepath  # Stocker le chemin
                                
                                # Stocker aussi le chemin pour les métadonnées (chargement à la demande)
                                # On stocke le chemin du fichier X pour charger les métadonnées plus tard si nécessaire
                                if key == 'displacement_x':
                                    comparisons[pair_id]['_metadata_file'] = filepath
                                
                                data_loaded = True
                                break
                        
                        if not data_loaded and key != 'valid_mask':  # valid_mask peut être absent
                            logger.debug(f"Fichier de déplacement {key} non trouvé dans {pair_path}")
        
        results = {
            'model_mapping': model_mapping,
            'matrices': matrices,
            'comparisons': comparisons,
            'output_dir': output_dir,
            'n_models': len(model_mapping)
        }
        
        logger.info(f"Résultats chargés: {len(matrices)} matrices, {len(comparisons)} comparaisons (chemins seulement pour les rasters, chargement à la demande)")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des résultats: {e}")
        return None


def plot_displacement_maps(comparison_data: Dict[str, Any], pair_id: str, 
                           output_path: Optional[str] = None,
                           vmin: Optional[float] = None, vmax: Optional[float] = None) -> Figure:
    """
    Trace les cartes de déplacement (X, Y, Z, magnitude).
    
    Note: La magnitude est 3D (X, Y, Z) pour mnt_ortho, 2D (X, Y) pour ortho seul.
    
    Args:
        comparison_data: Données de comparaison pour une paire (peut contenir des chemins ou des arrays)
        pair_id: Identifiant de la paire (ex: "0_vs_1")
        output_path: Chemin pour sauvegarder la figure (optionnel)
        
    Returns:
        Figure matplotlib
    """
    # Créer une nouvelle figure explicitement pour éviter les conflits avec Qt
    fig = Figure(figsize=(14, 12))
    axes = fig.subplots(2, 2)
    fig.suptitle(f'Cartes de déplacement - Paire {pair_id}', fontsize=16, fontweight='bold')
    
    # Carte X
    ax = axes[0, 0]
    dx_path_or_data = comparison_data.get('displacement_x')
    if dx_path_or_data is not None:
        dx = load_data_if_needed(dx_path_or_data)
        if dx is not None:
            valid_mask_path_or_data = comparison_data.get('valid_mask')
            valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
            if valid_mask is not None:
                dx_masked = np.where(valid_mask, dx, np.nan)
            else:
                dx_masked = dx
            
            im = ax.imshow(dx_masked, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Déplacement X (m)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Colonne (pixels)')
            ax.set_ylabel('Ligne (pixels)')
            fig.colorbar(im, ax=ax, label='m')
        else:
            ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Déplacement X (m)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Déplacement X (m)', fontsize=12, fontweight='bold')
    
    # Carte Y
    ax = axes[0, 1]
    dy_path_or_data = comparison_data.get('displacement_y')
    if dy_path_or_data is not None:
        dy = load_data_if_needed(dy_path_or_data)
        if dy is not None:
            valid_mask_path_or_data = comparison_data.get('valid_mask')
            valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
            if valid_mask is not None:
                dy_masked = np.where(valid_mask, dy, np.nan)
            else:
                dy_masked = dy
            
            im = ax.imshow(dy_masked, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Déplacement Y (m)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Colonne (pixels)')
            ax.set_ylabel('Ligne (pixels)')
            fig.colorbar(im, ax=ax, label='m')
        else:
            ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Déplacement Y (m)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Déplacement Y (m)', fontsize=12, fontweight='bold')
    
    # Carte Z
    ax = axes[1, 0]
    dz_path_or_data = comparison_data.get('displacement_z')
    if dz_path_or_data is not None:
        dz = load_data_if_needed(dz_path_or_data)
        if dz is not None:
            valid_mask_path_or_data = comparison_data.get('valid_mask')
            valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
            if valid_mask is not None:
                dz_masked = np.where(valid_mask, dz, np.nan)
            else:
                dz_masked = dz
            
            im = ax.imshow(dz_masked, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Déplacement Z (m)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Colonne (pixels)')
            ax.set_ylabel('Ligne (pixels)')
            fig.colorbar(im, ax=ax, label='m')
        else:
            ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Déplacement Z (m)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Déplacement Z (m)', fontsize=12, fontweight='bold')
    
    # Magnitude (3D pour mnt_ortho, 2D pour ortho seul)
    ax = axes[1, 1]
    mag_path_or_data = comparison_data.get('displacement_magnitude')
    if mag_path_or_data is not None:
        mag = load_data_if_needed(mag_path_or_data)
        if mag is not None:
            valid_mask_path_or_data = comparison_data.get('valid_mask')
            valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
            if valid_mask is not None:
                mag_masked = np.where(valid_mask, mag, np.nan)
            else:
                mag_masked = mag
            
            im = ax.imshow(mag_masked, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            # Déterminer si c'est 3D ou 2D selon la présence de Z
            dz_path_or_data = comparison_data.get('displacement_z')
            if dz_path_or_data is not None:
                ax.set_title('Magnitude 3D (m)', fontsize=12, fontweight='bold')
            else:
                ax.set_title('Magnitude 2D (m)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Colonne (pixels)')
            ax.set_ylabel('Ligne (pixels)')
            fig.colorbar(im, ax=ax, label='m')
        else:
            ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Magnitude (m)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Magnitude (m)', fontsize=12, fontweight='bold')
    
    # Utiliser fig.tight_layout() au lieu de plt.tight_layout() pour éviter les conflits Qt
    try:
        fig.tight_layout()
    except Exception as e:
        logger.warning(f"Erreur lors de tight_layout: {e}")
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_displacement_vectors(comparison_data: Dict[str, Any], pair_id: str,
                              subsample: int = 10, output_path: Optional[str] = None) -> Figure:
    """
    Trace un quiver plot des vecteurs de déplacement horizontaux (X, Y).
    
    Note: Affiche uniquement les composantes X et Y (magnitude 2D).
    
    Args:
        comparison_data: Données de comparaison pour une paire (peut contenir des chemins ou des arrays)
        pair_id: Identifiant de la paire
        subsample: Facteur de sous-échantillonnage pour les vecteurs
        output_path: Chemin pour sauvegarder la figure (optionnel)
        
    Returns:
        Figure matplotlib
    """
    # Créer une nouvelle figure explicitement pour éviter les conflits avec Qt
    fig = Figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    dx_path_or_data = comparison_data.get('displacement_x')
    dy_path_or_data = comparison_data.get('displacement_y')
    dx = load_data_if_needed(dx_path_or_data) if dx_path_or_data else None
    dy = load_data_if_needed(dy_path_or_data) if dy_path_or_data else None
    
    if dx is None or dy is None:
        ax.text(0.5, 0.5, 'Données de déplacement X/Y non disponibles', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Vecteurs de déplacement - Paire {pair_id}', fontsize=14, fontweight='bold')
        return fig
    
    # Charger le masque si nécessaire
    valid_mask_path_or_data = comparison_data.get('valid_mask')
    valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
    
    # Appliquer le masque
    if valid_mask is not None:
        dx_masked = np.where(valid_mask, dx, np.nan)
        dy_masked = np.where(valid_mask, dy, np.nan)
    else:
        dx_masked = dx
        dy_masked = dy
    
    # Sous-échantillonnage
    dx_sub = dx_masked[::subsample, ::subsample]
    dy_sub = dy_masked[::subsample, ::subsample]
    
    # Créer les grilles de coordonnées
    y, x = np.mgrid[0:dx_masked.shape[0]:subsample, 0:dx_masked.shape[1]:subsample]
    
    # Calculer la magnitude 2D pour la couleur
    magnitude_2d = np.sqrt(dx_sub**2 + dy_sub**2)
    
    # Quiver plot (scale réduit par 100 pour rendre les vecteurs plus visibles)
    quiver = ax.quiver(x, y, dx_sub, dy_sub, magnitude_2d, 
                       cmap='viridis', angles='xy', scale_units='xy', scale=0.01,
                       width=0.003, headwidth=3, headlength=4)
    
    ax.set_title(f'Vecteurs de déplacement (X, Y) - Paire {pair_id}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Colonne (pixels)')
    ax.set_ylabel('Ligne (pixels)')
    ax.set_aspect('equal')
    fig.colorbar(quiver, ax=ax, label='Magnitude 2D (m)')
    
    # Utiliser fig.tight_layout() au lieu de plt.tight_layout() pour éviter les conflits Qt
    try:
        fig.tight_layout()
    except Exception as e:
        logger.warning(f"Erreur lors de tight_layout: {e}")
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_displacement_histograms(comparison_data: Dict[str, Any], pair_id: str,
                                 output_path: Optional[str] = None) -> Figure:
    """
    Trace les histogrammes de distribution par composante (X, Y, Z, magnitude).
    
    Note: La magnitude est 3D (X, Y, Z) pour mnt_ortho, 2D (X, Y) pour ortho seul.
    
    Args:
        comparison_data: Données de comparaison pour une paire
        pair_id: Identifiant de la paire
        output_path: Chemin pour sauvegarder la figure (optionnel)
        
    Returns:
        Figure matplotlib
    """
    # Créer une nouvelle figure explicitement pour éviter les conflits avec Qt
    # Utiliser add_subplot au lieu de subplots pour plus de contrôle
    fig = Figure(figsize=(14, 10))
    fig.suptitle(f'Distributions des déplacements - Paire {pair_id}', fontsize=16, fontweight='bold')
    
    # Créer les axes manuellement pour éviter les problèmes avec subplots
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    axes = np.array([[ax1, ax2], [ax3, ax4]])
    
    valid_mask_path_or_data = comparison_data.get('valid_mask')
    valid_mask = load_data_if_needed(valid_mask_path_or_data) if valid_mask_path_or_data else None
    
    # Histogramme X
    ax = axes[0, 0]
    dx_path_or_data = comparison_data.get('displacement_x')
    dx = load_data_if_needed(dx_path_or_data) if dx_path_or_data else None
    if dx is not None:
        # Appliquer le masque d'abord pour obtenir les valeurs valides
        if valid_mask is not None:
            # S'assurer que le masque a la même taille que dx
            if valid_mask.shape == dx.shape:
                dx_masked = dx[valid_mask]
            else:
                # Si les tailles ne correspondent pas, utiliser le masque NaN
                dx_masked = dx[~np.isnan(dx) & (dx != 0)]
        else:
            dx_masked = dx[~np.isnan(dx)]
        
        # Filtrer les valeurs infinies et NaN
        dx_clean = dx_masked[np.isfinite(dx_masked)]
        
        if len(dx_clean) > 0:
            # Vérifier qu'il y a de la variance (pas toutes les valeurs identiques)
            if np.std(dx_clean) > 1e-10:  # Seuil très petit pour éviter les valeurs constantes
                # Sous-échantillonnage aléatoire pour réduire la charge mémoire
                max_samples = 10000  # Augmenter pour avoir plus de données
                if len(dx_clean) > max_samples:
                    dx_clean = np.random.choice(dx_clean, max_samples, replace=False)
                
                # Sous-échantillonnage final à 1000 valeurs pour l'histogramme
                if len(dx_clean) > 1000:
                    dx_clean = np.random.choice(dx_clean, 1000, replace=False)
            else:
                # Si toutes les valeurs sont identiques, prendre juste quelques échantillons
                dx_clean = dx_clean[:min(100, len(dx_clean))]
                
                # Nombre de bins adaptatif (max 50 pour 1000 échantillons)
                n_bins = min(50, max(20, int(np.sqrt(len(dx_clean)))))
                
                # Histogramme sans edgecolor pour réduire la charge de rendu
                n, bins, patches = ax.hist(dx_clean, bins=n_bins, alpha=0.7, color='blue', edgecolor=None)
                mean_val = np.mean(dx_clean)
                median_val = np.median(dx_clean)
                
                # Créer les lignes verticales de manière plus sûre
                legend_labels = []
                if np.isfinite(mean_val):
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Moyenne: {mean_val:.3f} m')
                if np.isfinite(median_val):
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Médiane: {median_val:.3f} m')
                
                # Légende simplifiée seulement si nécessaire
                if legend_labels:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=legend_labels[0] if len(legend_labels) > 0 else ''),
                        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label=legend_labels[1] if len(legend_labels) > 1 else '')
                    ]
                    ax.legend(handles=legend_elements[:len(legend_labels)], fontsize=9, loc='best')
                
            ax.set_xlabel('Déplacement X (m)')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution X', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution X', fontsize=12, fontweight='bold')
    
    # Histogramme Y
    ax = axes[0, 1]
    dy_path_or_data = comparison_data.get('displacement_y')
    dy = load_data_if_needed(dy_path_or_data) if dy_path_or_data else None
    if dy is not None:
        # Appliquer le masque d'abord
        if valid_mask is not None:
            if valid_mask.shape == dy.shape:
                dy_masked = dy[valid_mask]
            else:
                dy_masked = dy[~np.isnan(dy) & (dy != 0)]
        else:
            dy_masked = dy[~np.isnan(dy)]
        
        dy_clean = dy_masked[np.isfinite(dy_masked)]
        
        if len(dy_clean) > 0:
            if np.std(dy_clean) > 1e-10:
                max_samples = 10000
                if len(dy_clean) > max_samples:
                    dy_clean = np.random.choice(dy_clean, max_samples, replace=False)
                if len(dy_clean) > 1000:
                    dy_clean = np.random.choice(dy_clean, 1000, replace=False)
            else:
                dy_clean = dy_clean[:min(100, len(dy_clean))]
                
                # Nombre de bins adaptatif
                n_bins = min(50, max(20, int(np.sqrt(len(dy_clean)))))
                
                n, bins, patches = ax.hist(dy_clean, bins=n_bins, alpha=0.7, color='green', edgecolor=None)
                mean_val = np.mean(dy_clean)
                median_val = np.median(dy_clean)
                
                legend_labels = []
                if np.isfinite(mean_val):
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Moyenne: {mean_val:.3f} m')
                if np.isfinite(median_val):
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Médiane: {median_val:.3f} m')
                
                if legend_labels:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=legend_labels[0] if len(legend_labels) > 0 else ''),
                        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, label=legend_labels[1] if len(legend_labels) > 1 else '')
                    ]
                    ax.legend(handles=legend_elements[:len(legend_labels)], fontsize=9, loc='best')
                
            ax.set_xlabel('Déplacement Y (m)')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution Y', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution Y', fontsize=12, fontweight='bold')
    
    # Histogramme Z
    ax = axes[1, 0]
    dz_path_or_data = comparison_data.get('displacement_z')
    dz = load_data_if_needed(dz_path_or_data) if dz_path_or_data else None
    if dz is not None:
        # Appliquer le masque d'abord
        if valid_mask is not None:
            if valid_mask.shape == dz.shape:
                dz_masked = dz[valid_mask]
            else:
                dz_masked = dz[~np.isnan(dz) & (dz != 0)]
        else:
            dz_masked = dz[~np.isnan(dz)]
        
        dz_clean = dz_masked[np.isfinite(dz_masked)]
        
        if len(dz_clean) > 0:
            if np.std(dz_clean) > 1e-10:
                max_samples = 10000
                if len(dz_clean) > max_samples:
                    dz_clean = np.random.choice(dz_clean, max_samples, replace=False)
                if len(dz_clean) > 1000:
                    dz_clean = np.random.choice(dz_clean, 1000, replace=False)
            else:
                dz_clean = dz_clean[:min(100, len(dz_clean))]
                
                # Nombre de bins adaptatif
                n_bins = min(50, max(20, int(np.sqrt(len(dz_clean)))))
                
                n, bins, patches = ax.hist(dz_clean, bins=n_bins, alpha=0.7, color='red', edgecolor=None)
                mean_val = np.mean(dz_clean)
                median_val = np.median(dz_clean)
                
                legend_labels = []
                if np.isfinite(mean_val):
                    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Moyenne: {mean_val:.3f} m')
                if np.isfinite(median_val):
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Médiane: {median_val:.3f} m')
                
                if legend_labels:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, label=legend_labels[0] if len(legend_labels) > 0 else ''),
                        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label=legend_labels[1] if len(legend_labels) > 1 else '')
                    ]
                    ax.legend(handles=legend_elements[:len(legend_labels)], fontsize=9, loc='best')
                
            ax.set_xlabel('Déplacement Z (m)')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution Z', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution Z', fontsize=12, fontweight='bold')
    
    # Histogramme Magnitude (3D pour mnt_ortho, 2D pour ortho seul)
    ax = axes[1, 1]
    mag_path_or_data = comparison_data.get('displacement_magnitude')
    mag = load_data_if_needed(mag_path_or_data) if mag_path_or_data else None
    if mag is not None:
        # Appliquer le masque d'abord
        if valid_mask is not None:
            if valid_mask.shape == mag.shape:
                mag_masked = mag[valid_mask]
            else:
                mag_masked = mag[~np.isnan(mag) & (mag != 0)]
        else:
            mag_masked = mag[~np.isnan(mag)]
        
        mag_clean = mag_masked[np.isfinite(mag_masked)]
        
        if len(mag_clean) > 0:
            if np.std(mag_clean) > 1e-10:
                max_samples = 10000
                if len(mag_clean) > max_samples:
                    mag_clean = np.random.choice(mag_clean, max_samples, replace=False)
                if len(mag_clean) > 1000:
                    mag_clean = np.random.choice(mag_clean, 1000, replace=False)
            else:
                mag_clean = mag_clean[:min(100, len(mag_clean))]
                
                # Nombre de bins adaptatif
                n_bins = min(50, max(20, int(np.sqrt(len(mag_clean)))))
                
                n, bins, patches = ax.hist(mag_clean, bins=n_bins, alpha=0.7, color='purple', edgecolor=None)
                mean_val = np.mean(mag_clean)
                median_val = np.median(mag_clean)
                
                legend_labels = []
                if np.isfinite(mean_val):
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Moyenne: {mean_val:.3f} m')
                if np.isfinite(median_val):
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                    legend_labels.append(f'Médiane: {median_val:.3f} m')
                
                if legend_labels:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=legend_labels[0] if len(legend_labels) > 0 else ''),
                        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label=legend_labels[1] if len(legend_labels) > 1 else '')
                    ]
                    ax.legend(handles=legend_elements[:len(legend_labels)], fontsize=9, loc='best')
                
            # Déterminer si c'est 3D ou 2D selon la présence de Z
            if comparison_data.get('displacement_z') is not None:  # Peut être un chemin ou None
                ax.set_xlabel('Magnitude 3D (m)')
                ax.set_title('Distribution Magnitude 3D', fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel('Magnitude 2D (m)')
                ax.set_title('Distribution Magnitude 2D', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fréquence')
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution Magnitude', fontsize=12, fontweight='bold')
    
    # Utiliser fig.tight_layout() au lieu de plt.tight_layout() pour éviter les conflits Qt
    try:
        fig.tight_layout()
    except Exception as e:
        logger.warning(f"Erreur lors de tight_layout: {e}")
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison_matrix(matrix: np.ndarray, matrix_name: str, model_labels: List[str],
                          output_path: Optional[str] = None) -> Figure:
    """
    Trace une heatmap d'une matrice de comparaison.
    
    Args:
        matrix: Matrice N×N à visualiser
        matrix_name: Nom de la métrique (pour le titre)
        model_labels: Labels des modèles
        output_path: Chemin pour sauvegarder la figure (optionnel)
        
    Returns:
        Figure matplotlib
    """
    # Créer une nouvelle figure explicitement pour éviter les conflits avec Qt
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Créer la heatmap
    mask = np.isnan(matrix) | (matrix == 0)
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis', 
                mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=model_labels, yticklabels=model_labels, ax=ax)
    
    ax.set_title(f'Matrice de {matrix_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modèle j')
    ax.set_ylabel('Modèle i')
    
    # Utiliser fig.tight_layout() au lieu de plt.tight_layout() pour éviter les conflits Qt
    try:
        fig.tight_layout()
    except Exception as e:
        logger.warning(f"Erreur lors de tight_layout: {e}")
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig

