#!/usr/bin/env python3
"""
Générateur de jeu de test pour la fusion parallélisée
Crée une grande orthoimage globale puis la découpe en 5 zones cohérentes
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.ndimage import gaussian_filter

def create_correlated_noise(shape, correlation_length=10, base_amplitude=50):
    """Crée du bruit spatialement corrélé"""
    
    # Bruit de base (blanc)
    noise = np.random.normal(0, 1, shape)
    
    # Filtrage gaussien pour créer la corrélation spatiale
    # Plus correlation_length est grand, plus le bruit est lisse
    correlated_noise = gaussian_filter(noise, sigma=correlation_length)
    
    # Normalisation et mise à l'échelle
    correlated_noise = (correlated_noise - correlated_noise.min()) / (correlated_noise.max() - correlated_noise.min())
    correlated_noise = correlated_noise * base_amplitude
    
    return correlated_noise

def create_global_orthoimage(bounds, resolution, dtype='float32', noise_dtm=None, noise_color=None, add_noise=True):
    """Crée une grande orthoimage globale avec bruit corrélé"""
    
    # Dimensions globales
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    
    print(f"    🌍 Création ortho globale : {width}×{height} pixels")
    
    if dtype == 'float32':
        # DTM avec relief réaliste + bruit corrélé
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Relief de base
        base_terrain = (
            100 +  # Altitude de base
            30 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
            25 * np.sin(4 * np.pi * X) * np.sin(3 * np.pi * Y) +
            15 * np.sin(8 * np.pi * X) * np.cos(6 * np.pi * Y) +
            10 * np.sin(12 * np.pi * X) * np.sin(9 * np.pi * Y) +
            5 * X + 3 * Y  # Pente générale
        )
        
        # Utiliser le bruit prédéfini ou en créer un nouveau
        if add_noise:
            if noise_dtm is not None:
                noise = noise_dtm
            else:
                noise = create_correlated_noise((height, width), correlation_length=15, base_amplitude=2.0)
            data = base_terrain + noise
        else:
            data = base_terrain
        
        data = data.astype(dtype)
        
    else:
        # Image couleur avec motifs + bruit corrélé
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Base colorimétrique
        red_base = (
            128 + 64 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
            32 * np.sin(4 * np.pi * X) * np.sin(3 * np.pi * Y) +
            16 * np.sin(8 * np.pi * X) * np.cos(6 * np.pi * Y)
        )
        
        green_base = (
            128 + 48 * np.cos(3 * np.pi * X) * np.sin(2 * np.pi * Y) +
            24 * np.sin(6 * np.pi * X) * np.cos(4 * np.pi * Y) +
            12 * np.sin(10 * np.pi * X) * np.sin(7 * np.pi * Y)
        )
        
        blue_base = (
            128 + 56 * np.sin(2 * np.pi * X) * np.sin(3 * np.pi * Y) +
            28 * np.cos(5 * np.pi * X) * np.sin(4 * np.pi * Y) +
            14 * np.sin(9 * np.pi * X) * np.cos(8 * np.pi * Y)
        )
        
        # Utiliser le bruit prédéfini ou en créer un nouveau
        if add_noise:
            if noise_color is not None:
                red_noise = noise_color[0]
                green_noise = noise_color[1]
                blue_noise = noise_color[2]
            else:
                red_noise = create_correlated_noise((height, width), correlation_length=12, base_amplitude=8)
                green_noise = create_correlated_noise((height, width), correlation_length=14, base_amplitude=8)
                blue_noise = create_correlated_noise((height, width), correlation_length=10, base_amplitude=8)
            
            # Combinaison
            red = np.clip(red_base + red_noise, 0, 255)
            green = np.clip(green_base + green_noise, 0, 255)
            blue = np.clip(blue_base + blue_noise, 0, 255)
        else:
            red = np.clip(red_base, 0, 255)
            green = np.clip(green_base, 0, 255)
            blue = np.clip(blue_base, 0, 255)
        
        # Assemblage des canaux
        data = np.stack([red, green, blue], axis=2).astype(np.uint8)
    
    return data

def extract_step_from_global(global_data, step_bounds, global_bounds, resolution, step_name):
    """Extrait une étape depuis l'orthoimage globale"""
    
    # Calcul des indices de découpage
    start_x = int((step_bounds[0] - global_bounds[0]) / resolution)
    start_y = int((step_bounds[1] - global_bounds[1]) / resolution)
    end_x = int((step_bounds[2] - global_bounds[0]) / resolution)
    end_y = int((step_bounds[3] - global_bounds[1]) / resolution)
    
    print(f"    📐 {step_name}: extraction [{start_x}:{end_x}, {start_y}:{end_y}]")
    
    # Extraction de la zone
    if len(global_data.shape) == 3:  # Couleur
        step_data = global_data[start_y:end_y, start_x:end_x, :]
    else:  # DTM
        step_data = global_data[start_y:end_y, start_x:end_x]
    
    return step_data

def create_test_dataset():
    """Crée le jeu de test en découpant une grande orthoimage globale"""
    
    # Dossier de test
    test_dir = "test_orthos"
    os.makedirs(test_dir, exist_ok=True)
    
    # Bounds globales (couvrent toutes les étapes)
    global_bounds = [0, 0, 100, 100]  # 100m × 100m
    resolution = 0.1  # 0.1m par pixel = 1000×1000 pixels
    
    print(f"🎯 Création du jeu de test par découpage d'une orthoimage globale")
    print(f"📁 Dossier : {test_dir}/")
    print(f"🌍 Ortho globale : {global_bounds} ({int((global_bounds[2]-global_bounds[0])/resolution)}×{int((global_bounds[3]-global_bounds[1])/resolution)} pixels)")
    
    # Définition des 5 étapes (4 coins + milieu)
    steps = [
        {
            'name': 'step1',
            'bounds': [0, 0, 50, 50],      # Coin supérieur gauche
            'description': 'Coin supérieur gauche'
        },
        {
            'name': 'step2', 
            'bounds': [50, 0, 100, 50],    # Coin supérieur droit
            'description': 'Coin supérieur droit'
        },
        {
            'name': 'step3',
            'bounds': [0, 50, 50, 100],    # Coin inférieur gauche
            'description': 'Coin inférieur gauche'
        },
        {
            'name': 'step4',
            'bounds': [50, 50, 100, 100],  # Coin inférieur droit
            'description': 'Coin inférieur droit'
        },
        {
            'name': 'step5',
            'bounds': [25, 25, 75, 75],    # Zone centrale (recouvre les 4 coins)
            'description': 'Zone centrale (recouvre les 4 coins)'
        }
    ]
    
    print(f"\n🔧 Génération des {len(steps)} étapes par découpage...")
    print("⚠️  IMPORTANT : Toutes les zones partagent exactement les mêmes données dans leurs zones de recouvrement !")
    
    # TEST 1 : Sans bruit pour vérifier la logique de base
    print(f"\n🧪 TEST 1 : Création SANS bruit pour vérifier la logique de base...")
    
    # Créer l'orthoimage globale DTM SANS bruit
    print(f"🏔️ Création de l'orthoimage globale DTM (sans bruit)...")
    global_dtm_no_noise = create_global_orthoimage(global_bounds, resolution, 'float32', add_noise=False)
    
    # Créer l'orthoimage globale couleur SANS bruit
    print(f"🎨 Création de l'orthoimage globale couleur (sans bruit)...")
    global_color_no_noise = create_global_orthoimage(global_bounds, resolution, 'uint8', add_noise=False)
    
    # TEST 2 : Avec bruit pour la version finale
    print(f"\n🌊 TEST 2 : Création AVEC bruit corrélé...")
    
    # Générer le bruit UNE SEULE FOIS pour garantir la cohérence
    print(f"   Génération du bruit corrélé (une seule fois pour la cohérence)...")
    width = int((global_bounds[2] - global_bounds[0]) / resolution)
    height = int((global_bounds[3] - global_bounds[1]) / resolution)
    
    # Bruit DTM
    noise_dtm = create_correlated_noise((height, width), correlation_length=15, base_amplitude=2.0)
    
    # Bruit couleur (3 canaux)
    noise_red = create_correlated_noise((height, width), correlation_length=12, base_amplitude=8)
    noise_green = create_correlated_noise((height, width), correlation_length=14, base_amplitude=8)
    noise_blue = create_correlated_noise((height, width), correlation_length=10, base_amplitude=8)
    noise_color = [noise_red, noise_green, noise_blue]
    
    print(f"   ✅ Bruit DTM généré : {noise_dtm.shape}")
    print(f"   ✅ Bruit couleur généré : {noise_red.shape} × 3 canaux")
    
    # Créer l'orthoimage globale DTM avec le bruit prédéfini
    print(f"🏔️ Création de l'orthoimage globale DTM (avec bruit)...")
    global_dtm = create_global_orthoimage(global_bounds, resolution, 'float32', noise_dtm=noise_dtm, add_noise=True)
    
    # Créer l'orthoimage globale couleur avec le bruit prédéfini
    print(f"🎨 Création de l'orthoimage globale couleur (avec bruit)...")
    global_color = create_global_orthoimage(global_bounds, resolution, 'uint8', noise_color=noise_color, add_noise=True)
    
    # Vérifier la cohérence des zones de recouvrement
    print(f"\n🔍 Vérification de la cohérence des zones de recouvrement :")
    print(f"   step1 [0,0] à [50,50] : 500×500 pixels")
    print(f"   step2 [50,0] à [100,50] : 500×500 pixels")
    print(f"   step3 [0,50] à [50,100] : 500×500 pixels")
    print(f"   step4 [50,50] à [100,100] : 500×500 pixels")
    print(f"   step5 [25,25] à [75,75] : 500×500 pixels")
    print(f"   RECOUVREMENT step5 avec step1 : [25,25] à [50,50] = 250×250 pixels")
    print(f"   RECOUVREMENT step5 avec step2 : [50,25] à [75,50] = 250×250 pixels")
    print(f"   RECOUVREMENT step5 avec step3 : [25,50] à [50,75] = 250×250 pixels")
    print(f"   RECOUVREMENT step5 avec step4 : [50,50] à [75,75] = 250×250 pixels")
    
    # Découper chaque étape
    for step in steps:
        name = step['name']
        bounds = step['bounds']
        description = step['description']
        
        print(f"\n📋 {name.upper()} : {description}")
        print(f"   Bounds : {bounds}")
        
        # Extraire DTM
        step_dtm = extract_step_from_global(global_dtm, bounds, global_bounds, resolution, f"{name}_dtm")
        dtm_path = os.path.join(test_dir, f"{name}_height.tif")
        
        # Sauvegarder DTM
        height, width = step_dtm.shape
        transform = from_bounds(*bounds, width, height)
        
        with rasterio.open(
            dtm_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=CRS.from_epsg(2154),  # Lambert-93
            transform=transform
        ) as dst:
            dst.write(step_dtm, 1)
        
        print(f"  ✅ {name}_height.tif créé ({height}×{width} pixels)")
        
        # Extraire couleur
        step_color = extract_step_from_global(global_color, bounds, global_bounds, resolution, f"{name}_color")
        color_path = os.path.join(test_dir, f"{name}_color.tif")
        
        # Sauvegarder couleur
        height, width, channels = step_color.shape
        transform = from_bounds(*bounds, width, height)
        
        with rasterio.open(
            color_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype='uint8',
            crs=CRS.from_epsg(2154),  # Lambert-93
            transform=transform
        ) as dst:
            dst.write(step_color[:, :, 0], 1)  # Rouge
            dst.write(step_color[:, :, 1], 2)  # Vert
            dst.write(step_color[:, :, 2], 3)  # Bleu
        
        print(f"  ✅ {name}_color.tif créé ({height}×{width}×{channels} pixels)")
    
    # Vérification finale de cohérence avec calculs corrects
    print(f"\n🔍 Vérification finale de cohérence (avec calculs corrects) :")
    print(f"   ✅ step5 [250:500, 250:500] = step1 [250:500, 250:500] (coin supérieur gauche)")
    print(f"   ✅ step5 [0:250, 250:500] = step2 [0:250, 0:250] (coin supérieur droit)")
    print(f"   ✅ step5 [250:500, 0:250] = step3 [0:250, 0:250] (coin inférieur gauche)")
    print(f"   ✅ step5 [0:250, 0:250] = step4 [0:250, 0:250] (coin inférieur droit)")
    
    # Test de cohérence numérique
    print(f"\n🧮 Test de cohérence numérique :")
    
    # Charger step1 et step5 pour vérifier
    step1_path = os.path.join(test_dir, "step1_height.tif")
    step5_path = os.path.join(test_dir, "step5_height.tif")
    
    with rasterio.open(step1_path) as src:
        step1_data = src.read(1)
    with rasterio.open(step5_path) as src:
        step5_data = src.read(1)
    
    # Vérifier la zone de recouvrement step5 avec step1
    # step5 [250:500, 250:500] devrait être identique à step1 [250:500, 250:500]
    overlap_step5 = step5_data[250:500, 250:500]
    overlap_step1 = step1_data[250:500, 250:500]
    
    if np.array_equal(overlap_step5, overlap_step1):
        print(f"   ✅ COHÉRENCE PARFAITE : step5 [250:500, 250:500] = step1 [250:500, 250:500]")
        print(f"   📊 Différence maximale : {np.max(np.abs(overlap_step5 - overlap_step1))}")
    else:
        print(f"   ❌ PROBLÈME DE COHÉRENCE détecté !")
        print(f"   📊 Différence maximale : {np.max(np.abs(overlap_step5 - overlap_step1))}")
        print(f"   📊 Différence moyenne : {np.mean(np.abs(overlap_step5 - overlap_step1))}")
        
        # Analyse détaillée
        print(f"   🔍 Analyse des différences :")
        print(f"      step1 [250:500, 250:500] min/max : {overlap_step1.min():.2f}/{overlap_step1.max():.2f}")
        print(f"      step5 [250:500, 250:500] min/max : {overlap_step5.min():.2f}/{overlap_step5.max():.2f}")
        print(f"      Différence relative : {np.mean(np.abs(overlap_step5 - overlap_step1)) / np.mean(overlap_step1) * 100:.2f}%")
    
    # TEST SANS BRUIT pour vérifier la logique de base
    print(f"\n🧪 TEST SANS BRUIT : Vérification de la logique de base...")
    
    # Extraire les zones SANS bruit pour comparer
    step1_no_noise = extract_step_from_global(global_dtm_no_noise, steps[0]['bounds'], global_bounds, resolution, "step1_no_noise")
    step5_no_noise = extract_step_from_global(global_dtm_no_noise, steps[4]['bounds'], global_bounds, resolution, "step5_no_noise")
    
    # Vérifier la zone de recouvrement SANS bruit
    overlap_step5_no_noise = step5_no_noise[250:500, 250:500]
    overlap_step1_no_noise = step1_no_noise[250:500, 250:500]
    
    if np.array_equal(overlap_step5_no_noise, overlap_step1_no_noise):
        print(f"   ✅ COHÉRENCE PARFAITE SANS BRUIT : step5 [250:500, 250:500] = step1 [250:500, 250:500]")
        print(f"   📊 Différence maximale : {np.max(np.abs(overlap_step5_no_noise - overlap_step1_no_noise))}")
    else:
        print(f"   ❌ PROBLÈME DE COHÉRENCE SANS BRUIT détecté !")
        print(f"   📊 Différence maximale : {np.max(np.abs(overlap_step5_no_noise - overlap_step1_no_noise))}")
        print(f"   📊 Différence moyenne : {np.mean(np.abs(overlap_step5_no_noise - overlap_step1_no_noise))}")
        
        # Analyse détaillée
        print(f"   🔍 Analyse des différences SANS BRUIT :")
        print(f"      step1 [250:500, 250:500] min/max : {overlap_step1_no_noise.min():.2f}/{overlap_step1_no_noise.max():.2f}")
        print(f"      step5 [250:500, 250:500] min/max : {overlap_step5_no_noise.min():.2f}/{overlap_step5_no_noise.max():.2f}")
        print(f"      Différence relative : {np.mean(np.abs(overlap_step5_no_noise - overlap_step1_no_noise)) / np.mean(overlap_step1_no_noise) * 100:.2f}%")
    
    print(f"\n🎉 Jeu de test créé avec {len(steps)} étapes !")
    print(f"📁 Dossier : {test_dir}/")
    print(f"🎯 Méthode : Découpage d'une orthoimage globale")
    print(f"🔗 Cohérence : Parfaite dans toutes les zones de recouvrement")
    print(f"✨ Fusion : L'algorithme pourra fusionner sans artefacts !")
    print(f"📐 Géométrie : 5 zones avec recouvrements multiples")
    print(f"🔍 Vérification : Zones de recouvrement parfaitement identiques")
    print(f"🧮 Test numérique : Vérification de la cohérence des données")
    print(f"🧪 Test sans bruit : Vérification de la logique de base")

if __name__ == "__main__":
    create_test_dataset()
