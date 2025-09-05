import subprocess, sys, os, shutil

if __name__ == "__main__":
    # l_res = [i/2 for i in range(9, 21)] + [10 + i for i in range(1, 11)] + [i*10 for i in range(2, 11)]
    # l_res_mm = [31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    # l_res_mm = [i+0.5 for i in range(10, 50)]
    l_res_mm = [i/2 for i in range(101, 151)]

    l_res_m = [i/1000 for i in l_res_mm]

    dossier_resume = "C:/Users/hugor/Documents/THESE/02_pro/03_repos/photogrammetry_cursor/test_analysis/dossier_analyses"

    # Paramètres (à adapter)
    analysis_type = "mnt_ortho"  # "mnt", "ortho" ou "mnt_ortho"
    image1 = r"C:/Users/hugor/Documents/THESE/02_pro/03_repos/photogrammetry_cursor/test_analysis/ortho_unified_final_07_09.tif"
    image2 = r"C:/Users/hugor/Documents/THESE/02_pro/03_repos/photogrammetry_cursor/test_analysis/ortho_unified_final_10_11.tif"
    mnt1 = r"C:/Users/hugor/Documents/THESE/02_pro/03_repos/photogrammetry_cursor/test_analysis/mnt_unified_final_07_09.tif"   # requis si mnt_ortho
    mnt2 = r"C:/Users/hugor/Documents/THESE/02_pro/03_repos/photogrammetry_cursor/test_analysis/mnt_unified_final_10_11.tif"   # requis si mnt_ortho
             # en mètres
    pyr_scale = 0.8
    levels = 5
    iterations = 10
    poly_n = 7
    poly_sigma = 1.2

    for (resolution_m, resolution_mm) in zip(l_res_m, l_res_mm):

        cmd = [
            sys.executable, "photogeoalign.py",
            "--analysis",
            f"--type={analysis_type}",
            f"--image1={image1}",
            f"--image2={image2}",
            f"--resolution={resolution_m}",
            f"--pyr-scale={pyr_scale}",
            f"--levels={levels}",
            f"--iterations={iterations}",
            f"--poly-n={poly_n}",
            f"--poly-sigma={poly_sigma}",
        ]

        # Ajouter MNTs si mode mnt_ortho
        if analysis_type == "mnt_ortho":
            cmd += [f"--mnt1={mnt1}", f"--mnt2={mnt2}"]
        print("lancement du code pour la résolution", resolution_m, "m")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Code retour:", result.returncode)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        os.rename(os.path.join(os.path.dirname(image1), f'analysis_results/analysis_report_{analysis_type}.txt'), os.path.join(os.path.dirname(image1), f'analysis_results/analysis_report_{analysis_type}_{str(resolution_mm).replace(".", "v")}.txt'))
        shutil.move(os.path.join(os.path.dirname(image1), f'analysis_results/analysis_report_{analysis_type}_{str(resolution_mm).replace(".", "v")}.txt'), dossier_resume)

        path = os.path.join(os.path.dirname(image1), f'analysis_results')
        if os.path.exists(path):
            shutil.rmtree(path)