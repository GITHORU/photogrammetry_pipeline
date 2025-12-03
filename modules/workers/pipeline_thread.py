import logging
import os
import sys
from PySide6.QtCore import QThread, Signal
from ..core.micmac import (
    run_micmac_tapioca, run_micmac_tapas, run_micmac_c3dc,
    run_micmac_saisieappuisinit, run_micmac_gcpbascule_init,
    run_micmac_saisieappuispredic, run_micmac_gcpbascule_predic,
    run_micmac_xifgps2xml, run_micmac_oriconvert,
    run_micmac_centerbascule, run_micmac_campari
)
from .utils import QtLogHandler

class PipelineThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, 
                 use_rtk=False, rtk_positions_file=None, rtk_gps_weights=(0.05, 0.07),
                 run_tapioca=True, run_tapas=True, run_saisieappuisinit=True, run_saisieappuispredic=True, run_c3dc=True):
        super().__init__()
        self.input_dir = input_dir
        self.mode = mode
        self.zoomf = zoomf
        self.tapas_model = tapas_model
        self.tapioca_extra = tapioca_extra
        self.tapas_extra = tapas_extra
        self.saisieappuisinit_extra = saisieappuisinit_extra
        self.saisieappuispredic_extra = saisieappuispredic_extra
        self.c3dc_extra = c3dc_extra
        self.saisieappuisinit_pt = saisieappuisinit_pt
        self.use_rtk = use_rtk
        self.rtk_positions_file = rtk_positions_file
        self.rtk_gps_weights = rtk_gps_weights
        self.run_tapioca = run_tapioca
        self.run_tapas = run_tapas
        self.run_saisieappuisinit = run_saisieappuisinit
        self.run_saisieappuispredic = run_saisieappuispredic
        self.run_c3dc = run_c3dc

    def run(self):
        logger = logging.getLogger(f"PhotogrammetryPipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        # Handler pour écrire dans stdout (capturé par SLURM dans le fichier .out)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # Handler pour écrire dans le fichier .log
        log_path = os.path.join(self.input_dir, 'photogrammetry_pipeline.log')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Handler pour l'interface graphique (Qt)
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        try:
            self.log_signal.emit("Démarrage du pipeline...\n")
            
            # Pipeline RTK : étapes préliminaires (si RTK activé et Tapas va être lancé)
            rawgnss_n = None
            if self.use_rtk and self.run_tapas:
                if not self.rtk_positions_file:
                    raise RuntimeError("Fichier de positions RTK requis lorsque l'option RTK est activée.")
                self.log_signal.emit("=== Pipeline RTK activé ===\n")
                self.log_signal.emit("Étape 0.1 : Extraction GPS depuis EXIF...\n")
                run_micmac_xifgps2xml(self.input_dir, logger)
                self.log_signal.emit("XifGps2Xml terminé.\n")
                
                self.log_signal.emit("Étape 0.2 : Conversion GPS en orientations initiales...\n")
                rawgnss_n, neighbor_file = run_micmac_oriconvert(self.input_dir, logger, self.rtk_positions_file, self.tapas_model)
                self.log_signal.emit("OriConvert terminé.\n")
            
            # Tapioca
            if self.run_tapioca:
                if self.use_rtk:
                    run_micmac_tapioca(self.input_dir, logger, use_neighbor_file=True, extra_params=self.tapioca_extra)
                else:
                    run_micmac_tapioca(self.input_dir, logger, use_neighbor_file=False, extra_params=self.tapioca_extra)
                self.log_signal.emit("Tapioca terminé.\n")
            
            # Tapas
            if self.run_tapas:
                if self.use_rtk:
                    out_ori = run_micmac_tapas(self.input_dir, logger, self.tapas_model, use_arbitrary=True, extra_params=self.tapas_extra)
                    self.log_signal.emit("Tapas terminé (orientation Arbitrary).\n")
                    
                    # CenterBascule
                    self.log_signal.emit("Étape 3.1 : Recalage initial avec GPS RTK...\n")
                    ori_after_center = run_micmac_centerbascule(self.input_dir, logger, ori_in=out_ori, rawgnss_n=rawgnss_n, tapas_model=self.tapas_model)
                    self.log_signal.emit("CenterBascule terminé.\n")
                    
                    # Campari
                    self.log_signal.emit("Étape 3.2 : Ajustement bundle avec contraintes GPS RTK...\n")
                    final_ori = run_micmac_campari(self.input_dir, logger, ori_in=ori_after_center, tapas_model=self.tapas_model, 
                                                   rawgnss_n=rawgnss_n, gps_weights=self.rtk_gps_weights)
                    self.log_signal.emit("Campari terminé (orientation normalisée à {}).\n".format(final_ori))
                else:
                    run_micmac_tapas(self.input_dir, logger, self.tapas_model, use_arbitrary=False, extra_params=self.tapas_extra)
                    self.log_signal.emit("Tapas terminé.\n")
            
            # Étapes GCP (identiques dans les deux cas, utilisent maintenant tapas_model normalisé)
            ori_abs_init = None
            if self.run_saisieappuisinit:
                run_micmac_saisieappuisinit(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt, self.saisieappuisinit_extra)
                self.log_signal.emit("SaisieAppuisInitQT terminé.\n")
                ori_abs_init = run_micmac_gcpbascule_init(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (init) terminé.\n")
            if self.run_saisieappuispredic:
                run_micmac_saisieappuispredic(self.input_dir, logger, self.tapas_model, ori_abs_init, self.saisieappuisinit_pt, self.saisieappuispredic_extra)
                self.log_signal.emit("SaisieAppuisPredicQT terminé.\n")
                run_micmac_gcpbascule_predic(self.input_dir, logger, self.tapas_model, self.saisieappuisinit_pt)
                self.log_signal.emit("GCPBascule (predic) terminé.\n")
            
            # C3DC
            if self.run_c3dc:
                run_micmac_c3dc(self.input_dir, logger, mode=self.mode, zoomf=self.zoomf, tapas_model=self.tapas_model, extra_params=self.c3dc_extra)
                self.log_signal.emit("C3DC terminé.\n")
            
            self.finished_signal.emit(True, "Pipeline terminé avec succès !")
        except Exception as e:
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline : {e}") 