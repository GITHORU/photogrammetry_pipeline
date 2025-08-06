import logging
import os
from PySide6.QtCore import QThread, Signal
from ..core.micmac import (
    run_micmac_tapioca, run_micmac_tapas, run_micmac_c3dc,
    run_micmac_saisieappuisinit, run_micmac_gcpbascule_init,
    run_micmac_saisieappuispredic, run_micmac_gcpbascule_predic
)
from .utils import QtLogHandler

class PipelineThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, input_dir, mode, zoomf, tapas_model, tapioca_extra, tapas_extra, saisieappuisinit_extra, saisieappuispredic_extra, c3dc_extra, saisieappuisinit_pt, run_tapioca=True, run_tapas=True, run_saisieappuisinit=True, run_saisieappuispredic=True, run_c3dc=True):
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
        self.run_tapioca = run_tapioca
        self.run_tapas = run_tapas
        self.run_saisieappuisinit = run_saisieappuisinit
        self.run_saisieappuispredic = run_saisieappuispredic
        self.run_c3dc = run_c3dc

    def run(self):
        logger = logging.getLogger(f"PhotogrammetryPipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        qt_handler = QtLogHandler(self.log_signal)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(qt_handler)
        try:
            self.log_signal.emit("Démarrage du pipeline...\n")
            if self.run_tapioca:
                run_micmac_tapioca(self.input_dir, logger, self.tapioca_extra)
                self.log_signal.emit("Tapioca terminé.\n")
            if self.run_tapas:
                run_micmac_tapas(self.input_dir, logger, self.tapas_model, self.tapas_extra)
                self.log_signal.emit("Tapas terminé.\n")
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
            if self.run_c3dc:
                run_micmac_c3dc(self.input_dir, logger, mode=self.mode, zoomf=self.zoomf, tapas_model=self.tapas_model, extra_params=self.c3dc_extra)
                self.log_signal.emit("C3DC terminé.\n")
            self.finished_signal.emit(True, "Pipeline terminé avec succès !")
        except Exception as e:
            self.log_signal.emit(f"Erreur : {e}\n")
            self.finished_signal.emit(False, f"Erreur lors de l'exécution du pipeline : {e}") 