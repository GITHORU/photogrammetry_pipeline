from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDialogButtonBox

class JobExportDialog(QDialog):
    def __init__(self, parent=None, job_name="PhotoGeoAlign", output="PhotoGeoAlign.job", partition="ncpu", ntasks=32, time_limit="48:00:00", cli_cmd=""):
        super().__init__(parent)
        self.setWindowTitle("Exporter le batch SLURM (.job)")
        self.setModal(True)
        layout = QFormLayout(self)
        self.job_name_edit = QLineEdit(job_name)
        self.output_edit = QLineEdit(output)
        self.partition_combo = QComboBox()
        self.partition_combo.addItems(["ncpu", "ncpum", "ncpu_long", "ncpu_short"])
        self.partition_combo.setCurrentText(partition)
        self.ntasks_spin = QSpinBox()
        self.ntasks_spin.setMinimum(1)
        self.ntasks_spin.setMaximum(1024)
        self.ntasks_spin.setValue(ntasks)
        self.time_edit = QLineEdit(time_limit)
        self.time_edit.setToolTip("Format: HH:MM:SS ou DD-HH:MM:SS (ex: 48:00:00 pour 48h)")
        self.cli_cmd = cli_cmd
        layout.addRow("Nom du job :", self.job_name_edit)
        layout.addRow("Fichier de sortie :", self.output_edit)
        layout.addRow("Partition :", self.partition_combo)
        layout.addRow("Nombre de tâches :", self.ntasks_spin)
        layout.addRow("Temps limite :", self.time_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_values(self):
        return {
            "job_name": self.job_name_edit.text().strip() or "PhotoGeoAlign",
            "output": self.output_edit.text().strip() or "PhotoGeoAlign.job",
            "partition": self.partition_combo.currentText(),
            "ntasks": self.ntasks_spin.value(),
            "time_limit": self.time_edit.text().strip() or "48:00:00",
            "cli_cmd": self.cli_cmd
        } 