from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDialogButtonBox

class JobExportDialog(QDialog):
    def __init__(self, parent=None, job_name="PhotoGeoAlign", output="PhotoGeoAlign.out", partition="ncpu", ntasks=128, cli_cmd=""):
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
        self.cli_cmd = cli_cmd
        layout.addRow("Nom du job :", self.job_name_edit)
        layout.addRow("Fichier de sortie :", self.output_edit)
        layout.addRow("Partition :", self.partition_combo)
        layout.addRow("Nombre de tâches :", self.ntasks_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_values(self):
        return {
            "job_name": self.job_name_edit.text().strip() or "PhotoGeoAlign",
            "output": self.output_edit.text().strip() or "PhotoGeoAlign.out",
            "partition": self.partition_combo.currentText(),
            "ntasks": self.ntasks_spin.value(),
            "cli_cmd": self.cli_cmd
        } 