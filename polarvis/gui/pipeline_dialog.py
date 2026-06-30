from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QProgressBar, QHBoxLayout, QComboBox


class PipelineDialog(QDialog):
    # TODO transfer UI construction into a self-contained .ui file
    # TODO add verbosity selection

    def __init__(self, calibration_manager, parent=None):
        super().__init__(parent)
        self.calibration_manager = calibration_manager

        self.setModal(True)
        self.setWindowTitle("Run Pipeline")
        self.setFixedSize(300, 180)

        layout = QVBoxLayout()
        window_layout = QHBoxLayout()

        self.label = QLabel("Run image processing?")

        # Calibration selection
        cal_layout = QHBoxLayout()
        cal_label = QLabel("Calibration:")

        self.calibration_combo = QComboBox()

        cal_layout.addWidget(cal_label)
        cal_layout.addWidget(self.calibration_combo)

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        layout.addWidget(self.label)
        layout.addLayout(cal_layout)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.run_btn.clicked.connect(self.on_run_clicked)
        self.cancel_btn.clicked.connect(self.reject)

    def on_run_clicked(self):
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.label.setText("Processing...")
        self.progress.show()
        self.accept()

    def _populate_calibrations(self):
        cal_ids = self.calibration_manager.list_calibrations()

        if not cal_ids:
            self.calibration_combo.addItem("No calibrations available")
            self.calibration_combo.setEnabled(False)
            return

        self.calibration_combo.addItems(cal_ids)

        # Set default
        default_id = self.calibration_manager.get_default()
        index = self.calibration_combo.findText(default_id)
        if index >= 0:
            self.calibration_combo.setCurrentIndex(index)

    def get_calibration_id(self) -> str:
        return self.calibration_combo.currentText()