from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QProgressBar, QHBoxLayout, QSpinBox

class PipelineDialog(QDialog):
    # TODO transfer UI construction into a self-contained .ui file
    # TODO add verbosity selection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Run Pipeline")
        self.setFixedSize(300, 180)

        layout = QVBoxLayout()

        self.label = QLabel("Run image processing?")

        # Sliding window size selector
        window_layout = QHBoxLayout()
        window_label = QLabel("Window size:")
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(2, 10)
        self.window_size_spin.setValue(2)
        self.window_size_spin.setSingleStep(1)

        window_layout.addWidget(window_label)
        window_layout.addWidget(self.window_size_spin)

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        layout.addWidget(self.label)
        layout.addLayout(window_layout)
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

    def get_window_size(self) -> int:
        """Expose selected window size"""
        return self.window_size_spin.value()