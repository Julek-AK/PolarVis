
# Builtin

# External
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QFileDialog,
)

# Internal
from ..app.config.settings import settings


class ProcessingDialog(QDialog):
    def __init__(self, calibration_id, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Processing")

        self.process_button = QPushButton("Process")
        self.cancel_button = QPushButton("Cancel")

        self.process_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.calibration_label = QLabel(f"Calibration: {calibration_id}")
        self.time_label = QLabel("Estimated time: unknown")

        self._build_layout()

    def _build_layout(self):
        layout = QVBoxLayout()

        layout.addWidget(self.calibration_label)

        self.add_input_widgets(layout)

        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_bar)

        buttons = QHBoxLayout()
        buttons.addWidget(self.process_button)
        buttons.addWidget(self.cancel_button)

        layout.addLayout(buttons)

        self.setLayout(layout)

    def add_input_widgets(self, layout):
        """Override in subclasses"""
        pass

    def update_progress(self, value, estimate=None):
        self.progress_bar.setValue(value)

        if estimate:
            self.time_label.setText(f"Estimated time remaining: {estimate:.1f}s")

        
class SingleProcessDialog(ProcessingDialog):
    def __init__(self, calibration_id, file_manager, parent=None):
        self.file_manager = file_manager
        self.filename = None

        super().__init__(calibration_id, parent)

    def add_input_widgets(self, layout):

        row = QHBoxLayout()

        self.file_label = QLabel("No image selected")

        browse = QPushButton("Select Image")
        browse.clicked.connect(self.select_image)

        row.addWidget(self.file_label)
        row.addWidget(browse)

        layout.addLayout(row)

    def select_image(self):

        filename = self.file_manager.select_file(self, settings.get('paths.open_file'))

        if filename:
            self.filename = filename
            self.file_label.setText(str(filename))


class BatchProcessDialog(ProcessingDialog):
    def __init__(self, calibration_id, file_manager, parent=None):
        self.file_manager = file_manager
        self.directory = None
        self.image_list = []
    
        super().__init__(calibration_id, parent)

    def add_input_widgets(self, layout):

        row = QHBoxLayout()

        self.directory_label = QLabel("No directory selected")

        browse = QPushButton("Select Directory")
        browse.clicked.connect(self.select_directory)

        row.addWidget(self.directory_label)
        row.addWidget(browse)

        layout.addLayout(row)

    def select_directory(self):

        directory = self.file_manager.select_folder(self, settings.get('paths.open_folder'))