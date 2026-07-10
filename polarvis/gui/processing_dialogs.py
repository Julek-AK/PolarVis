
# Builtin

# External
from PyQt6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QProgressBar,
)

# Internal
from ..app.config.settings import settings
from ..core.visualisation_control import list_visualisations


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

        if directory:
            self.directory = directory
            self.directory_label.setText(str(directory))


class VideoProcessDialog(ProcessingDialog):
    def __init__(self, calibration_id, file_manager, parent=None):
        self.file_manager = file_manager
        self.input_path = None
        self.output_directory = None

        super().__init__(calibration_id, parent)

    def add_input_widgets(self, layout):

        # Input video
        input_layout = QHBoxLayout()

        self.input_label = QLabel("No video selected")
        self.input_button = QPushButton("Select video")
        self.input_button.clicked.connect(self.select_video)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_button)

        # Output directory
        output_layout = QHBoxLayout()

        self.output_label = QLabel("No output directory selected")
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_directory)

        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_button)

        # Visualisations
        self.visualisation_checks = {}

        vis_widget = QWidget()
        vis_layout = QVBoxLayout()

        vis_names = list_visualisations()
        for name in vis_names:

            checkbox = QCheckBox(name)
            checkbox.setChecked(name == vis_names[0])
            self.visualisation_checks[name] = checkbox

            vis_layout.addWidget(checkbox)

        vis_widget.setLayout(vis_layout)

        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        layout.addWidget(QLabel("Visualisations:"))
        layout.addWidget(vis_widget)

        self.process_button.clicked.connect(lambda: self.set_processing_state(True))

    def select_video(self):

        filename = self.file_manager.select_video(self, settings.get('paths.open_file'))

        if filename:
            self.input_path = filename
            self.input_label.setText(str(filename))

    def select_directory(self):

        directory = self.file_manager.select_folder(self, settings.get('paths.open_folder'))

        if directory:
            self.output_directory = directory
            self.output_label.setText(str(directory))

    def get_data(self):

        visualisations = [
            name
            for name, checkbox in self.visualisation_checks.items()
            if checkbox.isChecked()
        ]

        return self.input_path, self.output_directory, visualisations
    
    def set_processing_state(self, processing: bool):

        self.input_button.setEnabled(not processing)
        self.output_button.setEnabled(not processing)

        for checkbox in self.visualisation_checks.values():
            checkbox.setEnabled(not processing)

        self.process_button.setEnabled(not processing)

        self.cancel_button.setEnabled(processing)
