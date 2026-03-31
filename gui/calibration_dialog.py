from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox
)
from PyQt6 import QtCore
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict, Optional

# Internal
from processing.calibration import CalibrationInput
from core.calibration_manager import CalibrationManager


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CalibrationWorker(QtCore.QThread):
    """Creates a thread for calibration execution to make sure the app doesn't freeze."""
    finished = QtCore.pyqtSignal(object, dict)  # Calibration, metadata
    failed = QtCore.pyqtSignal(str)

    def __init__(self, manager: CalibrationManager, cal_input: CalibrationInput, metadata: dict):
        super().__init__()
        self.manager = manager
        self.cal_input = cal_input
        self.metadata = metadata

    def run(self):
        # try:
        cal = self.manager.calibrate(self.cal_input)
        self.finished.emit(cal, self.metadata)
        # except Exception as e:
            # self.failed.emit(str(e))


class CalibrationDialog(QDialog):
    # TODO transfer UI construction into a self-contained .ui file
    def __init__(self, manager: CalibrationManager, parent=None):
        super().__init__(parent)

        self.manager = manager
        self.setWindowTitle("Camera Calibration")

        self.paths: Dict[str, Path] = {}

        layout = QVBoxLayout(self)

        # Metadata inputs
        self.sensor = QLineEdit("Blackfly S BFS-U3-51S5M")

        self.lens = QLineEdit("TECHSPEC 16mm C Series Fixed Focal Length Lens")

        self.wavelength = QLineEdit("None")

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(200, 400)
        self.temperature.setValue(288.15)

        self.bit_depth = QSpinBox()
        self.bit_depth.setRange(8, 16)
        self.bit_depth.setValue(8)

        self.angles = QLineEdit("0,45,90,135")

        layout.addWidget(QLabel("Sensor model"))
        layout.addWidget(self.sensor)

        layout.addWidget(QLabel("Lens model"))
        layout.addWidget(self.lens)

        layout.addWidget(QLabel("Wavelength filter"))
        layout.addWidget(self.wavelength)

        layout.addWidget(QLabel("Acquisition temperature [K]"))
        layout.addWidget(self.temperature)

        layout.addWidget(QLabel("Bit depth"))
        layout.addWidget(self.bit_depth)

        layout.addWidget(QLabel("Angles [deg], comma separated"))
        layout.addWidget(self.angles)

        # Folder selectors
        self.folder_labels = {}

        self._add_folder_selector(layout, 'dark', "Dark frames")
        self._add_folder_selector(layout, 'flat', "Flat field")

        # Dynamic angle folders
        self.angle_container = QVBoxLayout()
        layout.addLayout(self.angle_container)

        self._update_angle_selectors()

        self.angles.textChanged.connect(self._update_angle_selectors)

        # Run button
        self.run_button = QPushButton("Run Calibration")
        self.run_button.clicked.connect(self._run_calibration)
        layout.addWidget(self.run_button)

    # Folder selector builder
    def _add_folder_selector(self, parent_layout, key, label):

        row = QHBoxLayout()

        btn = QPushButton(f"Select {label}")
        info = QLabel("No folder selected")

        btn.clicked.connect(lambda: self._select_folder(key, info))

        row.addWidget(btn)
        row.addWidget(info)

        parent_layout.addLayout(row)

        self.folder_labels[key] = info

    # Recursive clearing of the angle selectors
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)

            widget = item.widget()
            child_layout = item.layout()

            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    # Dynamic angle folders
    def _update_angle_selectors(self):

        self._clear_layout(self.angle_container)

        self.folder_labels = {k: v for k, v in self.folder_labels.items() if k in ['dark', 'flat']}

        try:
            angles = [a.strip() for a in self.angles.text().split(",") if a.strip()]
        except:
            return

        for angle in angles:
            key = f'angle_{angle}'
            self._add_folder_selector(self.angle_container, key, f"Angle {angle}°")

    # Folder selection + preview
    def _select_folder(self, key, label_widget):

        path = QFileDialog.getExistingDirectory(self, "Select folder")

        if not path:
            return

        path = Path(path)
        self.paths[key] = path

        try:
            files = sorted(
                f for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTS
            )
            n = len(files)

            if n == 0:
                label_widget.setText("Empty folder")
                return

            with Image.open(files[0]) as img:
                arr = np.array(img)

            label_widget.setText(f"{n} images | {arr.shape} | {arr.dtype}")

        except Exception as e:
            label_widget.setText(f"Error: {e}")

    # Build CalibrationInput
    def _build_input(self) -> CalibrationInput:

        def load_stack(path: Path):
            imgs = []
            for f in sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS):
                with Image.open(f) as img:
                    arr = self.manager.get_valid_array(img, f)
                    imgs.append(arr.astype(np.float64))
            return np.stack(imgs, axis=-1)

        dark = load_stack(self.paths['dark'])
        flat = load_stack(self.paths['flat'])

        angle_cases = {}
        for key, path in self.paths.items():
            if key.startswith('angle_'):
                angle = key.split("_")[1]
                angle_cases[angle] = load_stack(path)

        return CalibrationInput(
            dark_frame=dark,
            flat_field=flat,
            angle_cases=angle_cases,
            bit_depth=self.bit_depth.value()
        )

    # Run calibration
    def _run_calibration(self):

        try:
            cal_input = self._build_input()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        metadata = {
            'sensor_model': self.sensor.text(),
            'lens_model': self.lens.text(),
            'wavelength_filter': self.wavelength.text(),
            'temperature': self.temperature.value(),
            'bit_depth': self.bit_depth.value(),
            'angles': self.angles.text()
        }

        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")

        self.worker = CalibrationWorker(self.manager, cal_input, metadata)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    # Worker callbacks
    def _on_finished(self, calibration, metadata):

        self.run_button.setEnabled(True)
        self.run_button.setText("Run Calibration")

        QMessageBox.information(self, "Success", "Calibration completed")

        # Save immediately (or emit signal instead)
        self.manager.save_calibration(calibration, self)
        self.accept()

    def _on_failed(self, error_msg):

        self.run_button.setEnabled(True)
        self.run_button.setText("Run Calibration")

        QMessageBox.critical(self, "Calibration failed", error_msg)

