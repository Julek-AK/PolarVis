
# Builtin
from pathlib import Path

# External
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QRadioButton,
    QComboBox,
    QDialogButtonBox
)

# Internal
from ..core.export_control import ExportConfig
from ..app.config.settings import settings

class ExportDialog(QDialog):
    def __init__(self, cached_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.cached_results = cached_results
        self.save_directory = None
        self._build_ui()


    def _build_ui(self):

        layout = QVBoxLayout(self)

        # Results list
        layout.addWidget(QLabel("Select results to export:"))

        self.result_list = QListWidget()

        for result in self.cached_results:
            item = QListWidgetItem(result)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.result_list.addItem(item)

        layout.addWidget(self.result_list)

        # Save location
        location_layout = QHBoxLayout()

        self.location_field = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_location)

        location_layout.addWidget(self.location_field)
        location_layout.addWidget(browse_button)

        layout.addWidget(QLabel("Save location:"))
        layout.addLayout(location_layout)


        # Representation
        layout.addWidget(QLabel("Representation:"))

        self.stokes_radio = QRadioButton("Stokes Parameters")
        self.polarized_radio = QRadioButton("Polarized Description (I, DoLP, AoP)")
        self.stokes_radio.setChecked(True)

        layout.addWidget(self.stokes_radio)
        layout.addWidget(self.polarized_radio)


        # File format
        layout.addWidget(QLabel("File format:"))
        self.format_selector = QComboBox()
        self.format_selector.addItems([
            "CSV",
            "MAT",
            "NPY"
        ])
        self.format_selector.setCurrentText("NPY")

        layout.addWidget(self.format_selector)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def select_location(self):
        
        filepath = QFileDialog.getExistingDirectory(
            self,
            "Select Export Location",
            settings.get('paths.export_data'),
        )

        if filepath:
            self.location_field.setText(filepath)

    def get_configuration(self):

        selected = []

        for i in range(self.result_list.count()):
            item = self.result_list.item(i)

            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())

        representation = (
            "stokes"
            if self.stokes_radio.isChecked()
            else "polar"
        )

        return ExportConfig(
            results=selected,
            save_directory=Path(self.location_field.text()),
            representation=representation,
            file_format=self.format_selector.currentText()
        )


