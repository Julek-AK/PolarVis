
# Builtin

# External
from matplotlib import colormaps
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QLineEdit,
    QRadioButton,
    QPushButton,
    QFileDialog,
)

# Internal


class DirectorySelectorWidget(QWidget):
    """Widget for determining savepath directories"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.default_radio = QRadioButton("Use default")
        self.custom_radio = QRadioButton("Custom")

        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.default_radio)
        layout.addWidget(self.custom_radio)
        layout.addLayout(path_layout)

        self.default_radio.toggled.connect(self._update_enabled)
        self.browse_button.clicked.connect(self._browse)

        self.default_radio.setChecked(True)
        self._update_enabled()

    def _update_enabled(self):
        enabled = self.custom_radio.isChecked()

        self.path_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            self.path_edit.text()
        )

        if directory:
            self.path_edit.setText(directory)

    def value(self):
        if self.default_radio.isChecked():
            return 'default'

        return self.path_edit.text()

    def set_value(self, value):
        if value == 'default':
            self.default_radio.setChecked(True)
            self.path_edit.clear()
        else:
            self.custom_radio.setChecked(True)
            self.path_edit.setText(value)

        self._update_enabled()


ANGLES = [0, 45, 90, 135]
class ChannelOrderWidget(QWidget):
    """Widget for selecting the sensor array polarizer channels"""
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QGridLayout(self)

        labels = [
            "Top Left",
            "Top Right",
            "Bottom Left",
            "Bottom Right",
        ]

        self.boxes = []

        for i, label in enumerate(labels):

            combo = QComboBox()

            for angle in ANGLES:
                combo.addItem(f"{angle}°", angle)

            self.boxes.append(combo)

            row = (i // 2) * 2

            col = i % 2

            layout.addWidget(QLabel(label), row, col)
            layout.addWidget(combo, row + 1, col)

    def value(self):
        return [box.currentData() for box in self.boxes]

    def set_value(self, values):
        for combo, angle in zip(self.boxes, values):
            combo.setCurrentIndex(combo.findData(angle))


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()

    def load(self, settings: dict):
        """Populate widgets from a settings dictionary."""
        raise NotImplementedError

    def save(self, settings: dict):
        """Write widget values back into a settings dictionary."""
        raise NotImplementedError

    
class GeneralSettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

    def load(self, settings: dict):
        pass

    def save(self, settings: dict):
        pass


class DisplaySettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Theme"))

        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "system",
            "light",
            "dark",
        ])
        layout.addWidget(self.theme_combo)

        self.autoscale_checkbox = QCheckBox("Automatically scale interface")
        layout.addWidget(self.autoscale_checkbox)

        layout.addStretch()

    def load(self, settings: dict):
        self.theme_combo.setCurrentText(settings['theme'])
        self.autoscale_checkbox.setChecked(settings['autoscale'])

    def save(self, settings: dict):
        settings['theme'] = self.theme_combo.currentText()
        settings['autoscale'] = self.autoscale_checkbox.isChecked()


class ProcessingSettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)

        self.use_gpu_checkbox = QCheckBox("Use GPU acceleration")
        layout.addWidget(self.use_gpu_checkbox)
        layout.addWidget(QLabel("This requires cuda. Processing speedup results may vary."))

        layout.addWidget(QLabel("Processing batch size:"))
        layout.addWidget(self.batch_size)
        layout.addWidget(QLabel("Larger values improve processing times,"))
        layout.addWidget(QLabel("but cost more memory and lose more data in case of a crash."))

        layout.addStretch()

    def load(self, settings: dict):

        self.use_gpu_checkbox.setChecked(settings['use_gpu'])
        self.batch_size.setValue(settings['batch_size'])
        
    def save(self, settings: dict):

        settings['use_gpu'] = self.use_gpu_checkbox.isChecked()
        settings['batch_size'] = self.batch_size.value()


class CameraSettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.channel_widget = ChannelOrderWidget()

        self.image_height = QSpinBox()
        self.image_height.setRange(1, 5000)

        self.image_width = QSpinBox()
        self.image_width.setRange(1, 5000)

        layout.addWidget(QLabel("Default calibration channel order:"))
        layout.addWidget(self.channel_widget)
        layout.addWidget(QLabel("Default calibration height:"))
        layout.addWidget(self.image_height)
        layout.addWidget(QLabel("Default calibration width"))
        layout.addWidget(self.image_width)

        layout.addStretch()

    def load(self, settings):

        self.channel_widget.set_value(settings['channel_order'])
        self.image_height.setValue(settings['size'][0])
        self.image_width.setValue(settings['size'][1])

    def save(self, settings):

        settings['channel_order'] = (self.channel_widget.value())
        settings['size'] = [self.image_height.value(), self.image_width.value()]


class VisualizationSettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

        maps = sorted(colormaps())

        layout = QVBoxLayout(self)
        
        self.intensity_combo = QComboBox()
        self.intensity_combo.addItems(maps)
        self.dolp_combo = QComboBox()
        self.dolp_combo.addItems(maps)
        self.aop_combo = QComboBox()
        self.aop_combo.addItems(maps)

        layout.addWidget(QLabel("Intensity Colormap"))
        layout.addWidget(self.intensity_combo)
        layout.addWidget(QLabel("DoLP Colormap"))
        layout.addWidget(self.dolp_combo)
        layout.addWidget(QLabel("AoP Colormap"))
        layout.addWidget(self.aop_combo)
        layout.addWidget(QLabel("Enter any registered Matplotlib colormap."))

        self.legend = QComboBox()
        self.legend.addItems([
            'small',
            'large',
        ])

        layout.addWidget(QLabel("Legend Style"))
        layout.addWidget(self.legend)

    def load(self, settings: dict):
        
        self.intensity_combo.setCurrentText(settings['colormaps']['intensity'])
        self.dolp_combo.setCurrentText(settings['colormaps']['dolp'])
        self.aop_combo.setCurrentText(settings['colormaps']['aop'])
        self.legend.setCurrentText(settings['legend_style'])

    def save(self, settings: dict):

        settings['colormaps']['intensity'] = self.intensity_combo.currentText()
        settings['colormaps']['dolp'] = self.dolp_combo.currentText()
        settings['colormaps']['aop'] = self.aop_combo.currentText()
        settings['legend_style'] = self.legend.currentText()


class PathsSettingsPage(SettingsPage):
    def __init__(self):
        super().__init__()

        layout = QFormLayout(self)

        self.open_file = DirectorySelectorWidget()
        self.open_folder = DirectorySelectorWidget()
        self.save_visualization = DirectorySelectorWidget()
        self.export_data = DirectorySelectorWidget()

        layout.addRow("Open File", self.open_file)
        layout.addRow("Open Folder", self.open_folder)
        layout.addRow("Save Visualization", self.save_visualization)
        layout.addRow("Export Data", self.export_data)

    def load(self, settings: dict):

        self.open_file.set_value(settings['open_file'])
        self.open_folder.set_value(settings['open_folder'])
        self.save_visualization.set_value(settings['save_visualization'])
        self.export_data.set_value(settings['export_data'])

    def save(self, settings: dict):
        
        settings['open_file'] = self.open_file.value()
        settings['open_folder'] = self.open_folder.value()
        settings['save_visualization'] = self.save_visualization.value()
        settings['export_data'] = self.export_data.value()
