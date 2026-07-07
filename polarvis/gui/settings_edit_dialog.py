
# Builtin
from copy import deepcopy

# External
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QStackedWidget,
    QPushButton,
)

# Internal
from ..gui.settings_pages import *

SETTINGS_PAGES = {  # These names must match the ones found in setting defaults
    'general': GeneralSettingsPage,
    'display': DisplaySettingsPage,
    'processing': ProcessingSettingsPage,
    'camera': CameraSettingsPage,
    'visualization': VisualizationSettingsPage,
    'paths': PathsSettingsPage,
}


class SettingsEditDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)

        self.settings = deepcopy(settings)
        self.page_lookup = {}

        self.setWindowTitle("Edit Settings")
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        
        # Left side
        left_layout = QVBoxLayout()

        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")

        left_layout.addWidget(self.save_button)
        left_layout.addWidget(self.cancel_button)

        self.settings_list = QListWidget()
        left_layout.addWidget(self.settings_list)

        layout.addLayout(left_layout)

        # Right side
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        
        # Populate with settings
        for key, value in self.settings.items():
            if not isinstance(value, dict):
                continue
            
            # Left side
            label = key.replace('_', ' ').title()
            self.settings_list.addItem(label)
            
            # Right side
            page_cls = SETTINGS_PAGES.get(key)

            if page_cls is None:
                continue

            page = page_cls()
            self.page_lookup[key] = page

            page.load(value)

            self.pages.addWidget(page)

        # Connections
        self.settings_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.save_button.clicked.connect(self._save)
        self.cancel_button.clicked.connect(self.reject)

        if self.settings_list.count():
            self.settings_list.setCurrentRow(0)

        
    def _save(self):

        page_index = 0

        for key, value in self.settings.items():

            if not isinstance(value, dict):
                continue

            page = self.pages.widget(page_index)
            page.save(value)

            page_index += 1

        self.accept()

    @property
    def edited_settings(self):
        return self.settings