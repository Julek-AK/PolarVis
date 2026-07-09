
# Builtins


# External
from PyQt6 import QtGui
from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel
)

# Internal
from ..app.config.settings import settings
from ..gui.settings_edit_dialog import SettingsEditDialog


class SettingsDisplayPanel(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._build_ui()
        self._load_settings()


    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()

        title = QLabel("System Settings")
        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)

        header.addWidget(title)

        header.addStretch()

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self._on_edit_clicked)
        header.addWidget(self.edit_button)

        layout.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Setting", "Value"])

        layout.addWidget(self.tree)

    def _load_settings(self):
        
        self.tree.clear()

        self._add_tree_items(
            self.tree.invisibleRootItem(),
            settings._settings  # internal access for display only
        )

    def _add_tree_items(self, parent, data):

        for key, value in data.items():

            item = QTreeWidgetItem(parent)
            item.setText(0, key)

            if isinstance(value, dict):
                self._add_tree_items(item, value)

            else:
                item.setText(1, str(value))

    def _on_edit_clicked(self):
        dialog = SettingsEditDialog(settings._settings, self)

        if dialog.exec():

            settings.update(dialog.edited_settings)
            self._load_settings()
            print("New settings saved!")