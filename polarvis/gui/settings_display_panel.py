
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

        data = settings._settings  # internal access for display only

        flat = self._flatten_dict(data)

        for key, value in flat.items():
            item = QTreeWidgetItem([key, str(value)])
            self.tree.addTopLevelItem(item)

    def _flatten_dict(self, d, parent_key=""):
        items = {}

        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v

        return items
    
    def _on_edit_clicked(self):
        pass