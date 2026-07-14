# Builtins
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# External Libraries
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

# Internal Support


class CalibrationPanel(QtWidgets.QFrame):
    """
    UI panel for selecting and inspecting the currently active calibration.

    Expected calibration manager interface:
        - list_calibrations() -> list[str]
        - load_calibration_metadata(cal_id) -> dict
        - get_calibration_path(cal_id) -> Path (optional)

    The metadata dict is expected to contain things like:
        {
            "name": str,
            "description": str,
            "sensor": str,
            "lens": str,
            "created": iso_string,
            "version": int,
            "dimensions": [w, h],
            ...
        }

    This panel intentionally tolerates missing fields.
    """

    calibration_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_calibration_id: Optional[str] = None
        self.current_metadata: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._connect_signals()

    def connect_manager(self, calibration_manager):
        """Provide the managers after UI load."""
        self.calibration_manager = calibration_manager
        self.refresh()

    # ======================================================================
    # UI
    # ======================================================================

    def _build_ui(self):
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------

        title = QtWidgets.QLabel("Calibration")
        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)

        root_layout.addWidget(title)

        # ------------------------------------------------------------------
        # Selection row
        # ------------------------------------------------------------------

        selection_layout = QtWidgets.QHBoxLayout()

        self.calibration_combo = QtWidgets.QComboBox()
        self.calibration_combo.setMinimumWidth(120)

        self.refresh_button = QtWidgets.QPushButton("Refresh")

        selection_layout.addWidget(self.calibration_combo, 1)
        selection_layout.addWidget(self.refresh_button)

        root_layout.addLayout(selection_layout)

        # ------------------------------------------------------------------
        # Metadata display
        # ------------------------------------------------------------------

        self.info_tree = QtWidgets.QTreeWidget()
        self.info_tree.setColumnCount(2)
        self.info_tree.setHeaderLabels(["Property", "Value"])
        self.info_tree.setRootIsDecorated(False)
        self.info_tree.setAlternatingRowColors(True)
        self.info_tree.setMinimumHeight(160)

        self.info_tree.header().setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )

        self.info_tree.header().setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )

        root_layout.addWidget(self.info_tree)

        # ------------------------------------------------------------------
        # Status line
        # ------------------------------------------------------------------

        self.status_label = QtWidgets.QLabel("No calibration selected")
        self.status_label.setWordWrap(True)

        root_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.refresh_button.clicked.connect(self.refresh)

        self.calibration_combo.currentTextChanged.connect(
            self._on_calibration_selected
        )

    # ======================================================================
    # Public API
    # ======================================================================

    def refresh(self):
        """Reload calibration list from manager."""

        self.calibration_combo.blockSignals(True)

        self.calibration_combo.clear()

        if self.calibration_manager is None:
            self.status_label.setText("No calibration manager connected.")
            self.calibration_combo.blockSignals(False)
            return

        try:
            calibration_ids = self.calibration_manager.list_calibrations()

        except Exception as exc:
            self.status_label.setText(
                f"Failed to load calibration list:\n{exc}"
            )
            self.calibration_combo.blockSignals(False)
            return

        if not calibration_ids:
            self.status_label.setText("No calibration files found.")
            self.calibration_combo.blockSignals(False)
            return

        self.calibration_combo.addItems(calibration_ids)

        self.calibration_combo.blockSignals(False)

        # Force selected entry to be the default calibration and load it
        self.calibration_combo.setCurrentText('Default_factory')
        self._on_calibration_selected(self.calibration_combo.currentText())

    def get_current_calibration_id(self) -> Optional[str]:
        return self.current_calibration_id

    def get_current_metadata(self) -> Optional[Dict[str, Any]]:
        return self.current_metadata

    # ======================================================================
    # Internal
    # ======================================================================

    def _on_calibration_selected(self, calibration_id: str):

        try:
            self.calibration_manager.set_current_calibration(calibration_id)
            
        except Exception as exc:
            self.current_metadata = None
            self.info_tree.clear()
            self.status_label.setText(f"Failed to load calibration metadata:\n{exc}")
            return

        self.current_calibration_id = calibration_id
        metadata = self.calibration_manager.current_metadata
        self._populate_metadata(metadata)

        self.status_label.setText(f"Loaded calibration: {calibration_id}")

    def _populate_metadata(self, metadata: Dict[str, Any]):

        self.info_tree.clear()

        ordered_keys = [
            "name",
            "description",
            "version",
            "sensor",
            "lens",
            "dimensions",
            "pattern",
            "polarizer_angles",
            "created",
            "author",
        ]

        displayed = set()

        # Preferred ordering
        for key in ordered_keys:

            if key not in metadata:
                continue

            self._add_metadata_row(key, metadata[key])
            displayed.add(key)

        # Remaining metadata
        for key, value in metadata.items():

            if key in displayed:
                continue

            self._add_metadata_row(key, value)

    def _add_metadata_row(self, key: str, value: Any):

        item = QtWidgets.QTreeWidgetItem()

        item.setText(0, self._format_key(key))
        item.setText(1, self._format_value(value))

        self.info_tree.addTopLevelItem(item)

    # ======================================================================
    # Formatting helpers
    # ======================================================================

    @staticmethod
    def _format_key(key: str) -> str:
        return key.replace("_", " ").title()

    @staticmethod
    def _format_value(value: Any) -> str:

        if value is None:
            return "-"

        # NumPy arrays
        if isinstance(value, np.ndarray):
            return (
                f"ndarray "
                f"{value.shape} "
                f"{value.dtype}"
            )

        # Lists / tuples
        if isinstance(value, (list, tuple)):

            if len(value) > 10:
                return f"[{', '.join(map(str, value[:10]))}, ...]"

            return str(list(value))

        # Dicts
        if isinstance(value, dict):
            return f"dict ({len(value)} keys)"

        # Paths
        if isinstance(value, Path):
            return str(value)

        # Datetime
        if isinstance(value, datetime):
            return value.isoformat(sep=" ")

        return str(value)