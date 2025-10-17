
# Builtins

# External Libraries
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

# Internal Support
from core.visualisation_control import list_visualisations, generate_visualisation



class VisualisationPanel(QtWidgets.QFrame):
    pixelHovered = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = self.window()
        self.image_view = self.main_window.graphicsView

        self.current_image = None
        self.current_file = None
        self.current_array = None

        # --- UI setup ---
        layout = QtWidgets.QVBoxLayout(self)

        self.file_selector = QtWidgets.QComboBox()
        self.vis_selector = QtWidgets.QComboBox()
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.save_button = QtWidgets.QPushButton("Save Visualisation")

        layout.addWidget(QtWidgets.QLabel("Select Cached File:"))
        layout.addWidget(self.file_selector)
        layout.addWidget(QtWidgets.QLabel("Visualisation Type:"))
        layout.addWidget(self.vis_selector)
        layout.addWidget(self.preview_label, stretch=1)
        layout.addWidget(self.save_button)

        # --- Populate selectors ---
        self.vis_selector.addItems(list_visualisations())

        # --- Connect logic ---
        self.vis_selector.currentIndexChanged.connect(self.update_preview)
        self.file_selector.currentIndexChanged.connect(self.update_preview)
        self.save_button.clicked.connect(self.save_current_visualisation)
        self.image_view.pixelHovered.connect(self.on_pixel_hovered)

    def connect_managers(self, cache_manager, file_manager):
        """Provide the managers after UI load."""
        self.cache_manager = cache_manager
        self.file_manager = file_manager
        self.refresh_cache_list()

    # =============================================
    # CACHE + PREVIEW
    # =============================================
    def refresh_cache_list(self):
        """Reloads the combo box based on current cache contents."""
        cached_files = self.cache_manager.list_contents()
        self.file_selector.clear()
        self.file_selector.addItems([None] + [str(f.stem) for f in cached_files])

    def update_preview(self):
        file_name = self.file_selector.currentText()
        vis_name = self.vis_selector.currentText()
        if not file_name:
            return

        self.current_array = self.cache_manager.get_array(file_name)
        self.current_file = file_name

        image = generate_visualisation(vis_name, self.current_array)
        self.current_image = image
        self.main_window.graphicsView.display_pil_image(self.main_window, self.current_image)

    def on_pixel_hovered(self, x: int, y: int):
        if self.current_file is None:
            return
        img_data = self.current_array
        if not (0 <= x < img_data.shape[1] and 0 <= y < img_data.shape[0]):
            return

        i = img_data[y, x, 0] + img_data[y, x, 1]
        d = img_data[y, x, 1] / max(i, 1e-8)
        t = np.mod(img_data[y, x, 2], np.pi)
        self.pixelHovered.emit(float(i), float(d), float(t))

    # =============================================
    # SAVING
    # =============================================
    def save_current_visualisation(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "No visualisation generated yet.")
            return

        default_name = f"{self.current_file}_{self.vis_selector.currentText().replace(' ', '_')}"
        save_path = self.file_manager.select_save_location(self, default_name)
        self.file_manager.save_visualisation(save_path, self.current_image)
