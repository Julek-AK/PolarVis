
# Builtins

# External Libraries
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

# Internal Support
from ..core.visualisation_control import (
    list_visualisations,
    generate_visualisation,
    generate_legend
)
from ..app.config.settings import settings


class VisualisationPanel(QtWidgets.QFrame):
    pixelHovered = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = self.window()
        self.image_view = self.main_window.graphicsView

        self.current_result = None
        self.current_file = None
        self.current_array = None
        self.current_visualisation = None

        # UI setup
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Visualisation")
        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)

        self.file_selector = QtWidgets.QListWidget()
        self.file_selector.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.file_selector.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )

        vis_grid = QtWidgets.QGridLayout()
        self.vis_buttons = QtWidgets.QButtonGroup(self)
        self.vis_buttons.setExclusive(True)
        visualisations = list_visualisations()
        for idx, name in enumerate(visualisations):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)

            if idx == 0:
                button.setChecked(True)
                self.current_visualisation = name

            self.vis_buttons.addButton(button)

            row = idx // 2
            col = idx % 2
            vis_grid.addWidget(button, row, col)

        # self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.legend_checkbox = QtWidgets.QCheckBox("Include Legend")
        self.legend_checkbox.setChecked(True)
        self.save_button = QtWidgets.QPushButton("Save Visualisation")

        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("Cached File Selection:"))
        layout.addWidget(self.file_selector, stretch=1)
        layout.addWidget(QtWidgets.QLabel("Visualisation Type:"))
        layout.addLayout(vis_grid)
        # layout.addWidget(self.preview_label, stretch=1)
        layout.addWidget(QtWidgets.QLabel("Saving:"))
        layout.addWidget(self.legend_checkbox)
        layout.addWidget(self.save_button)

        # Connect logic
        self.vis_buttons.buttonClicked.connect(self.on_visualisation_selected)
        self.file_selector.itemSelectionChanged.connect(self.update_preview)
        self.save_button.clicked.connect(self.save_current_visualisation)
        self.image_view.pixelHovered.connect(self.on_pixel_hovered)

    def connect_managers(self, cache_manager, file_manager):
        """Provide the managers after UI load."""
        self.cache_manager = cache_manager
        self.file_manager = file_manager
        self.cache_manager.cacheChanged.connect(self.refresh_cache_list)
        self.refresh_cache_list()

    # =============================================
    # CACHE + PREVIEW
    # =============================================
    def refresh_cache_list(self):
        """Reloads the combo box based on current cache contents."""
        cached_files = self.cache_manager.list_contents()
        self.file_selector.clear()

        for file in cached_files:
            item = QtWidgets.QListWidgetItem(str(file.stem))
            self.file_selector.addItem(item)

    def on_visualisation_selected(self, button):
        self.current_visualisation = button.text()
        self.update_preview()

    def update_preview(self):
        item = self.file_selector.currentItem()
        if item is None: return

        file_name = item.text()
        vis_name = self.current_visualisation

        new_file = (file_name != self.current_file)
        if new_file:
            self.current_array = self.cache_manager.get_array(file_name)
            self.current_file = file_name

        result = generate_visualisation(vis_name, self.current_array)
        self.current_result = result

        self.image_view.display_pil_image(
            self.main_window,
            self.current_result.image,
            preserve_view=not new_file
        )

    def on_pixel_hovered(self, x: int, y: int):
        if self.current_file is None:
            return
        img_data = self.current_array
        if not (0 <= x < img_data.shape[1] and 0 <= y < img_data.shape[0]):
            return

        i = img_data[y, x, 0] / 2  # Times-two offset correction
        d = img_data[y, x, 1]
        t = img_data[y, x, 2]
        self.pixelHovered.emit(float(i), float(d), float(t))

    # =============================================
    # SAVING
    # =============================================
    def save_current_visualisation(self):
        if self.current_result is None:
            QtWidgets.QMessageBox.warning(self, "No image", "No visualisation generated yet.")
            return
        
        export_image = self.current_result.image.copy()

        if self.legend_checkbox.isChecked():

            export_image = generate_legend(
                name=self.current_visualisation,
                image=export_image,
                result=self.current_result
            )

        default_name = f"{self.current_file}_{self.current_visualisation.replace(' ', '_')}"
        save_path = self.file_manager.select_save_location(self, default_name, settings.get('paths.save_visualization'))

        if save_path:
            self.file_manager.save_visualisation(save_path, export_image)

