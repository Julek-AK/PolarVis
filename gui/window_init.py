"""
Dedicated class for initializing all the functionality of the MainWindow
"""

# Builtins
import sys

# External libraries
from PyQt6.QtWidgets import QGraphicsScene, QVBoxLayout

# Internal Support
from core.pipeline import Pipeline
from core.console_redirector import ConsoleRedirector
from core.file_manager import ImageFileManager, CacheManager
from gui.visualisation_panel import VisualisationPanel



class MainWindowConstructor:
    def __init__(self, window):
        self.window = window

    # =============================================
    # INITIALIZATION FUNCTIONS
    # =============================================
    def setup(self):
        self.init_menu_bar()
        self.init_console()
        self.init_image_display()
        
        # currently placeholders
        self.init_file_managers()
        self.init_pipelines()
        self.init_visualisation()

    def init_console(self):
        self.window.stdout_redirector = ConsoleRedirector()
        self.window.stderr_redirector = ConsoleRedirector()

        self.window.stdout_redirector.new_text.connect(self.window.append_console_text)
        self.window.stderr_redirector.new_text.connect(
            lambda text: self.window.append_console_text(f"[Error] {text}")
        )

        sys.stdout = self.window.stdout_redirector
        sys.stderr = self.window.stderr_redirector

    def init_menu_bar(self):
        # Image loading
        self.window.actionLoad_Image.triggered.connect(self.window.load_raw_image)

        # Image Processing
        self.window.actionSingle_Processing.triggered.connect(self.window.run_single_process)
        self.window.actionBatch_Processing.triggered.connect(self.window.run_batch_process)

    def init_image_display(self):
        self.window.scene = QGraphicsScene(self.window)
        self.window.graphicsView.setScene(self.window.scene)

    def init_file_managers(self):
        self.window.file_manager = ImageFileManager()
        self.window.cache_manager = CacheManager()

    def init_pipelines(self):
        self.window.pipeline = Pipeline()

    def init_visualisation(self):
        vis_panel = self.window.frame_view_filters
        info_panel = self.window.frame_pixel_information

        vis_panel.connect_managers(self.window.cache_manager, self.window.file_manager)
        vis_panel.pixelHovered.connect(info_panel.update_values)


    # =============================================
    # IMAGE PROCESSING
    # =============================================
        








