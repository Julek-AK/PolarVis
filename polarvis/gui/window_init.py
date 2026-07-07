"""
Dedicated class for initializing all the functionality of the MainWindow
"""

# Builtins
import sys

# External libraries
from PyQt6.QtWidgets import QGraphicsScene, QVBoxLayout, QApplication
from PyQt6.QtGui import QPalette, QColor

# Internal Support
from ..app.streams import TeeStream
from ..app.config.settings import settings

from ..core.pipeline import ImagePipeline
from ..core.console_redirector import ConsoleRedirector
from ..core.file_manager import ImageFileManager, CacheManager
from ..core.calibration_manager import CalibrationManager
from ..gui.visualisation_panel import VisualisationPanel
from ..gui.calibration_panel import CalibrationPanel


class MainWindowConstructor:
    def __init__(self, window):
        self.window = window

    def setup(self):
        self.init_console()
        
        self.init_managers()
        self.init_pipelines()
        self.init_visualisation()
        self.init_calibration()

        self.init_menu_bar()
        self.init_image_display()
        self.init_settings_panel()

    def init_console(self):
        gui_stdout = ConsoleRedirector()
        gui_stderr = ConsoleRedirector()

        gui_stdout.new_text.connect(self.window.append_console_text)
        gui_stderr.new_text.connect(
            lambda text: self.window.append_console_text(f"[Error] {text}")
        )

        self.window.stdout_redirector = gui_stdout
        self.window.stderr_redirector = gui_stderr

        sys.stdout = TeeStream(sys.__stdout__, gui_stdout)
        sys.stderr = TeeStream(sys.__stderr__, gui_stderr)

    def init_menu_bar(self):
        # File
        self.window.actionLoad_Image.triggered.connect(self.window.load_raw_image)
        self.window.actionExport_Data.triggered.connect(self.window.export)

        # Processing
        self.window.actionSingle_Processing.triggered.connect(self.window.run_single_process)
        self.window.actionBatch_Processing.triggered.connect(self.window.run_batch_process)
        self.window.actionVideo_Processing.triggered.connect(self.window.run_video_process)

        # Calibration
        self.window.actionCompute_Calibration.triggered.connect(self.window.compute_calibration)
        self.window.actionInfoCalibration.triggered.connect(self.window.show_calibration_info)

        # Cache
        self.window.actionInfoCache.triggered.connect(self.window.show_cache_info)
        self.window.actionBrowseCache.triggered.connect(self.window.browse_cache)
        self.window.actionClearCache.triggered.connect(self.window.clear_cache)

        # Help
        self.window.actionGitHub.triggered.connect(self.window.open_github)

    def init_image_display(self):
        self.window.scene = QGraphicsScene(self.window)
        self.window.graphicsView.setScene(self.window.scene)

    def init_managers(self):
        self.window.file_manager = ImageFileManager()
        self.window.cache_manager = CacheManager()
        self.window.calibration_manager = CalibrationManager()

    def init_pipelines(self):
        self.window.pipeline = ImagePipeline()

    def init_visualisation(self):
        vis_panel = self.window.frame_view_filters
        info_panel = self.window.frame_pixel_information

        vis_panel.connect_managers(self.window.cache_manager, self.window.file_manager)
        vis_panel.pixelHovered.connect(info_panel.update_values)
        
    def init_calibration(self):
        self.window.frame_calibration_panel.connect_manager(self.window.calibration_manager)

    def init_settings_panel(self):
        pass


class UISettingsController:
    def __init__(self, window):
        self.window = window
        self.settings = settings

    def apply(self) -> None:
        display = self.settings.get('display')

        self._apply_theme(display.get('theme', 'system'))
        self._apply_autoscale(display.get('autoscale', True))

    def _apply_theme(self, theme: str) -> None:
        if theme == 'dark':
            self._apply_dark_theme()
        elif theme == 'light':
            self._apply_light_theme()
        else:
            self._apply_system_theme()

    def _apply_dark_theme(self) -> None:
        palette = QPalette()

        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))

        QApplication.instance().setPalette(palette)

    def _apply_light_theme(self) -> None:
        QApplication.instance().setPalette(QApplication.style().standardPalette())

    def _apply_system_theme(self) -> None:
        QApplication.instance().setPalette(QApplication.style().standardPalette())

    def _apply_autoscale(self, enabled: bool) -> None:
        self.window.setProperty('autoscale_enabled', enabled)



