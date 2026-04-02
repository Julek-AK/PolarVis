# Builtins
import os
import sys
import subprocess

# External
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QApplication, QMessageBox

# Internal
from paths import UI_DIR

from gui.pipeline_dialog import PipelineDialog
from gui.calibration_dialog import CalibrationDialog
from gui.window_init import MainWindowConstructor


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(UI_DIR / "mainwindow2.ui", self)  # loads UI into this instance

        MainWindowConstructor(self).setup()


    # =============================================
    # CONSOLE
    # =============================================

    def append_console_text(self, text: str) -> None:
        self.consoleOutput.appendPlainText(text.strip())

    # =============================================
    # CACHE HANDLING
    # =============================================

    def show_cache_info(self) -> None:
        stats = self.cache_manager.get_stats()

        text = (
            f"Cache location:\n{stats['path']}\n\n"
            f"Files in cache: {stats['file_count']}\n"
            f"Total size: {stats['total_size_bytes']/1000000:.2f} MB"
        )

        QMessageBox.information(self, "Cache Information", text)

    def browse_cache(self) -> None:
        path = str(self.cache_manager.cache_dir)

        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def clear_cache(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Clear Cache",
            "This will permanently delete all cached files.\n\n"
            "This action cannot be undone.\n\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.cache_manager.clear(confirm=True)

    # =============================================
    # CALIBRATION
    # =============================================

    def compute_calibration(self) -> None:

        # Activate the dialog window        
        dialog = CalibrationDialog(self.calibration_manager, self)
        if dialog.exec() == 0:
            return


    # =============================================
    # IMAGE PROCESSING
    # =============================================

    def run_single_process(self) -> None:
        # File management
        filename = self.file_manager.select_file(self)
        if not filename:
            return
        
        img_id = self.file_manager.get_id(filename)
        img_arr = self.file_manager.load_image(filename)

        # Activate the dialog window        
        dialog = PipelineDialog(self)
        if dialog.exec() == 0:
            return
        
        # Calibration
        cal_id = dialog.get_calibration_id()
        # cal = self.calibration_manager.load_calibration(cal_id)
        cal, _ = self.calibration_manager.load_calibration("Default_factory_NA")

        # Check the cache
        ID = f"{img_id}__{cal_id}" 
        cached = self.cache_manager.get_array(ID)
        if cached is not None:
            QMessageBox.information(self, "Cache hit!", f"Result for ID '{ID}' is available in the cache")
            return

        # Callbacks
        def on_finished(sol_array):
             
            self.cache_manager.save_array(ID, sol_array)
            dialog.close()
            QMessageBox.information(self, "Success", f"Saved results for ID '{ID}'")

        def on_error(msg):
            dialog.close()
            QMessageBox.critical(self, "Error", msg)

        def on_warning(msg):
            QMessageBox.warning(self, "Warning", msg)

        # Initiate the pipeline
        self.pipeline.single_process(
            img_arr,
            cal,
            on_finished,
            on_error,
            on_warning,
        )

        dialog.show()

        # Refresh the gui with new cache information
        self.frame_view_filters.refresh_cache_list()

    def run_batch_process(self) -> None:
        raise NotImplementedError

    def run_video_process(self) -> None:
        raise NotImplementedError

    # =============================================
    # IMAGE VISUALISATION
    # =============================================
    def load_raw_image(self) -> None:
        filename = self.file_manager.select_file(self)

        if not filename: raise ValueError("[Image Loading] No correct file was found")

        self.graphicsView.display_image(self, filename)

    # here comes all the other stuff related to visualising



def run_main_window():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


