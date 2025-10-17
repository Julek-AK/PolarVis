# Builtins
import sys

# External
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QApplication, QMessageBox

# Internal
from paths import UI_DIR

from gui.pipeline_dialog import PipelineDialog
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
    # IMAGE PROCESSING
    # =============================================

    def run_single_process(self) -> None:
        # File management
        filename = self.file_manager.select_file(self)
        if not filename:
            QMessageBox.warning(self, "No File", "No valid file selected.")
            return
        
        ID = self.file_manager.get_id(filename)
        img_arr = self.file_manager.load_image(filename)

        # Check the cache
        cached = self.cache_manager.get_array(ID)
        if cached is not None:
            QMessageBox.information(self, "Cached", f"Result for ID '{ID}' is available in the cache")
            return

        # Activate the dialog window        
        dialog = PipelineDialog(self)
        if dialog.exec() == 0:
            return

        def on_finished(sol_array):
            dialog.close()
            self.cache_manager.save_array(ID, sol_array)
            QMessageBox.information(self, "Success", f"Saved results for ID '{ID}'")

        def on_error(msg):
            dialog.close()
            QMessageBox.critical(self, "Error", msg)

        # Initiate the pipeline
        self.pipeline.single_process(img_arr, on_finished, on_error)
        dialog.show()

    def run_batch_process(self) -> None:
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


