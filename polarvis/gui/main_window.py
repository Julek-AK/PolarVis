# Builtins
import os
import sys
import ctypes
import subprocess
import webbrowser


# Correct app icon maneuvers
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        'Julek-AK.PolarVis.Application'
    )

# External
from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QApplication, QMessageBox, QDialog

# Internal
from ..app.paths import UI_DIR, ICONS_DIR
from ..app.config.settings import settings

from ..gui.calibration_dialog import CalibrationDialog
from ..gui.processing_dialogs import SingleProcessDialog, BatchProcessDialog, VideoProcessDialog
from ..gui.window_init import MainWindowConstructor
from ..gui.window_init import UISettingsController
from ..gui.export_dialog import ExportDialog

from ..core.export_control import execute_export
from ..io.video import check_ffmpeg_available


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(UI_DIR / "mainwindow.ui", self)  # loads UI into this instance

        MainWindowConstructor(self).setup()
        # UISettingsController(self).apply()

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

    def show_calibration_info(self) -> None:
        stats = self.calibration_manager.get_stats()

        text = (
            f"Calibration storage location:\n{stats['path']}\n\n"
            f"Number of calibration files: {stats['file_count']}\n"
            f"Total size: {stats['total_size_bytes']/1000000:.2f} MB"
        )

        QMessageBox.information(self, "Calibration Information", text)

    def compute_calibration(self) -> None:

        # Activate the dialog window        
        dialog = CalibrationDialog(self.calibration_manager, self)
        if dialog.exec() == 0:
            return


    # =============================================
    # IMAGE PROCESSING
    # =============================================

    def run_single_process(self) -> None:        
        # Calibration
        cal, cal_id = self.calibration_manager.require_current_calibration()

        dialog = SingleProcessDialog(
            cal_id,
            self.file_manager,
            self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        filename = dialog.filename

        if not filename:
            return
        
        img_id = self.file_manager.get_id(filename)
        img_arr = self.file_manager.load_image(filename)

        # Check the cache
        ID = f"{img_id}__{cal_id}" 
        if self.cache_manager.cache_check(ID):
            QMessageBox.information(self, "Cache hit!", f"Result for ID '{ID}' is available in the cache")
            return

        # Callbacks
        def on_finished(result):
            self.cache_manager.save_array(ID, result)

            dialog.close()

            QMessageBox.information(self, "Success", f"Saved results for ID '{ID}'")

            self.frame_view_filters.refresh_cache_list()

        def on_error(msg):
            dialog.close()
            QMessageBox.critical(self, "Error", msg)

        dialog.process_button.setEnabled(False)

        # Setup pipline processing
        worker = self.pipeline.start_single_processing(img_arr, cal)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()

        dialog.show()

    def run_batch_process(self) -> None:
        # Calibration
        cal, cal_id = self.calibration_manager.require_current_calibration()

        dialog = BatchProcessDialog(
            cal_id,
            self.file_manager,
            self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Extract all the files
        directory = dialog.directory
        file_list = self.file_manager.browse_directory(directory)

        tasks = []
        for filename in file_list:
            img_id = self.file_manager.get_id(filename)

            # Cache check
            ID = f"{img_id}__{cal_id}" 
            if self.cache_manager.cache_check(ID):
                print(f"[Pipeline] The result for ID '{ID}' is available in cache!")
                continue
        
            img_arr = self.file_manager.load_image(filename)
            tasks.append((img_arr, img_id))

        # Callbacks
        def on_finished(result):

            dialog.close()

            QMessageBox.information(self, "Success", f"Completed batch processing!")

            self.frame_view_filters.refresh_cache_list()

        def on_error(msg):
            dialog.close()
            QMessageBox.critical(self, "Error", msg)

        # Setup pipline processing
        worker = self.pipeline.start_batch_processing(tasks, cal, cal_id)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()

        dialog.show()


    def run_video_process(self) -> None:

        QMessageBox.critical(
            self,
            "Not Implemented",
            "Unfortunately, video processing is currently broken "
            "and will hopefully be made working in a future patch. "
            "In the meantime, don't stop recording videos, "
            "you'll process them soon!"
        )

        raise NotImplementedError

        # FFmpeg check
        if not check_ffmpeg_available():

            QMessageBox.critical(
                self,
                "FFmpeg unavailable",
                "FFmpeg was not found on this system.\n\n"
                "Video processing requires FFmpeg to be installed "
                "and available on PATH."
            )

            return
    
        # Calibration
        cal, cal_id = self.calibration_manager.require_current_calibration()

        dialog = VideoProcessDialog(
            cal_id,
            self.file_manager,
            self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Get dialog data
        input_path, output_directory, visualisations = dialog.get_data()
        
        # Callbacks
        def on_finished(result):

            dialog.close()

            QMessageBox.information(self, "Success", "Completed video processing!")

        def on_error(msg):
            dialog.close()
            QMessageBox.critical(self, "Error", msg)

        # Start processing
        worker = self.pipeline.start_video_processing(input_path, output_directory, visualisations, cal)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()

        dialog.show()

    # =============================================
    # FILE HANDLING
    # =============================================
    def load_raw_image(self) -> None:
        filename = self.file_manager.select_file(self, settings.get('paths.open_file'))

        if not filename: raise ValueError("[Image Loading] No correct file was found.")

        self.graphicsView.display_image(self, filename)

    def export(self) -> None:
        cached = [str(file.stem) for file in self.cache_manager.list_contents()]
        dialog = ExportDialog(cached, self)

        if dialog.exec():

            config = dialog.get_configuration()
            execute_export(config, self.cache_manager)


    # =============================================
    # HELP
    # =============================================

    def open_github(self) -> None:
        if webbrowser.open("https://github.com/Julek-AK/PolarVis"):
            print("Opened the GitHub repository in browser.")
        else:
            raise RuntimeError("[Menu] Failed to open the default web browser.")


def run_main_window():
    app = QApplication(sys.argv)
    window = MainWindow()

    icon = QIcon(str(ICONS_DIR / "polarvis.ico"))

    app.setWindowIcon(icon)
    window.setWindowIcon(icon)

    window.show()
    sys.exit(app.exec())


