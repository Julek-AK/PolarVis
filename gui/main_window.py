# Builtins
import sys

# External
from PyQt6 import QtWidgets, uic

# Internal
from paths import UI_DIR
from core.console_redirector import ConsoleRedirector

from gui.window_init import MainWindowInitializer


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(UI_DIR / "mainwindow2.ui", self)  # loads UI into this instance

        MainWindowInitializer(self).setup()


    # Console Operations
    def append_console_text(self, text) -> None:
        self.consoleOutput.appendPlainText(text.strip())


    # Processing Operations
    def run_single_process(self) -> None:
        ...

    def run_batch_process(self) -> None:
        ...


    # Visualisation Operations
    ...



def run_main_window():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


