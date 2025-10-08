# Builtins
import sys

# External
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap

# Internal
from paths import UI_DIR
from core.console_redirector import ConsoleRedirector

from gui.menus import setup_menus


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_DIR / "mainwindow2.ui", self)  # loads UI into this instance
    
        # Create a scene for the graphics view
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)

        # Console redirector stuff
        self.stdout_redirector = ConsoleRedirector()
        self.stderr_redirector = ConsoleRedirector()

        self.stdout_redirector.new_text.connect(self.append_console_text)
        self.stderr_redirector.new_text.connect(
            lambda text: self.append_console_text(f"[Error] {text}")
        )

        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector

        setup_menus(self)

    def append_console_text(self, text):
        self.consoleOutput.appendPlainText(text.strip())

    def on_event(self, text):
        self.label.setText(text)
        self.label.adjustSize()






def run_main_window():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


