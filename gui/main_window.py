# Builtins
import sys

# External
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap

# Internal
from paths import UI_DIR


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_DIR / "testwindow.ui", self)  # loads UI into this instance

        # Example: connect some actions
        self.actionCopy.triggered.connect(lambda: self.on_event("You just tried to copy"))
        self.actionPaste.triggered.connect(lambda: self.on_event("You just tried to paste"))
        self.actionSave.triggered.connect(lambda: self.on_event("You just tried to save"))
        self.actionSingle_Processing.triggered.connect(lambda: self.on_event("You tried to execute single processing"))
        self.actionBatch_Processing.triggered.connect(lambda: self.on_event("You tried to execute batch processing"))
    
        # Create a scene for the graphics view
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)

        self.actionLoad_Image.triggered.connect(self.load_image)

    def on_event(self, text):
        self.label.setText(text)
        self.label.adjustSize()

    def load_image(self):
        # Open file dialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )

        if file_name:
            pixmap = QPixmap(file_name)

            # Clear any previous items
            self.scene.clear()

            # Add new image
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)




def run_main_window():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


