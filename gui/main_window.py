
from PyQt5 import QtWidgets, uic
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("../resources/ui/testwindow.ui", self)  # loads UI into this instance

        # Example: connect a button
        # self.pushButton.clicked.connect(self.on_button_click)

    def on_button_click(self):
        print("Button clicked!")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


