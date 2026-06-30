import sys
import traceback
from PyQt6.QtCore import QObject, pyqtSignal

class ConsoleRedirector(QObject):
    """Redirects any interraction with the python console to instead be displayed within the app"""
    new_text = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # skip empty newlines
            self.new_text.emit(text)

    def flush(self):
        pass


def gui_exception_hook(exctype, value, tb):
    """Captures exceptions from traceback"""
    msg = ''.join(traceback.format_exception(exctype, value, tb))
    print("[Unhandled Exception]", msg)

sys.excepthook = gui_exception_hook