"""
Dedicated class for initializing all the functionality of the MainWindow
"""

# Builtins
import sys
from functools import partial

# External libraries
from PyQt6.QtWidgets import QGraphicsScene

# Internal Support
from gui.image_view import load_image
from core.pipeline import SingleProcess
from core.console_redirector import ConsoleRedirector



class MainWindowInitializer:
    def __init__(self, window):
        self.window = window

    def setup(self):
        self.init_menu_bar()
        self.init_console()
        self.init_image_display()
        
        # currently placeholders
        self.init_file_managers()
        self.init_pipelines()

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
        raise NotImplementedError("this is forbidden from running until pipeline and file manager logic gets fixed and streamlined")
        # Image loading
        self.window.actionLoad_Image.triggered.connect(partial(load_image, self.window))

        # Image Processing
        pipeline = SingleProcess()  # TODO this is a placeholder
        self.window.actionSingle_Processing.triggered.connect(partial(pipeline.single_process, self.window))

    def init_image_display(self):
        self.window.scene = QGraphicsScene(self.window)
        self.window.graphicsView.setScene(self.window.scene)

    def init_file_managers(self):
        ...

    def init_pipelines(self):
        ...
        








