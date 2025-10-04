"""
Initates all connections related to the menu bar
"""


from gui.image_view import load_image
from core.pipeline import SingleProcessPipeline

def setup_menus(window):
           

    # Image loading
    window.actionLoad_Image.triggered.connect(lambda: load_image(window))

    # Image Processing
    pipeline = SingleProcessPipeline()
    window.actionSingle_Processing.triggered.connect(lambda: pipeline.single_process(window))