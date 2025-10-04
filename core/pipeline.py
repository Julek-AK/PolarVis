
import torch
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QFileDialog

from processing.torch_backend import resolve_intensities
from paths import CACHE_DIR




class SingleProcessPipeline():
    def __init__(self):     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


    def load_image(self, filepath):
        im = Image.open(filepath).convert("L")
        return np.array(im)


    def single_process(self, window):
        file_name, _ = QFileDialog.getOpenFileName(
            window,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )

        if file_name:
            img = self.load_image(file_name)
            sol_array = resolve_intensities(img, self.device, test=False, verbose=True)
            np.save(CACHE_DIR / "first_cache.npy", sol_array)