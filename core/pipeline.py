
import torch
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QFileDialog

from processing.torch_backend import resolve_intensities
from paths import CACHE_DIR
from core.file_manager import CacheManager, ImageFileManager



class Pipeline():
    """Parent class ensuring system architecture is in place and torch is configured correctly"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


class SingleProcess(Pipeline):
    """Simple pipeline constructed for processing individual images"""
    def __init__(self):
        super().__init__()     

    def load_image(self, filepath):
        im = Image.open(filepath).convert("L")
        return np.array(im)

    def single_process(self, file_name):
        cache_manager = CacheManager()  # TODO remove once proper cache manager architecture is in place

        if file_name:
            img = self.load_image(file_name)
            sol_array = resolve_intensities(img, self.device, test=False, verbose=True)
            
            cache_manager.save_array(file_name, sol_array)

class BatchProcess(Pipeline):
    ...