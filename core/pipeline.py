
import torch
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal

from processing.torch_backend import resolve_intensities


class PipelineWorker(QThread):
    """Threaded worker for safely processing an image"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img_array, device):
        super().__init__()
        self.img_array = img_array
        self.device = device

    def run(self):
        from processing.torch_backend import resolve_intensities
        try:
            sol_array = resolve_intensities(self.img_array, self.device, test=False, verbose=True)
            self.finished.emit(sol_array)
        except Exception as e:
            self.error.emit(str(e))


class Pipeline():
    """Maintains system architecture for image processing"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # def single_process(self, img_array: NDArray) -> NDArray:
    #     sol_array = resolve_intensities(img_array, self.device, test=False, verbose=True)
        
    #     return sol_array

    def single_process(self, img_array, on_finished, on_error):
        """Start threaded processing"""
        self.worker = PipelineWorker(img_array, self.device)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.worker.start()