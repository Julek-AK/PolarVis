
import torch
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal

from core.utils import raw_to_metapixel_channels
from core.image_validation import ValidationError, ValidationWarning, validate_calibration


class PipelineWorker(QThread):
    """Threaded worker for safely processing an image"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img_array, calibration, device):
        super().__init__()
        self.img_array = img_array
        self.calibration = calibration
        self.device = device

    # Calibrated metapixel proceesing
    def run(self):
        from processing.torch_backend import calibrated_resolve_polarization

        try:
            sol_array = calibrated_resolve_polarization(
                self.img_array,
                self.calibration,
            )
            self.finished.emit(sol_array)

        except Exception as e:
            self.error.emit(str(e))

    # resolve with a sliding sampling window
    # def run(self):
    #     # TODO add calibration adjusting
    #     from processing.torch_backend import resolve_sliding_polarization
    #     try:
    #         sol_array = resolve_sliding_polarization(
    #             self.img_array,
    #             self.window_size,
    #             self.device
    #         )
    #         self.finished.emit(sol_array)
    #     except Exception as e:
    #         self.error.emit(str(e))


class Pipeline():
    """Maintains system architecture for image processing"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


    def single_process(self, img_array, cal, on_finished, on_error, on_warning=None):
        """Start threaded processing"""

        try:
            validate_calibration(img_array, cal)

        except ValidationWarning as w:
            if on_warning:
                on_warning(str(w))
        
        except ValidationError as e:
            on_error(str(e))
            return

        self.worker = PipelineWorker(img_array, cal, self.device)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.worker.start()