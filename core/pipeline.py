
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


class ImagePipeline():
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


# TODO make this actually work not in a scuffed manner
class VideoPipeline():
    def __init__(self, cal) -> None:
        self.cal = cal

    def process(self, input_path, output_path):
        import cv2
        from processing.torch_backend import calibrated_resolve_polarization
        from processing.video_processing import arr_to_polarimetric, cleanup_frame

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # read first frame to determine output size
        success, frame = cap.read()
        if not success:
            raise RuntimeError("Could not read video")

        raw = cleanup_frame(frame)
        polarization = calibrated_resolve_polarization(raw, self.cal)
        vis = arr_to_polarimetric(polarization)

        h, w = vis.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        writer.write(vis)

        i = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            raw = cleanup_frame(frame)
            polarization = calibrated_resolve_polarization(raw, self.cal)
            vis = arr_to_polarimetric(polarization)

            writer.write(vis)
            print(f"Processed frame {i} out of {count}")
            i += 1

        cap.release()
        writer.release()