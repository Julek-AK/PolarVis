from ..processing.torch_backend import (
    calibrated_resolve_polarization,
    batch_resolve_polarization,
)

    
class SingleProcessor:
    def __init__(self, calibration, device):

        self.calibration = calibration
        self.device = device

    def process(self, image, progress=None):

        result = calibrated_resolve_polarization(image, self.calibration, self.device)

        if progress:
            progress(1, 1)

        return result
    

class BatchProcessor:
    def __init__(self, calibration, device):
        
        self.calibration = calibration
        self.device = device

    def process(self, images, progress=None):  # TODO invalid progress tracking
        
        result = batch_resolve_polarization(images, self.calibration, self.device)

        if progress:
            progress(1, 1)

        return result


class VideoProcessor:
    ...

    